import math
import warnings
import itertools
import random
import logging
import copy
from typing import Optional
import os
import tempfile
import dataclasses
import numpy as np
from PIL import Image
from random import randint
from argparse import ArgumentParser
import shlex

import torch
from torch import nn

from nerfbaselines import (
    Method, MethodInfo, RenderOutput, ModelInfo,
    Dataset,
    Cameras, camera_model_to_int,
)

from arguments import ModelParams, PipelineParams, OptimizationParams  # type: ignore
from gaussian_renderer import render  # type: ignore
from scene import GaussianModel  # type: ignore
import scene.dataset_readers  # type: ignore
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
from scene.dataset_readers import storePly, fetchPly  # type: ignore
from scene.dataset_readers import CameraInfo as _old_CameraInfo  # type: ignore
from utils.general_utils import safe_state  # type: ignore
from utils.graphics_utils import fov2focal  # type: ignore
from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.sh_utils import SH2RGB, eval_sh  # type: ignore
from scene import Scene, sceneLoadTypeCallbacks  # type: ignore
from utils import camera_utils  # type: ignore
from utils.general_utils import PILtoTorch  # type: ignore
from encoders import AppearanceTransform, initialize_weights


@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()

    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1

    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image


def normalize_to_01(tensor, eps=1e-8):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

#
# Patch Gaussian Splatting to include sampling masks
# Also, fix cx, cy (ignored in mip-splatting)
#
# Patch loadCam to include sampling mask
_old_loadCam = camera_utils.loadCam
def loadCam(args, id, cam_info, resolution_scale):
    camera = _old_loadCam(args, id, cam_info, resolution_scale)

    sampling_mask = None
    if cam_info.sampling_mask is not None:
        sampling_mask = PILtoTorch(cam_info.sampling_mask, (camera.image_width, camera.image_height))
    setattr(camera, "sampling_mask", sampling_mask)
    setattr(camera, "_patched", True)

    # Fix cx, cy (ignored in mip-splatting)
    camera.focal_x = fov2focal(cam_info.FovX, camera.image_width)
    camera.focal_y = fov2focal(cam_info.FovY, camera.image_height)
    camera.cx = cam_info.cx
    camera.cy = cam_info.cy
    camera.projection_matrix = getProjectionMatrixFromOpenCV(
        camera.image_width,
        camera.image_height,
        camera.focal_x,
        camera.focal_y,
        camera.cx,
        camera.cy,
        camera.znear,
        camera.zfar).transpose(0, 1).cuda()
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)

    return camera
camera_utils.loadCam = loadCam


# Patch CameraInfo to add sampling mask
class CameraInfo(_old_CameraInfo):
    def __new__(cls, *args, sampling_mask=None, cx, cy, **kwargs):
        self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
        self.sampling_mask = sampling_mask
        self.cx = cx
        self.cy = cy
        return self
scene.dataset_readers.CameraInfo = CameraInfo


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, sampling_mask=None, scale_coords=None):
    pose = np.copy(pose)
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
    if scale_coords is not None:
        T = T * scale_coords
    R = np.transpose(R)

    width, height = image_size
    fx, fy, cx, cy = intrinsics
    if image is None:
        image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    return CameraInfo(
        uid=idx, R=R, T=T,
        FovX=focal2fov(float(fx), float(width)),
        FovY=focal2fov(float(fy), float(height)),
        image=image, image_path=image_path, image_name=image_name,
        width=int(width), height=int(height),
        sampling_mask=sampling_mask,
        cx=cx, cy=cy)


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False, scale_coords=None):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None, translate=None), ply_path=None)
    assert np.all(dataset["cameras"].camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

    cam_infos = []
    for idx, extr in enumerate(dataset["cameras"].poses):
        del extr
        intrinsics = dataset["cameras"].intrinsics[idx]
        pose = dataset["cameras"].poses[idx]
        image_path = dataset["image_paths"][idx] if dataset["image_paths"] is not None else f"{idx:06d}.png"
        image_name = (
            os.path.relpath(str(dataset["image_paths"][idx]), str(dataset["image_paths_root"])) if dataset["image_paths"] is not None and dataset["image_paths_root"] is not None else os.path.basename(image_path)
        )

        w, h = dataset["cameras"].image_sizes[idx]
        im_data = dataset["images"][idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4]) * bg
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        if not white_background and dataset["metadata"].get("id") == "blender":
            warnings.warn("Blender scenes are expected to have white background. If the background is not white, please set white_background=True in the dataset loader.")
        elif white_background and dataset["metadata"].get("id") != "blender":
            warnings.warn("white_background=True is set, but the dataset is not a blender scene. The background may not be white.")
        image = Image.fromarray(im_data)
        sampling_mask = None
        # if dataset["masks"] is not None:
        #     sampling_mask = Image.fromarray((dataset["masks"][idx] * 255).astype(np.uint8))

        cam_info = _load_caminfo(
            idx, pose, intrinsics,
            image_name=image_name,
            image_path=image_path,
            image_size=(w, h),
            image=image,
            sampling_mask=sampling_mask,
            scale_coords=scale_coords,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset["points3D_xyz"]
    if scale_coords is not None:
        points3D_xyz = points3D_xyz * scale_coords
    points3D_rgb = dataset["points3D_rgb"]
    if points3D_xyz is None and dataset["metadata"].get("id", None) == "blender":
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L221C4-L221C4
        num_pts = 100_000
        logging.info(f"generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        points3D_xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        points3D_rgb = (SH2RGB(shs) * 255).astype(np.uint8)

    storePly(os.path.join(tempdir, "scene.ply"), points3D_xyz, points3D_rgb)
    pcd = fetchPly(os.path.join(tempdir, "scene.ply"))
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=cam_infos, test_cameras=[], nerf_normalization=nerf_normalization, ply_path=os.path.join(tempdir, "scene.ply"))
    return scene_info


def _config_overrides_to_args_list(args_list, config_overrides):
    for k, v in config_overrides.items():
        if str(v).lower() == "true":
            v = True
        if str(v).lower() == "false":
            v = False
        if isinstance(v, bool):
            if v:
                if f'--{k}' not in args_list:
                    args_list.append(f'--{k}')
            else:
                if f'--{k}' in args_list:
                    args_list.remove(f'--{k}')
        elif f'--{k}' in args_list:
            args_list[args_list.index(f'--{k}') + 1] = str(v)
        else:
            args_list.append(f"--{k}")
            args_list.append(str(v))


def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _format_output(output, options):
    del options
    return {
        k: v.cpu().numpy() for k, v in output.items()
    }


class AsymmetricGS(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.step = 0

        # Setup parameters
        self._args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        self._loaded_step = None
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())

            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            self._loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(str(checkpoint)) if x.startswith("chkpnt-"))[-1]

        # Fix old checkpoints
        if "--resolution" not in self._args_list:
            self._args_list.extend(("--resolution", "1"))

        if self.checkpoint is None and config_overrides is not None:
            _config_overrides_to_args_list(self._args_list, config_overrides)

        self._load_config()

        self._setup(train_dataset)

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--scale_coords", type=float, default=None, help="Scale the coords")
        args = parser.parse_args(self._args_list)
        self.dataset = lp.extract(args)
        self.dataset.scale_coords = args.scale_coords
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

    def _setup(self, train_dataset):
        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        self.gaussians_1 = GaussianModel(self.dataset.sh_degree)
        self.gaussians_2 = GaussianModel(self.dataset.sh_degree)
        self.scene_1 = self._build_scene(train_dataset, self.gaussians_1)
        self.scene_2 = self._build_scene(train_dataset, self.gaussians_2)
        if train_dataset is not None:
            self.gaussians_1.training_setup(self.opt)
            self.gaussians_2.training_setup(self.opt)
        filter_3D = None
        if train_dataset is None or self.checkpoint:
            info = self.get_info()
            _modeldata = torch.load(str(self.checkpoint) + f"/chkpnt-{info.get('loaded_step')}.pth", weights_only=False)
            if len(_modeldata) == 3:
                (model_params, filter_3D, self.step) = _modeldata
            else:
                warnings.warn("Old checkpoint format! The performance will be suboptimal. Please fix the checkpoint or restart the training.")
                (model_params, self.step) = _modeldata
            self.gaussians_1.restore(model_params, self.opt)
            # NOTE: this is not handled in the original code
            self.gaussians_1.filter_3D = filter_3D

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._viewpoint_stack_1 = []
        self._viewpoint_stack_2 = []
        self.trainCameras_1 = None
        self.highresolution_index_1 = None
        self.trainCameras_2 = None
        self.highresolution_index_2 = None

        if train_dataset is not None:
            self.trainCameras_1 = self.scene_1.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack_1):
                raise RuntimeError("could not patch loadCam!")
            # highresolution index
            self.highresolution_index_1 = []
            for index, camera in enumerate(self.trainCameras_1):
                if camera.image_width >= 800:
                    self.highresolution_index_1.append(index)

            self.trainCameras_2 = self.scene_2.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack_2):
                raise RuntimeError("could not patch loadCam!")
            # highresolution index
            self.highresolution_index_2 = []
            for index, camera in enumerate(self.trainCameras_2):
                if camera.image_width >= 800:
                    self.highresolution_index_2.append(index)

        if filter_3D is None:
            if self.trainCameras_1 is None:
                raise RuntimeError("Old checkpoint format! Please run nerfbaselines fix-checkpoint first.")
            self.gaussians_1.compute_3D_filter(cameras=self.trainCameras_1)

            if self.trainCameras_2 is None:
                raise RuntimeError("Old checkpoint format! Please run nerfbaselines fix-checkpoint first.")
            self.gaussians_2.compute_3D_filter(cameras=self.trainCameras_2)

        # todo tuning these models
        self.encoding_dim = 32
        appearance_n_fourier_freqs = 4
        self.appearance_transform = AppearanceTransform(global_encoding_dim=self.encoding_dim, local_encoding_dim=appearance_n_fourier_freqs*6).cuda()
        self.appearance_transform.apply(initialize_weights)
        self.appearance_transform_optimizer = torch.optim.Adam(self.appearance_transform.parameters(), lr=0.0005, eps=1e-15)

        # todo make this config
        if train_dataset is not None:
            self.scene_name = train_dataset["image_paths_root"].split("/")[-2]
            self.dataset_name = train_dataset["image_paths_root"].split("/")[-3]
            self.abs_root = "/".join(train_dataset["image_paths_root"].split("/")[:-3])

            # Load preprocessed raw masks for multi-cue adaptive mask
            self.raw_masks = np.load(os.path.join(self.abs_root, self.dataset_name, self.scene_name, f"multi_cue_masks_{self.scene_name}.npz"))
            # self.raw_masks = np.load(f"/home/lorentz/Project/Code/mip-splatting_ema/self_created_data_mask/semantic_sam_masks_unpooled_filtered_{self.scene_name}_23sam.npz")

            # Learnable mask
            train_image_number = len(train_dataset["images"])
            self.learnable_mask_logits = {
                cam.image_name: torch.ones((1, cam.image_height, cam.image_width)).cuda().requires_grad_()
                for cam in self.scene_1.train_cameras[1.0]
            }
            self.learnable_mask_optimizer = torch.optim.Adam([item for item in self.learnable_mask_logits.items()], lr=0.1, eps=1e-15)
            # Treat positions around correspondence points as static
            # self.correspond_points = np.load(os.path.join(self.abs_root, self.dataset_name, self.scene_name, f"dilated_correspond_points_{self.scene_name}.npz"))

            self.dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').cuda()

            # todo config appearance encoding
            # Adopting appearance modeling in phototourism dataset
            if self.dataset_name == "phototourism":
                self.use_color_transform = True
                self.global_encoding_1 = torch.normal(mean=0, std=0.01, size=(train_image_number, self.encoding_dim)).cuda().requires_grad_()
                self.global_encoding_optimizer_1 = torch.optim.Adam([{'params': [self.global_encoding_1]}], lr=0.001, eps=1e-15)
                self.global_encoding_2 = torch.normal(mean=0, std=0.01, size=(train_image_number, self.encoding_dim)).cuda().requires_grad_()
                self.global_encoding_optimizer_2 = torch.optim.Adam([{'params': [self.global_encoding_2]}], lr=0.001, eps=1e-15)
            else:
                self.use_color_transform = False
                self.global_encoding_1 = torch.zeros((train_image_number, self.encoding_dim)).cuda().requires_grad_(False)
                self.global_encoding_2 = torch.zeros((train_image_number, self.encoding_dim)).cuda().requires_grad_(False)

    def get_learnable_mask(self, id, size):
        learnable_mask_logit = self.learnable_mask_logits[id].view(-1, 1, size[0], size[1])
        activated_mask = torch.sigmoid(8 * learnable_mask_logit)
        return activated_mask.squeeze(0)

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be set by the registry
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color",),
        )

    def get_info(self) -> ModelInfo:
        hparams = flatten_hparams(dict(itertools.chain(vars(self.dataset).items(), vars(self.opt).items(), vars(self.pipe).items())))
        for k in ("source_path", "resolution", "eval", "images", "model_path", "data_device"):
            hparams.pop(k, None)
        return ModelInfo(
            num_iterations=self.opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )

    def _build_scene(self, dataset, gaussians):
        opt = copy.copy(self.dataset)
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(td + "/sparse")
            opt.source_path = td  # To trigger colmap loader
            opt.model_path = td if dataset is not None else str(self.checkpoint)
            backup = sceneLoadTypeCallbacks["Colmap"]
            try:
                info = self.get_info()
                def colmap_loader(*args, **kwargs):
                    del args, kwargs
                    return _convert_dataset_to_gaussian_splatting(dataset, td, white_background=self.dataset.white_background, scale_coords=self.dataset.scale_coords)
                sceneLoadTypeCallbacks["Colmap"] = colmap_loader
                loaded_step = info.get("loaded_step")
                assert dataset is not None or loaded_step is not None, "Either dataset or loaded_step must be set"
                scene = Scene(opt, gaussians, load_iteration=str(loaded_step) if dataset is None else None)
                # NOTE: This is a hack to match the RNG state of GS on 360 scenes
                _tmp = list(range((len(next(iter(scene.train_cameras.values()))) + 6) // 7))
                random.shuffle(_tmp)
                return scene
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup

    def optimize_embedding(self, dataset, *, embedding):
        device = self.gaussians_1.get_xyz.device
        camera = dataset["cameras"].item()
        assert np.all(camera.camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

        viewpoint_info = _load_caminfo(0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self.dataset.scale_coords)
        viewpoint_cam = loadCam(self.dataset, 0, viewpoint_info, 1.0)

        if True:
            global_encoding_np_1 = self._optimize_single_gaussian(self.gaussians_1, viewpoint_cam, dataset, device)
            return {
                "embedding": global_encoding_np_1,
            }
        else:
            raise NotImplementedError("Trying to optimize embedding with appearance_enabled=False")

    def _optimize_single_gaussian(self, gaussians, viewpoint_cam, dataset, device):
        gaussians.freeze()
        i = 0
        losses, psnrs, mses = [], [], []

        if self.dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
        else:
            subpixel_offset = None

        global_encoding_param = torch.nn.Parameter(torch.zeros_like(self.global_encoding_1[0]).to(device).requires_grad_(True))
        # todo add this to config learning rate
        optimizer = torch.optim.Adam([global_encoding_param], lr=0.1)

        gt_image = torch.tensor(convert_image_dtype(dataset["images"][i], np.float32), dtype=torch.float32, device=device).permute(2, 0, 1)
        gt_mask = torch.tensor(convert_image_dtype(dataset["masks"][i], np.float32), dtype=torch.float32, device=device)[..., None].permute(2, 0, 1) if dataset["masks"] is not None else None

        with torch.enable_grad():
            # todo add this to config
            app_optim_type = 'dssim+l1'
            loss_mult = None
            if app_optim_type.endswith("-scaled"):
                app_optim_type = app_optim_type[:-7]
            # todo add this to config
            global_encoding_optim_iters = 128
            for _ in range(global_encoding_optim_iters):
                optimizer.zero_grad()

                bg = torch.zeros((3,), dtype=torch.float32, device="cuda")
                image = self._render_with_appearance_encoding(viewpoint_cam, gaussians, global_encoding_param, bg, subpixel_offset)["render"]

                if gt_mask is not None:
                    image = scale_grads(image, gt_mask.float())
                if loss_mult is not None:
                    image = scale_grads(image, loss_mult)

                mse = torch.nn.functional.mse_loss(image, gt_image)

                if app_optim_type == "mse":
                    loss = mse
                elif app_optim_type == "dssim+l1":
                    Ll1 = torch.nn.functional.l1_loss(image, gt_image)
                    ssim_value = ssim(image, gt_image, size_average=True)
                    loss = (
                            (1.0 - self.opt.lambda_dssim) * Ll1 +
                            self.opt.lambda_dssim * (1.0 - ssim_value)
                    )
                else:
                    raise ValueError(f"Unknown appearance optimization type {app_optim_type}")
                loss.backward()
                optimizer.step()

                losses.append(loss.detach().cpu().item())
                mses.append(mse.detach().cpu().item())
                psnrs.append(20 * math.log10(1.0) - 10 * torch.log10(mse).detach().cpu().item())

        if gaussians.optimizer is not None:
            gaussians.optimizer.zero_grad()
        global_encoding = global_encoding_param
        global_encoding_np = global_encoding_param.detach().cpu().numpy()

        torch.cuda.empty_cache()
        gaussians.unfreeze()

        return global_encoding_np

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        # del options
        camera = camera.item()
        assert np.all(camera.camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

        with torch.no_grad():
            viewpoint_cam = _load_caminfo(0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self.dataset.scale_coords)
            viewpoint = loadCam(self.dataset, 0, viewpoint_cam, 1.0)

            if self.dataset.ray_jitter:
                subpixel_offset = torch.rand((int(viewpoint.image_height), int(viewpoint.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            else:
                subpixel_offset = None

            if self.use_color_transform:
                if options is not None:
                    _np_embedding_1 = (options or {}).get("embedding", None)
                else:
                    _np_embedding_1 = np.zeros((1, self.encoding_dim), dtype=np.float32)
                global_encoding_1 = torch.from_numpy(_np_embedding_1).cuda()

                image_1 = torch.clamp(self._render_with_appearance_encoding(viewpoint, self.gaussians_1, global_encoding_1, self.background, subpixel_offset)["render"], 0.0, 1.0)
            else:
                image_1 = torch.clamp(render(viewpoint, self.gaussians_1, self.pipe, self.background, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset)["render"], 0.0, 1.0)

            image = image_1
            color = image.detach().permute(1, 2, 0)
            return _format_output({"color": color}, options)

    def train_iteration(self, step):
        assert self.trainCameras_1 is not None, "Model was not initialized with a training dataset"
        assert self.trainCameras_2 is not None, "Model was not initialized with a training dataset"
        assert self.highresolution_index_1 is not None, "Model was not initialized with a training dataset"
        assert self.highresolution_index_2 is not None, "Model was not initialized with a training dataset"

        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        self.gaussians_1.update_learning_rate(iteration)
        self.gaussians_2.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians_1.oneupSHdegree()
            self.gaussians_2.oneupSHdegree()

        if not self._viewpoint_stack_1:
            loadCam.was_called = False  # type: ignore
            self._viewpoint_stack_1 = self.scene_1.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack_1):
                raise RuntimeError("could not patch loadCam!")
        if not self._viewpoint_stack_2:
            loadCam.was_called = False  # type: ignore
            self._viewpoint_stack_2 = self.scene_1.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack_2):
                raise RuntimeError("could not patch loadCam!")

        viewpoint_cam_1 = self._viewpoint_stack_1[randint(0, len(self._viewpoint_stack_1) - 1)]
        viewpoint_cam_2 = self._viewpoint_stack_2[randint(0, len(self._viewpoint_stack_2) - 1)]

        # Share render inputs
        bg = torch.rand((3), device="cuda") if getattr(self.opt, 'random_background', False) else self.background
        if self.dataset.ray_jitter:
            subpixel_offset_1 = torch.rand((int(viewpoint_cam_1.image_height), int(viewpoint_cam_1.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            subpixel_offset_2 = torch.rand((int(viewpoint_cam_2.image_height), int(viewpoint_cam_2.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
        else:
            subpixel_offset_1 = None
            subpixel_offset_2 = None

        if self.use_color_transform:
            # Phototourism
            image_name_1 = viewpoint_cam_1.image_name
            image_name_2 = viewpoint_cam_2.image_name

            # Render 1
            global_encoding_1 = self.global_encoding_1[viewpoint_cam_1.uid]
            render_pkg_1 = self._render_with_appearance_encoding(viewpoint_cam_1, self.gaussians_1, global_encoding_1, bg, subpixel_offset_1)
            image_1, viewspace_point_tensor_1, visibility_filter_1, radii_1 = render_pkg_1["render"], render_pkg_1["viewspace_points"], render_pkg_1["visibility_filter"], render_pkg_1["radii"]
            render_pkg_raw_1 = render(viewpoint_cam_1, self.gaussians_1, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_1)
            image_raw_1 = render_pkg_raw_1["render"]
            image_raw_2_g1 = render(viewpoint_cam_2, self.gaussians_1, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_2)["render"]

            # Render 2
            global_encoding_2 = self.global_encoding_2[viewpoint_cam_2.uid]
            render_pkg_2 = self._render_with_appearance_encoding(viewpoint_cam_2, self.gaussians_2, global_encoding_2, bg, subpixel_offset_2)
            image_2, viewspace_point_tensor_2, visibility_filter_2, radii_2 = render_pkg_2["render"], render_pkg_2["viewspace_points"], render_pkg_2["visibility_filter"], render_pkg_2["radii"]
            render_pkg_raw_2 = render(viewpoint_cam_2, self.gaussians_2, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_2)
            image_raw_2 = render_pkg_raw_2["render"]
            image_raw_1_g2 = render(viewpoint_cam_1, self.gaussians_2, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_1)["render"]

            # Masked reconstruction loss 1 (multi-cue adaptive mask by default)
            gt_image_1 = viewpoint_cam_1.original_image.cuda()
            sampling_mask_1 = self._mask_generation(self.pipe.apply_mask[0], image_1, gt_image_1, image_name_1)
            # sample gt_image with subpixel offset
            if self.dataset.resample_gt_image:
                gt_image_1 = create_offset_gt(gt_image_1, subpixel_offset_1)
                sampling_mask_1 = create_offset_gt(sampling_mask_1, subpixel_offset_1) if sampling_mask_1 is not None else None
            loss = self._masked_loss(image_1, gt_image_1, sampling_mask_1)

            # Masked reconstruction loss 2 (learnable mask by default)
            gt_image_2 = viewpoint_cam_2.original_image.cuda()
            sampling_mask_2 = self._mask_generation(self.pipe.apply_mask[1], image_2, gt_image_2, image_name_2)
            # sample gt_image with subpixel offset
            if self.dataset.resample_gt_image:
                gt_image_2 = create_offset_gt(gt_image_2, subpixel_offset_2)
                sampling_mask_2 = create_offset_gt(sampling_mask_2, subpixel_offset_2) if sampling_mask_2 is not None else None
            loss += self._masked_loss(image_2, gt_image_2, sampling_mask_2)

            # Mutual consistency
            if iteration > self.pipe.warmup:
                loss += self.opt.lambda_mul * l1_loss(image_raw_2, image_raw_2_g1)
                loss += self.opt.lambda_mul * l1_loss(image_raw_1_g2, image_raw_1)

            # Mask learning loss
            if self.pipe.apply_mask[0] == 2:
                loss += self.opt.lambda_mask * self._mask_error(sampling_mask_1, image_1, gt_image_1, image_name_1)
            if self.pipe.apply_mask[1] == 2:
                loss += self.opt.lambda_mask * self._mask_error(sampling_mask_2, image_2, gt_image_2, image_name_2)
                
        else:
            # Onthego and Robustnerf
            image_name_1 = viewpoint_cam_1.image_name
            image_name_2 = viewpoint_cam_2.image_name

            # Render 1
            render_pkg_1 = render(viewpoint_cam_1, self.gaussians_1, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_1)
            image_1, viewspace_point_tensor_1, visibility_filter_1, radii_1 = render_pkg_1["render"], render_pkg_1["viewspace_points"], render_pkg_1["visibility_filter"], render_pkg_1["radii"]
            image_2_g1 = render(viewpoint_cam_2, self.gaussians_1, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_2)["render"]

            # Render 2
            render_pkg_2 = render(viewpoint_cam_2, self.gaussians_2, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_2)
            image_2, viewspace_point_tensor_2, visibility_filter_2, radii_2 = render_pkg_2["render"], render_pkg_2["viewspace_points"], render_pkg_2["visibility_filter"], render_pkg_2["radii"]
            image_1_g2 = render(viewpoint_cam_1, self.gaussians_2, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset_1)["render"]

            # Masked reconstruction loss 1 (multi-cue adaptive mask by default)
            gt_image_1 = viewpoint_cam_1.original_image.cuda()
            sampling_mask_1 = self._mask_generation(self.pipe.apply_mask[0], image_1, gt_image_1, image_name_1)
            # sample gt_image with subpixel offset
            if self.dataset.resample_gt_image:
                gt_image_1 = create_offset_gt(gt_image_1, subpixel_offset_1)
                sampling_mask_1 = create_offset_gt(sampling_mask_1, subpixel_offset_1) if sampling_mask_1 is not None else None
            loss = self._masked_loss(image_1, gt_image_1, sampling_mask_1)

            # Masked reconstruction loss 2 (learnable mask by default)
            gt_image_2 = viewpoint_cam_2.original_image.cuda()
            sampling_mask_2 = self._mask_generation(self.pipe.apply_mask[1], image_2, gt_image_2, image_name_2)
            # sample gt_image with subpixel offset
            if self.dataset.resample_gt_image:
                gt_image_2 = create_offset_gt(gt_image_2, subpixel_offset_2)
                sampling_mask_2 = create_offset_gt(sampling_mask_2, subpixel_offset_2) if sampling_mask_2 is not None else None
            loss += self._masked_loss(image_2, gt_image_2, sampling_mask_2)

            # Mutual consistency
            if iteration > self.pipe.warmup:
                loss += self.opt.lambda_mul * l1_loss(image_2, image_2_g1)
                loss += self.opt.lambda_mul * l1_loss(image_1_g2, image_1)

            # Mask learning loss
            if self.pipe.apply_mask[0] == 2:
                loss += self.opt.lambda_mask * self._mask_error(sampling_mask_1, image_1, gt_image_1, image_name_1)
            if self.pipe.apply_mask[1] == 2:
                loss += self.opt.lambda_mask * self._mask_error(sampling_mask_2, image_2, gt_image_2, image_name_2)

        loss.backward()

        with torch.no_grad():
            psnr_value = 10 * torch.log10(1 / torch.mean((image_1 - gt_image_1) ** 2))

            if self.use_color_transform:
                psnr_value_cross = 10 * torch.log10(1 / torch.mean((image_raw_1_g2 - image_raw_1) ** 2))
            else:
                psnr_value_cross = 10 * torch.log10(1 / torch.mean((image_1 - image_1_g2) ** 2))


            metrics = {
                # "l1_loss_1": Ll1_1.detach().cpu().item(),
                # "l1_loss_2": Ll1_2.detach().cpu().item(),
                "loss": loss.detach().cpu().item(),
                "psnr": psnr_value.detach().cpu().item(),
                "psnr_cross": psnr_value_cross.detach().cpu().item(),
                "num_gaussians": len(self.gaussians_1._xyz),
            }

            # Densification
            self._densification(iteration, self.gaussians_1, visibility_filter_1, radii_1, viewspace_point_tensor_1, self.scene_1, self.trainCameras_1)
            self._densification(iteration, self.gaussians_2, visibility_filter_2, radii_2, viewspace_point_tensor_2, self.scene_2, self.trainCameras_2)

            # Optimizer step
            if iteration < self.opt.iterations:
                self.gaussians_1.optimizer.step()
                self.gaussians_1.optimizer.zero_grad(set_to_none=True)
                self.gaussians_2.optimizer.step()
                self.gaussians_2.optimizer.zero_grad(set_to_none=True)

                # todo config this
                if self.use_color_transform:
                    self.appearance_transform_optimizer.step()
                    self.appearance_transform_optimizer.zero_grad(set_to_none=True)
                    self.global_encoding_optimizer_1.step()
                    self.global_encoding_optimizer_1.zero_grad(set_to_none=True)
                    self.global_encoding_optimizer_2.step()
                    self.global_encoding_optimizer_2.zero_grad(set_to_none=True)

                # skip static points
                # if self.pipe.apply_mask[0] == 2:
                #     self.learnable_mask_logits[image_name_1].grad *= torch.from_numpy(
                #         1.0 - self.correspond_points[image_name_1]).to(gt_image_1.dtype).cuda()

                # if self.pipe.apply_mask[1] == 2:
                #     self.learnable_mask_logits[image_name_2].grad *= torch.from_numpy(
                #         1.0 - self.correspond_points[image_name_2]).to(gt_image_2.dtype).cuda()

                if self.pipe.apply_mask[0] == 2 or self.pipe.apply_mask[1] == 2:
                    self.learnable_mask_optimizer.step()
                    self.learnable_mask_optimizer.zero_grad(set_to_none=True)

        self.step = self.step + 1
        return metrics

    def _mask_error(self, sampling_mask_1, image_1, gt_image_1, image_name_1):
        # transform to [0, 1]
        transformed_feature_residual = self._feature_residual(image_1.detach(), gt_image_1)
        transformed_feature_residual = normalize_to_01(transformed_feature_residual)
        # skip static points
        # transformed_feature_residual -= torch.from_numpy(self.correspond_points[image_name_1]).to(gt_image_1.dtype).cuda()
        transformed_feature_residual = transformed_feature_residual.clamp(min=0.0, max=1.0)

        return l1_loss(sampling_mask_1, 1.0 - transformed_feature_residual)

    def _mask_generation(self, mask_method, image, gt_image, image_name):
        if mask_method == 0:
            return None
        elif mask_method == 1:
            return self._sampling_mask_from_residual(image, gt_image, image_name)
        elif mask_method == 2:
            return self.get_learnable_mask(image_name, (gt_image.size()[1], gt_image.size()[2]))
        return None

    def _masked_loss(self, render_image, gt_image, sampling_mask):
        # Apply mask
        if sampling_mask is not None:
            render_image = render_image * sampling_mask + (1.0 - sampling_mask) * render_image.detach()
        # Masked L1 adn SSIM
        Ll1 = l1_loss(render_image, gt_image)
        ssim_value = ssim(render_image, gt_image)
        # Weighting
        return (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)

    def _sampling_mask_from_residual(self, render_image, gt_image, image_name):
        '''
        Multi-cue adaptive mask
        '''
        render_image = render_image.detach()
        gt_image = gt_image.detach()
        image_size = (gt_image.size(1), gt_image.size(2))

        # Pixel level residual
        residual_map_1 = torch.mean(torch.abs(render_image - gt_image), dim=0, keepdim=True)
        high_residual_map_1 = (residual_map_1 > torch.quantile(residual_map_1, 0.9)) * 1.0 # todo need to tunning threshold so far 0.5 ok

        # Feature level residual
        residual_map_2 = self._feature_residual(render_image, gt_image)
        high_residual_map_2 = (residual_map_2 > torch.quantile(residual_map_2, 0.9)) * 1.0 # residual_map.mean()
        high_residual_map = (high_residual_map_1 + high_residual_map_2) > 1
        high_residual_mask_threshold = high_residual_map.sum() / (image_size[0] * image_size[1])

        # Load raw masks
        raw_sampling_masks_resized = torch.from_numpy(self.raw_masks[image_name]).to(gt_image.dtype).cuda()

        # Select masks contain high residual area (reaching certain density threshold)
        high_residual_within_masks = (raw_sampling_masks_resized * high_residual_map).sum(dim=(-2, -1))
        masks_area = raw_sampling_masks_resized.sum(dim=(-2, -1))
        high_residual_within_masks_per_area = high_residual_within_masks / masks_area
        high_residual_masks_resized = raw_sampling_masks_resized[high_residual_within_masks_per_area > high_residual_mask_threshold]

        return 1.0 - high_residual_masks_resized.any(dim=0, keepdims=True) * 1.0

    def _feature_residual(self, render_image, gt_image):
        # todo modify this
        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).cuda() / 255
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).cuda() / 255

        image1 = (render_image - img_norm_mean[None, :, None, None]) / img_norm_std[None, :, None, None]
        image2 = (gt_image - img_norm_mean[None, :, None, None]) / img_norm_std[None, :, None, None]

        # resize for dino
        h = image1.size(-2)
        w = image1.size(-1)
        nh = (h // 14 + 1) * 14
        nw = (w // 14 + 1) * 14

        image1 = nn.functional.interpolate(image1, (nh, nw)).to(torch.float32)
        image2 = nn.functional.interpolate(image2, (nh, nw)).to(torch.float32)

        e1 = self.dinov2_vits14_reg.patch_embed(image1)
        e2 = self.dinov2_vits14_reg.patch_embed(image2)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        output = 1.0 - cos(e1, e2)
        output = output.view(-1, nh // 14, nw // 14)
        return nn.functional.interpolate(output.unsqueeze(0), (h, w)).squeeze(0)

    def _render_with_appearance_encoding(self, viewpoint_cam, gaussians, global_encoding, bg, subpixel_offset):
        # Evaluate color
        shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
        dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        eval_color = torch.clamp_min(sh2rgb + 0.5, 0.0)

        # color transform given appearance encodings
        override_color = self.appearance_transform(eval_color, global_encoding.repeat(eval_color.size(0), 1), gaussians.get_local_encoding).clamp(min=0.0, max=1.0)

        # rendering with transformed color
        return render(viewpoint_cam, gaussians, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset, override_color=override_color)

    def _densification(self, iteration, gaussians, visibility_filter, radii, viewspace_point_tensor, correspond_scene, trainCameras):
        if iteration < self.opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, correspond_scene.cameras_extent, size_threshold)
                gaussians.compute_3D_filter(cameras=trainCameras)

            if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                gaussians.reset_opacity()

        if iteration % 100 == 0 and iteration > self.opt.densify_until_iter:
            if iteration < self.opt.iterations - 100:
                # don't update in the end of training
                gaussians.compute_3D_filter(cameras=trainCameras)

    def save(self, path: str):
        if self.checkpoint:
            return

        self.gaussians_1.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud.ply"))
        self.gaussians_2.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud_2.ply"))
        torch.save((self.gaussians_1.capture(), self.gaussians_1.filter_3D, self.step), str(path) + f"/chkpnt-{self.step}.pth")
        torch.save(self.learnable_mask_logits, str(path) + f"/learnable_mask-{self.step}.pth")

        if self.use_color_transform:
            torch.save(self.global_encoding_1, str(path) + f"/global_appearance_1-{self.step}.pth")
            torch.save(self.global_encoding_2, str(path) + f"/global_appearance_2-{self.step}.pth")
            torch.save(self.appearance_transform.state_dict(), str(path) + f"/appearance_transform-{self.step}.pth")

        with open(str(path) + "/args.txt", "w", encoding="utf8") as f:
            f.write(" ".join(shlex.quote(x) for x in self._args_list))

    def export_demo(self, path: str, *, options=None):
        from ._gaussian_splatting_demo import export_demo

        options = (options or {}).copy()
        options["antialiased"] = True
        options["kernel_2D_size"] = self.dataset.kernel_size
        export_demo(path,
                    options=options,
                    xyz=self.gaussians_1.get_xyz.detach().cpu().numpy(),
                    scales=self.gaussians_1.get_scaling_with_3D_filter.detach().cpu().numpy(),
                    opacities=self.gaussians_1.get_opacity_with_3D_filter.detach().cpu().numpy(),
                    quaternions=self.gaussians_1.get_rotation.detach().cpu().numpy(),
                    spherical_harmonics=self.gaussians_1.get_features.transpose(1, 2).detach().cpu().numpy())
