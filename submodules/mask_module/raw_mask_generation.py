import os
import cv2
import numpy as np
from utils.colmap_utils import read_images_binary, read_points3D_binary
import matplotlib.image as mpimg
import time
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator


def max_pool_to_size(img, target_h, target_w):
    """
    Downsamples a 2D (or 3D) NumPy array to (target_h, target_w) using max pooling.
    Supports grayscale or multi-channel images (H, W) or (H, W, C).
    """
    H, W = img.shape[:2]
    stride_h = H // target_h
    stride_w = W // target_w

    pooled = np.zeros((target_h, target_w) + img.shape[2:], dtype=img.dtype)

    for i in range(target_h):
        for j in range(target_w):
            h_start = i * stride_h
            w_start = j * stride_w
            h_end = (i + 1) * stride_h if i < target_h - 1 else H
            w_end = (j + 1) * stride_w if j < target_w - 1 else W

            patch = img[h_start:h_end, w_start:w_end, ...]
            pooled[i, j] = np.max(patch, axis=(0, 1)) if img.ndim == 3 else np.max(patch)

    return pooled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-the-go processing")

    parser.add_argument("--dataset", default="on-the-go", type=str, help="Supported dataset: on-the-go, robustnerf, phototourism")
    parser.add_argument("--scene", default="spot-undistorted", type=str, help="scenes in each dataset")
    parser.add_argument('--match', default=3, type=int)
    parser.add_argument('--scale', default=1, type=int)
    args = parser.parse_args()

    dataset = args.dataset
    scene = args.scene
    occurance_filtering_threshold = args.match
    scale_factor = args.scale

    # initialize model
    semantic_sam_model = build_semantic_sam(model_type='L', ckpt='swinl_only_sam_many2many.pth')
    mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam_model, level=[2], points_per_side=64)

    # root and files
    root = f"../../dataset/{dataset}/{scene}"
    with open(os.path.join(root, "train_list.txt"), "r") as f:
        train_list = [line.strip() for line in f]

    # loading colmap data
    images_bin = read_images_binary(os.path.join(root, 'sparse', 'images.bin'))
    points3D = read_points3D_binary(os.path.join(root, 'sparse', 'points3D.bin'))
    image_idx_dict = {image.name: idx for idx, image in images_bin.items()}

    # filtering invalid and under matched points (< 3)
    pointid_occurance_dict = {idx: point.image_ids.shape[0] for idx, point in points3D.items()}
    unreliable_points = [-1]
    for pointid, occurance in pointid_occurance_dict.items():
        if occurance < occurance_filtering_threshold:
            unreliable_points.append(pointid)

    start_time = time.time()
    multi_cue_masks = {}
    for file in train_list:
        print(file)

        # raw semantic masks generation
        original_image, input_image = prepare_image(image_pth=os.path.join(root, 'images', file))
        raw_masks = mask_generator.generate(input_image)
        raw_masks_np = np.stack([mask["segmentation"] for mask in raw_masks])

        # Valid correspondence points
        # A pixel is considered a valid correspondence if its match count exceeds a threshold,
        # indicating it likely belongs to a static region.
        # We set the stereo correspondence map value to 1 at the pixel location if there are more than 2 matches.
        h = original_image.height
        w = original_image.width
        image_colmap = images_bin[image_idx_dict[file]]
        points_mask = np.isin(image_colmap.point3D_ids, unreliable_points)
        points = image_colmap.xys
        points = points[~points_mask]
        points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
        points = np.round(points // scale_factor, decimals=1).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], a_min=0, a_max=w - 1)
        points[:, 1] = np.clip(points[:, 1], a_min=0, a_max=h - 1)
        point_map = np.zeros((1, h, w))
        point_map[:, points[:, 1], points[:, 0]] = 1

        # mask resize to origin size
        masks = np.transpose(raw_masks_np, (1, 2, 0))
        masks_unpooled = cv2.resize(masks * 1.0, (w, h), interpolation=cv2.INTER_NEAREST) > 0.8
        # cv2 resize remove 1 channel
        if len(masks_unpooled.shape) == 2:
            masks_unpooled = masks_unpooled.reshape(h, w, 1)
        masks_unpooled = np.transpose(masks_unpooled, (2, 0, 1))

        # Adaptively filtering masks covered static regions
        # Global density
        valid_correspondence_aver_density = points.shape[0] / (h * w)
        threshold = 0.1 * valid_correspondence_aver_density  # todo config the scaler
        # Per mask density
        valid_correspondence_counts = np.sum(point_map * masks_unpooled, axis=(1, 2))
        mask_areas = np.sum(masks_unpooled, axis=(1, 2))
        valid_correspondence_densities = valid_correspondence_counts / mask_areas
        # todo handling empty mask
        masks_unpooled_filtered = masks_unpooled[valid_correspondence_densities < threshold]

        # record the filtered resized mask
        multi_cue_masks[file] = masks_unpooled_filtered

    print("%s --- %.2f seconds" % (scene, time.time() - start_time))
    np.savez_compressed(os.path.join(root, f"multi_cue_masks_{scene}"), **multi_cue_masks)

