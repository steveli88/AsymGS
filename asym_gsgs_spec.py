import os
from nerfbaselines import register, MethodSpec


_MIPNERF360_NOTE = """Authors evaluated on larger images which were downscaled to the target size (avoiding JPEG compression artifacts) instead of using the official provided downscaled images. As mentioned in the 3DGS paper, this increases results slightly ~0.5 dB PSNR."""
_paper_results = {
    # Mip-NeRF 360
    # 360 scenes: bicycle flowers garden stump treehill room counter kitchen bonsai
    # 360 PSNRs: 25.72 21.93 27.76 26.94 22.98 31.74 29.16 31.55 32.31
    # 360 SSIMs: 0.780 0.623 0.875 0.786 0.655 0.928 0.916 0.933 0.948
    # 360 LPIPS: 0.206 0.331 0.103 0.209 0.320 0.192 0.179 0.113 0.173
    "mipnerf360/bicycle": {"psnr": 25.72, "ssim": 0.780, "lpips_vgg": 0.206, "note": _MIPNERF360_NOTE},
    "mipnerf360/flowers": {"psnr": 21.93, "ssim": 0.623, "lpips_vgg": 0.331, "note": _MIPNERF360_NOTE},
    "mipnerf360/garden": {"psnr": 27.76, "ssim": 0.875, "lpips_vgg": 0.103, "note": _MIPNERF360_NOTE},
    "mipnerf360/stump": {"psnr": 26.94, "ssim": 0.786, "lpips_vgg": 0.209, "note": _MIPNERF360_NOTE},
    "mipnerf360/treehill": {"psnr": 22.98, "ssim": 0.655, "lpips_vgg": 0.320, "note": _MIPNERF360_NOTE},
    "mipnerf360/room": {"psnr": 31.74, "ssim": 0.928, "lpips_vgg": 0.192, "note": _MIPNERF360_NOTE},
    "mipnerf360/counter": {"psnr": 29.16, "ssim": 0.916, "lpips_vgg": 0.179, "note": _MIPNERF360_NOTE},
    "mipnerf360/kitchen": {"psnr": 31.55, "ssim": 0.933, "lpips_vgg": 0.113, "note": _MIPNERF360_NOTE},
    "mipnerf360/bonsai": {"psnr": 32.31, "ssim": 0.948, "lpips_vgg": 0.173, "note": _MIPNERF360_NOTE},
}


AsymmetricGSSpec: MethodSpec = {
    "id": "asym_gsgs",
    "method_class": "asym_gsgs:AsymmetricGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": "",
    },
    # "metadata": {
    #     "name": "Mip-Splatting",
    #     "description": """A modification of Gaussian Splatting designed to better handle aliasing artifacts.""",
    #     "paper_title": "Mip-Splatting: Alias-free 3D Gaussian Splatting",
    #     "paper_authors": ["Zehao Yu", "Anpei Chen", "Binbin Huang", "Torsten Sattler", "Andreas Geiger"],
    #     "paper_link": "https://arxiv.org/pdf/2311.16493.pdf",
    #     "paper_results": _paper_results,
    #     "link": "https://niujinshuchong.github.io/mip-splatting/",
    #     "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/autonomousvision/mip-splatting/main/LICENSE.md"}],
    # },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True, },
        "on-the-go": { "@apply": [{"dataset": "on-the-go"}], "lambda_mul": 1.0, },
        "robustnerf": { "@apply": [{"dataset": "robustnerf"}], "lambda_mul": 1.0, },
        "large": {
            "@apply": [{"dataset": "phototourism"}],
            "@description": "A version of the method designed for large scenes.",
            "iterations": 100_000,
            "densify_from_iter": 500,
            "densify_until_iter": 30_000,
            "densification_interval": 1000,
            "opacity_reset_interval": 100_000,
            "position_lr_max_steps": 100_000,
            "position_lr_final": 0.000_001_6,
            "position_lr_init": 0.000_16,
            "scaling_lr": 0.005,
            "warmup": 1000,
            # "iterations": 200_000,
            # "densify_from_iter": 4_000,
            # "densify_until_iter": 100_000,
            # "densification_interval": 400,
            # "opacity_reset_interval": 15_000,
            # "position_lr_max_steps": 200_000,
            # "position_lr_final": 0.000_000_16,
            # "position_lr_init": 0.000_016,
            # "scaling_lr": 0.000_5,
            # "kernel_size": 0.0,
        },
    },
    "implementation_status": {
        "blender": "reproducing",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
        "seathru-nerf": "working",
    }
}

register(AsymmetricGSSpec)
