<p align="center">
  <h1 align="center">Robust Neural Rendering in the Wild <br> with Asymmetric Dual 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="">Chengqi Li</a>
    ·
    <a href="">Zhihao Shi</a>
    ·
    <a href="">Yangdi Lu</a>
    ·
    <a href="">Wenbo He</a>
    ·
    <a href="">Xiangyu Xu</a>

  </p>
  <h2 align="center">NeurIPS 2025 Spotlight</h2>
  <h3 align="center">
   <a href="https://arxiv.org/abs/2506.03538">Paper</a> | 
   <a href="">Project Page</a> | 
   <a href="https://github.com/steveli88/AsymmetricGS">Code</a> 
  </h3>
  <div align="center"></div>
</p>
<br/>

<p align="center">
</p>
<p align="justify">
In this work, we present Asymmetric Dual 3DGS, 
a robust and efficient framework for 3D scene reconstruction in unconstrained, 
in-the-wild environments. 
Our method employs two 3DGS models guided by distinct masking strategies to enforce cross-model consistency, 
effectively mitigating artifacts caused by low-quality observations. 
To further improve training efficiency, 
we introduce a dynamic EMA proxy that significantly reduces computational cost with minimal impact on performance. 
Extensive experiments on three challenging real-world datasets validate the effectiveness and generality of our approach. 
</p>
<br/>

## Installation
Clone the repository and create a `python == 3.11` Anaconda environment with CUDA toolkit 12.6 installed using
```bash
git clone https://github.com/steveli88/AsymmetricGS.git
cd AsymmetricGS

conda create -y -n AsymmetricGS python=3.11
conda activate AsymmetricGS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install plyfile
pip install tqdm
pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install nerfbaselines
```

## Dataset

### On-the-go dataset & RobustNeRF dataset
Download [raw On-the-go dataset](https://rwn17.github.io/nerf-on-the-go/) and [raw RobustNeRF dataset](https://robustnerf.github.io/) to the dataset folder. 
Then running the following script to undistort the raw images.
```bash
sh scripts/dataset_preparation.sh
```

### PhotoTourism dataset
Using `NerfBaselines` to download scenes from PhotoTourism dataset with initial point clouds and camera parameters ready.
```bash
nerfbaselines download-dataset external://phototourism/brandenburg-gate -o dataset/phototourism/brandenburg-gate
nerfbaselines download-dataset external://phototourism/sacre-coeur -o dataset/phototourism/sacre-coeur
nerfbaselines download-dataset external://phototourism/trevi-fountain -o dataset/phototourism/trevi-fountain
```
Alternatively, we can also download raw images from the official website and perform `COLMAP` to obtain point clouds and camera parameters.

## Mask preprocess for our multi-cue adaptive mask
We introduce Multi-Cue Adaptive Masking, 
which combines the strengths of residual- and segmentation-based approaches 
while incorporating a complementary hard mask 
that captures error patterns distinct from the self-supervised soft mask. 
Specifically, we first employ Semantic-SAM to generate raw masks. 
Masks covering static regions are then filtered out using stereo-based correspondence 
(derived from COLMAP results in dataset preparation). 
The remaining masks are integrated with residual information during training to identify distractor areas.

Installing requirements for `Semantic-SAM`
```
cd submodules

pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install timm
pip install transformers
pip install kornia

git clone git@github.com:facebookresearch/Mask2Former.git
TORCH_CUDA_ARCH_LIST='8.9' FORCE_CUDA=1 python Mask2Former/mask2former/modeling/pixel_decoder/ops/setup.py build install

wget -P mask_module https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth
```

Mask preprocessing
```
cd submodules/mask_module



```



## Training


<!--
<p align="center">
  <a href="">
    <img src="./media/bicycle_3dgs_vs_ours.gif" alt="Logo" width="95%">
  </a>
</p>

  <img width="51%" alt="WildGaussians model appearance" src=".assets/cover-trevi.webp" />
  <img width="43%" alt="WildGaussians remove occluders" src=".assets/cover-onthego.webp" />
<br>

# Update
We integrated an improved densification metric proposed in [Gaussian Opacity Fields](https://niujinshuchong.github.io/gaussian-opacity-fields/), which significantly improves the novel view synthesis results, please check the [paper](https://arxiv.org/pdf/2404.10772.pdf) for details. Please download the lastest code and reinstall `diff-gaussian-rasterization` to try it out. 

# Installation
Clone the repository and create an anaconda environment using
```
git clone git@github.com:autonomousvision/mip-splatting.git
cd mip-splatting

conda create -y -n mip-splatting python=3.8
conda activate mip-splatting

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

# Dataset
## Blender Dataset
Please download and unzip nerf_synthetic.zip from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Then generate multi-scale blender dataset with
```
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale
```

## Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and request the authors for the treehill and flowers scenes.

# Training and Evaluation
```
# single-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py 

# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py 

# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py 

# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 
```

# Online viewer
After training, you can fuse the 3D smoothing filter to the Gaussian parameters with
```
python create_fused_ply.py -m {model_dir}/{scene} --output_ply fused/{scene}_fused.ply"
```
Then use our [online viewer](https://niujinshuchong.github.io/mip-splatting-demo) to visualize the trained model.
-->

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/).
Please follow the license of 3DGS and Mip-Splatting. We thank all the authors for their great work and released code.

## Citation
If you find our code or paper useful, please cite
```bibtex
@misc{li2025robustneuralrenderingwild,
      title={Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting}, 
      author={Chengqi Li and Zhihao Shi and Yangdi Lu and Wenbo He and Xiangyu Xu},
      year={2025},
      eprint={2506.03538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03538}, 
}
```
