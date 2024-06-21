# Depth Anything V2 for Metric Depth Estimation

![teaser](./assets/compare_zoedepth.png)

We here provide a simple codebase to fine-tune our Depth Anything V2 pre-trained encoder for metric depth estimation. Built on our powerful encoder, we use a simple DPT head to regress the depth. We fine-tune our pre-trained encoder on synthetic Hypersim / Virtual KITTI datasets for indoor / outdoor metric depth estimation, respectively.


# Pre-trained Models

We provide **six metric depth models** of three scales for indoor and outdoor scenes, respectively.

| Base Model | Params | Indoor (Hypersim) | Outdoor (Virtual KITTI 2) |
|:-|-:|:-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true) | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true) | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true) | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true) |

*We recommend to first try our larger models (if computational cost is affordable) and the indoor version.*

## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Use our models
```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
```

### Running script on images

Here, we take the `vitl` encoder as an example. You can also use `vitb` or `vits` encoders.

```bash
# indoor scenes
python run.py \
  --encoder vitl \
  --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 \
  --img-path <path> --outdir <outdir> [--input-size <size>] [--save-numpy]

# outdoor scenes
python run.py \
  --encoder vitl \
  --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth \
  --max-depth 80 \
  --img-path <path> --outdir <outdir> [--input-size <size>] [--save-numpy]
```

### Project 2D images to point clouds:

```bash
python depth_to_pointcloud.py \
  --encoder vitl \
  --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 \
  --img-path <path> --outdir <outdir>
```

### Reproduce training

Please first prepare the [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets. Then:

```bash
bash dist_train.sh
```


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
