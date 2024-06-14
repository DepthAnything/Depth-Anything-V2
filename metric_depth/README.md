# Depth Anything V2 for Metric Depth Estimation

![teaser](./assets/compare_zoedepth.png)

We here provide a simple codebase to fine-tune our Depth Anything V2 pre-trained encoder for metric depth estimation. Built on our powerful encoder, we use a simple DPT head to regress the depth. We fine-tune our pre-trained encoder on synthetic Hypersim / Virtual KITTI datasets for indoor / outdoor metric depth estimation, respectively.


## Usage

### Inference

Please first download our pre-trained metric depth models and put them under the `checkpoints` directory: 
- [Indoor model from Hypersim](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true)
- [Outdoor model from Virtual KITTI 2](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true)

```bash
# indoor scenes
python run.py \
  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 --img-path <path> --outdir <outdir> [--input-size <size>] [--save-numpy]

# outdoor scenes
python run.py \
  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_vkitti_vitl.pth \
  --max-depth 80 --img-path <path> --outdir <outdir> [--input-size <size>] [--save-numpy]
```

You can also project 2D images to point clouds:
```bash
python depth_to_pointcloud.py \
  --encoder vitl --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 --img-path <path> --outdir <outdir>
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
```
