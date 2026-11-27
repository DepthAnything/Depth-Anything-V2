<div align="center">
<h1>Depth Anything V2</h1>

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://bingykang.github.io/)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup>
<br>
[**Zhen Zhao**](http://zhaozhen.me/) · [**Xiaogang Xu**](https://xiaogang00.github.io/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

<sup>1</sup>香港大学&emsp;&emsp;&emsp;<sup>2</sup>TikTok
<br>
&dagger;项目负责人&emsp;*通讯作者

<a href="https://arxiv.org/abs/2406.09414"><img src='https://img.shields.io/badge/arXiv-Depth Anything V2-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything V2-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>

<p>[ <a href="README.md">English</a> | <strong>中文</strong> ]</p>
</div>

本项目呈现了 Depth Anything V2。相较于 [V1](https://github.com/LiheYoung/Depth-Anything)，V2 在精细几何细节与鲁棒性方面取得了显著提升。与基于 Stable Diffusion（SD）的模型相比，它具备更快的推理速度、更少的参数量以及更高的深度估计精度。

![效果概览](assets/teaser.png)


## 最新动态 (News)
- **2025-01-22：** [Video Depth Anything](https://videodepthanything.github.io) 正式发布，可为超长视频（例如超过 5 分钟）生成时序高度一致的连续深度图。
- **2024-12-22：** [Prompt Depth Anything](https://promptda.github.io/) 正式发布，支持通过低分辨率 LiDAR 传感器提示 DA 模型，实现 4K 超高清分辨率的度量深度估计。
- **2024-07-06：** [Transformers](https://github.com/huggingface/transformers/) 已原生支持 Depth Anything V2，详见[使用说明文档](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2)便捷调用。
- **2024-06-25：** Depth Anything 已集成至 [Apple Core ML Models](https://developer.apple.com/machine-learning/models/) 官方模型库，详见使用指引（[V1](https://huggingface.co/apple/coreml-depth-anything-small)、[V2](https://huggingface.co/apple/coreml-depth-anything-v2-small)）。
- **2024-06-22：** 基于 Depth-Anything-V2-Small 与 Base 架构的[更小尺寸度量深度模型](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models)正式发布。
- **2024-06-20：** 仓库与项目主页被 GitHub 误标记下线 6 天，对由此带来的不便深表歉意。
- **2024-06-14：** 论文、项目主页、源码、模型权重、在线 Demo 与评测基准全面发布。


## 预训练模型 (Pre-trained Models)

提供**四种不同参数规格**的模型，用于高鲁棒性相对深度估计：

| 模型 | 参数量 | 权重检查点 |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [下载](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [下载](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [下载](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | 即将推出 |


## 使用指南 (Usage)

### 准备环境 (Preparation)

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

下载上文[预训练模型列表](#预训练模型-pre-trained-models)中的权重文件，并放置在 `checkpoints` 目录下。

### 使用模型进行推理 (Use our models)

```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
```

如果不想克隆本代码仓库，也可以直接通过 [Transformers](https://github.com/huggingface/transformers/) 加载模型。以下为简要代码片段，更多详情请参阅[官方文档页面](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2)。

- 注意 1：请确保能够稳定连接 Hugging Face，并且已安装最新版本的 Transformers。
- 注意 2：由于 OpenCV（本项目所用）与 Pillow（HF 所用）之间存在[上采样实现差异](https://github.com/huggingface/transformers/pull/31522#issuecomment-2184123463)，预测结果可能会有微小出入。因此更推荐通过上文介绍的原生方式调用模型。

```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('your/image/path')
depth = pipe(image)["depth"]
```

### 针对*图像*运行推理脚本 (Running script on images)

```bash
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <path> --outdir <outdir> \
  [--input-size <size>] [--pred-only] [--grayscale]
```

参数选项说明：
- `--img-path`：可以 1) 指向存放待处理图像的图片目录，2) 指向单张图片文件，或者 3) 指向包含所有图片路径的文本列表文件。
- `--input-size`（可选）：默认模型推理输入分辨率为 `518`。***可调大该尺寸以获得更精细的细节结果。***
- `--pred-only`（可选）：仅保存预测出的深度图，不拼接原始图片。
- `--grayscale`（可选）：保存灰度深度图，不应用伪彩色调色板。

例如：
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

### 针对*视频*运行推理脚本 (Running script on videos)

```bash
python run_video.py \
  --encoder <vits | vitb | vitl | vitg> \
  --video-path assets/examples_video --outdir video_depth_vis \
  [--input-size <size>] [--pred-only] [--grayscale]
```

***尺寸更大的模型在视频上具有更出色的时序一致性。***

### Gradio 本地交互演示 (Gradio demo)

在本地启动 Gradio 演示界面：

```bash
python app.py
```

亦可体验[在线演示 Demo](https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2)。

***注意：与 V1 相比，对 DINOv2-DPT 网络架构做出了微小调整（源于此 [Issue #81](https://github.com/LiheYoung/Depth-Anything/issues/81)）。*** 在 V1 中，解码器*无意间*使用了 DINOv2 最后四层的特征进行解码。而在 V2 中，改为了使用[中间层特征（intermediate features）](https://github.com/DepthAnything/Depth-Anything-V2/blob/2cbc36a8ce2cec41d38ee51153f112e87c8e42d8/depth_anything_v2/dpt.py#L164-L169)。尽管该调整并未额外提升细节或精度，但团队决定遵循这一通用设计范式。


## 度量深度估计微调模型 (Fine-tuned to Metric Depth Estimation)

详情请参阅[度量深度估计指南](./metric_depth)。


## DA-2K 评估基准 (DA-2K Evaluation Benchmark)

详情请参阅 [DA-2K 评估基准说明](./DA-2K.md)。


## 社区生态支持 (Community Support)

**由衷感谢开源社区对 Depth Anything 系列项目的鼎力支持！**

- Apple Core ML:
    - https://developer.apple.com/machine-learning/models
    - https://huggingface.co/apple/coreml-depth-anything-v2-small
    - https://huggingface.co/apple/coreml-depth-anything-small
- Transformers:
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything
- TensorRT:
    - https://github.com/spacewalk01/depth-anything-tensorrt
    - https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python
- ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- ComfyUI: https://github.com/kijai/ComfyUI-DepthAnythingV2
- Transformers.js（网页端实时深度估计）: https://huggingface.co/spaces/Xenova/webgpu-realtime-depth-estimation
- Android 移动端:
  - https://github.com/shubham0204/Depth-Anything-Android
  - https://github.com/FeiGeChuanShu/ncnn-android-depth_anything


## 致谢 (Acknowledgement)

衷心感谢出色的 Hugging Face 团队（[@Pedro Cuenca](https://huggingface.co/pcuenq)、[@Niels Rogge](https://huggingface.co/nielsr)、[@Merve Noyan](https://huggingface.co/merve)、[@Amy Roberts](https://huggingface.co/amyeroberts) 等）在 Transformers 与 Apple Core ML 模型支持方面付出的巨大努力。

同时感谢 [DINOv2](https://github.com/facebookresearch/dinov2) 团队为社区贡献了如此令人瞩目的基座模型。


## 许可证 (LICENSE)

Depth-Anything-V2-Small 模型采用 Apache-2.0 许可证。Depth-Anything-V2-Base / Large / Giant 模型采用 CC-BY-NC-4.0 许可证。


## 引用 (Citation)

如果觉得本项目对您的研究有所帮助，请考虑引用：

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

---

> **免责声明**：本文档为开源社区提供的中文翻译版，仅供参考与学习交流。如翻译内容与英文原文存在出入，请以原版英文 [README.md](README.md) 为准。
