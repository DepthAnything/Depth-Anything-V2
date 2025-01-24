import argparse
import cv2
import glob
import matplotlib as plt
import numpy as np
import os
import torch
import torch.onnx

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])

    args = parser.parse_args()
    
    # we are undergoing company review procedures to release Depth-Anything-Giant checkpoint
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20  # 20 for indoor model, 80 for outdoor model

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to('cpu').eval()

    # Define dummy input data
    dummy_input = torch.ones((3, args.input_size, args.input_size)).unsqueeze(0)

    # Provide an example input to the model, this is necessary for exporting to ONNX
    example_output = depth_anything.forward(dummy_input)

    onnx_path = f'depth_anything_v2_metric_{dataset}_{args.encoder}.onnx'

    # Export the PyTorch model to ONNX format
    torch.onnx.export(depth_anything, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"], verbose=True)

    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    main()