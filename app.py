import gradio as gr
import cv2
import matplotlib
import numpy as np
import os
from PIL import Image
import spaces
import torch
import tempfile
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from depth_anything_v2.dpt import DepthAnythingV2

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder2name = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large',
    'vitg': 'Giant', # we are undergoing company review procedures to release our giant model checkpoint
}
encoder = 'vitl'
model_name = encoder2name[encoder]
model = DepthAnythingV2(**model_configs[encoder])
filepath = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_name}", filename=f"depth_anything_v2_{encoder}.pth", repo_type="model")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

title = "# Depth Anything V2"
description = """Official demo for **Depth Anything V2**.
Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), and [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""

@spaces.GPU
def predict_depth(image):
    return model.infer_image(image)

def process_video(video_path):
    input_size = 518
    temp_output_path = tempfile.mktemp(suffix='.mp4')
    
    raw_video = cv2.VideoCapture(video_path)
    frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
    
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break
        
        depth = model.infer_image(raw_frame, input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        colored_depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
        out.write(colored_depth)
    
    raw_video.release()
    out.release()
    
    return temp_output_path

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Tabs():
        with gr.TabItem("Image"):
            gr.Markdown("### Depth Prediction demo")
            with gr.Row():
                input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
                depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
            submit = gr.Button(value="Compute Depth")
            gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download",)
            raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download",)
            
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')

            def on_submit(image):
                original_image = image.copy()

                h, w = image.shape[:2]

                depth = predict_depth(image[:, :, ::-1])

                raw_depth = Image.fromarray(depth.astype('uint16'))
                tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                raw_depth.save(tmp_raw_depth.name)

                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

                gray_depth = Image.fromarray(depth)
                tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                gray_depth.save(tmp_gray_depth.name)

                return [(original_image, colored_depth), tmp_gray_depth.name, tmp_raw_depth.name]
            
            submit.click(on_submit, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file])
            
            example_files = os.listdir('assets/examples')
            example_files.sort()
            example_files = [os.path.join('assets/examples', filename) for filename in example_files]
            examples = gr.Examples(examples=example_files, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file, raw_file], fn=on_submit)
        
        with gr.TabItem("Video"):
            gr.Markdown("### Video Depth Prediction demo")
            input_video = gr.Video(label="Input Video")
            output_video = gr.Video(label="Output Video")
            process_video_btn = gr.Button(value="Process Video")
            
            process_video_btn.click(process_video, inputs=[input_video], outputs=[output_video])
            
            example_files = os.listdir('assets/examples_video')
            example_files.sort()
            example_files = [os.path.join('assets/examples_video', filename) for filename in example_files]
            examples = gr.Examples(examples=example_files, inputs=[input_video], outputs=[output_video], fn=process_video)

if __name__ == '__main__':
    demo.queue().launch(share=True)
