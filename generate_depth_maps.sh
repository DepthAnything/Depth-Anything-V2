
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true
mv depth_anything_v2_metric_hypersim_vitl.pth?download=true depth_anything_v2_metric_hypersim_vitl.pth
mkdir checkpoints
mv depth_anything_v2_metric_hypersim_vitl.pth checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
python get_depth_maps.py --use-metric-depth-model
python save_depth_maps.py