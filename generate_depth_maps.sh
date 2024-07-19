#!/bin/bash

wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true
mv depth_anything_v2_metric_hypersim_vitl.pth?download=true depth_anything_v2_metric_hypersim_vitl.pth
mkdir checkpoints
mv depth_anything_v2_metric_hypersim_vitl.pth checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
ln -s checkpoints/depth_anything_v2_metric_hypersim_vitl.pth checkpoints/depth_anything_v2_metric_hypersim_vitl.pth

for i in {0..1024}
do
  python get_depth_maps.py --data-shard $i --use-metric-depth-model --batch-size 4 --data-dir /ariesdv0/zhanling/oxe-data-converted
  python save_depth_maps.py --data-shard $i --data-dir /ariesdv0/zhanling/oxe-data-converted --output-dir /ariesdv0/zhanling/oxe-data-converted/fractal20220817_depth_data/0.1.0
done
