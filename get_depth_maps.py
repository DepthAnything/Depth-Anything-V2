from PIL import Image
import os, json
import pdb
import torch
import numpy as np
import cv2
from depth_anything_v2.dpt import DepthAnythingV2
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as MetricDepthAnythingV2
import matplotlib
from torch.nn import functional as F
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import pickle
import argparse

def save_pickle():
    
    # Example directory containing images
    image_directory = 'depth_imgs'

    # List to store image data
    images_data = {}

    # Iterate over each image file in the directory
    for filename in os.listdir(image_directory):
        filepath = os.path.join(image_directory, filename)
        
        img = Image.open(filepath)
        images_data[filepath] = img
    
    pickle_file = params.pickle_file_path
    os.system('rm -r depth_imgs')

    # Serialize and save images_data using pickle
    with open(pickle_file, 'wb') as f:
        pickle.dump(images_data, f)

def add_timestep_index(example, index):
    
    def add_step_index(example, index):
        example['timestep'] = index
        return example
    
    timestep_index = tf.range(index['timestep_length'])
    timestep_index = tf.data.Dataset.from_tensor_slices(timestep_index)
    example['steps'] = tf.data.Dataset.zip((example['steps'], timestep_index))
    example['steps'] = example['steps'].map(add_step_index)
    example['idx'] = index['idx']
    
    return example


def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=0,
                        help='Shard of the dataset to save', choices=[i for i in range(1025)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--pickle_file_path', type=str, default='depth_imgs.pkl')
    parser.add_argument('--use-metric-depth-model', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()
    
    return args

    
if __name__ == '__main__':
    
    BATCH_SIZE = 2
    params = params()
    DEVICE = f'cuda:{params.device}' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    os.makedirs('depth_imgs', exist_ok=True)
    
    dataset = tfds.load('fractal20220817_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))], 
                 'timestep_length': [len(item['steps']) for item in dataset]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.zip((dataset, data_idx))
    dataset = dataset.map(add_timestep_index, num_parallel_calls=1)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    if not params.use_metric_depth_model:
        depth_anything = DepthAnythingV2(**model_configs['vitl'])
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
    else:
        depth_anything = MetricDepthAnythingV2(**{**model_configs['vitl'], 'max_depth': 20})
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    print('Starting to extract depth maps...')
    start_time = time.time()
    
    img_idx = 0

    for example in dataset:
        
        task = [d['observation']['natural_language_instruction'].numpy().decode('utf-8') for d in example['steps'].take(1)][0]
        example_idx = int(example['idx'].numpy())
        
        for batch in example['steps'].batch(BATCH_SIZE):
        
            images = batch['observation']['image']
            images = torch.stack([depth_anything.image2tensor(img.numpy())[0].squeeze(0) for
                                      img in images], dim=0)
            N = len(batch['observation']['natural_language_instruction'])
            actions = batch['action']
            
            action_list = []
            for i in range(N):
                action_str = np.concatenate([actions[a_type][i].numpy() for a_type in actions
                            if a_type not in ['terminate_episode', 'world_vector']], axis=0).tolist()
                action_list.append('_'.join([str(a) for a in action_str]))
                
            depth_images = depth_anything.forward(images.to(DEVICE))
            
            ts_list = [int(b.numpy()) for b in batch['timestep']]
            
            for depth_img, action_str, ts in zip(depth_images, action_list, ts_list):
                
                depth = F.interpolate(torch.unsqueeze(depth_img.clone().detach(), dim=0)[:, None], (518, 518),
                                        mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = 1 - depth
                depth = depth.astype(np.uint8)
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                
                img_name = f"{task}_{example_idx}_{ts}.png"
                
                print(f'Saving depth image: {img_name}')
                cv2.imwrite(f"depth_imgs/{img_name}", depth)
                img_idx += 1
            
        
        if img_idx % 1000 == 0:
            print(f'Looked at {img_idx} images...')
    
    print(f'Saving {img_idx} images to pickle file...')
    save_pickle()
    print(f"Time taken: {time.time() - start_time}")