import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import json
import argparse
import pickle
from collections import defaultdict

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_record_features():
    feature_dict = {}
    
    for raw_record in record_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        for key, feature in example.features.feature.items():
            
            # Determine the data type based on which list is populated
            if feature.HasField('bytes_list'):
                data_type = 'byte' if len(feature.bytes_list.value) == 1 else 'bytes list'
            elif feature.HasField('float_list'):
                data_type = 'float' if len(feature.float_list.value) == 1 else 'float list'
            elif feature.HasField('int64_list'):
                data_type = 'int64' if len(feature.int64_list.value) == 1 else 'int64 list'
            else:
                data_type = 'unknown'
            
            feature_dict[key] = data_type
    feature_dict['steps/observation/depth'] = 'bytes list'
    return feature_dict


def serialize_example(example):
    
    feature = {}
    
    for key in ['aspects', 'attributes']:
        
        for key_2 in list(example[key].keys()):
            
            feature_name = '/'.join([key, key_2])
            feature[feature_name] = method_dict[feature_dict[feature_name]](example[key][key_2])
    
    steps_dict = defaultdict(list)
    for item in example['steps']:
        
        for steps_key in item:
            
            if steps_key not in ['action', 'observation']:
                steps_dict['/'.join(['steps', steps_key])].append(item[steps_key])
            else:
                
                for key_2 in item[steps_key]:
                    
                    if key_2 in ['depth', 'image']:
                        steps_dict['/'.join(['steps', steps_key, key_2])].append(tf.io.encode_png(item[steps_key][key_2]).numpy())
                    elif key_2 == 'natural_language_instruction':
                        steps_dict['/'.join(['steps', steps_key, key_2])].append(item[steps_key][key_2].numpy())
                    else:
                        tensor = item[steps_key][key_2]
                        flattened_tensor = tf.reshape(tensor, [-1]).numpy().tolist()
                        steps_dict['/'.join(['steps', steps_key, key_2])].extend(flattened_tensor)
    
    for key in steps_dict:
        feature[key] = method_dict[feature_dict[key]](steps_dict[key])
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(dataset):
    
    os.makedirs(params.depth_data_dir, exist_ok=True)
    tfrecord_file = os.path.join(
        params.depth_data_dir,
        f'fractal20220817_depth_data-train.tfrecord-{shard_str}-of-01024'
    )
    
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for example in dataset:
            serialized_example = serialize_example(example)
            writer.write(serialized_example)


def add_timestep_index(example, index):
    
    def add_step_index(example, ts, idx):
        example['timestep'] = ts
        example['idx'] = idx
        return example
    
    timestep_index = tf.range(index['timestep_length'])
    timestep_index = tf.data.Dataset.from_tensor_slices(timestep_index)
    idx = tf.fill([index['timestep_length']], index['idx'])
    idx = tf.data.Dataset.from_tensor_slices(idx)
    example['steps'] = tf.data.Dataset.zip((example['steps'], timestep_index, idx))
    example['steps'] = example['steps'].map(add_step_index)
    
    return example


def add_depth_image(example):
    
    def add_depth_image_2(example):
        
        # Convert idx and timestep to string tensors
        idx_str = tf.strings.as_string(example['idx'])
        timestep_str = tf.strings.as_string(example['timestep'])
        
        # Ensure task is a string tensor
        task = tf.cast(example['observation']['natural_language_instruction'], tf.string)
        filename = tf.strings.join(['depth_imgs/', task, "_", idx_str, "_", timestep_str, ".png"])
        
        def read_image(filename):
            return tf.convert_to_tensor(np.array(data[filename.numpy().decode('utf-8')]))
        
        image = tf.py_function(read_image, [filename], tf.uint8)
        example['observation']['depth'] = image
        del example['timestep']
        del example['idx']
        return example
    
    example['steps'] = example['steps'].map(add_depth_image_2)
    
    return example


def add_timestep(dataset):
    
    # Apply the function to each (example, index) pair in the dataset
    data_dict = {'idx': [idx for idx in range(len(dataset))], 'timestep_length': [len(item['steps']) for item in dataset]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset_with_idx = tf.data.Dataset.zip((dataset, data_idx))
    dataset_with_idx = dataset_with_idx.map(add_timestep_index, num_parallel_calls=1)

    return dataset_with_idx

def save_dataset_info():
    
    features_path = os.path.join(
        params.data_dir,
        'fractal20220817_data',
        '0.1.0',
        'features.json'
    )
    dset_info_path = os.path.join(
        params.data_dir,
        'fractal20220817_data',
        '0.1.0',
        'dataset_info.json'
    )
    
    with open(features_path, 'r') as f:
        features = json.load(f)
    with open(dset_info_path, 'r') as f:
        dset_info = json.load(f)
    
    dset_info['name'] = 'fractal20220817_depth_data'
    depth_feature_dict = {
        'image': 
        {
            'dtype': 'uint8',
            'encodingFormat': 'png',
            'shape': {'dimensions': ['518', '518', '3']}
        },
        'pythonClassName': 'tensorflow_datasets.core.features.image_feature.Image'
    }
    features['featuresDict']['features']['steps']\
                ['sequence']['feature']['featuresDict']\
                ['features']['observation']\
                ['featuresDict']['features']\
                ['depth'] = depth_feature_dict
    
    with open(os.path.join(params.depth_data_dir, 'features.json'), 'w') as f:
        json.dump(features, f, indent=6)
    with open(os.path.join(params.depth_data_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dset_info, f, indent=6)

def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=0,
                        help='Shard of the dataset to save', choices=[i for i in range(1025)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--depth-data-dir', type=str, default='/data/shresth/octo-data/fractal20220817_depth_data/0.1.0')
    parser.add_argument('--pickle_file_path', type=str, default='depth_imgs.pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    params = params()
    
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    # Load pickle file and dataset/record dataset
    dataset = tfds.load('fractal20220817_data', data_dir=params.data_dir,
                        split=split)
    with open(params.pickle_file_path, 'rb') as f:
        
        data = pickle.load(f)
    
    record_dataset = tf.data.TFRecordDataset(
        os.path.join(
            params.data_dir, 'fractal20220817_data', '0.1.0',
            f'fractal20220817_data-train.tfrecord-{shard_str}-of-01024'
        )
    )
    
    print('Adding depth features to dataset...')
    dataset = add_timestep(dataset)
    dataset = dataset.map(add_depth_image)
    
    method_dict = {
        'byte': _bytes_feature,
        'bytes list': _bytes_list_feature,
        'float': _float_feature,
        'float list': _float_list_feature,
        'int64': _int64_feature,
        'int64 list': _int64_list_feature
    }
    feature_dict = get_record_features()
    print(f'Serializing and writing dataset to tfrecord...')
    write_tfrecord(dataset)
    print(f'Updating feature and info dictionary...')
    save_dataset_info()
    os.system(f'rm {params.pickle_file_path}') 