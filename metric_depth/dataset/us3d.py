import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet


class US3D(Dataset):

    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size

        with open(filelist_path, 'r') as f:
            lines = f.read().splitlines()
        self.filelist = [line.strip().split() for line in lines]

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=(mode == 'train'),
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    @staticmethod
    def transform_height(z):
        return np.sign(z) * np.log1p(np.abs(z))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        rec = self.filelist[idx]
        image_path = rec[0]
        height_path = rec[1]
        semantic_path = rec[2] if len(rec) > 2 else None

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        height_map = cv2.imread(height_path, cv2.IMREAD_UNCHANGED)
        if height_map is None:
            raise FileNotFoundError(f"Height map not found: {height_path}")

        height_map = height_map.astype('float32')

        height_map = self.transform_height(height_map)

        semantics = None
        if semantic_path:
            semantics = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            if semantics is not None:
                semantics = semantics.astype('int64')

        sample = {'image': image, 'depth': height_map}
        if semantics is not None:
            sample['semantics'] = semantics

        sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['depth'] = torch.from_numpy(sample['depth']).float()

        sample['valid_mask'] = torch.isfinite(sample['depth']) & (sample['depth'] > 0)

        if semantics is not None:
            sample['semantics'] = torch.from_numpy(sample['semantics']).long()

        sample['image_path'] = image_path

        return sample
