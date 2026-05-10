import torch
from torch.utils.data import Dataset
import numpy as np

class VectorDoubleDataset(Dataset):
    """
    Dataset class for handling vector data with weak and strong augmentations.
    Instead of PIL images and torchvision transforms, it handles numpy arrays/tensors
    and applies simple noise-based augmentations or similar.
    """
    def __init__(self, data, targets, masks=None, transform_weak=None, transform_strong=None):
        """
        Initialize the dataset.

        Parameters:
        - data: The input data (N, D) numpy array or tensor.
        - targets: The labels (N, C) numpy array or tensor.
        - masks: Optional masks (N, C) numpy array or tensor.
        - transform_weak: Optional transformation for weak augmentation.
        - transform_strong: Optional transformation for strong augmentation.
        """
        self.data = data
        self.targets = targets
        self.masks = masks
        self.n = len(targets)
        self.index = list(range(self.n))
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __getitem__(self, i):
        vec = self.data[i]
        
        # Apply transforms if they exist
        vec_weak = self.transform_weak(vec) if self.transform_weak else vec
        vec_strong = self.transform_strong(vec) if self.transform_strong else vec

        # Convert to tensors if they aren't already
        if not isinstance(vec_weak, torch.Tensor):
            vec_weak = torch.tensor(vec_weak, dtype=torch.float32)
        if not isinstance(vec_strong, torch.Tensor):
            vec_strong = torch.tensor(vec_strong, dtype=torch.float32)
        
        target = self.targets[i]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)

        if self.masks is not None:
            mask = self.masks[i]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
            return vec_weak, vec_strong, target, mask, self.index[i]

        return vec_weak, vec_strong, target, self.index[i]

    def __len__(self):
        return self.n

    def update_label(self, noise_label):
        self.targets[:] = noise_label[:]

class VectorCustomDataset(Dataset):
    """
    Custom dataset class for handling vector data and targets.
    """
    def __init__(self, data, targets, masks=None, transform=None):
        self.data = data
        self.targets = targets
        self.masks = masks
        self.n = len(targets)
        self.index = list(range(self.n))
        self.transform = transform

    def __getitem__(self, i):
        vec = self.data[i]
        if self.transform is not None:
            vec = self.transform(vec)
        
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
            
        target = self.targets[i]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)

        if self.masks is not None:
            mask = self.masks[i]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
            return vec, target, mask, self.index[i]
            
        return vec, target, self.index[i]

    def __len__(self):
        return self.n

    def update_label(self, noise_label):
        self.targets[:] = noise_label[:]

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
