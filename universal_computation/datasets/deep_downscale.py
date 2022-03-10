from einops import rearrange
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from universal_computation.datasets.dataset import Dataset

class DeepDownscaleTorchDataset(torch.utils.data.Dataset):
    
    def __init__(self, X_file, Y_file, transform=None, target_transform=None):
        with open(X_file, 'rb') as f:
            self.X = np.load(f)
        with open(Y_file, 'rb') as f:
            self.Y = np.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.transform(y)
            y = torch.squeeze(y)
        
        return x, y


class DeepDownscaleDataset(Dataset):
    # https://arxiv.org/abs/1808.05264
    
    def __init__(self, batch_size, patch_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.batch_size = batch_size
        self.patch_size = patch_size

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0., std=1.)])

        self.d_train = DataLoader(
            DeepDownscaleTorchDataset(
                'data/deep_downscale/train_x.npy',
                'data/deep_downscale/train_y.npy',
                transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True)

        self.d_val = DataLoader(
            DeepDownscaleTorchDataset(
                'data/deep_downscale/val_x.npy',
                'data/deep_downscale/val_y.npy',
                transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True)

        self.d_test = DataLoader(
            DeepDownscaleTorchDataset(
                'data/deep_downscale/test_x.npy',
                'data/deep_downscale/test_y.npy',
                transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True)

        self.train_enum = enumerate(self.d_train)
        self.val_enum = enumerate(self.d_val)
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.train_enum = enumerate(self.d_train)
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.val_enum, (None, (None, None)))
            if x is None:
                self.val_enum = enumerate(self.d_val)
                _, (x, y) = next(self.val_enum)

        if self.patch_size is not None:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=self.patch_size, p2=self.patch_size)
            y = rearrange(y, 'b h w -> b (h w)')

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1
        
        return x, y
