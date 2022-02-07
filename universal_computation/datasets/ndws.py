from einops import rearrange
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from universal_computation.datasets.dataset import Dataset


class NDWSDatasetHelper(torch.utils.data.Dataset):
    def __init__(self, filename, transform=None):
        self.data = []
        data_dict = np.load(filename)
        for _, arr in data_dict.items():
            self.data.append(arr)
        self.data = np.concatenate(self.data, axis=0)
        self.data = rearrange(self.data, 'b h w c -> b c h w')
        self.data = torch.tensor(self.data)

        # Just one transform because the same crop has to be applied to both x
        # and y
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        x = item[:-1, :, :]
        y = item[-1, :, :]
        return x, y


class NDWSDataset(Dataset):
    def __init__(self, batch_size, patch_size=None, data_aug=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.patch_size = patch_size

        if data_aug:
            transform = transforms.Compose([
                transforms.RandomCrop(32)
            ])
        else:  # TODO: make data_aug actually different
            transform = transforms.Compose([
                transforms.RandomCrop(32)
            ])

        self.d_train = DataLoader(
            NDWSDatasetHelper('data/ndws/ndws_train.npz', transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True
        )
        self.d_val = DataLoader(
            NDWSDatasetHelper('data/ndws/ndws_val.npz', transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True
        )
        self.d_test = DataLoader(
            NDWSDatasetHelper('data/ndws/ndws_test.npz', transform=transform),
            batch_size=batch_size, drop_last=True, shuffle=True
        )

        self.train_enum = enumerate(self.d_train)
        self.val_enum = enumerate(self.d_val)
        self.test_enum = enumerate(self.d_test)

    def reset_train(self):
        self.train_enum = enumerate(self.d_train)

    def reset_test(self):
        self.test_enum = enumerate(self.d_test)

    def get_batch(self, batch_size=None, train=True):
        if train:
            _, (x, y) = next(self.train_enum, (None, (None, None)))
            if x is None:
                self.reset_train()
                _, (x, y) = next(self.train_enum)
        else:
            _, (x, y) = next(self.test_enum, (None, (None, None)))
            if x is None:
                self.reset_test()
                _, (x, y) = next(self.test_enum)

        # PyTorch CE loss requires targets to be of type long
        y = y.type(torch.long)

        # PyTorch CE loss requires classes in the range [0, # classes]
        # but the original dataset defines -1 to be a class
        y += 1

        if self.patch_size is not None:
            x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=self.patch_size, p2=self.patch_size)
            y = rearrange(y, 'b h w-> b (h w)')

        x = x.to(device=self.device)
        y = y.to(device=self.device)

        self._ind += 1

        return x, y
