import pathlib
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T


def get_loader(root_dir, transform, train=True, batchsize=16):
    if train:
        data = MNIST(root_dir, train,
                        transform=transform, download=True)
        train_idx, val_idx = [], []
        for i in range(len(data)):
            if (i + 1) % 5 == 0:
                val_idx.append(i)
            else:
                train_idx.append(i)
        train_data = Subset(data, train_idx)
        val_data = Subset(data, val_idx)

        train_loader = DataLoader(train_data, batchsize, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_data, batchsize, shuffle=False, num_workers=2)

        return train_loader, val_loader
    else:
        data = MNIST(root_dir, train,
                     transform=transform, download=True)
        test_loader = DataLoader(data, batchsize, shuffle=False, num_workers=2)
        return test_loader
