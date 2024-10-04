import pickle
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class Normalizer:
    # percentiles calculated in George's notebook
    qu_bound = 0.0032
    tau_bound = 0.0010
    hi_bound = 47526304.0156

    def norm_to_pm_1(self, x, bound):
        # x: (N, C, H, W)
        lbound = -bound
        x = torch.clamp(x, min=lbound, max=bound)
        x = 2 * (x - lbound) / (bound - lbound) - 1
        x = torch.clamp(x, min=-1.0, max=1.0)
        return x

    def denorm_pm_1(self, x, bound):
        # x: (N, C, H, W)
        lbound = -bound
        return 0.5 * (x + 1) * (bound - lbound) + lbound

    def norm_to_01(self, x, bound):
        # x: (N, H, W)
        x = torch.clamp(x, min=0, max=bound)
        x = x / bound
        x = torch.clamp(x, min=0.0, max=1.0)
        return x

    def denorm_01(self, x, bound):
        # x: (N, H, W)
        return x * bound

    def norm(self, x):
        # x: (N, C, H, W)
        if x.size(1) == 2:
            # QU only
            return self.norm_to_pm_1(x, self.qu_bound)
        x[:, 0, ...] = self.norm_to_01(x[:, 0, ...], self.tau_bound)
        x[:, 1:3, ...] = self.norm_to_pm_1(x[:, 1:3, ...], self.hi_bound)
        x[:, 3:, ...] = self.norm_to_pm_1(x[:, 3:, ...], self.qu_bound)
        return x

    def denorm(self, x):
        if x.size(1) == 2:
            # QU only
            return self.denorm_pm_1(x, self.qu_bound)
        x[:, 0, ...] = self.denorm_01(x[:, 0, ...], self.tau_bound)
        x[:, 1:3, ...] = self.denorm_pm_1(x[:, 1:3, ...], self.hi_bound)
        x[:, 3:, ...] = self.denorm_pm_1(x[:, 3:, ...], self.qu_bound)
        return x


class CustomRotate(torch.nn.Module):
    def __call__(self, *args):
        k = random.randint(0, 3)
        outs = []
        for arg in args:
            outs.append(torch.rot90(arg, dims=[-2, -1], k=k))
        return outs


class CustomFlip(torch.nn.Module):
    def __call__(self, *args):
        vflip = random.random() < 0.5
        hflip = random.random() < 0.5
        outs = []
        for arg in args:
            if vflip:
                arg = v2.functional.vertical_flip(arg)
            if hflip:
                arg = v2.functional.horizontal_flip(arg)
            outs.append(arg)
        return outs


class AddGaussianNoise:
    def __init__(self, means, stds, scale_factor=0.1):
        self.stds = stds
        self.means = means
        self.scale_factor = scale_factor

    def __call__(self, tensor):
        if tensor.ndim == 4:
            return tensor + (
                torch.randn(tensor.size()) * self.stds[None, :, None, None]
                + self.means[None, :, None, None]
            ) * self.scale_factor
        elif tensor.ndim == 3:
            return tensor + (
                torch.randn(tensor.size()) * self.stds[:, None, None] + self.means[:, None, None]
            ) * self.scale_factor


class InferenceDataset(Dataset):
    def __init__(self, patches_path: str = 'data/inference_patches.pkl'):
        self.norm = Normalizer()
        with open(patches_path, 'rb') as f:
            patches = pickle.load(f)
        self.patches = torch.cat([torch.tensor(p) for p in patches.values()], dim=0)
        self.normer = Normalizer()
        self.patches = self.normer.norm(self.patches).float()
        self.res_to_npatches = {res: len(p) for res, p in patches.items()}

    def __len__(self):
        return self.patches.size(0)

    def __getitem__(self, idx):
        # if isinstance(idx, int):
        #     idx = [idx]
        return self.patches[idx]

    def denorm(self, x):
        return self.normer.denorm(x)


class TrainDataset(Dataset):
    def __init__(
        self,
        patches_path: str = 'data/training_patches.pkl',
        add_noise=False,
        noise_scale_factor=0.1,
    ):
        with open(patches_path, 'rb') as f:
            patches = pickle.load(f)
        self.patches_in = torch.cat([torch.tensor(p['in']) for p in patches.values()], dim=0)
        self.patches_out = torch.cat([torch.tensor(p['out']) for p in patches.values()], dim=0)
        assert self.patches_in.size(0) == self.patches_out.size(0)
        self.normer = Normalizer()
        self.patches_in = self.normer.norm(self.patches_in).float()
        self.patches_out = self.normer.norm(self.patches_out).float()
        self.base_transforms = v2.Compose([
            CustomRotate(),
            CustomFlip()
        ])
        self.noise_transform = AddGaussianNoise(
                self.patches_in.mean(dim=[0, 2, 3]),
                self.patches_in.std(dim=[0, 2, 3]),
                scale_factor=noise_scale_factor,
            ) if add_noise else None

    def __len__(self):
        return self.patches_in.size(0)

    def __getitem__(self, idx):
        # if isinstance(idx, int):
        #     idx = [idx]
        inputs, targets = self.base_transforms(self.patches_in[idx], self.patches_out[idx])
        if self.noise_transform is not None:
            inputs = self.noise_transform(inputs)
        return inputs, targets
