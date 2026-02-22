"""
PalmNet dataset loader.

Adapted from CCNet/models/dataset.py with:
  - Deprecation warnings fixed (is not 1  →  != 1, resample= → interpolation=)
  - Works with torchvision >= 0.9.0
  - No CCNet-specific architecture code

Data format (same as CCNet):
    <image_path> <integer_label>
    e.g.  ./TongJi/session1/00001.bmp 0

Training mode  : returns two independently-augmented views of images from
                 the same identity class (required for Supervised Contrastive).
Eval/test mode : returns the same image twice (no augmentation difference).
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI:
    """
    Normalise a single-channel ROI image to zero-mean / unit-std,
    computed only over non-black (> 0) pixels.

    Input tensor: (1, H, W)  float32
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        c, h, w = tensor.size()
        if c != 1:
            raise TypeError("NormSingleROI only supports single-channel images.")

        flat = tensor.view(1, h * w)
        mask = flat > 0
        active = flat[mask]

        if active.numel() > 1:
            m = active.mean()
            s = active.std()
            flat[mask] = (active - m) / (s + 1e-6)

        return flat.view(1, h, w)



def _make_transforms(imside: int, train: bool) -> T.Compose:
    if not train:
        return T.Compose([
            T.Resize(imside),
            T.ToTensor(),
            NormSingleROI(),
        ])

    BICUBIC = T.InterpolationMode.BICUBIC
    return T.Compose([
        T.Resize(imside),

        T.RandomResizedCrop(size=imside, scale=(0.75, 1.0), ratio=(0.95, 1.05)),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.5),
        T.RandomChoice([
            T.RandomRotation(degrees=15, interpolation=BICUBIC, expand=False,
                             center=(0.5 * imside, 0.0)),
            T.RandomRotation(degrees=15, interpolation=BICUBIC, expand=False,
                             center=(0.0, 0.5 * imside)),
            T.RandomRotation(degrees=15, interpolation=BICUBIC, expand=False),
        ]),

        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.2)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),

        T.ToTensor(),
        NormSingleROI(),

        T.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ])


class PalmDataset(data.Dataset):
    """
    Palmprint ROI dataset.

    Args:
        txt     : path to a text file with lines ``<image_path> <label>``.
        train   : True → apply random augmentation and return two views per
                  sample (for Supervised Contrastive Learning).
                  False → no augmentation; both returned views are identical.
        imside  : images are resized to (imside × imside).
    """

    def __init__(self, txt: str, train: bool = True, imside: int = 128):
        self.train  = train
        self.imside = imside
        self.tf     = _make_transforms(imside, train)

        self.images_path:  list[str] = []
        self.images_label: list[str] = []
        self._read_txt(txt)

        self._label_to_indices: dict[str, list[int]] = {}
        for i, lbl in enumerate(self.images_label):
            self._label_to_indices.setdefault(lbl, []).append(i)

    def _read_txt(self, txt: str):
        with open(txt, "r") as f:
            for line in f:
                path, label = line.strip().split(" ")
                self.images_path.append(path)
                self.images_label.append(label)

    def _load(self, path: str) -> torch.Tensor:
        return self.tf(Image.open(path).convert("L"))

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, index: int):
        label = self.images_label[index]
        img1  = self._load(self.images_path[index])

        if self.train:
            candidates = self._label_to_indices[label]
            idx2 = index
            while idx2 == index and len(candidates) > 1:
                idx2 = int(np.random.choice(candidates))
            img2 = self._load(self.images_path[idx2])
        else:
            img2 = img1  

        return [img1, img2], int(label)
