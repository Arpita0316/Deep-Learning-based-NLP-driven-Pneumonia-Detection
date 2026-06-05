"""
src/dataset.py — Dataset classes and transform pipelines
"""

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── Standard transforms ───────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.3),
        transforms.RandomAffine(degrees=0, shear=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def get_tta_transforms(img_size: int = 224, n_augments: int = 5):
    """Test-Time Augmentation: returns list of transform pipelines."""
    base = [transforms.Resize((img_size, img_size))]
    norm = [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]
    augments = [
        transforms.Compose(base + norm),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + norm),
        transforms.Compose(base + [transforms.RandomRotation(5)] + norm),
        transforms.Compose(base + [transforms.ColorJitter(brightness=0.1, contrast=0.2)] + norm),
        transforms.Compose([transforms.Resize((img_size+16, img_size+16)),
                            transforms.CenterCrop(img_size)] + norm),
    ]
    return augments[:n_augments]


# ── Binary Dataset ─────────────────────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    Binary classification dataset.
    Folder structure: root/{NORMAL,PNEUMONIA}/*.jpeg
    """
    CLASS_TO_IDX = {'NORMAL': 0, 'PNEUMONIA': 1}
    IDX_TO_CLASS = {0: 'NORMAL', 1: 'PNEUMONIA'}
    EXTENSIONS   = ('*.jpeg', '*.jpg', '*.png')

    def __init__(self, root_dir, transform=None, return_path: bool = False):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.return_path = return_path
        self.samples: list = []

        for cls, idx in self.CLASS_TO_IDX.items():
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for ext in self.EXTENSIONS:
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), idx))

        if not self.samples:
            raise FileNotFoundError(f'No images found in {root_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, label, path
        return image, label

    def class_counts(self) -> dict:
        counts = {}
        for _, label in self.samples:
            cls = self.IDX_TO_CLASS[label]
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Returns inverse-frequency weights for CrossEntropyLoss."""
        counts = self.class_counts()
        total  = len(self.samples)
        n_cls  = len(self.CLASS_TO_IDX)
        weights = [total / (n_cls * counts.get(cls, 1))
                   for cls in self.CLASS_TO_IDX]
        return torch.tensor(weights, dtype=torch.float32)


# ── Multiclass Dataset ─────────────────────────────────────────────────────────

class MultiClassXRayDataset(Dataset):
    """
    3-class dataset: NORMAL / BACTERIAL pneumonia / VIRAL pneumonia.
    Infers subtype from filename: 'bacteria_*' vs 'virus_*'.
    """
    CLASS_TO_IDX = {'NORMAL': 0, 'BACTERIAL': 1, 'VIRAL': 2}
    IDX_TO_CLASS = {0: 'NORMAL', 1: 'BACTERIAL', 2: 'VIRAL'}
    EXTENSIONS   = ('*.jpeg', '*.jpg', '*.png')

    def __init__(self, root_dir, transform=None):
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples: list = []

        for ext in self.EXTENSIONS:
            for p in (self.root_dir / 'NORMAL').glob(ext):
                self.samples.append((str(p), 0))

        for ext in self.EXTENSIONS:
            for p in (self.root_dir / 'PNEUMONIA').glob(ext):
                name = p.name.lower()
                if 'bacteria' in name:
                    self.samples.append((str(p), 1))
                elif 'virus' in name:
                    self.samples.append((str(p), 2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=3)
        total  = len(self.samples)
        weights = total / (3 * counts)
        return torch.tensor(weights, dtype=torch.float32)
