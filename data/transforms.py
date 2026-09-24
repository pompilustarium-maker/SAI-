"""
Transforms and Augmentation Pipeline for WikiArt Multi-Task Learning.
Provides robust augmentations tailored for art style and period generalization,
preventing overfitting to specific canvas resolutions and lighting conditions.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from PIL import Image

try:
    from torchvision.transforms import v2 as T
    HAS_V2 = True
except ImportError:
    from torchvision import transforms as T
    HAS_V2 = False

# ImageNet default normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> T.Compose:
    """
    Constructs high-generalization training augmentations for art classification.
    Includes random cropping, color jitter (simulating varied lighting and degradation),
    slight affine transformations (brushstroke invariance), and horizontal flips.
    """
    if HAS_V2:
        return T.Compose([
            T.ToImage() if hasattr(T, 'ToImage') else T.Lambda(lambda x: x),
            T.Resize((int(img_size * 1.14), int(img_size * 1.14)), antialias=True),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToDtype(torch.float32, scale=True) if hasattr(T, 'ToDtype') else T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_val_transforms(img_size: int = 224) -> T.Compose:
    """
    Deterministic validation and test transforms.
    Preserves aspect ratio before center cropping to target resolution.
    """
    if HAS_V2:
        return T.Compose([
            T.ToImage() if hasattr(T, 'ToImage') else T.Lambda(lambda x: x),
            T.Resize((int(img_size * 1.14), int(img_size * 1.14)), antialias=True),
            T.CenterCrop(img_size),
            T.ToDtype(torch.float32, scale=True) if hasattr(T, 'ToDtype') else T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class MultiTaskMixupCutmix:
    """
    Applies Mixup and Cutmix simultaneously across Multi-Task classification targets.
    Both task target distributions (Genre & Century) are linearly blended using the same lambda.
    """
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        num_genres: int = 19,
        num_centuries: int = 8,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_genres = num_genres
        self.num_centuries = num_centuries

    def _rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        W = size[2]
        H = size[3]
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)

        return bbx1, bby1, bbx2, bby2

    def __call__(
        self,
        images: torch.Tensor,
        targets_genre: torch.Tensor,
        targets_century: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > self.prob:
            # One-hot encode targets without mixing
            one_hot_g = torch.nn.functional.one_hot(targets_genre, num_classes=self.num_genres).float()
            one_hot_c = torch.nn.functional.one_hot(targets_century, num_classes=self.num_centuries).float()
            return images, one_hot_g, one_hot_c

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        use_cutmix = torch.rand(1).item() > 0.5
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        lam = torch.distributions.Beta(alpha, alpha).sample().item()

        one_hot_g = torch.nn.functional.one_hot(targets_genre, num_classes=self.num_genres).float()
        one_hot_c = torch.nn.functional.one_hot(targets_century, num_classes=self.num_centuries).float()

        if use_cutmix:
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images_mixed = images.clone()
            images_mixed[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            # Adjust lambda according to exact pixel area
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        else:
            images_mixed = lam * images + (1.0 - lam) * images[index]

        mixed_targets_g = lam * one_hot_g + (1.0 - lam) * one_hot_g[index]
        mixed_targets_c = lam * one_hot_c + (1.0 - lam) * one_hot_c[index]

        return images_mixed, mixed_targets_g, mixed_targets_c
