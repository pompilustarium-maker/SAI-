"""
Multi-Head Art Classification Network.
Combines modern Vision Backbones (ConvNeXt, Swin Transformer, EfficientNetV2)
with specialized MLP projection heads for joint Genre and Historical Century prediction.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

import torchvision.models as tv_models


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for handling severe class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ArtClassificationHead(nn.Module):
    """
    Multi-Layer Perceptron (MLP) classification head with normalization,
    GELU activations, and progressive dropout regularization.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        dropout1: float = 0.35,
        dropout2: float = 0.20
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(dropout1),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout2),

            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadArtNet(nn.Module):
    """
    Multi-Task Deep Learning Model for Artwork Classification:
      - Shared Vision Backbone (ConvNeXt-Tiny by default, Swin, EfficientNetV2)
      - Shared Global Pooling & Bottleneck
      - Head 1: Genre Classifier
      - Head 2: Century / Historical Period Classifier
    """
    def __init__(
        self,
        num_genres: int,
        num_centuries: int,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_genres = num_genres
        self.num_centuries = num_centuries

        # Initialize backbone
        self.backbone, self.in_features = self._build_backbone(backbone_name, pretrained)

        # Global average pool if backbone outputs 4D tensor
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Multi-Head MLPs
        self.genre_head = ArtClassificationHead(
            in_features=self.in_features,
            num_classes=num_genres,
            dropout1=dropout,
            dropout2=dropout * 0.6
        )

        self.century_head = ArtClassificationHead(
            in_features=self.in_features,
            num_classes=num_centuries,
            dropout1=dropout,
            dropout2=dropout * 0.6
        )

    def _build_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        name_lower = name.lower()

        # Check timm first if installed
        if HAS_TIMM:
            try:
                # Create backbone without classifier head
                backbone = timm.create_model(
                    name,
                    pretrained=pretrained,
                    num_classes=0,  # pooled features output
                    features_only=False
                )
                in_features = backbone.num_features
                return backbone, in_features
            except Exception:
                pass

        # Native torchvision fallbacks
        weights = "DEFAULT" if pretrained else None
        if "convnext_tiny" in name_lower:
            m = tv_models.convnext_tiny(weights=weights)
            in_features = m.classifier[2].in_features
            m.classifier = nn.Identity()
            return m, in_features
        elif "convnext_small" in name_lower:
            m = tv_models.convnext_small(weights=weights)
            in_features = m.classifier[2].in_features
            m.classifier = nn.Identity()
            return m, in_features
        elif "swin_t" in name_lower or "swin_tiny" in name_lower:
            m = tv_models.swin_t(weights=weights)
            in_features = m.head.in_features
            m.head = nn.Identity()
            return m, in_features
        elif "efficientnet_v2_s" in name_lower:
            m = tv_models.efficientnet_v2_s(weights=weights)
            in_features = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, in_features
        else:
            # ResNet50 fallback if unknown
            m = tv_models.resnet50(weights=weights)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
            return m, in_features

    def get_last_conv_layer(self) -> Optional[nn.Module]:
        """Returns target feature map layer for Grad-CAM generation."""
        if hasattr(self.backbone, "features"):
            # ConvNeXt / EfficientNet / VGG in torchvision
            return self.backbone.features[-1]
        elif hasattr(self.backbone, "layer4"):
            # ResNet in torchvision
            return self.backbone.layer4[-1]
        elif hasattr(self.backbone, "stages"):
            # ConvNeXt in timm
            return self.backbone.stages[-1]
        return None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts deep pooled representation [B, D]."""
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            feats = self.global_pool(feats)
            feats = torch.flatten(feats, 1)
        return feats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing logits for both heads simultaneously:
        Returns:
            genre_logits: [B, num_genres]
            century_logits: [B, num_centuries]
        """
        feats = self.forward_features(x)
        genre_logits = self.genre_head(feats)
        century_logits = self.century_head(feats)
        return genre_logits, century_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns softmax normalized probabilities for both tasks."""
        self.eval()
        g_logits, c_logits = self(x)
        g_probs = F.softmax(g_logits, dim=-1)
        c_probs = F.softmax(c_logits, dim=-1)
        return g_probs, c_probs

    # --- Transfer Learning & Two-Stage Training Control ---

    def freeze_backbone(self) -> None:
        """Freezes all backbone parameters for head warm-up stage."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Ensure heads remain trainable
        for param in self.genre_head.parameters():
            param.requires_grad = True
        for param in self.century_head.parameters():
            param.requires_grad = True

    def unfreeze_top_stages(self, num_stages: int = 2) -> None:
        """
        Unfreezes the top N stages/blocks of the backbone for fine-tuning
        while keeping earlier low-level feature extractors frozen.
        """
        # Determine child blocks/stages in backbone
        named_children = list(self.backbone.named_children())
        if named_children:
            # Unfreeze the last num_stages child modules
            for name, child in named_children[-num_stages:]:
                for param in child.parameters():
                    param.requires_grad = True
        else:
            # Fallback: unfreeze all
            self.unfreeze_all()

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters across backbone and heads."""
        for param in self.parameters():
            param.requires_grad = True


class MultiTaskCriterion(nn.Module):
    """
    Weighted Multi-Task Loss supporting Class Weights and Focal Loss:
    Total Loss = lambda_g * Loss_Genre + lambda_c * Loss_Century
    """
    def __init__(
        self,
        genre_weight: float = 1.0,
        century_weight: float = 0.8,
        genre_class_weights: Optional[torch.Tensor] = None,
        century_class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = False,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.genre_weight = genre_weight
        self.century_weight = century_weight

        if use_focal:
            self.genre_loss_fn = FocalLoss(alpha=genre_class_weights, gamma=2.0, label_smoothing=label_smoothing)
            self.century_loss_fn = FocalLoss(alpha=century_class_weights, gamma=2.0, label_smoothing=label_smoothing)
        else:
            self.genre_loss_fn = nn.CrossEntropyLoss(
                weight=genre_class_weights,
                label_smoothing=label_smoothing
            )
            self.century_loss_fn = nn.CrossEntropyLoss(
                weight=century_class_weights,
                label_smoothing=label_smoothing
            )

    def forward(
        self,
        genre_logits: torch.Tensor,
        century_logits: torch.Tensor,
        genre_targets: torch.Tensor,
        century_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_g = self.genre_loss_fn(genre_logits, genre_targets)
        loss_c = self.century_loss_fn(century_logits, century_targets)
        total_loss = (self.genre_weight * loss_g) + (self.century_weight * loss_c)
        return total_loss, loss_g, loss_c
