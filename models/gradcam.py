"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Multi-Head Art Networks.
Enables visual explainability by highlighting brushstroke textures, color gradients,
and compositional regions driving both Genre and Century predictions.
"""

from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskGradCAM:
    """
    Grad-CAM engine supporting dual-task target selection:
    Target 'genre' or 'century' output heads.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_last_conv_layer"):
                target_layer = model.get_last_conv_layer()
            if target_layer is None:
                # Fallback: search backwards through modules for Conv2d
                for module in reversed(list(model.modules())):
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                        break

        if target_layer is None:
            raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")

        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        task: str = "genre",
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1].

        Args:
            input_tensor: [1, 3, H, W] normalized image tensor
            task: 'genre' or 'century'
            target_class_idx: class index to explain (None = model's top-1 prediction)

        Returns:
            cam: 2D numpy array [H, W] normalized in [0.0, 1.0]
        """
        self.model.zero_grad()

        # Forward pass
        genre_logits, century_logits = self.model(input_tensor)

        if task.lower() == "genre":
            logits = genre_logits
        else:
            logits = century_logits

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=-1).item()

        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients during backward hook.")

        # Pool gradients across spatial dimensions
        # activations: [1, C, H', W'], gradients: [1, C, H', W']
        gradients = self.gradients
        activations = self.activations

        # Handle transformer / permutation outputs if needed
        if activations.ndim == 3:
            # e.g., Swin Transformer token shape [B, L, C] -> reshape
            B, L, C = activations.shape
            side = int(L ** 0.5)
            activations = activations.transpose(1, 2).view(B, C, side, side)
            gradients = gradients.transpose(1, 2).view(B, C, side, side)

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H', W']

        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input spatial size
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_cam_on_image(
    original_img: Union[Image.Image, np.ndarray],
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet"
) -> Image.Image:
    """
    Overlays Grad-CAM heatmap over original RGB image.
    Supports pure numpy colormaps without requiring GUI dependencies.
    """
    if isinstance(original_img, Image.Image):
        img_np = np.array(original_img.convert("RGB")).astype(np.float32) / 255.0
    else:
        img_np = original_img.astype(np.float32)
        if img_np.max() > 1.0:
            img_np /= 255.0

    H, W = img_np.shape[0], img_np.shape[1]

    # Resize CAM to match image dimensions if different
    if cam.shape != (H, W):
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam = np.array(cam_pil.resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0

    # Fast custom JET colormap lookup
    def apply_jet(val):
        four_val = 4.0 * val
        r = np.clip(np.minimum(four_val - 1.5, -four_val + 4.5), 0, 1)
        g = np.clip(np.minimum(four_val - 0.5, -four_val + 3.5), 0, 1)
        b = np.clip(np.minimum(four_val + 0.5, -four_val + 2.5), 0, 1)
        return np.stack([r, g, b], axis=-1)

    heatmap = apply_jet(cam)

    # Blend
    blended = (1.0 - alpha) * img_np + alpha * heatmap
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
