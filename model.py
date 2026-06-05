"""
src/model.py — PneumoniaNet Model Definition
EfficientNetB3 backbone with MC Dropout + Grad-CAM support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PneumoniaNet(nn.Module):
    """
    EfficientNetB3 backbone with:
    - Custom multi-layer classification head
    - Monte Carlo Dropout for uncertainty estimation
    - Grad-CAM hooks on the final convolutional layer

    Args:
        num_classes   (int): Number of output classes (2 = binary, 3 = multiclass)
        dropout_rate  (float): Dropout probability
        pretrained    (bool): Load ImageNet pretrained weights
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,       # Remove default head
            global_pool=''       # Keep spatial features
        )
        feature_dim = self.backbone.num_features   # 1536 for B3

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

        # Grad-CAM internals
        self._feature_maps = None
        self._gradients    = None
        self._register_gradcam_hooks()

    # ── Grad-CAM hooks ─────────────────────────────────────────────────────────
    def _register_gradcam_hooks(self):
        def fwd(module, inp, out):
            self._feature_maps = out

        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        target = self.backbone.conv_head
        target.register_forward_hook(fwd)
        target.register_full_backward_hook(bwd)

    # ── Forward ────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)   # (B, C, H, W)
        out  = self.global_pool(feat)
        out  = self.classifier(out)
        return out

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    def grad_cam(self, x: torch.Tensor, class_idx: int = None):
        """
        Returns normalised CAM heatmap (H, W) for a single image tensor.
        """
        self.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x = x.requires_grad_(True)

        output = self(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.zero_grad()
        output[0, class_idx].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._feature_maps).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().detach().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    # ── MC Dropout ────────────────────────────────────────────────────────────
    def enable_mc_dropout(self):
        """Keep dropout active during eval mode for MC Dropout inference."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_predict(self, x: torch.Tensor, n_passes: int = 30, device='cpu'):
        """
        Monte Carlo Dropout inference.

        Returns:
            mean_probs  : np.ndarray (num_classes,) — mean prediction
            uncertainty : np.ndarray (num_classes,) — std across passes
            entropy     : float — predictive entropy
        """
        import numpy as np

        self.enable_mc_dropout()
        x = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
        probs_list = []

        with torch.no_grad():
            for _ in range(n_passes):
                out   = self(x)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
                probs_list.append(probs)

        arr     = np.array(probs_list)
        mean_p  = arr.mean(axis=0)
        std_p   = arr.std(axis=0)
        entropy = -np.sum(mean_p * np.log(mean_p + 1e-8))

        self.eval()
        return mean_p, std_p, entropy


def build_model(num_classes: int = 2, dropout_rate: float = 0.3,
                pretrained: bool = True, checkpoint: str = None,
                device: str = 'cpu') -> PneumoniaNet:
    """Factory function to build and optionally load a checkpoint."""
    model = PneumoniaNet(num_classes, dropout_rate, pretrained).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f'Loaded checkpoint: {checkpoint}')
    return model
