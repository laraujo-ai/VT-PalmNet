"""
PalmNet — lightweight palmprint recognition for commercial use.

Architecture
------------
Input (1, 128, 128)  grayscale palm ROI
  ↓  GaborLayer           — 9-orientation learnable Gabor bank (stride 2)
  ↓  MobileNetV3-Small    — Apache 2.0 (torchvision). First conv patched to 9-ch.
  ↓  FC + BN + Hardswish  — 576 → embed_dim projection
  ↓  L2-normalise          — inference embedding (default: 256-dim)
  ↓  ArcMarginProduct      — ArcFace head, used only during training

Design choices
--------------
* No CompetitiveBlock (CCNet's proprietary contribution).
* GaborLayer is a classical signal-processing technique, not paper-specific.
* MobileNetV3-Small uses depthwise-separable convolutions + built-in SE
  attention — optimised for ARM/edge hardware.
* ArcFace implementation follows ronghuaiyang/arcface-pytorch (MIT licence).

Licence summary
---------------
GaborLayer  : public domain (classical Gabor filter maths)
MobileNetV3 : Apache 2.0 (torchvision / PyTorch team)
ArcFace     : MIT (ronghuaiyang/arcface-pytorch)
PyTorch     : BSD-3
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models import mobilenet_v3_small


# ---------------------------------------------------------------------------
# Learnable Gabor filter bank
# ---------------------------------------------------------------------------

class GaborLayer(nn.Module):
    """
    Multi-orientation learnable Gabor filter bank.

    The Gabor function is a standard, unpatentable signal-processing technique:

        x_r = x·cos(θ) + y·sin(θ)
        y_r = -x·sin(θ) + y·cos(θ)
        gb  = -exp(-0.5·((γ·x_r)² + y_r²) / (8·σ²)) · cos(2π·f·x_r)
        gb  = gb - mean(gb)          # zero-mean normalisation

    Parameters are learnable (σ, f, γ); orientations are fixed and equally
    spaced in [0, π).

    Input  : (B, 1,  H,  W)
    Output : (B, num_orientations,  H//stride,  W//stride)
    """

    def __init__(
        self,
        num_orientations: int = 9,
        kernel_size: int = 17,
        stride: int = 2,
        padding: int = 8,
    ):
        super().__init__()
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Learnable envelope / frequency parameters
        self.sigma = nn.Parameter(torch.tensor(9.2))
        self.freq  = nn.Parameter(torch.tensor(0.057))
        self.gamma = nn.Parameter(torch.tensor(2.0))

        # Fixed, equally-spaced orientations in [0, π)
        angles = torch.arange(num_orientations).float() * math.pi / num_orientations
        self.register_buffer("theta", angles)  # not learned

    def _make_filters(self) -> torch.Tensor:
        half = self.kernel_size // 2
        coords = torch.arange(
            -half, half + 1, dtype=torch.float32, device=self.sigma.device
        )
        y_c, x_c = torch.meshgrid(coords, coords, indexing="ij")  # [k, k]

        # Expand to [num_orientations, 1, k, k]
        x = x_c.unsqueeze(0).unsqueeze(0).expand(self.num_orientations, 1, -1, -1)
        y = y_c.unsqueeze(0).unsqueeze(0).expand(self.num_orientations, 1, -1, -1)

        cos_t = torch.cos(self.theta).view(-1, 1, 1, 1)
        sin_t = torch.sin(self.theta).view(-1, 1, 1, 1)
        x_r = x * cos_t + y * sin_t   # rotated coordinates
        y_r = -x * sin_t + y * cos_t

        gb = -torch.exp(
            -0.5 * ((self.gamma * x_r) ** 2 + y_r ** 2) / (8.0 * self.sigma ** 2)
        ) * torch.cos(2.0 * math.pi * self.freq * x_r)

        gb = gb - gb.mean(dim=[2, 3], keepdim=True)  # zero-mean per filter
        return gb  # [num_orientations, 1, k, k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._make_filters(), stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# ArcFace head
# Based on: https://github.com/ronghuaiyang/arcface-pytorch  (MIT Licence)
# ---------------------------------------------------------------------------

class ArcMarginProduct(nn.Module):
    """
    ArcFace additive angular margin loss head.

    Training : applies angular margin m to the true-class cosine logit.
    Inference : returns s · cosine(W, x) — acts as a linear classifier.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if not self.training:
            return self.s * cosine

        assert labels is not None, "labels required in training mode"
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi  = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


# ---------------------------------------------------------------------------
# PalmNet
# ---------------------------------------------------------------------------

class PalmNet(nn.Module):
    """
    PalmNet: commercial palmprint recognition network.

    Shape flow (default config):
        (B, 1, 128, 128)
        → GaborLayer                 (B,  9,  64,  64)
        → MobileNetV3-Small          (B, 576,   2,   2)
        → AdaptiveAvgPool2d(1)       (B, 576,   1,   1)
        → flatten                    (B, 576)
        → FC + BN + Hardswish        (B, embed_dim)     [default 256]
        → L2-normalise               (B, embed_dim)     ← inference output
        → ArcMarginProduct           (B, num_classes)   ← training only

    Args:
        num_classes : number of palm identities for classification head.
        embed_dim   : dimension of the output embedding (default 256).
        s           : ArcFace scale (default 30).
        m           : ArcFace angular margin in radians (default 0.5).
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        s: float = 30.0,
        m: float = 0.5,
    ):
        super().__init__()

        # Stage 1 — Learnable Gabor bank (9 orientations, stride 2)
        # Input:  (B, 1, 128, 128) → Output: (B, 9, 64, 64)
        self.gabor = GaborLayer(
            num_orientations=9,
            kernel_size=17,
            stride=2,
            padding=8,
        )

        # Stage 2 — MobileNetV3-Small backbone (Apache 2.0)
        # Patch the first conv from 3-channel (RGB) → 9-channel (Gabor).
        mbv3 = mobilenet_v3_small(weights=None)
        first_conv = mbv3.features[0][0]          # Conv2d(3, 16, 3, stride=2)
        mbv3.features[0][0] = nn.Conv2d(
            9,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        self.backbone = mbv3.features             # all InvRes + SE blocks
        self.avgpool  = mbv3.avgpool              # AdaptiveAvgPool2d(1)

        # Stage 3 — Embedding head: 576 → embed_dim
        self.embed_head = nn.Sequential(
            nn.Linear(576, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Hardswish(),
        )

        # Stage 4 — ArcFace head (training only)
        self.arc_head = ArcMarginProduct(embed_dim, num_classes, s=s, m=m)

    # ------------------------------------------------------------------

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding.  Shape: (B, embed_dim)."""
        x = self.gabor(x)       # (B, 9, 64, 64)
        x = self.backbone(x)    # (B, 576, 2, 2)  — exact size depends on input
        x = self.avgpool(x)     # (B, 576, 1, 1)
        x = x.flatten(1)        # (B, 576)
        x = self.embed_head(x)  # (B, embed_dim)
        return F.normalize(x, dim=1)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward — returns (logits, embedding)."""
        emb    = self.get_embedding(x)
        logits = self.arc_head(emb, labels)
        return logits, emb

    def getFeatureCode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for get_embedding — compatible with CCNet evaluation scripts."""
        return self.get_embedding(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    net = PalmNet(num_classes=600)
    net.eval()

    dummy = torch.randn(4, 1, 128, 128)
    with torch.no_grad():
        logits, emb = net(dummy, labels=None)

    print("Input shape  :", dummy.shape)
    print("Logits shape :", logits.shape)   # (4, 600)
    print("Embed shape  :", emb.shape)      # (4, 256)
    print("Embed norm   :", emb.norm(dim=1))  # should be ~1.0

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total params : {total_params:,}")
