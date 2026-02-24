"""
PalmNet — multi-scale orientation feature network for palm recognition.

Architecture
------------
Input (B, 1, 128, 128)  grayscale palm ROI
  ↓  OB1  [LGC(9 ch, k=35)  + soft-select + ChannelGate + PPU]  ×2 orders
  ↓  OB2  [LGC(36 ch, k=17) + soft-select + ChannelGate + PPU]  ×2 orders
  ↓  OB3  [LGC(9 ch, k=7)   + soft-select + ChannelGate + PPU]  ×2 orders
  ↓  concat OB1 // OB2 // OB3
  ↓  FC(feat_dim → fc_hidden) + BN + HardSwish + FC(fc_hidden → embed_dim)
  ↓  L2-normalised embedding
  ↓  ArcMarginProduct — training head
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GaborConv2d(nn.Module):
    """
    Learnable Gabor filter bank.

    Generates a bank of oriented Gabor filters whose shape parameters
    (sigma, frequency, aspect ratio) are learned during training.

    Args:
        channel_in  : number of input channels (typically 1)
        channel_out : number of filter orientations
        kernel_size : spatial size of each filter
        init_ratio  : scale factor for the initial receptive-field parameters
    """

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super().__init__()

        self.channel_in  = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.init_ratio  = max(init_ratio, 1e-6)

        self.gamma = nn.Parameter(torch.FloatTensor([2.0]))
        self.sigma = nn.Parameter(torch.FloatTensor([9.2 * self.init_ratio]))
        self.theta = nn.Parameter(
            torch.arange(0, channel_out).float() * math.pi / channel_out,
            requires_grad=False,
        )
        self.f   = nn.Parameter(torch.FloatTensor([0.057 / self.init_ratio]))
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def _gen_gabor_bank(self):
        ksize  = self.kernel_size
        half   = ksize // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32, device=self.sigma.device)

        C, N = self.channel_out, self.channel_in
        y = coords.view(1, -1).repeat(C, N, ksize, 1)
        x = coords.view(-1, 1).repeat(C, N, 1, ksize)

        cos_t = torch.cos(self.theta).view(-1, 1, 1, 1)
        sin_t = torch.sin(self.theta).view(-1, 1, 1, 1)
        x_r = x * cos_t + y * sin_t
        y_r = -x * sin_t + y * cos_t

        gb = -torch.exp(
            -0.5 * ((self.gamma * x_r) ** 2 + y_r ** 2) / (8 * self.sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * self.f.view(-1, 1, 1, 1) * x_r + self.psi.view(-1, 1, 1, 1))

        return gb - gb.mean(dim=[2, 3], keepdim=True)

    def forward(self, x):
        return F.conv2d(x, self._gen_gabor_bank(), stride=self.stride, padding=self.padding)


class ChannelGate(nn.Module):
    """
    Channel attention via global average pooling followed by a two-layer MLP
    that produces a per-channel sigmoid gate.

    Args:
        channel   : number of input channels
        reduction : bottleneck ratio for the MLP hidden dimension
    """

    def __init__(self, channel, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OrientationBlock(nn.Module):
    """
    Two-order orientation feature extractor.

    Applies a learnable Gabor filter bank and uses a soft-selection mechanism
    (weighted combination of channel-wise and spatial softmax responses) to
    highlight dominant orientations and locations.  Channel attention and a
    spatial compression stage (PPU) are applied to each order independently;
    the two resulting flat vectors are concatenated as output.

    Args:
        channel_in    : number of input channels
        n_orientations: number of Gabor filter orientations
        ksize         : Gabor kernel size
        stride        : unused externally; internal stride is fixed to 2
        padding       : Gabor padding
        weight        : channel-softmax weight; spatial axes share (1 - weight) / 2 each
        init_ratio    : Gabor scale factor
        o1            : PPU output channels per order (controls feature vector length)
        o2            : unused; kept for API compatibility
    """

    def __init__(self, channel_in, n_orientations, ksize, stride, padding,
                 weight=0.8, init_ratio=1, o1=32, o2=12):
        super().__init__()

        self.weight_chan = weight
        self.weight_spa  = (1.0 - weight) / 2.0

        self.gabor1 = GaborConv2d(channel_in,      n_orientations, ksize, stride=2,
                                  padding=ksize // 2, init_ratio=init_ratio)
        self.gabor2 = GaborConv2d(n_orientations,  n_orientations, ksize, stride=2,
                                  padding=ksize // 2, init_ratio=init_ratio)

        self.softmax_c = nn.Softmax(dim=1)
        self.softmax_h = nn.Softmax(dim=2)
        self.softmax_w = nn.Softmax(dim=3)

        self.gate1 = ChannelGate(n_orientations)
        self.gate2 = ChannelGate(n_orientations)

        self.ppu1    = nn.Conv2d(n_orientations, o1 // 2, 5, 2, 0)
        self.ppu2    = nn.Conv2d(n_orientations, o1 // 2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

    def _soft_select(self, x):
        return (self.weight_chan * self.softmax_c(x)
                + self.weight_spa * (self.softmax_h(x) + self.softmax_w(x)))

    def forward(self, x):
        g1 = self.gabor1(x)

        x1 = self._soft_select(g1)
        x1 = self.gate1(x1)
        x1 = self.maxpool(self.ppu1(x1))

        x2 = self.gabor2(g1)
        x2 = self._soft_select(x2)
        x2 = self.gate2(x2)
        x2 = self.maxpool(self.ppu2(x2))

        return torch.cat([x1.flatten(1), x2.flatten(1)], dim=1)


class ArcMarginProduct(nn.Module):
    """
    Additive angular margin classification head (ArcFace).

    Applies a cosine similarity classifier with an additive margin m in the
    angular space during training to improve embedding discriminability.

    Args:
        in_features  : embedding dimension
        out_features : number of identity classes
        s            : feature norm scale factor
        m            : additive angular margin (radians)
        easy_margin  : use a relaxed margin boundary
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
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

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if not self.training:
            return self.s * cosine

        assert labels is not None
        sine  = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = torch.where(cosine > self.th, phi, cosine - self.mm) if not self.easy_margin \
                else torch.where(cosine > 0, phi, cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s


class PalmNet(nn.Module):
    """
    Multi-scale palm recognition network.

    Three parallel OrientationBlocks process the input at different spatial
    scales (coarse, medium, fine).  Their flat outputs are concatenated and
    projected to a compact L2-normalised embedding used for identity matching.

    Args:
        num_classes  : number of palm identities
        embed_dim    : output embedding dimension (default 512)
        ppu_channels : PPU output channels per order in each OrientationBlock.
                       Controls the raw feature size before the FC layers.
                       32 → full size, 16 → ~4× smaller, 8 → ~16× smaller.
        fc_hidden    : hidden width of the first FC layer (default 4096).
                       Reduce to 2048 or 1024 for a lighter model.
        weight       : channel soft-selection weight for all blocks (default 0.8)
        s, m         : ArcFace scale / margin
    """

    def __init__(self, num_classes: int, embed_dim: int = 512,
                 ppu_channels: int = 32, fc_hidden: int = 4096,
                 weight: float = 0.8, s: float = 30.0, m: float = 0.5):
        super().__init__()

        self.ob1 = OrientationBlock(channel_in=1, n_orientations=9,  ksize=35,
                                    stride=3, padding=17, init_ratio=1.0,  weight=weight, o1=ppu_channels)
        self.ob2 = OrientationBlock(channel_in=1, n_orientations=36, ksize=17,
                                    stride=3, padding=8,  init_ratio=0.5,  weight=weight, o1=ppu_channels)
        self.ob3 = OrientationBlock(channel_in=1, n_orientations=9,  ksize=7,
                                    stride=3, padding=3,  init_ratio=0.25, weight=weight, o1=ppu_channels)

        with torch.no_grad():
            dummy    = torch.zeros(1, 1, 128, 128)
            feat_dim = (self.ob1(dummy).shape[1] +
                        self.ob2(dummy).shape[1] +
                        self.ob3(dummy).shape[1])

        self.fc1      = nn.Linear(feat_dim, fc_hidden)
        self.bn1      = nn.BatchNorm1d(fc_hidden)
        self.fc2      = nn.Linear(fc_hidden, embed_dim)
        self.drop     = nn.Dropout(p=0.5)
        self.arc_head = ArcMarginProduct(embed_dim, num_classes, s=s, m=m)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embedding.  Shape: (B, embed_dim)."""
        feat = torch.cat([self.ob1(x), self.ob2(x), self.ob3(x)], dim=1)
        feat = F.hardswish(self.bn1(self.fc1(feat)))
        return F.normalize(self.fc2(feat), dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """Training forward — returns (logits, embedding)."""
        emb    = self.get_embedding(x)
        logits = self.arc_head(self.drop(emb), labels)
        return logits, emb

    def getFeatureCode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for get_embedding — compatible with evaluation scripts."""
        return self.get_embedding(x)


if __name__ == "__main__":
    net   = PalmNet(num_classes=600, embed_dim=512, weight=0.8)
    dummy = torch.randn(4, 1, 128, 128)
    net.eval()
    with torch.no_grad():
        logits, emb = net(dummy)

    print("Input  :", dummy.shape)
    print("Logits :", logits.shape)
    print("Embed  :", emb.shape)
    print("Norm   :", emb.norm(dim=1))
    print(f"Params : {sum(p.numel() for p in net.parameters()):,}")
