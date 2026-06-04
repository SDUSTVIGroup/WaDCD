import torch
import torch.nn as nn

from .util import normalization
from wfanet_wavelet import DWT_2D, IDWT_2D


class FAB(nn.Module):
    """Simple frequency-domain residual block used inside WaveletSDEMBlock.

    Operates on wavelet coefficients (B, 4C, H/2, W/2).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = normalization(channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.norm2 = normalization(channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        return out + identity


class WaveletSDEMBlock(nn.Module):
    """Wavelet-based SDEM-style enhancement block.

    This block works in the spatial domain, but internally performs:
      x (B,C,H,W)
        -> DWT_2D -> (B,4C,H/2,W/2)
        -> n_fab * FAB(4C)
        -> 1x1 conv + sigmoid to get gating map (B,4C,H/2,W/2)
        -> gate * coeffs
        -> IDWT_2D -> (B,C,H,W)
        -> x + residual

    It is designed to be inserted into the UNet bottleneck, independent from timestep embedding.
    """

    def __init__(self, channels: int, n_fab: int = 3):
        super().__init__()
        self.channels = channels
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()

        freq_channels = channels * 4
        self.fab_blocks = nn.Sequential(
            *[FAB(freq_channels) for _ in range(n_fab)]
        )

        self.gate_conv = nn.Conv2d(freq_channels, freq_channels, kernel_size=1)
        self.gate_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.channels, f"WaveletSDEMBlock expects {self.channels} channels, got {C}"

        # Wavelet decomposition
        coeffs = self.dwt(x)  # (B,4C,H/2,W/2)

        # Frequency-domain refinement
        coeffs_refined = self.fab_blocks(coeffs)

        # Sigmoid gating on coefficients
        gate = self.gate_act(self.gate_conv(coeffs_refined))
        coeffs_gated = coeffs_refined * gate

        # Inverse wavelet transform back to spatial domain
        residual = self.idwt(coeffs_gated)

        return x + residual
