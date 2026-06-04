import torch
import torch.nn as nn

__all__ = ["HaarDWT", "HaarIWT", "dwt_split", "iwt_merge"]

def _haar_split(x: torch.Tensor):
        """
        标准正交Haar二维小波分解。
        x: (B,C,H,W) with even H, W
        返回: LL, LH, HL, HH，每个(B,C,H/2,W/2)
        定义:
            x1 = x[0::2, 0::2], x2 = x[0::2, 1::2], x3 = x[1::2, 0::2], x4 = x[1::2, 1::2]
            LL = (x1 + x2 + x3 + x4) / 2
            LH = (x1 - x2 + x3 - x4) / 2
            HL = (x1 + x2 - x3 - x4) / 2
            HH = (x1 - x2 - x3 + x4) / 2
        这样保证能量保持与可逆。
        """
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 0::2]
        x4 = x[:, :, 1::2, 1::2]
        half = 0.5
        LL = (x1 + x2 + x3 + x4) * half
        LH = (x1 - x2 + x3 - x4) * half
        HL = (x1 + x2 - x3 - x4) * half
        HH = (x1 - x2 - x3 + x4) * half
        return LL, HL, LH, HH

class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        LL, HL, LH, HH = _haar_split(x)
        return torch.cat([LL, HL, LH, HH], dim=1)

class HaarIWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "Channel must be multiple of 4"
        C = C4 // 4
        LL = x[:, :C]
        HL = x[:, C:2*C]
        LH = x[:, 2*C:3*C]
        HH = x[:, 3*C:]
        # 逆变换：
        # x1 = (LL + LH + HL + HH) / 2
        # x2 = (LL - LH + HL - HH) / 2
        # x3 = (LL + LH - HL - HH) / 2
        # x4 = (LL - LH - HL + HH) / 2
        half = 0.5
        x1 = (LL + LH + HL + HH) * half
        x2 = (LL - LH + HL - HH) * half
        x3 = (LL + LH - HL - HH) * half
        x4 = (LL - LH - HL + HH) * half
        out = torch.empty((B, C, H*2, W*2), device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 0::2, 1::2] = x2
        out[:, :, 1::2, 0::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out

# Helper that returns tensor + slices for convenience

def dwt_split(x: torch.Tensor):
    LL, HL, LH, HH = _haar_split(x)
    return torch.cat([LL, HL, LH, HH], dim=1), (LL, HL, LH, HH)

def iwt_merge(LL: torch.Tensor, HL: torch.Tensor, LH: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
    half = 0.5
    x1 = (LL + LH + HL + HH) * half
    x2 = (LL - LH + HL - HH) * half
    x3 = (LL + LH - HL - HH) * half
    x4 = (LL - LH - HL + HH) * half
    B, C, H, W = LL.shape
    out = torch.empty((B, C, H*2, W*2), device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x1
    out[:, :, 0::2, 1::2] = x2
    out[:, :, 1::2, 0::2] = x3
    out[:, :, 1::2, 1::2] = x4
    return out
