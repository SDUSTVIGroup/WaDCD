import torch
import torch.nn as nn
from typing import Optional
from wfanet_wavelet import DWT_2D, IDWT_2D
from .util import normalization
from ldm.modules.attention import SpatialTransformer


class MultiheadAttention2D(nn.Module):
    """Simple 2D multi-head attention operating on (B,C,H,W).

    This is a lightweight variant used inside WaveletMFFA.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.to_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, q, k, v):
        # q/k/v: (B,C,H,W)
        B, C, H, W = q.shape
        def reshape_heads(x):
            x = x.view(B, self.num_heads, self.head_dim, H * W)
            return x

        q = reshape_heads(self.to_q(q))  # (B,heads,dim,HW)
        k = reshape_heads(self.to_k(k))
        v = reshape_heads(self.to_v(v))

        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhdn,bhdm->bhnm", q * scale, k)  # (B,heads,N,N)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)  # (B,heads,dim,N)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return out


class WaveletMFFA(nn.Module):
    """WFANet-style Multi-Frequency Fusion Attention in wavelet domain.

    Input: coeffs (B,4C,H/2,W/2) from DWT_2D.
    Output: band-wise DoD (B,4C,H/2,W/2).
    """

    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, num_heads: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels or in_channels

        # FATG-style band-wise conv + BN + 非线性，用于显式编码 LL / LH / HL / HH 频率特征。
        def make_fatg_block(ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(),
            )

        self.fatg_ll = make_fatg_block(in_channels)
        self.fatg_lh = make_fatg_block(in_channels)
        self.fatg_hl = make_fatg_block(in_channels)
        self.fatg_hh = make_fatg_block(in_channels)

        # Projections for LL / HF / fused value（在 FATG 之后）。
        self.proj_ll = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1)
        self.proj_hf = nn.Conv2d(in_channels * 3, self.hidden_channels, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels * 4, self.hidden_channels, kernel_size=1)

        self.attn = MultiheadAttention2D(self.hidden_channels, num_heads=num_heads)

        # ADFR-style per-band reconstruction heads：从注意力输出重构四个子带 DoD。
        self.recon_ll = nn.Conv2d(self.hidden_channels, in_channels, kernel_size=3, padding=1)
        self.recon_lh = nn.Conv2d(self.hidden_channels, in_channels, kernel_size=3, padding=1)
        self.recon_hl = nn.Conv2d(self.hidden_channels, in_channels, kernel_size=3, padding=1)
        self.recon_hh = nn.Conv2d(self.hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = coeffs.shape
        C = C4 // 4
        ll, lh, hl, hh = torch.split(coeffs, C, dim=1)

        # 1) FATG：对每个子带做 conv+BN+SiLU，显式编码频率特征
        ll_f = self.fatg_ll(ll)
        lh_f = self.fatg_lh(lh)
        hl_f = self.fatg_hl(hl)
        hh_f = self.fatg_hh(hh)

        # 2) 生成 Q / K / V
        q = self.proj_hf(torch.cat([lh_f, hl_f, hh_f], dim=1))          # 高频 query
        k = self.proj_ll(ll_f)                                          # 结构 key
        v = self.proj_v(torch.cat([ll_f, lh_f, hl_f, hh_f], dim=1))     # 融合 value

        # 3) MFFA 注意力
        attn_out = self.attn(q, k, v)

        # 4) ADFR-style 重构：为每个 band 单独生成 DoD 分量
        dod_ll = self.recon_ll(attn_out)
        dod_lh = self.recon_lh(attn_out)
        dod_hl = self.recon_hl(attn_out)
        dod_hh = self.recon_hh(attn_out)

        dod_bands = torch.cat([dod_ll, dod_lh, dod_hl, dod_hh], dim=1)
        return dod_bands


class WaveletSDEMGate(nn.Module):
    """LL-guided high-frequency gate corresponding to the paper's WGDRB.

    Wavelet coefficient order used by DWT_2D:
        [LL, LH, HL, HH]
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        if channels % 4 != 0:
            raise ValueError(
                f"Expected wavelet channels divisible by 4, got {channels}"
            )

        self.channels = channels
        self.latent_channels = channels // 4

        hidden = max(self.latent_channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            # FC1 after GAP: C -> C/reduction
            nn.Conv2d(
                self.latent_channels,
                hidden,
                kernel_size=1,
            ),
            nn.SiLU(),

            # FC2: C/reduction -> 3C
            nn.Conv2d(
                hidden,
                3 * self.latent_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        dod_bands: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> torch.Tensor:
        del dod_bands  # WGDRB directly gates the original wavelet coefficients.

        c = self.latent_channels

        if coeffs.ndim != 4 or coeffs.shape[1] != 4 * c:
            raise ValueError(
                f"Expected coefficients with {4 * c} channels, "
                f"got shape {tuple(coeffs.shape)}"
            )

        # DWT_2D output order: [LL, LH, HL, HH]
        z_ll = coeffs[:, :c]
        z_hf = coeffs[:, c:]  # concat(LH, HL, HH)

        # g_t = sigmoid(FC2(SiLU(FC1(GAP(z_t^LL)))))
        gate = self.mlp(self.pool(z_ll))

        # Only high-frequency subbands are gated.
        gated_hf = gate * z_hf

        # LL remains unchanged before IDWT.
        return torch.cat([z_ll, gated_hf], dim=1)


class WaveletDoDPredictor(nn.Module):
    """Wavelet DoD Predictor wrapper around a base UNet.

    Implements: z_t -> DWT -> MFFA -> (optional SDEM gate) -> IDWT -> eta_hat.
    The interface matches the original UNet: forward(z_t, t, context=None, mask=None).
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_mffa: bool = True,
        use_sdem_gate: bool = False,
        num_heads: int = 4,
        combine_mode: str = "wave-only",
    ):
        super().__init__()
        self.base_model = base_model
        self.use_mffa = use_mffa
        self.use_sdem_gate = use_sdem_gate  # 允许单独启用 SDEM gate
        # 组合方式：
        #   "unet-only" : 仅使用 base UNet 的 DoD 预测
        #   "wave-only" : 仅使用 Wavelet DoD Predictor（当前默认行为）
        #   "sum"       : eta = eta_unet + eta_wave，UNet 为主，wavelet 提供残差增强
        assert combine_mode in {"unet-only", "wave-only", "sum"}, f"invalid combine_mode: {combine_mode}"
        self.combine_mode = combine_mode

        # 波形变换在 latent 维度上工作，通道数与 latent 一致
        # 在第一次前向时根据输入通道数 lazy init MFFA/SDEM
        self.dwt = DWT_2D()
        self.idwt = IDWT_2D()
        self.mffa: Optional[WaveletMFFA] = None
        self.sdem_gate: Optional[WaveletSDEMGate] = None
        self.dod_head: Optional[nn.Conv2d] = None  # wavelet分支DoD预测头

        self.num_heads = num_heads

    def _lazy_build(self, in_channels: int):
        if self.mffa is None and self.use_mffa:
            self.mffa = WaveletMFFA(in_channels, hidden_channels=in_channels, num_heads=self.num_heads)
        if self.sdem_gate is None and self.use_sdem_gate:
            self.sdem_gate = WaveletSDEMGate(in_channels * 4)
        if self.dod_head is None:
            self.dod_head = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None, mask=None):
        """Forward like UNetModel: returns eta_hat.

        支持 MFFA/SDEM gate 独立或联合使用。
        """
        # 显式指定 unet-only 且不需要 wavelet 分支时，直接用 base UNet
        if (not self.use_mffa and not self.use_sdem_gate) or self.combine_mode == "unet-only":
            return self.base_model(x, t, context=context, mask=mask)

        B, C, H, W = x.shape
        # lazy 初始化 MFFA/SDEM，并保证它们与 base_model 保持相同设备和 dtype
        self._lazy_build(C)
        device = x.device
        dtype = x.dtype
        if self.mffa is not None:
            self.mffa.to(device=device, dtype=dtype)
        if self.sdem_gate is not None:
            self.sdem_gate.to(device=device, dtype=dtype)

        # 1) Wavelet decomposition of latent
        coeffs = self.dwt(x)  # (B,4C,H/2,W/2)

        # 2) MFFA predicts band-wise DoD（可选）
        if self.use_mffa and self.mffa is not None:
            dod_bands = self.mffa(coeffs)
        else:
            dod_bands = coeffs

        # 3) Optional SDEM gate（可选）
        if self.use_sdem_gate and self.sdem_gate is not None:
            dod_bands = self.sdem_gate(dod_bands, coeffs)

        # 4) Merge to spatial DoD via IDWT
        eta_wave = self.idwt(dod_bands)
        # 5) Wavelet分支加DoD预测头
        eta_wave = self.dod_head(eta_wave)

        if self.combine_mode == "wave-only":
            # 仅使用 wavelet DoD
            return eta_wave
        elif self.combine_mode == "sum":
            # UNet 为主，wavelet 提供残差增强
            eta_unet = self.base_model(x, t, context=context, mask=mask)
            return eta_unet + eta_wave
        else:
            # 理论上不会到这里，之前 assert 已覆盖
            return eta_wave
