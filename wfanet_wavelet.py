import torch
import torch.nn as nn
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            dx_ll, dx_lh, dx_hl, dx_hh = dx[:, :C], dx[:, C:C * 2], dx[:, C * 2:C * 3], dx[:, C * 3:]

            dx_x_ll = torch.nn.functional.conv_transpose2d(dx_ll, w_ll.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_lh = torch.nn.functional.conv_transpose2d(dx_lh, w_lh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hl = torch.nn.functional.conv_transpose2d(dx_hl, w_hl.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hh = torch.nn.functional.conv_transpose2d(dx_hh, w_hh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            return dx_x_ll + dx_x_lh + dx_x_hl + dx_x_hh, None, None, None, None
        else:
            return dx, None, None, None, None


class DWT_2D(nn.Module):
    """2D Haar DWT: (B,C,H,W) -> (B,4C,H/2,W/2), channels [LL, LH, HL, HH]."""

    def __init__(self):
        super(DWT_2D, self).__init__()
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)

        self.register_buffer("w_ll", w_ll)
        self.register_buffer("w_lh", w_lh)
        self.register_buffer("w_hl", w_hl)
        self.register_buffer("w_hh", w_hh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        _, C, _, _ = x.shape
        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        x_ll, x_lh, x_hl, x_hh = (
            x[:, : C // 4],
            x[:, C // 4 : C * 2 // 4],
            x[:, C * 2 // 4 : C * 3 // 4],
            x[:, C * 3 // 4 :],
        )
        x_1_ll = torch.nn.functional.conv_transpose2d(
            x_ll,
            w_ll.expand(C // 4, -1, -1, -1),
            stride=2,
            groups=C // 4,
        )
        x_1_lh = torch.nn.functional.conv_transpose2d(
            x_lh,
            w_lh.expand(C // 4, -1, -1, -1),
            stride=2,
            groups=C // 4,
        )
        x_1_hl = torch.nn.functional.conv_transpose2d(
            x_hl,
            w_hl.expand(C // 4, -1, -1, -1),
            stride=2,
            groups=C // 4,
        )
        x_1_hh = torch.nn.functional.conv_transpose2d(
            x_hh,
            w_hh.expand(C // 4, -1, -1, -1),
            stride=2,
            groups=C // 4,
        )
        return x_1_ll + x_1_lh + x_1_hl + x_1_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            (filters,) = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            C //= 4

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(
                dx,
                w_ll.unsqueeze(1).expand(C, -1, -1, -1) / 4,
                stride=2,
                groups=C,
            )
            x_lh = torch.nn.functional.conv2d(
                dx,
                w_lh.unsqueeze(1).expand(C, -1, -1, -1) / 4,
                stride=2,
                groups=C,
            )
            x_hl = torch.nn.functional.conv2d(
                dx,
                w_hl.unsqueeze(1).expand(C, -1, -1, -1) / 4,
                stride=2,
                groups=C,
            )
            x_hh = torch.nn.functional.conv2d(
                dx,
                w_hh.unsqueeze(1).expand(C, -1, -1, -1) / 4,
                stride=2,
                groups=C,
            )
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    """2D Haar IDWT: (B,4C,H/2,W/2) -> (B,C,H,W)."""

    def __init__(self):
        super(IDWT_2D, self).__init__()
        w_ll = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32, requires_grad=False)

        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return IDWT_Function.apply(x, self.filters)
