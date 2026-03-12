import torch

from torch import Tensor


def index_mode(
    x: Tensor, target_indices: Tensor, target_size: int, ignore_value_zero=False
):
    assert x.dtype == torch.int32 or x.dtype == torch.int64
    assert x.numel() == target_indices.numel()

    x_high = torch.max(x) + 1
    if x_high == 1:
        x_mode = torch.zeros(size=(target_size,), dtype=torch.int64, device=x.device)
    else:
        x_lift = x + target_indices * x_high
        x_count = torch.bincount(x_lift, minlength=target_size * x_high)
        del x_lift

        x_count = torch.reshape(x_count, (target_size, -1))
        if ignore_value_zero:
            # compute x_mode by max count of nonzero labels in the grid
            x_count_nonzero = x_count[:, 1:]
            x_max = torch.amax(x_count_nonzero, dim=-1)
            x_argmax = torch.argmax(x_count_nonzero, dim=-1).add_(1)
            x_mode = torch.where(x_max > 0, x_argmax, 0)
        else:
            x_mode = torch.argmax(x_count, dim=-1)

    return x_mode
