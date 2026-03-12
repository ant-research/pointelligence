import torch


def check_all_close(a, b, tag, rtol=1e-4, atol=None, mute=False):
    diff = (a - b).abs()
    if not mute:
        print(
            f"{tag}:\tmax_diff: {diff.max().item():.6e}\tmean_diff: {diff.mean().item():.6e}"
        )

    atol = atol if atol is not None else rtol * a.abs().max().item()
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        print(f"mean_value: {a.abs().mean().item():.6e}")
        print(a.flatten()[:32])
        print(b.flatten()[:32])
        return False
    return True
