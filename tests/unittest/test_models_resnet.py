import pytest
import torch

from models import resnet18, resnet50


def _run_resnet(model_fn, num_samples, num_points_max, in_channels, device):
    """Run a forward pass of a resnet model and return output shapes for sanity checking."""
    torch.cuda.empty_cache()

    sample_sizes = torch.randint(
        low=1, high=num_points_max, size=(num_samples,), device=device
    )
    num_points = torch.sum(sample_sizes)

    x = torch.randn((num_points, in_channels), device=device, dtype=torch.float32)
    points = torch.rand((num_points, 3), device=device, dtype=torch.float32)
    grid_size = 1 / 512

    model = model_fn(in_channels=in_channels).to(device)
    output = model(x, points, sample_sizes, grid_size)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resnet18():
    _run_resnet(resnet18, num_samples=4, num_points_max=10000, in_channels=3, device="cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resnet50():
    _run_resnet(resnet50, num_samples=4, num_points_max=10000, in_channels=3, device="cuda:0")


if __name__ == "__main__":
    test_resnet18()
    test_resnet50()
