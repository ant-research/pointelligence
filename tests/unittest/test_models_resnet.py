import torch

from models import resnet18, resnet50


def test_resnets():
    num_samples = 8
    num_points_max = 100000
    in_channels = 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    sample_sizes = torch.randint(
        low=0, high=num_points_max, size=(num_samples,), device=device
    )
    num_points = torch.sum(sample_sizes)

    x = torch.randn((num_points, in_channels), device=device, dtype=torch.float32)
    points = torch.rand((num_points, 3), device=device, dtype=torch.float32)
    grid_size = 1 / 512

    resnet_18 = resnet18(in_channels=in_channels).to(device)
    output_resnet_18 = resnet_18(x, points, sample_sizes, grid_size)

    resnet_50 = resnet50(in_channels=in_channels).to(device)
    output_resnet_50 = resnet_50(x, points, sample_sizes, grid_size)

    pass


if __name__ == "__main__":
    test_resnets()
