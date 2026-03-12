import torch
import torch.nn as nn


class RaggedNorm(nn.Module):
    """
    Unified Engine supporting pluggable reduction implementations.
    """

    def __init__(
        self,
        num_features,
        norm_type="batch",
        num_groups=1,
        eps=1e-5,
        affine=True,
        reduce_fn=None,
    ):
        """
        Args:
            reduce_fn: Optional callable for reduction.
                       If None, uses torch.segment_reduce (axis=0).
                       If provided, uses reduce_fn(input, reduce=..., lengths=...) (no axis).
        """
        super().__init__()
        self.norm_type = norm_type.lower()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        self.reduce_fn = reduce_fn

        if self.norm_type == "batch":
            self.bn = nn.BatchNorm1d(num_features, eps=eps, affine=affine)
        else:
            if self.affine:
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

    def _reduce(self, x, reduce_op, lengths):
        """Dispatches to the correct reduction implementation."""
        if self.reduce_fn is not None:
            # User's custom implementation (assumes axis=0)
            return self.reduce_fn(x, reduce=reduce_op, lengths=lengths)
        else:
            # PyTorch standard implementation (requires axis=0)
            return torch.segment_reduce(x, reduce=reduce_op, lengths=lengths, axis=0)

    def forward(self, x, lengths=None):
        if self.norm_type == "batch":
            return self.bn(x)

        if lengths is None:
            raise ValueError(f"lengths argument is required for {self.norm_type} norm")

        if self.norm_type == "instance":
            x_norm = self._instance_norm(x, lengths)
        elif self.norm_type == "group":
            x_norm = self._group_norm(x, lengths)
        elif self.norm_type == "layer":
            x_norm = self._layer_norm(x, lengths)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

        if self.affine:
            return x_norm * self.weight + self.bias
        return x_norm

    def _instance_norm(self, x, lengths):
        # 1. Mean
        mean = self._reduce(x, "mean", lengths)
        mean_expanded = torch.repeat_interleave(mean, lengths, dim=0)

        # 2. Variance
        x_centered = x - mean_expanded
        var = self._reduce(x_centered.pow(2), "mean", lengths)
        std_expanded = torch.repeat_interleave((var + self.eps).sqrt(), lengths, dim=0)

        return x_centered / std_expanded

    def _group_norm(self, x, lengths):
        C_per_G = self.num_features // self.num_groups
        x_g = x.view(x.size(0), self.num_groups, C_per_G)

        # 1. Mean
        x_g_mean = x_g.mean(dim=2)
        mean = self._reduce(x_g_mean, "mean", lengths)
        mean_expanded = torch.repeat_interleave(mean, lengths, dim=0).unsqueeze(2)

        # 2. Variance
        x_centered = x_g - mean_expanded
        var_spatial = x_centered.pow(2).mean(dim=2)
        var = self._reduce(var_spatial, "mean", lengths)
        std_expanded = torch.repeat_interleave(
            (var + self.eps).sqrt(), lengths, dim=0
        ).unsqueeze(2)

        return (x_centered / std_expanded).view(x.size(0), self.num_features)

    def _layer_norm(self, x, lengths):
        # 1. Mean
        x_mean_c = x.mean(dim=1, keepdim=True)
        mean = self._reduce(x_mean_c, "mean", lengths)
        mean_expanded = torch.repeat_interleave(mean, lengths, dim=0)

        # 2. Variance
        x_centered = x - mean_expanded
        var_c = x_centered.pow(2).mean(dim=1, keepdim=True)
        var = self._reduce(var_c, "mean", lengths)
        std_expanded = torch.repeat_interleave((var + self.eps).sqrt(), lengths, dim=0)

        return x_centered / std_expanded


# --- The 4 Wrappers ---


class RaggedBatchNorm(RaggedNorm):
    def __init__(self, num_features, eps=1e-5, affine=True, reduce_fn=None):
        super().__init__(
            num_features, norm_type="batch", eps=eps, affine=affine, reduce_fn=reduce_fn
        )


class RaggedInstanceNorm(RaggedNorm):
    def __init__(self, num_features, eps=1e-5, affine=False, reduce_fn=None):
        super().__init__(
            num_features,
            norm_type="instance",
            eps=eps,
            affine=affine,
            reduce_fn=reduce_fn,
        )


class RaggedLayerNorm(RaggedNorm):
    def __init__(self, num_features, eps=1e-5, affine=True, reduce_fn=None):
        super().__init__(
            num_features, norm_type="layer", eps=eps, affine=affine, reduce_fn=reduce_fn
        )


class RaggedGroupNorm(RaggedNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, reduce_fn=None):
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        super().__init__(
            num_channels,
            norm_type="group",
            num_groups=num_groups,
            eps=eps,
            affine=affine,
            reduce_fn=reduce_fn,
        )
