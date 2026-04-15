"""Sequential container that threads MetaData alongside features.

Like nn.Sequential, but passes multiple arguments (features, MetaData)
through each layer, supporting the (Tensor, MetaData) -> (Tensor, MetaData)
calling convention used by point convolution blocks.
"""

import torch.nn as nn


class MultiSequential(nn.Sequential):
    def forward(self, *inputs):
        # If inputs is passed as (x, y), inputs becomes ((x, y),) inside *args
        # We need to normalize the input for the loop.
        # If multiple inputs were passed, use them as the starting tuple.
        # If a single input was passed, just use that.

        # Determine starting point
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = inputs

        for module in self:
            if isinstance(x, tuple):
                # If x is a tuple, unpack it as multiple arguments for the module
                x = module(*x)
            else:
                # If x is a single tensor, pass it normally
                x = module(x)
        return x
