# *_*coding:utf-8 *_*
import torch.nn as nn

try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive.
        """

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, input):
            return input