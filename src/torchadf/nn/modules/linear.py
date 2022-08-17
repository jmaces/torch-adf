import math

import torch

from torch.nn import Module, init
from torch.nn.parameter import Parameter

from .. import functional as F


class Identity(Module):
    """Identity Operator.

    Assumed Density Filtering (ADF) version of the PyTorch `Identity` layer.
    A placeholder identity operator that is argument-insensitive.

    Parameters
    ----------
    *args: any arguments (unused).
    **kwargs: any keyword arguments (unused).

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, in_mean, in_var):
        return in_mean, in_var


class Linear(Module):
    """A dense (fully connected) linear layer.

    Assumed Density Filtering (ADF) version of `torch.nn.Linear`.

    Parameters
    ----------
    in_features : torch.Tensor
        Size of each input samples.
    out_features : torch.Tensor
        Size of each output sample.
    bias : bool, optional
        Add a learnable bias to the linear layer (Default True).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Linear, self).__init__()
        self.mode = mode.lower()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, in_mean, in_var):
        return F.linear(in_mean, in_var, self.weight, self.bias, self.mode)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
