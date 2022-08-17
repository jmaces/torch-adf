from torch.nn import Module

from .. import functional as F


class Flatten(Module):
    """Flattens the input along a contiguous range of dimensions.

    Assumed Density Filtering (ADF) version of `torch.nn.Flatten`.
    The dimensions to be flattened refer to the first input (mean) Tensor.
    Respective dimensions for the second input (covariance) Tensor are
    inferred according to the covariance propagation mode.
    (For the full covariance mode this can be ambiguous if the number of
    leading "batch" dimensions is unknown, hence we assume that any dimensions
    before the specified start_dim are batch dimensions.)

    Parameters
    ----------
    start_dim : int, optional
        First dimension to flatten (Default 1).
    end_dim: int, optional
        Last dimension to flatten (Default -1).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(self, start_dim=1, end_dim=-1, mode="diag"):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.mode = mode.lower()

    def forward(self, in_mean, in_var):
        return F.flatten(
            in_mean, in_var, self.start_dim, self.end_dim, self.mode
        )

    def extra_repr(self):
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


class Unflatten(Module):
    """Unflattens a dimension of the input over multiple dimensions.

    Assumed Density Filtering (ADF) version of `torch.nn.Unflatten`.
    The dimension to be unflattened and target shape refer to the first input
    (mean) Tensor. Respective dimensions for the second input (covariance)
    Tensor are inferred according to the covariance propagation mode.
    (For the full covariance mode this can be ambiguous if the number of
    leading "batch" dimensions is unknown, hence we assume that any dimensions
    before the specified unflatten dimension are batch dimensions.)

    Parameters
    ----------
    dim : int
        Dimension to unflatten.
    unflattened_size: tuple of int
        Shape into which the selected dimension should be unflattened.
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(self, dim, unflattened_size, mode="diag"):
        super(Unflatten, self).__init__()
        self.mode = mode.lower()

        if isinstance(dim, int):
            self._require_tuple_int(unflattened_size)
        elif isinstance(dim, str):
            self._require_tuple_tuple(unflattened_size)
        else:
            raise TypeError("invalid argument type for dim parameter")

        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_tuple(self, input):
        raise NotImplementedError(
            "Unflatten is not yet implemented for " "named Tensor dimensions."
        )

    def _require_tuple_int(self, input):
        if isinstance(input, (tuple, list)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "
                        + "but found element of type {} at pos {}".format(
                            type(elem).__name__, idx
                        )
                    )
            return
        raise TypeError(
            "unflattened_size must be a tuple of ints, "
            + "but found type {}".format(type(input).__name__)
        )

    def forward(self, in_mean, in_var):
        return F.unflatten(
            in_mean, in_var, self.dim, self.unflattened_size, self.mode
        )

    def extra_repr(self):
        return "dim={}, unflattened_size={}".format(
            self.dim, self.unflattened_size
        )
