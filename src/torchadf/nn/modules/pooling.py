from torch.nn.modules.pooling import _AvgPoolNd
from torch.nn.modules.utils import _pair, _single, _triple

from .. import functional as F


class AvgPool1d(_AvgPoolNd):
    """A 1D average pooling layer.

    Assumed Density Filtering (ADF) version of `torch.nn.AvgPool1d`.


    Parameters
    ----------
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Stride of the pooling window (Default = kernel_size).
    padding : int or tuple of int, optional
        Implicit zero padding on both sides of input (Default 0).
    ceil_mode : bool, optional
        Use ceil instead of floor to compute output shape (Default False).
    count_include_pad : bool, optional
        Include zero-padding in average calculations (Default True).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        mode="diag",
    ):
        super(AvgPool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.mode = mode.lower()

    def forward(self, in_mean, in_var):
        return F.avg_pool1d(
            in_mean,
            in_var,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.mode,
        )


class AvgPool2d(_AvgPoolNd):
    """A 2D average pooling layer.

    Assumed Density Filtering (ADF) version of `torch.nn.AvgPool2d`.


    Parameters
    ----------
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Stride of the pooling window (Default = kernel_size).
    padding : int or tuple of int, optional
        Implicit zero padding on both sides of input (Default 0).
    ceil_mode : bool, optional
        Use ceil instead of floor to compute output shape (Default False).
    count_include_pad : bool, optional
        Include zero-padding in average calculations (Default True).
    divisor_override : optional,
        Will be used as divisor if specified, otherwise ``kernel_size`` will be
        used (Default None).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        mode="diag",
    ):
        super(AvgPool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.mode = mode.lower()

    def forward(self, in_mean, in_var):
        return F.avg_pool2d(
            in_mean,
            in_var,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
            self.mode,
        )


class AvgPool3d(_AvgPoolNd):
    """A 3D average pooling layer.

    Assumed Density Filtering (ADF) version of `torch.nn.AvgPool3d`.


    Parameters
    ----------
    kernel_size : int or tuple of int
        Size of the pooling window.
    stride : int or tuple of int, optional
        Stride of the pooling window (Default = kernel_size).
    padding : int or tuple of int, optional
        Implicit zero padding on both sides of input (Default 0).
    ceil_mode : bool, optional
        Use ceil instead of floor to compute output shape (Default False).
    count_include_pad : bool, optional
        Include zero-padding in average calculations (Default True).
    divisor_override : optional,
        Will be used as divisor if specified, otherwise ``kernel_size`` will be
        used (Default None).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        mode="diag",
    ):
        super(AvgPool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.mode = mode.lower()

    def forward(self, in_mean, in_var):
        return F.avg_pool3d(
            in_mean,
            in_var,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
            self.mode,
        )
