from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.utils import _pair, _single, _triple

from .. import functional as F


class Conv1d(_ConvNd):
    """A 1D convolution layer.

    Assumed Density Filtering (ADF) version of `torch.nn.Conv1d`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    bias : bool, optional
        Add a learnable convolution bias (Default True).
    padding_mode : {"zeros", "reflect", "replicate", "circular"}, optional
        Padding mode (Default "zeros").
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        self.mode = mode.lower()
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(self, in_mean, in_var, weight, bias):
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "Convolutions for padding modes other than zero-padding have"
                "not yet been implemented for ADF layers."
            )
        return F.conv1d(
            in_mean,
            in_var,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mode,
        )

    def forward(self, in_mean, in_var):
        return self._conv_forward(in_mean, in_var, self.weight, self.bias)


class Conv2d(_ConvNd):
    """A 2D convolution layer.

    Assumed Density Filtering (ADF) version of `torch.nn.Conv2d`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    bias : bool, optional
        Add a learnable convolution bias (Default True).
    padding_mode : {"zeros", "reflect", "replicate", "circular"}, optional
        Padding mode (Default "zeros").
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.mode = mode.lower()
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(self, in_mean, in_var, weight, bias):
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "Convolutions for padding modes other than zero-padding have"
                "not yet been implemented for ADF layers."
            )
        return F.conv2d(
            in_mean,
            in_var,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mode,
        )

    def forward(self, in_mean, in_var):
        return self._conv_forward(in_mean, in_var, self.weight, self.bias)


class Conv3d(_ConvNd):
    """A 3D convolution layer.

    Assumed Density Filtering (ADF) version of `torch.nn.Conv3d`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    bias : bool, optional
        Add a learnable convolution bias (Default True).
    padding_mode : {"zeros", "reflect", "replicate", "circular"}, optional
        Padding mode (Default "zeros").
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        self.mode = mode.lower()
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(self, in_mean, in_var, weight, bias):
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "Convolutions for padding modes other than zero-padding have"
                "not yet been implemented for ADF layers."
            )
        return F.conv3d(
            in_mean,
            in_var,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.mode,
        )

    def forward(self, in_mean, in_var):
        return self._conv_forward(in_mean, in_var, self.weight, self.bias)


class ConvTranspose1d(_ConvTransposeNd):
    """A 1D tranpose convolution layer.

    Assumed Density Filtering (ADF) version of `torch.nn.ConvTranspose1d`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    output_padding : int or tuple of int, optional
        Additional padding for the output (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    bias : bool, optional
        Add a learnable convolution bias (Default True).
    padding_mode : {"zeros", "reflect", "replicate", "circular"}, optional
        Padding mode (Default "zeros").
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        self.mode = "diag"
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def forward(self, in_mean, in_var, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        num_spatial_dims = 1
        output_padding = self._output_padding(
            in_mean,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        return F.conv_transpose1d(
            in_mean,
            in_var,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
            self.mode,
        )


class ConvTranspose2d(_ConvTransposeNd):
    """A 2D tranpose convolution layer.

    Assumed Density Filtering (ADF) version of `torch.nn.ConvTranspose2d`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    output_padding : int or tuple of int, optional
        Additional padding for the output (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    bias : bool, optional
        Add a learnable convolution bias (Default True).
    padding_mode : {"zeros", "reflect", "replicate", "circular"}, optional
        Padding mode (Default "zeros").
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        self.mode = "diag"
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def forward(self, in_mean, in_var, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        num_spatial_dims = 2
        output_padding = self._output_padding(
            in_mean,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        return F.conv_transpose2d(
            in_mean,
            in_var,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
            self.mode,
        )


class ConvTranspose3d(_ConvTransposeNd):
    """A 3D tranpose convolution layer.

    Assumed Density Filtering (ADF) version of `torch.nn.ConvTranspose3d`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the convolution kernel.
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    output_padding : int or tuple of int, optional
        Additional padding for the output (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    bias : bool, optional
        Add a learnable convolution bias (Default True).
    padding_mode : {"zeros", "reflect", "replicate", "circular"}, optional
        Padding mode (Default "zeros").
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        mode="diag",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        self.mode = "diag"
        super(ConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def forward(self, in_mean, in_var, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        num_spatial_dims = 3
        output_padding = self._output_padding(
            in_mean,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        return F.conv_transpose3d(
            in_mean,
            in_var,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
            self.mode,
        )
