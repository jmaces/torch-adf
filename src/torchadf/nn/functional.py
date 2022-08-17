import operator

from functools import reduce

import torch.nn.functional as F

from torch.distributions import Normal
from torch.nn.modules.utils import _pair, _single, _triple


# utility product over iterables
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


# ----- ----- Convolutional ----- -----


def conv1d(
    in_mean,
    in_var,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mode="diag",
):
    """Applies a convolution function for 1D inputs.

    Assumed Density Filtering (ADF) version of `torch.nn.functional.conv1d`.


    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
    weight : torch.Tensor or torch.nn.parameter.Parameter
        The convolution filter weights.
    bias : torch.Tensor or torch.nn.parameter.Parameter, optional
        The additive convolution bias (Default None).
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : string or int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    out_mean = F.conv1d(
        in_mean, weight, bias, stride, padding, dilation, groups
    )
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.conv1d(
            in_var, weight.square(), None, stride, padding, dilation, groups
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        in_var = in_var.movedim(-1, -3)  # move rank dimension out of the way
        unflatten_size = in_var.shape[:-2]
        in_var = in_var.flatten(0, -3)  # compress leading batch dimensions
        out_var = F.conv1d(
            in_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress batch dim
        out_var = out_var.movedim(-3, -1)  # move back rank dimension
    elif mode.lower() == "full":
        in_var = in_var.movedim((-2, -1), (-4, -3))  # move cov dims away
        unflatten_size = in_var.shape[:-2]
        in_var = in_var.flatten(0, -3)  # compress leading dimensions
        out_var = F.conv1d(
            in_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
        out_var = out_var.movedim((-4, -3), (-2, -1))  # move cov dims back
        unflatten_size = out_var.shape[:-2]
        out_var = out_var.flatten(0, -3)  # compress leading dimensions
        out_var = F.conv1d(
            out_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


def conv2d(
    in_mean,
    in_var,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mode="diag",
):
    """Applies a convolution function for 2D inputs.

    Assumed Density Filtering (ADF) version of `torch.nn.functional.conv2d`.


    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
    weight : torch.Tensor or torch.nn.parameter.Parameter
        The convolution filter weights.
    bias : torch.Tensor or torch.nn.parameter.Parameter, optional
        The additive convolution bias (Default None).
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : string or int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    out_mean = F.conv2d(
        in_mean, weight, bias, stride, padding, dilation, groups
    )
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.conv2d(
            in_var, weight.square(), None, stride, padding, dilation, groups
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        in_var = in_var.movedim(-1, -4)  # move rank dimension out of the way
        unflatten_size = in_var.shape[:-3]
        in_var = in_var.flatten(0, -4)  # compress leading batch dimensions
        out_var = F.conv2d(
            in_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress batch dim
        out_var = out_var.movedim(-4, -1)  # move back rank dimension
    elif mode.lower() == "full":
        in_var = in_var.movedim(
            (-3, -2, -1), (-6, -5, -4)
        )  # move cov dims away
        unflatten_size = in_var.shape[:-3]
        in_var = in_var.flatten(0, -4)  # compress leading dimensions
        out_var = F.conv2d(
            in_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
        out_var = out_var.movedim(
            (-6, -5, -4), (-3, -2, -1)
        )  # move cov dims back
        unflatten_size = out_var.shape[:-3]
        out_var = out_var.flatten(0, -4)  # compress leading dimensions
        out_var = F.conv2d(
            out_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


def conv3d(
    in_mean,
    in_var,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    mode="diag",
):
    """Applies a convolution function for 3D inputs.

    Assumed Density Filtering (ADF) version of `torch.nn.functional.conv3d`.


    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
    weight : torch.Tensor or torch.nn.parameter.Parameter
        The convolution filter weights.
    bias : torch.Tensor or torch.nn.parameter.Parameter, optional
        The additive convolution bias (Default None).
    stride : int or tuple of int, optional
        Stride of the convolution kernel (Default 1).
    padding : string or int or tuple of int, optional
        Implicit padding on both sides of input (Default 0).
    dilation : int or tuple of int, optional
        Spacing between kernel elements (Default 1).
    groups : int, optional
        Split convolution into a number independent groups (Default 1).
        Number of input channels has to be divisible by the number of groups.
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    out_mean = F.conv3d(
        in_mean, weight, bias, stride, padding, dilation, groups
    )
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.conv3d(
            in_var, weight.square(), None, stride, padding, dilation, groups
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        in_var = in_var.movedim(-1, -5)  # move rank dimension out of the way
        unflatten_size = in_var.shape[:-4]
        in_var = in_var.flatten(0, -5)  # compress leading batch dimensions
        out_var = F.conv3d(
            in_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress batch dim
        out_var = out_var.movedim(-5, -1)  # move back rank dimension
    elif mode.lower() == "full":
        in_var = in_var.movedim(
            (-4, -3, -2, -1), (-8, -7, -6, -5)
        )  # move cov dims away
        unflatten_size = in_var.shape[:-4]
        in_var = in_var.flatten(0, -5)  # compress leading dimensions
        out_var = F.conv3d(
            in_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
        out_var = out_var.movedim(
            (-8, -7, -6, -5), (-4, -3, -2, -1)
        )  # move cov dims back
        unflatten_size = out_var.shape[:-4]
        out_var = out_var.flatten(0, -5)  # compress leading dimensions
        out_var = F.conv3d(
            out_var, weight, None, stride, padding, dilation, groups
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


def conv_transpose1d(
    in_mean,
    in_var,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    mode="diag",
):
    # TODO: add transpose convs
    raise NotImplementedError("Transpose convolution is not yet implemented.")


def conv_transpose2d(
    in_mean,
    in_var,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    mode="diag",
):
    # TODO: add transpose convs
    raise NotImplementedError("Transpose convolution is not yet implemented.")


def conv_transpose3d(
    in_mean,
    in_var,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
    mode="diag",
):
    # TODO: add transpose convs
    raise NotImplementedError("Transpose convolution is not yet implemented.")


# ----- ----- Poolings ----- -----


def avg_pool1d(
    in_mean,
    in_var,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    mode="diag",
):
    """Applies an average pooling function for 1D inputs.

    Assumed Density Filtering (ADF) version of
    `torch.nn.functional.avg_pool1d`.


    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
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

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    out_mean = F.avg_pool1d(
        in_mean, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
    kernel_numel = prod(_single(kernel_size))
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.avg_pool1d(
            in_var / kernel_numel,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        in_var = in_var.movedim(-1, -3)  # move rank dimension out of the way
        unflatten_size = in_var.shape[:-2]
        in_var = in_var.flatten(0, -3)  # compress leading batch dimensions
        out_var = F.avg_pool1d(
            in_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress batch dim
        out_var = out_var.movedim(-3, -1)  # move back rank dimension
    elif mode.lower() == "full":
        in_var = in_var.movedim((-2, -1), (-4, -3))  # move cov dims away
        unflatten_size = in_var.shape[:-2]
        in_var = in_var.flatten(0, -3)  # compress leading dimensions
        out_var = F.avg_pool1d(
            in_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
        out_var = out_var.movedim((-4, -3), (-2, -1))  # move cov dims back
        unflatten_size = out_var.shape[:-2]
        out_var = out_var.flatten(0, -3)  # compress leading dimensions
        out_var = F.avg_pool1d(
            out_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


def avg_pool2d(
    in_mean,
    in_var,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    mode="diag",
):
    """Applies an average pooling function for 2D inputs.

    Assumed Density Filtering (ADF) version of
    `torch.nn.functional.avg_pool2d`.


    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
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

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    out_mean = F.avg_pool2d(
        in_mean, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
    kernel_numel = prod(_pair(kernel_size))
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.avg_pool2d(
            in_var / kernel_numel,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        in_var = in_var.movedim(-1, -4)  # move rank dimension out of the way
        unflatten_size = in_var.shape[:-3]
        in_var = in_var.flatten(0, -4)  # compress leading batch dimensions
        out_var = F.avg_pool2d(
            in_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress batch dim
        out_var = out_var.movedim(-4, -1)  # move back rank dimension
    elif mode.lower() == "full":
        in_var = in_var.movedim(
            (-3, -2, -1), (-6, -5, -4)
        )  # move cov dims away
        unflatten_size = in_var.shape[:-3]
        in_var = in_var.flatten(0, -4)  # compress leading dimensions
        out_var = F.avg_pool2d(
            in_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
        out_var = out_var.movedim(
            (-6, -5, -4), (-3, -2, -1)
        )  # move cov dims back
        unflatten_size = out_var.shape[:-3]
        out_var = out_var.flatten(0, -4)  # compress leading dimensions
        out_var = F.avg_pool2d(
            out_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


def avg_pool3d(
    in_mean,
    in_var,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    mode="diag",
):
    """Applies an average pooling function for 3D inputs.

    Assumed Density Filtering (ADF) version of
    `torch.nn.functional.avg_pool3d`.


    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
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

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    out_mean = F.avg_pool3d(
        in_mean, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
    kernel_numel = prod(_triple(kernel_size))
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.avg_pool3d(
            in_var / kernel_numel,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        in_var = in_var.movedim(-1, -5)  # move rank dimension out of the way
        unflatten_size = in_var.shape[:-4]
        in_var = in_var.flatten(0, -5)  # compress leading batch dimensions
        out_var = F.avg_pool3d(
            in_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress batch dim
        out_var = out_var.movedim(-5, -1)  # move back rank dimension
    elif mode.lower() == "full":
        in_var = in_var.movedim(
            (-4, -3, -2, -1), (-8, -7, -6, -5)
        )  # move cov dims away
        unflatten_size = in_var.shape[:-4]
        in_var = in_var.flatten(0, -5)  # compress leading dimensions
        out_var = F.avg_pool3d(
            in_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
        out_var = out_var.movedim(
            (-8, -7, -6, -5), (-4, -3, -2, -1)
        )  # move cov dims back
        unflatten_size = out_var.shape[:-4]
        out_var = out_var.flatten(0, -5)  # compress leading dimensions
        out_var = F.avg_pool3d(
            out_var, kernel_size, stride, padding, ceil_mode, count_include_pad
        )
        out_var = out_var.unflatten(0, unflatten_size)  # uncompress leading
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


# ----- ----- Non-Linear Activation ----- -----


_normal_dist = Normal(0.0, 1.0)


# private helper for relu: cumulative density of standard normal distribution
def _cdf(x):
    return _normal_dist.cdf(x)


# private helper for relu: probability density of standard normal distribution
def _pdf(x):
    return _normal_dist.log_prob(x).exp()


def relu(in_mean, in_var, mode="diag"):
    """Applies a rectified linear unit activation.

    Assumed Density Filtering (ADF) version of `torch.nn.functional.relu`.

    Since the standard ReLU function operates component-wise there is some
    ambiguity in how to interpret the input dimensions if the number of leading
    "batch" dimensions is unknown. The following convention is used:

    - if ``in_mean.ndim==3`` the mean input is interpreted as
      ``(batch_dim, num_channels, in_dim)``, e.g., for 1D convolutions
    - if ``in_mean.ndim==4`` the mean input is interpreted as
      ``(batch_dim, num_channels, height, width)``, e.g., for 2D convolutions
    - if ``in_mean.ndim==5`` the mean input is interpreted as
      ``(batch_dim, num_channels, depth, height, width)``, e.g., for 3D
      convolutions
    - otherwise the mean input is interpreted as ``(*,in_dim)``, e.g., for
      dense linear layers

    The corresponding (co-)variance shapes are interpreted accordingly,
    depending on the covariance propagation mode.

    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor.
    out_var : torch.Tensor
        The transformed (co-)variance tensor.
    """
    EPS = 1e-7  # for numerical stability, avoid dividing by zero
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        std = (in_var + EPS).sqrt()
        div = in_mean / std
        pdf = _pdf(div)
        cdf = _cdf(div)
        out_mean = F.relu(in_mean * cdf + std * pdf)
        out_var = F.relu(
            in_mean.square() * cdf
            + in_var * cdf
            + in_mean * std * pdf
            - out_mean.square()
        )
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        if in_mean.ndim >= 3 and in_mean.ndim <= 5:  # interpret as conv inputs
            unflatten_size = in_mean.shape[1:]
            in_mean, in_var = flatten(in_mean, in_var, 1, -1, mode)
        else:  # interpret as batches of vectors
            unflatten_size = None  # no reshaping necessary
        std = (in_var.square().sum(-1) + EPS).sqrt()
        div = in_mean / std
        pdf = _pdf(div)
        cdf = _cdf(div)
        out_mean = F.relu(in_mean * cdf + std * pdf)
        out_var = in_var * cdf.unsqueeze(-1)
        if unflatten_size:  # undo previous reshapes if necessary
            out_mean, out_var = unflatten(
                out_mean, out_var, 1, unflatten_size, mode
            )
    elif mode.lower() == "full":
        if in_mean.ndim >= 3 and in_mean.ndim <= 5:  # interpret as conv inputs
            unflatten_size = in_mean.shape[1:]
            in_mean, in_var = flatten(in_mean, in_var, 1, -1, mode)
        else:  # interpret as batches of vectors
            unflatten_size = None  # no reshaping necessary
        std = (in_var.diagonal(0, -2, -1) + EPS).sqrt()
        div = in_mean / std
        pdf = _pdf(div)
        cdf = _cdf(div)
        out_mean = F.relu(in_mean * cdf + std * pdf)
        out_var = in_var * cdf.unsqueeze(-2) * cdf.unsqueeze(-1)
        if unflatten_size:  # undo previous reshapes if necessary
            out_mean, out_var = unflatten(
                out_mean, out_var, 1, unflatten_size, mode
            )
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


# ----- ----- Fully-Connected (Affine) Linear ----- -----


def linear(in_mean, in_var, weight, bias=None, mode="diag"):
    """Applies a dense (fully connected) linear transform.

    Assumed Density Filtering (ADF) version of `torch.nn.functional.linear`.

    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor. Expected shape is ``(*, in_dim)``.
    in_var : torch.Tensor
        Input (co-)variance tensor. Expected shape is ``(*, in_dim)``,
        ``(*, in_dim, rank)``, or ``(*, in_dim, in_dim)`` depending on the
        mode.
    weight : torch.Tensor or torch.nn.parameter.Parameter
        Weight matrix of the affine linear transform.
    bias : torch.Tensor or torch.nn.parameter.Parameter, optional
        Bias vecotr of the affine linear transform (Default None).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The transformed mean tensor of shape ``(*, out_dim)``.
    out_var : torch.Tensor
        The transformed (co-)variance tensor of shape ``(*, out_dim)``,
        ``(*, out_dim, rank)``, or ``(*, out_dim, out_dim)`` depending on the
        mode.
    """
    out_mean = F.linear(in_mean, weight, bias)
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = F.linear(in_var, weight.square())
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        out_var = F.linear(in_var.transpose(-1, -2), weight).transpose(-1, -2)
    elif mode.lower() == "full":
        out_var = F.linear(in_var, weight)
        out_var = F.linear(out_var.transpose(-1, -2), weight)
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


# ----- ----- Utility ----- -----


def flatten(in_mean, in_var, start_dim, end_dim, mode="diag"):
    """Flattens the inputs along a contiguous range of dimensions.

    Assumed Density Filtering (ADF) version of `torch.flatten`.
    The dimensions to be flattened refer to the first input (mean) Tensor.
    Respective dimensions for the second input (covariance) Tensor are
    inferred according to the covariance propagation mode.
    (For the full covariance mode this can be ambiguous if the number of
    leading "batch" dimensions is unknown, hence we assume that any dimensions
    before the specified start_dim are batch dimensions.)

    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
    start_dim : int, optional
        First dimension to flatten (Default 1).
    end_dim: int, optional
        Last dimension to flatten (Default -1).
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The reshaped mean tensor.
    out_var : torch.Tensor
        The reshaped (co-)variance tensor.

    """
    out_mean = in_mean.flatten(start_dim, end_dim)
    if mode.lower() == "diag" or mode.lower() == "diagonal":
        out_var = in_var.flatten(start_dim, end_dim)
    elif mode.lower() == "lowrank" or mode.lower() == "half":
        v_start_dim, v_end_dim = start_dim, end_dim
        if v_start_dim < 0:
            v_start_dim -= 1  # shift to account for extra rank dim
        if v_end_dim < 0:
            v_end_dim -= 1  # shift to account for extra rank dim
        out_var = in_var.flatten(v_start_dim, v_end_dim)
    elif mode.lower() == "full":
        v_start_dim, v_end_dim = start_dim, end_dim
        if v_start_dim < 0:
            v_start_dim = in_mean.ndim + v_start_dim  # convert to true pos
        if v_end_dim < 0:
            v_end_dim = in_mean.ndim + v_end_dim  # convert to true pos
        if (in_var.ndim - v_start_dim) % 2 != 0:
            raise ValueError(
                "Invalid dimensions. We do not know how to interpret "
                "flattening dimensions {} to {} for a full covariance "
                "input of {} dimensions".format(
                    start_dim, end_dim, in_var.ndim
                )
            )

        # determine split index between first and second covariance half
        half_index = (in_var.ndim - v_start_dim) // 2 + v_start_dim

        # flatten the second half first, to not change positions in first
        out_var = in_var.flatten(
            half_index, half_index + v_end_dim - v_start_dim
        )

        # flatten the first half of the covariance
        out_var = out_var.flatten(v_start_dim, v_end_dim)
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var


def unflatten(in_mean, in_var, dim, unflattened_size, mode="diag"):
    """Unflattens a dimension of the inputs over multiple dimensions.

    Assumed Density Filtering (ADF) version of the reverse reshaping operation
    corresponding to `flatten`.
    The dimension to be unflattened and target shape refer to the first input
    (mean) Tensor. Respective dimensions for the second input (covariance)
    Tensor are inferred according to the covariance propagation mode.
    (For the full covariance mode this can be ambiguous if the number of
    leading "batch" dimensions is unknown, hence we assume that any dimensions
    before the specified unflatten dimension are batch dimensions.)

    Parameters
    ----------
    in_mean : torch.Tensor
        Input mean tensor.
    in_var : torch.Tensor
        Input (co-)variance tensor.
    dim : int
        Dimension to unflatten.
    unflattened_size: tuple of int
        Shape into which the selected dimension should be unflattened.
    mode : {"diag", "diagonal", "lowrank", "half", "full"}, optional
        Covariance propagation mode (Default "diag").

    Returns
    -------
    out_mean : torch.Tensor
        The reshaped mean tensor.
    out_var : torch.Tensor
        The reshaped (co-)variance tensor.

    """
    out_mean = in_mean.unflatten(dim, unflattened_size)
    if mode.lower() == "diag" or mode == "diagonal":
        out_var = in_var.unflatten(dim, unflattened_size)
    elif mode.lower() == "lowrank" or mode == "half":
        v_dim = dim
        if v_dim < 0:
            v_dim -= 1  # shift to account for extra rank dim
        out_var = in_var.unflatten(v_dim, unflattened_size)
    elif mode.lower() == "full":
        v_dim = dim
        if v_dim < 0:
            v_dim = in_mean.ndim + v_dim  # convert to true pos
        if (in_var.ndim - v_dim) % 2 != 0:
            raise ValueError(
                "Invalid dimensions. We do not know how to interpret "
                "unflattening dimension {} for a full covariance "
                "input of {} dimensions".format(dim, in_var.ndim)
            )

        # determine split index between first and second cavariance half
        half_index = (in_var.ndim - v_dim) // 2 + v_dim

        # flatten the second half first, to not change positions in first
        out_var = in_var.unflatten(half_index, unflattened_size)

        # flatten the first half of the covariance
        out_var = out_var.unflatten(v_dim, unflattened_size)
    else:
        raise ValueError(
            "Invalid covariance propagation mode: {}".format(mode)
        )
    return out_mean, out_var
