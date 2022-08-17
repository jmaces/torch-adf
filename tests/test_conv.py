"""Tests for `torchadf.nn.modules.conv`. """

import hypothesis.strategies as st
import numpy as np
import pytest
import torch

from hypothesis import given, settings

from torchadf.nn.modules.conv import Conv1d, Conv2d, Conv3d

from .strategies import batched_float_array


# convolution layer tests
@settings(deadline=None)
@pytest.mark.parametrize("padding", ["same", "valid"])
@given(
    st.integers(min_value=1, max_value=64),
    st.tuples(st.integers(min_value=1, max_value=8))
    | st.integers(min_value=1, max_value=8),
    st.tuples(st.integers(min_value=1, max_value=8))
    | st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=2, max_data_dims=2),
)
def test_convolution_1d(padding, filters, kernel_size, strides, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    if isinstance(strides, tuple):
        strides = np.minimum(strides, means.shape[2:])
    else:
        strides = min(strides, min(means.shape[2:]))
    if isinstance(kernel_size, tuple):
        kernel_size = np.minimum(kernel_size, means.shape[2:])
    else:
        kernel_size = min(kernel_size, min(means.shape[2:]))
    if padding == "same":
        strides = 1  # strided convs not supported for padding=same
    layer = Conv1d(
        means.shape[1],
        filters,
        kernel_size,
        strides,
        padding,
        mode=mode,
        dtype=means_tensor.dtype,
    )
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(strides, tuple):
        kernel_size = kernel_size[0]
    if padding == "same":
        out_size = np.ceil(means.shape[-1] / strides)
    elif padding == "valid":
        out_size = np.ceil((means.shape[-1] - kernel_size + 1) / strides)
    assert means.shape[0] == means_out.shape[0]
    assert out_size == means_out.shape[-1]
    assert filters == means_out.shape[-2]
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters == covariances_out.shape[-2]
        assert out_size == covariances_out.shape[-1]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[-1] == covariances_out.shape[-1]
        assert filters == covariances_out.shape[-3]
        assert out_size == covariances_out.shape[-2]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters == covariances_out.shape[-2]
        assert out_size == covariances_out.shape[-1]
        assert filters == covariances_out.shape[-4]
        assert out_size == covariances_out.shape[-3]


@settings(deadline=None)
@pytest.mark.parametrize("padding", ["same", "valid"])
@given(
    st.integers(min_value=1, max_value=64),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=3, max_data_dims=3),
)
def test_convolution_2d(padding, filters, kernel_size, strides, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    if isinstance(strides, tuple):
        strides = np.minimum(strides, means.shape[2:])
    else:
        strides = min(strides, min(means.shape[2:]))
    if isinstance(kernel_size, tuple):
        kernel_size = np.minimum(kernel_size, means.shape[2:])
    else:
        kernel_size = min(kernel_size, min(means.shape[2:]))
    if padding == "same":
        strides = 1  # strided convs not supported for padding=same
    layer = Conv2d(
        means.shape[1],
        filters,
        kernel_size,
        strides,
        padding,
        mode=mode,
        dtype=means_tensor.dtype,
    )
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    if padding == "same":
        out_size = np.ceil(np.asarray(means.shape[2:]) / strides)
    elif padding == "valid":
        out_size = np.ceil(
            (np.asarray(means.shape[2:]) - kernel_size + 1) / strides
        )
    assert means.shape[0] == means_out.shape[0]
    assert filters == means_out.shape[-3]
    assert out_size[0] == means_out.shape[-2]
    assert out_size[1] == means_out.shape[-1]
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters == covariances_out.shape[-3]
        assert out_size[0] == covariances_out.shape[-2]
        assert out_size[1] == covariances_out.shape[-1]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[-1] == covariances_out.shape[-1]
        assert filters == covariances_out.shape[-4]
        assert out_size[0] == covariances_out.shape[-3]
        assert out_size[1] == covariances_out.shape[-2]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters == covariances_out.shape[-3]
        assert out_size[0] == covariances_out.shape[-2]
        assert out_size[1] == covariances_out.shape[-1]
        assert filters == covariances_out.shape[-6]
        assert out_size[0] == covariances_out.shape[-5]
        assert out_size[1] == covariances_out.shape[-4]


@settings(deadline=None)
@pytest.mark.parametrize("padding", ["same", "valid"])
@given(
    st.integers(min_value=1, max_value=64),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    batched_float_array(min_data_dims=4, max_data_dims=4),
)
def test_convolution_3d(padding, filters, kernel_size, strides, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    if isinstance(strides, tuple):
        strides = np.minimum(strides, means.shape[2:])
    else:
        strides = min(strides, min(means.shape[2:]))
    if isinstance(kernel_size, tuple):
        kernel_size = np.minimum(kernel_size, means.shape[2:])
    else:
        kernel_size = min(kernel_size, min(means.shape[2:]))
    if padding == "same":
        strides = 1  # strided convs not supported for padding=same
    layer = Conv3d(
        means.shape[1],
        filters,
        kernel_size,
        strides,
        padding,
        mode=mode,
        dtype=means_tensor.dtype,
    )
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    if padding == "same":
        out_size = np.ceil(np.asarray(means.shape[2:]) / strides)
    elif padding == "valid":
        out_size = np.ceil(
            (np.asarray(means.shape[2:]) - kernel_size + 1) / strides
        )
    assert means.shape[0] == means_out.shape[0]
    assert filters == means_out.shape[-4]
    assert out_size[0] == means_out.shape[-3]
    assert out_size[1] == means_out.shape[-2]
    assert out_size[2] == means_out.shape[-1]
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters == covariances_out.shape[-4]
        assert out_size[0] == covariances_out.shape[-3]
        assert out_size[1] == covariances_out.shape[-2]
        assert out_size[2] == covariances_out.shape[-1]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[-1] == covariances_out.shape[-1]
        assert filters == covariances_out.shape[-5]
        assert out_size[0] == covariances_out.shape[-4]
        assert out_size[1] == covariances_out.shape[-3]
        assert out_size[2] == covariances_out.shape[-2]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters == covariances_out.shape[-4]
        assert out_size[0] == covariances_out.shape[-3]
        assert out_size[1] == covariances_out.shape[-2]
        assert out_size[2] == covariances_out.shape[-1]
        assert filters == covariances_out.shape[-8]
        assert out_size[0] == covariances_out.shape[-7]
        assert out_size[1] == covariances_out.shape[-6]
        assert out_size[2] == covariances_out.shape[-5]
