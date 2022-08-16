"""Tests for `torchadf.nn.modules.pooling`. """

import hypothesis.strategies as st
import numpy as np
import torch

from hypothesis import given, settings

from torchadf.nn.modules.pooling import AvgPool1d, AvgPool2d, AvgPool3d

from .strategies import batched_float_array


# pooling layer tests
@settings(deadline=None)
@given(
    st.integers(min_value=0, max_value=7),
    st.tuples(st.integers(min_value=1, max_value=8))
    | st.integers(min_value=1, max_value=8),
    st.tuples(st.integers(min_value=1, max_value=8))
    | st.integers(min_value=1, max_value=8)
    | st.none(),
    batched_float_array(min_data_dims=2, max_data_dims=2),
)
def test_average_pool_1d(padding, pool_size, strides, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    if strides:
        if isinstance(strides, tuple):
            strides = np.minimum(strides, means.shape[2:])
        else:
            strides = min(strides, min(means.shape[2:]))
    if isinstance(pool_size, tuple):
        pool_size = np.minimum(pool_size, means.shape[2:])
        padding = min(padding, min(pool_size) // 2)
    else:
        pool_size = min(pool_size, min(means.shape[2:]))
        padding = min(padding, pool_size // 2)
    layer = AvgPool1d(pool_size, strides, padding, mode=mode)
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    if strides is None:
        strides = pool_size
    if isinstance(strides, tuple):
        strides = strides[0]
    out_size = tuple(
        np.floor(
            (np.array(means.shape[2:]) + 2 * padding - pool_size) / strides + 1
        )
    )
    assert means.shape[:2] == means_out.shape[:2]
    assert out_size == means_out.shape[2:]
    if mode == "diag":
        assert covariances.shape[:2] == covariances_out.shape[:2]
        assert out_size == covariances_out.shape[2:]
    elif mode == "half":
        assert covariances.shape[:2] == covariances_out.shape[:2]
        assert out_size == covariances_out.shape[2:-1]
        assert covariances.shape[-1] == covariances_out.shape[-1]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size == covariances_out.shape[2:3]
        assert covariances.shape[3] == covariances_out.shape[3]
        assert out_size == covariances_out.shape[4:5]


@settings(deadline=None)
@given(
    st.integers(min_value=0, max_value=7),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8),
    st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=8),
    )
    | st.integers(min_value=1, max_value=8)
    | st.none(),
    batched_float_array(min_data_dims=3, max_data_dims=3),
)
def test_average_pool_2d(padding, pool_size, strides, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    if strides:
        if isinstance(strides, tuple):
            strides = np.minimum(strides, means.shape[2:])
        else:
            strides = min(strides, min(means.shape[2:]))
    if isinstance(pool_size, tuple):
        pool_size = np.minimum(pool_size, means.shape[2:])
        padding = min(padding, min(pool_size) // 2)
    else:
        pool_size = min(pool_size, min(means.shape[2:]))
        padding = min(padding, pool_size // 2)
    layer = AvgPool2d(pool_size, strides, padding, mode=mode)
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    if strides is None:
        strides = pool_size
    out_size = tuple(
        np.floor(
            (np.array(means.shape[2:]) + 2 * padding - pool_size) / strides + 1
        )
    )
    assert means.shape[:2] == means_out.shape[:2]
    assert out_size == means_out.shape[2:]
    if mode == "diag":
        assert covariances.shape[:2] == covariances_out.shape[:2]
        assert out_size == covariances_out.shape[2:]
    elif mode == "half":
        assert covariances.shape[:2] == covariances_out.shape[:2]
        assert out_size == covariances_out.shape[2:-1]
        assert covariances.shape[-1] == covariances_out.shape[-1]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size == covariances_out.shape[2:4]
        assert covariances.shape[4] == covariances_out.shape[4]
        assert out_size == covariances_out.shape[5:7]


@settings(deadline=None)
@given(
    st.integers(min_value=0, max_value=7),
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
    | st.integers(min_value=1, max_value=8)
    | st.none(),
    batched_float_array(min_data_dims=4, max_data_dims=4),
)
def test_average_pool_3d(padding, pool_size, strides, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    if strides:
        if isinstance(strides, tuple):
            strides = np.minimum(strides, means.shape[2:])
        else:
            strides = min(strides, min(means.shape[2:]))
    if isinstance(pool_size, tuple):
        pool_size = np.minimum(pool_size, means.shape[2:])
        padding = min(padding, min(pool_size) // 2)
    else:
        pool_size = min(pool_size, min(means.shape[2:]))
        padding = min(padding, pool_size // 2)
    layer = AvgPool3d(pool_size, strides, padding, mode=mode)
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    if strides is None:
        strides = pool_size
    out_size = tuple(
        np.floor(
            (np.array(means.shape[2:]) + 2 * padding - pool_size) / strides + 1
        )
    )
    assert means.shape[:2] == means_out.shape[:2]
    assert out_size == means_out.shape[2:]
    if mode == "diag":
        assert covariances.shape[:2] == covariances_out.shape[:2]
        assert out_size == covariances_out.shape[2:]
    elif mode == "half":
        assert covariances.shape[:2] == covariances_out.shape[:2]
        assert out_size == covariances_out.shape[2:-1]
        assert covariances.shape[-1] == covariances_out.shape[-1]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[1] == covariances_out.shape[1]
        assert out_size == covariances_out.shape[2:5]
        assert covariances.shape[5] == covariances_out.shape[5]
        assert out_size == covariances_out.shape[6:9]
