"""Tests for `torchadf.nn.modules.container`. """

import hypothesis.strategies as st
import torch

from hypothesis import given, settings

from torchadf.nn.modules.container import Sequential
from torchadf.nn.modules.conv import Conv1d, Conv2d, Conv3d
from torchadf.nn.modules.linear import Linear

from .strategies import batched_float_array


# sequential of linear layers tests
@settings(deadline=None)
@given(
    batched_float_array(max_data_dims=1),
    st.tuples(
        st.integers(min_value=1, max_value=128),
        st.integers(min_value=1, max_value=128),
        st.integers(min_value=1, max_value=128),
    ),
)
def test_sequential_linear(x, units):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    units = (means.shape[-1],) + units
    layers = [
        Linear(units[k], units[k + 1], mode=mode, dtype=means_tensor.dtype)
        for k in range(len(units) - 1)
    ]
    model = Sequential(*layers)
    means_out, covariances_out = model(means_tensor, covariances_tensor)
    means_out = means_out.detach().numpy()
    covariances_out = covariances_out.detach().numpy()
    assert means.shape[:-1] == means_out.shape[:-1]
    assert units[-1] == means_out.shape[-1]
    if mode == "diag":
        assert covariances.shape[:-1] == covariances_out.shape[:-1]
        assert units[-1] == covariances_out.shape[-1]
    elif mode == "half":
        assert covariances.shape[:-2] == covariances_out.shape[:-2]
        assert units[-1] == covariances_out.shape[-2]
        assert covariances.shape[-1] == covariances_out.shape[-1]
    elif mode == "full":
        assert covariances.shape[:-2] == covariances_out.shape[:-2]
        assert units[-1] == covariances_out.shape[-1]
        assert units[-1] == covariances_out.shape[-2]


# sequential of convolution layer tests
@settings(deadline=None)
@given(
    st.tuples(
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=1, max_value=64),
        st.integers(min_value=1, max_value=64),
    ),
    batched_float_array(min_data_dims=2, max_data_dims=2),
)
def test_sequential_convolution_1d(filters, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    filters = (means.shape[1],) + filters
    layers = [
        Conv1d(
            filters[k],
            filters[k + 1],
            3,
            1,
            "same",
            mode=mode,
            dtype=means_tensor.dtype,
        )
        for k in range(len(filters) - 1)
    ]
    model = Sequential(*layers)
    means_out, covariances_out = model(means_tensor, covariances_tensor)
    assert means.shape[0] == means_out.shape[0]
    assert filters[-1] == means_out.shape[-2]
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters[-1] == covariances_out.shape[-2]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[-1] == covariances_out.shape[-1]
        assert filters[-1] == covariances_out.shape[-3]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters[-1] == covariances_out.shape[-2]
        assert filters[-1] == covariances_out.shape[-4]


@settings(deadline=None)
@given(
    st.tuples(
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=32),
        st.integers(min_value=1, max_value=32),
    ),
    batched_float_array(min_data_dims=3, max_data_dims=3),
)
def test_squential_convolution_2d(filters, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    filters = (means.shape[1],) + filters
    layers = [
        Conv2d(
            filters[k],
            filters[k + 1],
            3,
            1,
            "same",
            mode=mode,
            dtype=means_tensor.dtype,
        )
        for k in range(len(filters) - 1)
    ]
    model = Sequential(*layers)
    means_out, covariances_out = model(means_tensor, covariances_tensor)
    assert means.shape[0] == means_out.shape[0]
    assert filters[-1] == means_out.shape[-3]
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters[-1] == covariances_out.shape[-3]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[-1] == covariances_out.shape[-1]
        assert filters[-1] == covariances_out.shape[-4]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters[-1] == covariances_out.shape[-3]
        assert filters[-1] == covariances_out.shape[-6]


@settings(deadline=None)
@given(
    st.tuples(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=1, max_value=16),
    ),
    batched_float_array(min_data_dims=4, max_data_dims=4),
)
def test_sequential_convolution_3d(filters, x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    filters = (means.shape[1],) + filters
    layers = [
        Conv3d(
            filters[k],
            filters[k + 1],
            3,
            1,
            "same",
            mode=mode,
            dtype=means_tensor.dtype,
        )
        for k in range(len(filters) - 1)
    ]
    model = Sequential(*layers)
    means_out, covariances_out = model(means_tensor, covariances_tensor)
    assert means.shape[0] == means_out.shape[0]
    assert filters[-1] == means_out.shape[-4]
    if mode == "diag":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters[-1] == covariances_out.shape[-4]
    elif mode == "half":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert covariances.shape[-1] == covariances_out.shape[-1]
        assert filters[-1] == covariances_out.shape[-5]
    elif mode == "full":
        assert covariances.shape[0] == covariances_out.shape[0]
        assert filters[-1] == covariances_out.shape[-4]
        assert filters[-1] == covariances_out.shape[-8]
