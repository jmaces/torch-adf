"""Tests for `torchadf.nn.modules.linear`. """

import hypothesis.strategies as st
import torch

from hypothesis import given, settings

from torchadf.nn.modules.linear import Identity, Linear

from .strategies import assert_eq, batched_float_array


# linear layers tests
@settings(deadline=None)
@given(
    batched_float_array(max_data_dims=1),
    st.integers(min_value=1, max_value=128),
)
def test_identity(x, arbitrary):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    layer = Identity(arbitrary, mode=mode)
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    means_out = means_out.detach().numpy()
    covariances_out = covariances_out.detach().numpy()
    assert means.shape == means_out.shape
    assert covariances.shape == covariances_out.shape
    assert_eq(means, means_out)
    assert_eq(covariances, covariances_out)


@settings(deadline=None)
@given(
    batched_float_array(max_data_dims=1),
    st.integers(min_value=1, max_value=128),
)
def test_linear(x, units):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    layer = Linear(means.shape[-1], units, mode=mode, dtype=means_tensor.dtype)
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    means_out = means_out.detach().numpy()
    covariances_out = covariances_out.detach().numpy()
    assert means.shape[:-1] == means_out.shape[:-1]
    assert units == means_out.shape[-1]
    if mode == "diag":
        assert covariances.shape[:-1] == covariances_out.shape[:-1]
        assert units == covariances_out.shape[-1]
    elif mode == "half":
        assert covariances.shape[:-2] == covariances_out.shape[:-2]
        assert units == covariances_out.shape[-2]
        assert covariances.shape[-1] == covariances_out.shape[-1]
    elif mode == "full":
        assert covariances.shape[:-2] == covariances_out.shape[:-2]
        assert units == covariances_out.shape[-1]
        assert units == covariances_out.shape[-2]
