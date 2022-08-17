"""Tests for `torchadf.nn.modules.flatten`. """

import torch

from hypothesis import given, settings

from torchadf.nn.modules.flatten import Flatten, Unflatten

from .strategies import assert_eq, batched_float_array


# reshaping layers tests
@settings(deadline=None)
@given(batched_float_array(min_data_dims=2))
def test_flatten(x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    layer = Flatten(mode=mode)
    means_out, covariances_out = layer(means_tensor, covariances_tensor)
    means_out = means_out.detach().numpy()
    covariances_out = covariances_out.detach().numpy()
    assert means.shape[0] == means_out.shape[0]
    assert covariances.shape[0] == covariances_out.shape[0]
    assert 2 == len(means_out.shape)
    if mode == "diag":
        assert 2 == len(covariances_out.shape)
    elif mode == "half":
        assert 3 == len(covariances_out.shape)
    elif mode == "full":
        assert 3 == len(covariances_out.shape)
    assert_eq(means.flatten(), means_out.flatten())
    assert_eq(covariances.flatten(), covariances_out.flatten())


@settings(deadline=None)
@given(batched_float_array(min_data_dims=2))
def test_flatten_unflatten(x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    unflatten_size = means_tensor.shape[1:]
    fl_layer = Flatten(mode=mode)
    ufl_layer = Unflatten(1, unflatten_size, mode=mode)
    means_out, covariances_out = ufl_layer(
        *fl_layer(means_tensor, covariances_tensor)
    )
    means_out = means_out.detach().numpy()
    covariances_out = covariances_out.detach().numpy()
    assert means.shape == means_out.shape
    assert covariances.shape == covariances_out.shape
    assert_eq(means.flatten(), means_out.flatten())
    assert_eq(covariances.flatten(), covariances_out.flatten())
