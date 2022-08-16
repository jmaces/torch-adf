"""Tests for `torchadf.nn.modules.activation`. """

import numpy as np
import torch

from hypothesis import given

from torchadf.nn.modules.activation import ReLU

from .strategies import assert_leq, batched_float_array


# activation tests
@given(batched_float_array())
def test_relu(x):
    means, covariances, mode = x
    means_tensor = torch.tensor(means)  # same dtype of means
    covariances_tensor = torch.tensor(covariances)  # same dtype of covariances
    means_out, covariances_out = ReLU(mode)(means_tensor, covariances_tensor)
    means_out = means_out.detach().numpy()
    covariances_out = covariances_out.detach().numpy()
    assert means.shape == means_out.shape
    assert covariances.shape == covariances_out.shape
    assert means.dtype.name == means_out.dtype.name
    assert covariances.dtype.name == covariances_out.dtype.name
    assert_leq(np.zeros_like(means_out), means_out)
    assert_leq(means, means_out)
    if mode == "diag" or mode == "diagonal":
        variances_out = covariances_out
    elif mode == "half" or mode == "lowrank":
        cov_shape = covariances_out.shape
        variances_out = np.reshape(
            np.sum(
                np.square(covariances_out),
                axis=-1,
            ),
            means_out.shape,
        )
    elif mode == "full":
        cov_shape = covariances_out.shape
        cov_rank = len(cov_shape) - 1
        variances_out = np.reshape(
            np.diagonal(
                np.reshape(
                    covariances_out,
                    (
                        cov_shape[0],
                        np.prod(cov_shape[1 : cov_rank // 2 + 1]),
                        np.prod(cov_shape[cov_rank // 2 + 1 :]),
                    ),
                ),
                axis1=-2,
                axis2=-1,
            ),
            means_out.shape,
        )
    assert means_out.shape == variances_out.shape
    assert_leq(np.zeros_like(variances_out), variances_out)
