Overview
========

``torch-adf`` provides implementations of probabilistic assumed density filtering
(ADF) based versions of layers commonly used in neural networks. They are to be
used within the `PyTorch <https://pytorch.org/>`_
framework. Unlike the standard (deterministic) PyTorch layers that propagate
point estimates, ADF layers propagate a probability distribution parametrized
by its mean and (co-)variance.

We think it is best to show the core concepts of the package by
a simple exemplary demonstration.

For this let us define a simple feed-forward model with fully-connected
ADF layers.

We begin by importing the relevant PyTorch and ``torch-adf`` components.

.. doctest:: OVERVIEW

    >>> import torch
    >>> from torchadf.nn import Sequential, Linear, ReLU

For a simple feed-forward network we can use a sequential model.
However, as we want to pass two rather than one tensor between layers, we can
not use the standard `torch.nn.Sequential` model but have to use a
`torchadf.nn.modules.container.Sequential` model instead.
We add a fully-connected hidden layer followed by a rectified linear
unit (ReLU) activation and then another fully-connected output layer.

.. doctest:: OVERVIEW

    >>> model = Sequential(Linear(32, 64), ReLU(), Linear(64, 1))

We have defined a fully-connected feed-forward neural network with
32-dimensional input space, a 64-dimensional hidden representation space, and a
1-dimensional output space. The model summary looks like this:

.. doctest:: OVERVIEW

    >>> print(model)  # doctest: +NORMALIZE_WHITESPACE
    Sequential(
      (0): Linear(in_features=32, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=1, bias=True)
    )

Let us try to feed in a mini-batch of ten input tensors. Since we did not
specify the covariance propagation mode for any of the layers, the default
"diagonal" is used. This means that only the variances but no covariances are
propagated. Therefore, the two input tensor for means and (co-)-variances need
to be of the same shape.

.. doctest:: OVERVIEW

    >>> in_means = torch.randn(10, 32)
    >>> in_covs = torch.randn(10, 32).square()  # variances are non-negative
    >>> out_means, out_covs = model(in_means, in_covs)
    >>> out_means.shape
    torch.Size([10, 1])
    >>> out_covs.shape
    torch.Size([10, 1])

This model can now be used like any other PyTorch model: It can be trained after
providing a loss function and an optimizer, it can be saved and restored, it
can be used to make predictions, ...
