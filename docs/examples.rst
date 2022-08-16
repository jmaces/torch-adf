Examples
========

Here we provide some more detailed examples showcasing some of the more advanced
aspects of using ``torch-adf``. For a simple use case and example of how to
get started we recommend to first have a look at our `overview` section.

.. contents:: Advanced topics addressed in our examples
    :depth: 1
    :local:
    :backlinks: none

Modules and Functions
---------------------

In the same way as their PyTorch counterparts, most ADF layers are
available in a functional form and as modules (e.g. to be used within a `torch.nn.Sequential` model or as submodules within any other `torch.nn.Module`).


(Co-)Variance Computation Modes
-------------------------------

There are three (co-)variance computation modes available for ADF layers.
All layers within a model must use the same mode to guarantee matching input/output shapes.
As common in PyTorch we adopt the "channel first" ordering of dimensions, that is
``(batch_size, num_dims)``, ``(batch_size, num_channels, num_dims)``,
``(batch_size, num_channels, height, width)``, and
``(batch_size, num_channels, depth, height, width)`` for inputs that could
be passed to a linear layer, or 1D, 2D, and 3D convolutions respectively.

The three modes are:

    "diag" or "diagonal" mode
        This is the default for all layers. Here only
        variances but no covariances are propagated, i.e. in other words the dimensions
        of the inputs/activations/outputs are treated as independent/uncorrelated and only the
        diagonal of the covariance matrix is propagated. This independence is of course usually
        not really satisfied but in many scenarios a good enough approximation.
        In this mode the tensors for ``mean`` and ``variance`` in each layer have the same shape.
    "half" or "lowrank" mode
        This mode makes use of the symmetric factorization of the
        covariance matrix. Only one of the factors is propagated through the layers. The full output
        covariance matrix can be retrieved as the product of this factor with its transpose. Reducing the
        inner dimension of the matrix factors (which is kept constant throughout layers) allows the propagation
        of low-rank approximations to the covariance matrix and reduces the computational costs.
        In this mode the covariance factor has one additional dimension compared to the mean tensor.
        It is appended to the end and has a size is equal to the chosen rank, e.g.,
        ``(batch_size, num_channels, height, width, rank)`` for the 2D convolution case.
    "full" mode
        This propagates the full covariance matrix and is computationally costly, in particular memory
        consumption can be problematic. Use this only for small layers and models. The covariance matrix requires the
        squared size of the mean tensor. For this, all dimensions (except the leading dimensions considered as
        batch dimensions) are repeated and appended to the end, e.g.,
        ``(batch_size, num_channels, height, width, num_channels, height, width)``
        for the 2D convolution case.

As an example we will create `Linear` layers for all three modes. It works
analogously for all other layers.

.. doctest:: COV_MODES

    >>> from torchadf.nn import Linear

First, we use the "diagonal" mode. Both inputs will need the same shape.

.. doctest:: COV_MODES

    >>> in_dim, out_dim = 64, 32
    >>> layer = Linear(in_dim, out_dim, mode="diag")

This layer expects two inputs, each of size ``(*,64)``, and yields two outputs, each of size ``(*,32)``.
Next, we use the "half" mode. Here the second input needs one additional
dimension for the rank of the matrix factorization.

.. doctest:: COV_MODES

    >>> in_dim, out_dim = 64, 32
    >>> layer = Linear(in_dim, out_dim, mode="half")

This layer expects two inputs of size ``(*,64)`` and ``(*,64,rank)`` and yields two outputs of size ``(*,32)`` and ``(*,32,rank)``.
Note that the rank dimension never changes and is passed on from inputs to outputs.
Finally, we use the "full" mode. Here the second input requires the squared
size of the first input.

    .. doctest:: COV_MODES

        >>> in_dim, out_dim = 64, 32
        >>> layer = Linear(in_dim, out_dim, mode="full")

This layer expects two inputs of size ``(*,64)`` and ``(*,64,64)`` and yields two outputs of size ``(*,32)`` and ``(*,32,32)``.
