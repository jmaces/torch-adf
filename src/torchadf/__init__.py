"""PyTorch implementation of Assumed Density Filtering (ADF) based
probabilistic neural networks.

This package provides implementations of several ADF based probabilistic
buildings blocks commonly used in neural networks. They are to be used within
the framework of PyTorch. Unlike standard (deterministic) PyTorch layers
that propagate point estimates, the corresponding probabilsitic ADF layers
propagate a distribution parametrized by its mean and (co-)variance.

"""

# package meta data
__version__ = "22.1.0.dev"  # 0Y.Minor.Micro CalVer format
__title__ = "torch-adf"
__description__ = "Assumed Density Filtering (ADF) Probabilistic Networks"
__url__ = "https://github.com/jmaces/torch-adf"
__uri__ = __url__
# __doc__ = __description__ + " <" + __uri__ + ">"

__author__ = "Jan Maces"
__email__ = "janmaces[at]gmail.com"

__license__ = "MIT"
__copyright__ = "Copyright 2022 Jan Maces"
