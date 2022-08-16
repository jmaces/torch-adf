from torch.nn import Module

from .. import functional as F


class ReLU(Module):
    """Rectified Linear Unit.

    Assumed Density Filtering (ADF) version of the PyTorch `ReLU` activation.
    Transforms input means and (co-)variances according to ``math(x, 0)``.

    Parameters
    ----------
    mode: {"diag", "diagonal", "lowrank", "half", "full"}
        Covariance computation mode. Default is "diag".

    """

    def __init__(self, mode="diag"):
        super(ReLU, self).__init__()
        self.mode = mode.lower()

    def forward(self, in_mean, in_var):
        return F.relu(in_mean, in_var, self.mode)
