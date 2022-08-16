from torch.nn import Sequential as tSequential


class Sequential(tSequential):
    """A sequential container for passing along pairs of inputs to outputs.

    Assumed Density Filtering (ADF) version of `torch.nn.Sequential`.
    It propagates two instead of one tensor in a feed-forward manner.
    """

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, in_mean, in_var):
        for module in self:
            in_mean, in_var = module(in_mean, in_var)
        return in_mean, in_var
