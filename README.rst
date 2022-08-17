============================================================================
``torch-adf``: Assumed Density Filtering (ADF) Probabilistic Neural Networks
============================================================================

.. add project badges here
.. image:: https://readthedocs.org/projects/torch-adf/badge/?version=latest
    :target: https://torch-adf.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/jmaces/torch-adf/actions/workflows/pr-check.yml/badge.svg?branch=main
    :target: https://github.com/jmaces/torch-adf/actions/workflows/pr-check.yml?branch=main
    :alt: CI Status

.. image:: https://codecov.io/gh/jmaces/torch-adf/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/jmaces/torch-adf
  :alt: Code Coverage

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code Style: Black


.. teaser-start

``torch-adf`` provides implementations for probabilistic
`PyTorch <https://pytorch.org/>`_ neural network layers,
which are based on assumed density filtering.
Assumed density filtering (ADF) is a general concept from Bayesian inference, but in the case of feed-forward neural networks that we consider here
it is a way to approximately propagate a random distribution through the neural network.

The layers in this package have the same names and arguments as their corresponding
PyTorch versions. We use Gaussian distributions for our ADF approximations, which are
described by their means and (co-)variances. So unlike the standard PyTorch layers,
each ``torch-adf`` layer takes two inputs and produces two outputs (one for the means
and one for the (co-)variances).

.. teaser-end


.. example

``torch-adf`` layers can be used exactly like the corresponding PyTorch
layers within a model. However, as mentioned above, ADF layers take two inputs and produce two outputs
instead of one, so it is not possible to simply mix ADF and standard layers within the same model.

.. code-block:: python

    from torchadf.nn import Sequential
    from torchadf.nn import Linear

    in_dim, out_dim = 64, 32
    adflayer = Linear(in_dim, out_dim)
    model = Sequential(adflayer)

The `Overview <https://torch-adf.readthedocs.io/en/latest/overview.html>`_ and
`Examples <https://torch-adf.readthedocs.io/en/latest/examples.html>`_ sections
of our documentation provide more realistic and complete examples.

.. project-info-start

Project Information
===================

``torch-adf`` is released under the `MIT license <https://github.com/jmaces/torch-adf/blob/main/LICENSE>`_,
its documentation lives at `Read the Docs <https://torch-adf.readthedocs.io/en/latest/>`_,
the code on `GitHub <https://github.com/jmaces/torch-adf>`_,
and the latest release can be found on `PyPI <https://pypi.org/project/torch-adf/>`_.
It’s tested on Python 3.6+.

If you'd like to contribute to ``torch-adf`` you're most welcome.
We have written a `short guide <https://github.com/jmaces/torch-adf/blob/main/.github/CONTRIBUTING.rst>`_ to help you get you started!

.. project-info-end


.. literature-start

Further Reading
===============

Additional information on the algorithmic aspects of ``torch-adf`` can be found
in the following works:


- Jochen Gast, Stefan Roth,
  "Lightweight Probabilistic Deep Networks",
  2018
- Jan Macdonald, Stephan Wäldchen, Sascha Hauch, Gitta Kutyniok,
  "A Rate-Distortion Framework for Explaining Neural Network Decisions",
  2019

.. literature-end


Acknowledgments
===============

During the setup of this project we were heavily influenced and inspired by
the works of `Hynek Schlawack <https://hynek.me/>`_ and in particular his
`attrs <https://www.attrs.org/en/stable/>`_ package and blog posts on
`testing and packaing <https://hynek.me/articles/testing-packaging/>`_
and `deploying to PyPI <https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/>`_.
Thank you for sharing your experiences and insights.
