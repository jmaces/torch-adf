API Reference
=============

.. automodule:: torchadf

What follows is the *API explanation*. This mostly just lists functions and
their options and is intended for *quickly looking up* things.

If you like a more *hands-on introduction*, have a look at our `examples`.


torchadf.nn
-----------

.. automodule:: torchadf.nn

Collects aliases of all the supported layers in submodules of the package.
See below for details on the individual submodules.


torchadf.nn.modules.container
------------------------------

.. automodule:: torchadf.nn.modules.container

Below is a list of model containers in the package.

.. autosummary::
   :toctree: api
   :template: class.rst

   Sequential


torchadf.nn.modules.activation
------------------------------

.. automodule:: torchadf.nn.modules.activation

Below is a list of supported activation function layers in the package.

.. autosummary::
   :toctree: api
   :template: class.rst

   ReLU


torchadf.nn.modules.conv
------------------------

.. automodule:: torchadf.nn.modules.conv

Below is a list of supported convolution layers in the package.

.. autosummary::
  :toctree: api
  :template: class.rst

  Conv1d
  Conv2d
  Conv3d


torchadf.nn.modules.flatten
---------------------------

.. automodule:: torchadf.nn.modules.flatten

Below is a list of reshaping layers in the package.

.. autosummary::
   :toctree: api
   :template: class.rst

   Flatten
   Unflatten


torchadf.nn.modules.linear
--------------------------

.. automodule:: torchadf.nn.modules.linear

Below is a list of linear layers in the package.

.. autosummary::
  :toctree: api
  :template: class.rst

  Identity
  Linear


torchadf.nn.modules.pooling
---------------------------

.. automodule:: torchadf.nn.modules.pooling

Below is a list of supported pooling layers in the package.

.. autosummary::
   :toctree: api
   :template: class.rst

   AvgPool1d
   AvgPool2d
   AvgPool3d


torchadf.nn.functional
----------------------

.. automodule:: torchadf.nn.functional

Below is a list of the functional counterparts to all supported layers in the package.

.. autosummary::
   :toctree: api

   conv1d
   conv2d
   conv3d
   avg_pool1d
   avg_pool2d
   avg_pool3d
   relu
   linear
   flatten
   unflatten
