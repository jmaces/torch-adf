Changelog
=========

Versions follow `CalVer <https://calver.org>`_  (0Y.Minor.Micro format).

Changes for the upcoming release can be found in the `"changelog.d" directory <https://github.com/jmaces/torch-adf/tree/main/changelog.d>`_ in our repository.

..
   Do *NOT* add changelog entries here!

   This changelog is managed by towncrier and is compiled at release time.

   See our contribution guide for details.

.. towncrier release notes start

22.1.0 (2022-08-17)
-------------------

Changes
^^^^^^^

- Initial release.`#1 <https://github.com/jmaces/torch-adf/issues/1>`_

  - Functional API

    + torchadf.nn.functional.conv1d
    + torchadf.nn.functional.conv2d
    + torchadf.nn.functional.conv3d

    + torchadf.nn.functional.avg_pool1d
    + torchadf.nn.functional.avg_pool2d
    + torchadf.nn.functional.avg_pool3d

    + torchadf.nn.functional.relu

    + torchadf.nn.functional.linear

    + torchadf.nn.functional.flatten
    + torchadf.nn.functional.unflatten


  - Modules

    + torchadf.nn.modules.activation.ReLU

    + torchadf.nn.modules.container.Sequential

    + torchadf.nn.modules.conv.Conv1d
    + torchadf.nn.modules.conv.Conv2d
    + torchadf.nn.modules.conv.Conv3d

    + torchadf.nn.modules.flatten.Flatten
    + torchadf.nn.modules.flatten.Unflatten

    + torchadf.nn.modules.linear.Identity
    + torchadf.nn.modules.linear.Linear

    + torchadf.nn.modules.pooling.AvgPool1d
    + torchadf.nn.modules.pooling.AvgPool2d
    + torchadf.nn.modules.pooling.AvgPool3d
