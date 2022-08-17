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

  Functional API
  **************

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


  Modules
  *******

  + torch.nn.modules.activation.ReLU

  + torch.nn.modules.container.Sequential

  + torch.nn.modules.conv.Conv1d
  + torch.nn.modules.conv.Conv2d
  + torch.nn.modules.conv.Conv3d

  + torch.nn.modules.flatten.Flatten
  + torch.nn.modules.flatten.Unflatten

  + torch.nn.modules.linear.Identity
  + torch.nn.modules.linear.Linear

  + torch.nn.modules.pooling.AvgPool1d
  + torch.nn.modules.pooling.AvgPool2d
  + torch.nn.modules.pooling.AvgPool3d



----
