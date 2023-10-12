.. micromind documentation master file, created by
   sphinx-quickstart on Fri Feb 24 15:07:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to micromind's documentation!
=====================================

.. image:: https://img.shields.io/badge/python-3.9%20|%203.10-blue
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://github.com/fpaissan/micromind/blob/main/LICENSE

.. image:: https://img.shields.io/pypi/v/micromind

This is the official repository of `micromind`, a toolkit that aims to bridge two communities: artificial intelligence and embedded systems. `micromind` is based on `PyTorch <https://pytorch.org>`_ and provides exportability for the supported models in ONNX, Intel OpenVINO, and TFLite.

Key Features
------------

- Smooth flow from research to deployment;
- Support for multimedia analytics recipes (image classification, sound event detection, etc);
- Detailed API documentation;
- Tutorials for embedded deployment.

Installation
------------

Using Pip
~~~~~~~~~

First of all, install `Python 3.8 or later <https://www.python.org>`_. Open a terminal and run:

.. code:: shell

    pip install micromind

for the basic install. To install `micromind` with the full exportability features, run

.. code:: shell

    pip install micromind[conversion]


Basic how-to
------------

If you want to launch a simple training on an image classification model, you just need to define a class that extends `MicroMind`, defining the modules you want to use, such as a `PhiNet`, the forward method of the model and the way in which to calculate your loss function.

.. code-block:: python

   class ImageClassification(MicroMind):
      def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)

         self.modules["classifier"] = PhiNet(
            (3, 32, 32), include_top=True, num_classes=10
         )

      def forward(self, batch):
         return self.modules["classifier"](batch[0])

      def compute_loss(self, pred, batch):
         return nn.CrossEntropyLoss()(pred, batch[1])

If you want to export the model in the format you like best between **ONNX**, **TFLite** and **OpenVINO**, just run this simple code.

.. code-block:: python

   m = ImageClassification(hparams)
   m.export("output_onnx", "onnx", (3, 32, 32))


Here is the link to the Python `file <https://github.com/micromind-toolkit/micromind/blob/mm_refactor/examples/mind.py>`_ inside our repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 2
   :caption: How To Contribute

   contribution
