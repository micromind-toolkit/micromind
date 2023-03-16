[![Python version: 3.9 | 3.10](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lucadellalib/bayestorch/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![PyPI version](https://img.shields.io/pypi/v/micromind)

This is the official repo of `micromind`, a toolkit that aims at bridging two communities: artificial intelligence and embedded systems. `micromind` is based on [PyTorch](https://pytorch.org) and provides exportability for the supported models in ONNX, Intel OpenVINO, and TFLite.

---------------------------------------------------------------------------------------------------------

## 💡 Key features

- Smooth flow from research to deployment
- Support for multimedia analytics recipes (image classification, sound event detection, etc)
- Detailed API documentation
- Tutorials for embedded deployment

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

### Using Pip

First of all, install [Python 3.9 or later](https://www.python.org). Open a terminal and run:

```
pip install micromind
```
for the basic install. To install `micromind` with the full exportability features, run

```
pip install micromind[conversion]
```

### From source

First of all, install [Python 3.9 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a
terminal and run:

```
pip install -e .
```
for the basic install. To install `micromind` with the full exportability features, run

```
pip install -e .[conversion]
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[francescopaissan@gmail.com](mailto:francescopaissan@gmail.com)

---------------------------------------------------------------------------------------------------------
