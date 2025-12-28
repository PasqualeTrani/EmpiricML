# Install

## Installation

EmpiricML is available on PyPI and can be installed using pip:

```bash
pip install empiricml
```

**Requirements**: Python 3.11 or higher.

## Dependencies

When you install EmpiricML, the following dependencies will be automatically installed:

- `numpy>=1.26.4`
- `pandas>=2.2.2`
- `pyarrow>=15.0.2`
- `polars>=1.31.0`
- `matplotlib>=3.9.0`
- `scikit-learn>=1.7.1`
- `lightgbm>=4.6.0`
- `xgboost>=3.1.2`
- `catboost>=1.2.8`
- `skorch>=1.3.1`

## PyTorch Support

EmpiricML supports PyTorch models through the `TorchWrapper` class. However, **PyTorch is not installed automatically** to allow users to choose the specific version that matches their hardware configuration (CPU vs GPU, CUDA version, etc.).

To use PyTorch models with EmpiricML:

1.  **Install PyTorch separately** by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).
2.  **Use the `TorchWrapper` class** to integrate your PyTorch models into the EmpiricML workflow.
