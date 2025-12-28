# Installation

## Requirements

- Python 3.11 or higher
- pip (Python package installer)

## Install via pip

The recommended way to install EmpiricML is via pip:

```bash
pip install EmpiricML
```

## Install from Source

To install the latest development version from GitHub:

```bash
git clone https://github.com/PasqualeTrani/EmpiricML.git
cd EmpiricML
pip install -e .
```

## Dependencies

EmpiricML automatically installs the following dependencies:

| Package | Minimum Version | Description |
|---------|----------------|-------------|
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.2.2 | Data manipulation |
| polars | 1.31.0 | Fast DataFrames |
| scikit-learn | 1.7.1 | Machine learning algorithms |
| lightgbm | 4.6.0 | Gradient boosting |
| xgboost | 3.1.2 | Extreme gradient boosting |
| catboost | 1.2.8 | Categorical boosting |
| matplotlib | 3.9.0 | Visualization |
| skorch | 1.3.1 | PyTorch scikit-learn wrapper |

## Optional Dependencies

For PyTorch-based models, you'll also need to install PyTorch:

```bash
pip install torch
```

## Verify Installation

After installation, verify that EmpiricML is correctly installed:

```python
import EmpiricML as eml
print(eml.__version__)
```

## Troubleshooting

### Common Issues

!!! warning "CatBoost Installation on Windows"
    If you encounter issues installing CatBoost on Windows, try:
    ```bash
    pip install catboost --no-cache-dir
    ```

!!! tip "Virtual Environment"
    We recommend using a virtual environment to avoid dependency conflicts:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install EmpiricML
    ```
