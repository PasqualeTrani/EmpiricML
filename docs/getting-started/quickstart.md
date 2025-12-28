# Quick Start

Get up and running with EmpiricML in just a few minutes!

## Your First Model

This guide will walk you through building your first machine learning model with EmpiricML.

### Step 1: Import EmpiricML

```python
import EmpiricML as eml
import pandas as pd
```

### Step 2: Load Your Data

```python
# Load data from a CSV file
df = pd.read_csv("your_data.csv")

# Or create sample data
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
```

### Step 3: Train a Model

```python
# Create and train a LightGBM model
model = eml.LGBMWrapper()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Step 4: Evaluate Performance

```python
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, predictions))
```

## Working with Different Models

EmpiricML supports multiple model types:

=== "LightGBM"

    ```python
    model = eml.LGBMWrapper()
    model.fit(X_train, y_train)
    ```

=== "XGBoost"

    ```python
    model = eml.XGBWrapper()
    model.fit(X_train, y_train)
    ```

=== "CatBoost"

    ```python
    model = eml.CatBoostWrapper()
    model.fit(X_train, y_train)
    ```

=== "PyTorch Neural Network"

    ```python
    model = eml.TorchWrapper(model_class=YourNeuralNetwork)
    model.fit(X_train, y_train)
    ```

## Next Steps

- Learn about [Basic Concepts](concepts.md)
- Explore the [User Guide](../guide/overview.md)
- Check out the [API Reference](../api/core.md)
