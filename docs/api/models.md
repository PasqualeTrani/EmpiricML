# Models API Reference

This section documents all model wrappers in EmpiricML.

## LightGBM Wrapper

```python
from EmpiricML import LGBMWrapper

model = LGBMWrapper()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## XGBoost Wrapper

```python
from EmpiricML import XGBWrapper

model = XGBWrapper()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## CatBoost Wrapper

```python
from EmpiricML import CatBoostWrapper

model = CatBoostWrapper()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## PyTorch Wrapper

```python
from EmpiricML import TorchWrapper

model = TorchWrapper(model_class=YourNeuralNetwork)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

*Detailed API documentation coming soon...*
