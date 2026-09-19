# empml.errors

| Object | Description |
| :--- | :--- |
| `RunExperimentConfigException` | Exception raised for invalid experiment configurations. |
| `RunExperimentOnTestException` | Exception raised when errors occur during testing on the test set. |
| `TargetTransformError` | Exception raised when a target transformation cannot be applied or inverted. |

## RunExperimentConfigException
Exception raised for invalid experiment configurations.

## RunExperimentOnTestException
Exception raised when errors occur during testing on the test set.

## TargetTransformError
Exception raised when a target transformation cannot be applied or inverted.

Subclasses `ValueError`. Raised when the target holds values outside the transformation's domain, when the
target column is missing, or when a prediction cannot be mapped back. See [`empml.target`](target.md).
