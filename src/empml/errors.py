class RunExperimentConfigException(Exception):
    """Exception raised if there is a configuration error for running an experiment."""

    pass


class RunExperimentOnTestException(Exception):
    """Exception raised if there is a an error when running an experiment on the test set."""

    pass


class TargetTransformError(ValueError):
    """Exception raised when a target transformation cannot be applied or inverted."""

    pass
