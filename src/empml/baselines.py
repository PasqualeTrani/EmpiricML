"""
Baseline model catalog for quick benchmarking in the Lab.

Model libraries are imported only when a catalog is requested.
"""

from typing import Any

from empml.pipeline import Pipeline
from empml.wrappers import SKlearnWrapper


def _classification_estimators() -> dict[str, Any]:
    from catboost import CatBoostClassifier as ctb
    from lightgbm import LGBMClassifier as lgb
    from sklearn.ensemble import HistGradientBoostingClassifier as hgb
    from sklearn.ensemble import RandomForestClassifier as rf
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression as log_reg
    from sklearn.neighbors import KNeighborsClassifier as knn
    from sklearn.neural_network import MLPClassifier as mlp
    from sklearn.pipeline import Pipeline as SKlearnPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC as svc
    from sklearn.tree import DecisionTreeClassifier as dtree
    from xgboost import XGBClassifier as xgb

    return {
        "logistic_regression_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("clf", log_reg(max_iter=1000)),
            ]
        ),
        "knn_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("clf", knn()),
            ]
        ),
        "svm_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("clf", svc()),
            ]
        ),
        "random_forest_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("clf", rf(random_state=0, n_jobs=-1)),
            ]
        ),
        "decision_tree_base": SKlearnPipeline(
            [("impute", SimpleImputer()), ("clf", dtree())]
        ),
        "lightgbm_base": lgb(verbose=-1, random_state=0),
        "xgboost_base": xgb(verbosity=0, random_state=0),
        "catboost_base": ctb(verbose=0, random_state=0),
        "hgb_base": hgb(),
        "mlp_base": SKlearnPipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("clf", mlp(hidden_layer_sizes=(64, 32))),
            ]
        ),
    }


def _regression_estimators() -> dict[str, Any]:
    from catboost import CatBoostRegressor as ctb
    from lightgbm import LGBMRegressor as lgb
    from sklearn.ensemble import HistGradientBoostingRegressor as hgb
    from sklearn.ensemble import RandomForestRegressor as rf
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression as lr
    from sklearn.neighbors import KNeighborsRegressor as knn
    from sklearn.neural_network import MLPRegressor as mlp
    from sklearn.pipeline import Pipeline as SKlearnPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR as svr
    from sklearn.tree import DecisionTreeRegressor as dtree
    from xgboost import XGBRegressor as xgb

    return {
        "linear_regression_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("reg", lr()),
            ]
        ),
        "knn_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("reg", knn()),
            ]
        ),
        "svm_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("reg", svr()),
            ]
        ),
        "random_forest_base": SKlearnPipeline(
            [
                ("impute", SimpleImputer()),
                ("reg", rf(random_state=0, n_jobs=-1)),
            ]
        ),
        "decision_tree_base": SKlearnPipeline(
            [("impute", SimpleImputer()), ("reg", dtree())]
        ),
        "lightgbm_base": lgb(verbose=-1),
        "xgboost_base": xgb(verbosity=0),
        "catboost_base": ctb(verbose=0),
        "hgb_base": hgb(),
        "mlp_base": SKlearnPipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("reg", mlp(hidden_layer_sizes=(64, 32))),
            ]
        ),
    }


def baseline_pipelines(
    problem_type: str,
    features: str,
    target: str,
    preprocess_pipe: Pipeline | None = None,
) -> list[Pipeline]:
    """
    One pipeline per baseline model.

    ``problem_type`` 'classification' (any case) selects classifiers; any other
    value selects regressors.
    """
    if problem_type.lower() == "classification":
        estimators = _classification_estimators()
    else:
        estimators = _regression_estimators()

    preprocess_steps = [("preprocess", preprocess_pipe)] if preprocess_pipe else []
    return [
        Pipeline(
            [
                *preprocess_steps,
                (
                    "model",
                    SKlearnWrapper(
                        estimator=estimator, features=features, target=target
                    ),
                ),
            ],
            name=name,
            description=f"{name} with features = {features}",
        )
        for name, estimator in estimators.items()
    ]
