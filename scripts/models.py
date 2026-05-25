from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ModelBundle:
    """Container for model instances (not fitted)."""

    logistic_regression: LogisticRegression
    random_forest: RandomForestClassifier


def build_logistic_regression(
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
) -> LogisticRegression:
    """Create a baseline Logistic Regression model (not fitted)."""

    return LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        solver="lbfgs",
    )


def build_random_forest(
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced_subsample",
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Create a Random Forest model (not fitted)."""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_models(random_state: int = 42) -> ModelBundle:
    """Build both baseline and main models (not fitted)."""

    lr = build_logistic_regression(random_state=random_state)
    rf = build_random_forest(random_state=random_state)
    return ModelBundle(logistic_regression=lr, random_forest=rf)
