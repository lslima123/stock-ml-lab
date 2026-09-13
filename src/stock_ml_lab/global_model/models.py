from __future__ import annotations

from stock_ml_lab.classification.models import make_logistic_classifier
from stock_ml_lab.models.linear import make_ridge_pipeline


GLOBAL_REGRESSION_MODEL = "ridge"
GLOBAL_CLASSIFICATION_MODEL = "logistic"

GLOBAL_HYPERPARAMETERS = {
    ("regression", "ridge"): {"alpha": 1.0},
    ("classification", "logistic"): {"C": 1.0},
}


def build_global_estimator(task: str, model: str, *, random_state: int = 42):
    model = model.strip().lower()
    if task == "regression" and model == "ridge":
        return make_ridge_pipeline(alpha=1.0)
    if task == "classification" and model == "logistic":
        return make_logistic_classifier(C=1.0, random_state=random_state)
    raise ValueError(
        "M12 global scope supports regression/ridge and classification/logistic."
    )
