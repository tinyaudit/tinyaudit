"""Fitted-model fixtures shared by the XAI tests.

These wrap real scikit-learn estimators in ``SklearnModel`` so the explainers
are exercised against the frozen ``AuditedModel`` contract, never a fake.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyaudit.models import SklearnModel


@pytest.fixture
def lr_model(binary_classification) -> tuple[SklearnModel, pd.DataFrame]:
    X, y, _sensitive = binary_classification
    est = LogisticRegression(max_iter=500).fit(X, y)
    return SklearnModel(est), X


@pytest.fixture
def tree_model(binary_classification) -> tuple[SklearnModel, pd.DataFrame]:
    X, y, _sensitive = binary_classification
    est = DecisionTreeClassifier(random_state=0, max_depth=4).fit(X, y)
    return SklearnModel(est), X


@pytest.fixture
def mlp_model(binary_classification) -> tuple[SklearnModel, pd.DataFrame]:
    X, y, _sensitive = binary_classification
    est = MLPClassifier(hidden_layer_sizes=(8,), max_iter=400, random_state=0)
    est.fit(X, y)
    return SklearnModel(est), X
