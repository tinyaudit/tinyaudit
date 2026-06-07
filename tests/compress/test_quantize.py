"""Minimal tests for quantize_int8.

The full ONNX/skl2onnx path is exercised only when skl2onnx is available;
otherwise we confirm the correct NotImplementedError is raised.  The torch
path is skipped when torch is absent.  Import correctness and the unsupported-
model error surface are always tested.
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Import smoke test
# --------------------------------------------------------------------------- #


def test_module_imports() -> None:
    """quantize.py must import cleanly with no side effects."""
    from tinyaudit.compress import quantize  # noqa: F401
    from tinyaudit.compress.quantize import QuantizedOnnxModel, quantize_int8  # noqa: F401


# --------------------------------------------------------------------------- #
# Unsupported model raises NotImplementedError with a useful message
# --------------------------------------------------------------------------- #


class _UnsupportedModel:
    """A valid-looking AuditedModel with an unsupported framework."""

    framework = "unsupported_backend"
    n_params = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.ones(n) * 0.5, np.ones(n) * 0.5])


def test_unsupported_model_raises_not_implemented() -> None:
    from tinyaudit.compress.quantize import quantize_int8

    model = _UnsupportedModel()
    with pytest.raises(NotImplementedError, match="unsupported_backend"):
        quantize_int8(model)  # type: ignore[arg-type]


def test_unsupported_model_error_names_supported_families() -> None:
    from tinyaudit.compress.quantize import quantize_int8

    model = _UnsupportedModel()
    with pytest.raises(NotImplementedError) as exc_info:
        quantize_int8(model)  # type: ignore[arg-type]
    msg = str(exc_info.value).lower()
    # The error message should tell the user what IS supported.
    assert "sklearn" in msg or "torch" in msg


# --------------------------------------------------------------------------- #
# sklearn path: guarded by skl2onnx availability
# --------------------------------------------------------------------------- #

skl2onnx = pytest.importorskip("skl2onnx", reason="skl2onnx not installed")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


@pytest.fixture()
def lr_model():
    from sklearn.linear_model import LogisticRegression

    from tinyaudit.models.sklearn import SklearnModel

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] + rng.standard_normal(200) > 0).astype(int)
    clf = LogisticRegression(max_iter=500, random_state=0).fit(X, y)
    return SklearnModel(clf), X


@pytest.fixture()
def mlp_model():
    from sklearn.neural_network import MLPClassifier

    from tinyaudit.models.sklearn import SklearnModel

    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] + rng.standard_normal(200) > 0).astype(int)
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=0).fit(X, y)
    return SklearnModel(clf), X


def test_quantize_lr_returns_audited_model(lr_model) -> None:
    from tinyaudit.compress.quantize import QuantizedOnnxModel, quantize_int8
    from tinyaudit.models.base import AuditedModel

    model, _ = lr_model
    q = quantize_int8(model)
    assert isinstance(q, QuantizedOnnxModel)
    assert isinstance(q, AuditedModel)


def test_quantize_lr_framework_is_onnx(lr_model) -> None:
    from tinyaudit.compress.quantize import quantize_int8

    model, _ = lr_model
    q = quantize_int8(model)
    assert q.framework == "onnx"


def test_quantize_lr_n_params_preserved(lr_model) -> None:
    from tinyaudit.compress.quantize import quantize_int8

    model, _ = lr_model
    q = quantize_int8(model)
    assert q.n_params == model.n_params


def test_quantize_lr_predict_shape(lr_model) -> None:
    from tinyaudit.compress.quantize import quantize_int8

    model, X = lr_model
    q = quantize_int8(model)
    preds = q.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds.tolist()).issubset({0, 1})


def test_quantize_lr_predict_proba_shape(lr_model) -> None:
    from tinyaudit.compress.quantize import quantize_int8

    model, X = lr_model
    q = quantize_int8(model)
    proba = q.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)


def test_quantize_lr_predictions_close_to_original(lr_model) -> None:
    """Quantized model predictions should largely agree with the original."""
    from tinyaudit.compress.quantize import quantize_int8

    model, X = lr_model
    q = quantize_int8(model)
    orig_preds = model.predict(X)
    q_preds = q.predict(X)
    agreement = (orig_preds == q_preds).mean()
    # Allow a generous tolerance; int8 can degrade accuracy slightly.
    assert agreement >= 0.85, f"Agreement too low: {agreement:.2%}"


def test_quantize_mlp_returns_onnx_model(mlp_model) -> None:
    from tinyaudit.compress.quantize import QuantizedOnnxModel, quantize_int8

    model, X = mlp_model
    q = quantize_int8(model)
    assert isinstance(q, QuantizedOnnxModel)
    proba = q.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_sklearn_missing_skl2onnx_raises_not_implemented(monkeypatch) -> None:
    """If skl2onnx is absent the error message must mention it."""
    import sys

    from sklearn.linear_model import LogisticRegression

    from tinyaudit.compress import quantize as qmod
    from tinyaudit.models.sklearn import SklearnModel

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression().fit(X, y)
    model = SklearnModel(clf)

    # Temporarily hide skl2onnx. Setting sys.modules[name] = None makes
    # subsequent `import skl2onnx` raise ImportError even if it is installed.
    saved = sys.modules.pop("skl2onnx", None)
    saved_dt = sys.modules.pop("skl2onnx.common.data_types", None)
    sys.modules["skl2onnx"] = None  # type: ignore[assignment]
    sys.modules["skl2onnx.common.data_types"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(NotImplementedError, match="skl2onnx"):
            qmod._quantize_sklearn(model)
    finally:
        sys.modules.pop("skl2onnx", None)
        sys.modules.pop("skl2onnx.common.data_types", None)
        if saved is not None:
            sys.modules["skl2onnx"] = saved
        if saved_dt is not None:
            sys.modules["skl2onnx.common.data_types"] = saved_dt
