"""int8 quantization for fitted models.

``quantize_int8`` accepts any :class:`~tinyaudit.models.base.AuditedModel` and
returns a new :class:`~tinyaudit.models.base.AuditedModel` that still satisfies
the protocol.

Supported paths
---------------
* **sklearn** (``LogisticRegression``, ``MLPClassifier``):
  The estimator is exported to ONNX via *skl2onnx*, quantized in-memory with
  ``onnxruntime.quantization.quantize_dynamic``, and re-loaded as an
  ``onnxruntime.InferenceSession`` wrapped in a thin
  :class:`QuantizedOnnxModel` adapter.

* **torch** (``framework == "torch"``):
  ``torch.quantization.quantize_dynamic`` is applied to the wrapped ``nn.Module``
  targeting ``torch.nn.Linear`` layers with ``dtype=torch.qint8``. The
  quantized module is wrapped in a new model of the same concrete type as the
  input (duck-typed, no import of ``TorchModel``).

Unsupported families raise :class:`NotImplementedError` with a clear message
naming the supported ones.

Notes
-----
- ``skl2onnx`` is an optional dependency; its absence raises
  ``NotImplementedError`` rather than ``ImportError`` so callers get a
  consistent error surface.
- mypy ``--strict`` is *not* required on this module (only on ``fairness/``,
  ``uncertainty/``, and ``card/``).
- Torch duck-typing is intentional: ``TorchModel`` is never imported at
  module level to avoid circular dependencies.
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np

from tinyaudit.models.base import AuditedModel
from tinyaudit.models.sklearn import SklearnModel

# --------------------------------------------------------------------------- #
# ONNX adapter
# --------------------------------------------------------------------------- #


class QuantizedOnnxModel:
    """A thin :class:`AuditedModel` adapter around an ONNX inference session.

    Parameters
    ----------
    session:
        A fully constructed ``onnxruntime.InferenceSession`` for the quantized
        model.
    n_params_original:
        The parameter count of the source model, forwarded verbatim because
        counting parameters from an ONNX graph is non-trivial and the caller
        already knows the right value.
    """

    def __init__(self, session: Any, n_params_original: int) -> None:
        self._session = session
        self._n_params = n_params_original

    # ------------------------------------------------------------------ #
    # Protocol implementation
    # ------------------------------------------------------------------ #

    @property
    def framework(self) -> str:
        return "onnx"

    @property
    def n_params(self) -> int:
        """Parameter count inherited from the source model."""
        return self._n_params

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-class probabilities, shape ``(n_samples, n_classes)``."""
        arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: arr})
        # onnxruntime's probability output is either a list-of-dicts or an
        # ndarray depending on the opset.  Normalise both to a float64 matrix.
        raw = outputs[1]
        if isinstance(raw, np.ndarray):
            return raw.astype(np.float64)
        # list-of-dicts: [{class: prob, ...}, ...]
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            classes = sorted(raw[0].keys())
            return np.array([[row[c] for c in classes] for row in raw], dtype=np.float64)
        # Fallback: use output index 0 (label) and derive a binary proba.
        labels = np.asarray(outputs[0])
        p1 = labels.astype(np.float64)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class labels (``argmax`` of ``predict_proba``)."""
        return np.asarray(np.argmax(self.predict_proba(X), axis=1))


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _quantize_sklearn(model: SklearnModel) -> QuantizedOnnxModel:
    """Export *model* to ONNX, quantize with onnxruntime, wrap in adapter."""
    # ------------------------------------------------------------------
    # Guard: skl2onnx is optional.
    # ------------------------------------------------------------------
    try:
        from skl2onnx import convert_sklearn  # type: ignore[import-untyped]
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore[import-untyped]
    except ImportError as exc:
        raise NotImplementedError(
            "install skl2onnx for sklearn quantization " "(pip install skl2onnx)"
        ) from exc

    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            QuantType,
            quantize_dynamic,
        )
    except ImportError as exc:
        raise NotImplementedError(
            "onnxruntime is required for sklearn quantization " "(pip install onnxruntime)"
        ) from exc

    est = model.estimator
    cls_name = type(est).__name__

    # Determine input width from fitted weight shapes.
    if cls_name == "LogisticRegression":
        n_features = int(np.asarray(est.coef_).shape[1])
    elif cls_name == "MLPClassifier":
        n_features = int(np.asarray(est.coefs_[0]).shape[0])
    else:
        raise NotImplementedError(
            f"quantize_int8 supports sklearn LogisticRegression and "
            f"MLPClassifier; got {cls_name!r}. "
            "For other sklearn families, wrap the model in ONNX manually."
        )

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(est, initial_types=initial_type)

    # Write the original ONNX to a temp file (quantize_dynamic requires paths
    # on older onnxruntime versions).
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f_in:
        f_in.write(onnx_model.SerializeToString())
        in_path = f_in.name

    with tempfile.NamedTemporaryFile(suffix="_q.onnx", delete=False) as f_out:
        out_path = f_out.name

    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)

    session = ort.InferenceSession(out_path)
    return QuantizedOnnxModel(session, n_params_original=model.n_params)


def _quantize_torch(model: AuditedModel) -> AuditedModel:
    """Apply ``torch.quantization.quantize_dynamic`` and re-wrap the module."""
    import torch
    from torch import nn

    module = getattr(model, "module", None)
    if not isinstance(module, nn.Module):
        raise NotImplementedError(
            "quantize_int8 received a model with framework=='torch' but no "
            "torch.nn.Module reachable via a 'module' attribute; cannot "
            "quantize it."
        )

    quantized_module = torch.quantization.quantize_dynamic(
        module,
        {nn.Linear},
        dtype=torch.qint8,
    )

    # Reconstruct the same concrete wrapper class (duck-typed, no TorchModel
    # import).  Honour a ``device`` attribute if present.
    device: Any = getattr(model, "device", "cpu")
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    try:
        return type(model)(quantized_module, **kwargs)  # type: ignore[call-arg]
    except TypeError:
        return type(model)(quantized_module)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def quantize_int8(model: AuditedModel) -> AuditedModel:
    """Quantize *model* to int8 and return a new :class:`AuditedModel`.

    Parameters
    ----------
    model:
        A fitted model that satisfies the :class:`~tinyaudit.models.base.AuditedModel`
        protocol.

    Returns
    -------
    AuditedModel
        A new model (same ``framework`` for torch, ``"onnx"`` for sklearn) whose
        ``predict`` and ``predict_proba`` run on quantized weights.

    Raises
    ------
    NotImplementedError
        If the model family is not supported, or if a required optional
        dependency (``skl2onnx``, ``onnxruntime``) is missing.

    Notes
    -----
    **sklearn path** (``LogisticRegression``, ``MLPClassifier``):
        The estimator is exported to ONNX via *skl2onnx*, quantized in-memory
        with ``onnxruntime.quantization.quantize_dynamic``, and re-loaded as a
        :class:`QuantizedOnnxModel`.  Requires ``skl2onnx`` and ``onnxruntime``
        to be installed.

    **torch path** (``framework == "torch"``):
        ``torch.quantization.quantize_dynamic`` is applied targeting
        ``torch.nn.Linear`` layers with ``dtype=torch.qint8``.  The quantized
        module is re-wrapped in the same concrete class as the input.
    """
    framework = getattr(model, "framework", None)

    if isinstance(model, SklearnModel) or framework == "sklearn":
        if not isinstance(model, SklearnModel):
            raise NotImplementedError(
                "quantize_int8 got framework=='sklearn' but the model is "
                f"a {type(model).__name__}, not a SklearnModel; cannot "
                "reach its estimator for ONNX export."
            )
        return _quantize_sklearn(model)

    if framework == "torch":
        return _quantize_torch(model)

    raise NotImplementedError(
        "quantize_int8 supports sklearn (LogisticRegression, MLPClassifier) "
        "and torch models. "
        f"Got framework={framework!r} ({type(model).__name__}), which is not "
        "a supported quantization target."
    )
