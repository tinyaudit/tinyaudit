# Architecture and Module Spec

This is the build blueprint. It says what each module owns, what it exposes,
and the signatures other modules depend on. Implement against these contracts
so the pieces fit together without rework.

## Public API

The package exposes exactly two names at the top level. Keep it that small.

```python
# src/tinyaudit/__init__.py
from tinyaudit.card.schema import AuditCard
from tinyaudit.pipeline import audit

__all__ = ["audit", "AuditCard"]
```

```python
def audit(
    model: Any,                       # AuditedModel, or a fitted sklearn estimator (auto-wrapped)
    data: tuple[pd.DataFrame, pd.Series],
    sensitive: str | pd.Series | pd.DataFrame | np.ndarray,
    methods: list[str] | None = None,            # default: all stages
    compression: str | None = None,
    seed: int = 0,
    *,
    dataset: str = "dataset",         # label recorded on the card and manifest
) -> AuditCard: ...
```

`audit()` orchestrates the pipeline stages, writes one JSON manifest, and
returns an `AuditCard`. The CLI calls this same function so a run reproduces
from a config file. `sensitive` is a column name in `data[0]` or an aligned
Series/array of the protected attribute (pass one attribute at a time, e.g.
`S["sex"]`). A bare fitted scikit-learn estimator is accepted and wrapped in
`SklearnModel` automatically. Only implemented stages run; any requested stage
that is not yet built is recorded as skipped in the manifest and its card
block stays `None`. Everything below is internal.

## Module map

| Module | Owns | Key public symbols |
| --- | --- | --- |
| `pipeline.py` | Stage orchestration, manifest writing | `audit()` |
| `models/base.py` | The model contract | `AuditedModel` protocol |
| `models/sklearn.py` | scikit-learn adapter | `SklearnModel` |
| `models/torch.py` | PyTorch adapter | `TorchModel` |
| `data/adult.py` | UCI Adult loader | `load_adult()` |
| `data/folktables.py` | ACSIncome loader | `load_folktables()` |
| `data/compas.py` | COMPAS loader | `load_compas()` |
| `fairness/parity.py` | Point metrics | `demographic_parity_difference()`, `equalized_odds_difference()`, `disparate_impact_ratio()` |
| `fairness/_frames.py` | Per-group aggregation | `MetricFrame` |
| `uncertainty/mc_dropout.py` | MC Dropout estimator | `MCDropout` |
| `uncertainty/ensemble.py` | Deep ensemble estimator | `DeepEnsemble` |
| `uncertainty/early_exit.py` | QUTE-style estimator | `EarlyExitEnsemble` |
| `uncertainty/metrics.py` | Uncertainty-aware fairness | `group_predictive_entropy()`, `ece_per_group()`, `selective_fairness_auc()` |
| `xai/shap_wrap.py` | SHAP adapter | `shap_attributions()` |
| `xai/lime_wrap.py` | LIME adapter | `lime_attributions()` |
| `xai/occlusion.py` | Lightweight explainer | `occlusion_attributions()` |
| `compress/quantize.py` | int8 quantization | `quantize_int8()` |
| `compress/prune.py` | Magnitude pruning | `magnitude_prune()` |
| `profile/footprint.py` | Cost measurement | `profile_model()` |
| `card/schema.py` | The card data model | `AuditCard` and sub-models |
| `card/render.py` | Card rendering | `render_pdf()`, `render_html()` |
| `cli.py` | Config-driven entry point | `main()` |

## The model contract

Every adapter conforms to one protocol. Metrics and estimators depend only on
this, never on a concrete framework.

```python
# src/tinyaudit/models/base.py
from typing import Protocol
import numpy as np

class AuditedModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    @property
    def n_params(self) -> int: ...
    @property
    def framework(self) -> str: ...  # "sklearn" | "torch" | "onnx"
```

`SklearnModel` and `TorchModel` wrap a trained estimator and satisfy the
protocol. The torch wrapper also exposes the hooks MC Dropout and the
early-exit head need (dropout-on-at-inference, intermediate exits).

## Data loaders

Each loader is deterministic and returns the same shape.

```python
def load_adult(
    split: str = "test",          # "train" | "test"
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (features, label, sensitive)."""
```

The third return value is a DataFrame with exactly the columns `sex` and
`race`, carrying the raw protected attributes (sensitive columns are excluded
from `features`) so the audit can group on them after the model has predicted.
`race` stays multi-valued; it is never binarized. Preprocessing lives here, in
code, and is the single source of truth. The loaders own their validation:
schema and dtype checks, explicit categorical encodings, Folktables row-count
verification, and the COMPAS `race` spot-check.

## Fairness metrics

Pure functions, implemented from scratch, no Fairlearn in this path.

```python
def demographic_parity_difference(
    y_pred: np.ndarray, sensitive: np.ndarray
) -> float: ...

def equalized_odds_difference(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray
) -> float: ...

def disparate_impact_ratio(
    y_pred: np.ndarray, sensitive: np.ndarray
) -> float: ...
```

`MetricFrame` does the per-group splitting and aggregation so the three
functions stay short. The disparate-impact direction convention (numerator, denominator,
favored outcome) is fixed in the docstring and in a dedicated test. Do not
change it without changing both.

## Uncertainty estimators

One interface, three implementations.

```python
class UncertaintyEstimator(Protocol):
    def fit(self, model: AuditedModel, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict_dist(self, X: np.ndarray) -> UncertaintyOutput: ...

@dataclass
class UncertaintyOutput:
    mean_proba: np.ndarray
    predictive_entropy: np.ndarray
    predictive_variance: np.ndarray
    mutual_information: np.ndarray
```

`MCDropout` applies to the MLP only. `DeepEnsemble` trains five seeds across
every model family. `EarlyExitEnsemble` is the QUTE-style on-device proof
point. Each estimator is profiled through `profile/footprint.py`; its own cost
is part of the audit output.

The uncertainty-aware fairness metrics consume `UncertaintyOutput`:

```python
def group_predictive_entropy(out: UncertaintyOutput, sensitive: np.ndarray) -> dict: ...
def ece_per_group(out: UncertaintyOutput, y_true: np.ndarray, sensitive: np.ndarray) -> dict: ...
def selective_fairness_auc(out: UncertaintyOutput, y_true: np.ndarray, sensitive: np.ndarray) -> float: ...
```

Handle the edge cases explicitly: empty calibration bins and single-class
groups.

## Explainers

All three return the same array shape so the card renderer is uniform.

```python
def shap_attributions(model, X, background_size=1000) -> np.ndarray: ...
def lime_attributions(model, X) -> np.ndarray: ...
def occlusion_attributions(model, X) -> np.ndarray: ...  # MCU-feasible
```

SHAP picks the explainer by framework (Kernel for the MLP, Tree for the
decision tree, Linear for logistic regression). The pipeline computes per-group
attributions and flags importance flips across sensitive groups.

## Compression

Both functions take an `AuditedModel` and return a new `AuditedModel` that
still satisfies the protocol, so every downstream stage runs unchanged.

```python
def quantize_int8(model: AuditedModel) -> AuditedModel: ...
def magnitude_prune(model: AuditedModel, sparsity: float) -> AuditedModel: ...
```

Sparsity sweep values: 0.30, 0.50, 0.70, 0.90. int8 uses onnxruntime for
sklearn and ONNX, `torch.quantization` for the MLP.

## Profiler

`Footprint` is a pydantic model so it nests directly inside `AuditCard` with
no conversion layer.

```python
class Footprint(BaseModel):
    n_params: int
    peak_ram_bytes: int
    flops: int
    wall_clock_s_per_sample: float

def profile_model(model: AuditedModel, X: np.ndarray) -> Footprint: ...
```

Peak RAM uses `tracemalloc`. FLOPs use thop or fvcore. These numbers back the
"sized for microcontroller-class hardware" claim, so record them for every
model and every compression setting.

## Card schema and renderer

`AuditCard` is a pydantic model built from the manifest, never hand-edited. It
holds the footprint, the six metric values with traffic-light bands, the
per-group breakdowns, and the explainer summary.

```python
class AuditCard(BaseModel):
    dataset: str
    model: str
    compression: str | None
    footprint: Footprint
    fairness: FairnessBlock          # DP, EO, DI + bands
    uncertainty: UncertaintyBlock    # group entropy, ECE/group, selective AUC
    explainability: XaiBlock
    manifest_path: str

    def to_pdf(self, path: str) -> None: ...
    def to_html(self) -> str: ...
```

`render.py` turns the schema into one page via Jinja2 to HTML to PDF
(WeasyPrint). The template is `card/templates/audit_card.html.j2`. The page is
traffic-light coded and readable at a glance by a non-expert.

## Manifest

Every stage appends to one JSON file per run:
`experiments/results/<run_id>/manifest.json`. It records the config hash, the
library versions, the seed, and one block per stage. Every figure has a sibling
CSV in the same directory. The card is generated from this file, so the
manifest is the contract between the pipeline and the renderer.

## Build order

Dependencies point downward, so build bottom-up: `models/base.py` and the
adapters first, then `data/`, then `profile/`, then `fairness/`, then
`uncertainty/`, then `xai/`, then `compress/`, then `card/`, then `pipeline.py`
to wire them, and `cli.py` last. See [pipeline.md](pipeline.md) for the runtime
dataflow and [conventions.md](conventions.md) for the testing and CI standards
each module ships with.
