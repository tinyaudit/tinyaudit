"""Config-driven entry point.

``tinyaudit run --config run.yaml`` reproduces an audit from a config file
(YAML if PyYAML is available, JSON always). The config names a dataset, a
baseline model, the sensitive attribute, and an output path; the CLI is a
thin wrapper so the result matches calling :func:`tinyaudit.audit` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from tinyaudit.data import load_adult, load_compas, load_folktables
from tinyaudit.pipeline import audit

_DATASETS = {
    "adult": load_adult,
    "compas": load_compas,
    "folktables": load_folktables,
}
_MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "tree": lambda: DecisionTreeClassifier(random_state=0),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0),
}
console = Console()


def _load_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise click.ClickException(
                "PyYAML is not installed; use a .json config or install the full stack"
            ) from exc
        return dict(yaml.safe_load(text))
    return dict(json.loads(text))


@click.group()
def main() -> None:
    """tinyaudit command-line interface."""


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
def run(config: str) -> None:
    """Run an audit from a config file and write the card."""
    cfg = _load_config(config)
    dataset = cfg.get("dataset", "adult")
    model_key = cfg.get("model", "logreg")
    sensitive = cfg["sensitive"]
    seed = int(cfg.get("seed", 0))
    out = cfg.get("output", "audit_card.html")
    compression = cfg.get("compression")

    if dataset not in _DATASETS:
        raise click.ClickException(f"unknown dataset {dataset!r}; known: {list(_DATASETS)}")
    if model_key not in _MODELS:
        raise click.ClickException(f"unknown model {model_key!r}; known: {list(_MODELS)}")

    loader = _DATASETS[dataset]
    X_train, y_train, _ = loader(split="train", seed=seed)
    X_test, y_test, s_test = loader(split="test", seed=seed)

    estimator = _MODELS[model_key]()
    estimator.fit(X_train.to_numpy(), y_train.to_numpy())

    card = audit(
        estimator,
        data=(X_test, y_test),
        sensitive=s_test[sensitive],
        seed=seed,
        dataset=dataset,
        compression=compression,
    )

    out_path = Path(out)
    if out_path.suffix == ".pdf":
        card.to_pdf(str(out_path))
    else:
        out_path.write_text(card.to_html())

    console.print(f"[bold green]audit complete[/] -> {out_path}")
    console.print(f"manifest: {card.manifest_path}")
    for m in card.fairness.metrics:
        console.print(f"  {m.name}: {m.value:.4f} ({m.band})", markup=False)


if __name__ == "__main__":
    main()
