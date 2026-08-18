"""Uncertainty-aware fairness metrics.

These functions consume an :class:`~tinyaudit.uncertainty.types.UncertaintyOutput`
and a sensitive-attribute array and return scalar or per-group summaries that
quantify how uncertainty interacts with group fairness.

All three functions are pure (no side effects, no mutations) and work with any
number of sensitive groups.
"""

from __future__ import annotations

import math

import numpy as np

from tinyaudit.uncertainty.types import NDArrayAny, UncertaintyOutput

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _str_key(label: object) -> str:
    """Stringify a group label for dict keys."""
    return str(label)


def _unique_groups(sensitive: NDArrayAny) -> NDArrayAny:
    """Return sorted unique group values (preserves object-array labels)."""
    return np.asarray(np.unique(sensitive))


def _dp_diff_on_subset(
    mean_proba: NDArrayAny, mask: NDArrayAny, sensitive: NDArrayAny
) -> float | None:
    """Demographic-parity difference on the samples selected by *mask*.

    Uses the hard-prediction argmax of ``mean_proba`` so no ``y_pred``
    array is needed.  Returns ``None`` if any group has zero samples after
    masking (coverage threshold causes a group to disappear).
    """
    sub_proba = mean_proba[mask]
    sub_sens = sensitive[mask]
    y_pred = np.argmax(sub_proba, axis=1).astype(np.intp)

    groups = _unique_groups(sub_sens)
    if groups.size < 2:
        return None

    rates: list[float] = []
    for g in groups:
        g_mask = sub_sens == g
        n_g = int(np.count_nonzero(g_mask))
        if n_g == 0:
            return None
        rate = float(np.mean(y_pred[g_mask] == 1))
        rates.append(rate)

    return float(max(rates) - min(rates))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def group_predictive_entropy(out: UncertaintyOutput, sensitive: NDArrayAny) -> dict[str, float]:
    """Mean predictive entropy per sensitive group.

    Returns a dict mapping group label (stringified) to mean predictive
    entropy for samples in that group.  Empty groups are skipped entirely
    (they do not appear in the output dict).  Single-class groups are handled
    without special cases because entropy is still computable over one sample;
    the function will never raise.

    Parameters
    ----------
    out:
        Uncertainty output from any estimator.
    sensitive:
        1-D array of group labels aligned with ``out.predictive_entropy``.

    Returns
    -------
    dict[str, float]
        ``{group_label_str: mean_entropy}``.  ``nan`` is returned for a group
        only when the entropy values themselves are ``nan`` (propagated from
        the estimator).
    """
    entropy = out.predictive_entropy  # shape (n,)
    result: dict[str, float] = {}
    for g in _unique_groups(sensitive):
        mask = sensitive == g
        n_g = int(np.count_nonzero(mask))
        if n_g == 0:
            # skip truly-empty groups (should not happen with np.unique, but
            # guard defensively)
            continue
        result[_str_key(g)] = float(np.mean(entropy[mask]))
    return result


def _bin_edges(confidence: NDArrayAny, n_bins: int, binning: str) -> NDArrayAny:
    """Bin boundaries over the confidence scale.

    ``equal_width`` cuts [0, 1] into ``n_bins`` equal slices. It is the
    conventional choice and the default, but real classifiers pile almost all
    of their mass into one or two slices, leaving most bins empty and the
    estimate driven by whichever few bins happen to be populated.

    ``equal_mass`` places the edges at quantiles of the observed confidences,
    so every bin holds roughly the same *number* of predictions. Duplicate
    edges are collapsed, which is what happens when a model is degenerate and
    emits a single repeated confidence: the group then has one bin, and ECE
    reduces to the absolute gap between mean confidence and accuracy.
    """
    if binning == "equal_width":
        return np.asarray(np.linspace(0.0, 1.0, n_bins + 1))
    if binning == "equal_mass":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(confidence, quantiles))
        if edges.size < 2:
            # Every confidence identical: one bin spanning that single value.
            value = float(edges[0]) if edges.size else 0.0
            return np.asarray([value, value])
        return np.asarray(edges)
    raise ValueError(f"binning must be 'equal_width' or 'equal_mass'; got {binning!r}")


def ece_per_group(
    out: UncertaintyOutput,
    y_true: NDArrayAny,
    sensitive: NDArrayAny,
    n_bins: int = 10,
    binning: str = "equal_width",
) -> dict[str, float]:
    """Expected Calibration Error (ECE) per sensitive group.

    ECE is computed over the maximum predicted probability (confidence score),
    by default with 10 equal-width bins:

    .. math::

        \\text{ECE} = \\sum_b \\frac{|B_b|}{n} \\left|
            \\text{acc}(B_b) - \\text{conf}(B_b)
        \\right|

    Edge cases handled:

    * Empty bins: skipped (no contribution to the weighted sum).
    * Single-class groups: ECE is still computed if the group has at least 2
      samples.
    * Groups with fewer than 2 samples: ``nan`` is returned rather than
      crashing.

    Parameters
    ----------
    out:
        Uncertainty output; ``mean_proba`` is used for confidence.
    y_true:
        True binary labels (0/1), aligned with ``out.mean_proba``.
    sensitive:
        1-D group label array.
    n_bins:
        Number of confidence bins. Defaults to 10.
    binning:
        ``"equal_width"`` (default) for fixed slices of [0, 1], or
        ``"equal_mass"`` for quantile edges placing roughly equal counts in
        each bin. Equal-mass edges are computed per group, from that group's
        own confidences, so each group's bins are equally populated.

    Returns
    -------
    dict[str, float]
        ``{group_label_str: ECE}``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1; got {n_bins}")

    # Confidence = max predicted probability; hard prediction = argmax.
    confidence = np.max(out.mean_proba, axis=1)  # (n,)
    y_pred = np.argmax(out.mean_proba, axis=1)  # (n,)

    result: dict[str, float] = {}
    for g in _unique_groups(sensitive):
        mask = sensitive == g
        n_g = int(np.count_nonzero(mask))

        if n_g < 2:
            result[_str_key(g)] = float("nan")
            continue

        g_conf = confidence[mask]
        g_pred = y_pred[mask]
        g_true = y_true[mask]

        bin_edges = _bin_edges(g_conf, n_bins, binning)
        n_edges = int(bin_edges.size) - 1

        ece_acc = 0.0
        for b in range(n_edges):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            # Include the right edge in the last bin so the top confidence
            # (and the degenerate lo == hi case) is never dropped.
            if b < n_edges - 1:
                bin_mask = (g_conf >= lo) & (g_conf < hi)
            else:
                bin_mask = (g_conf >= lo) & (g_conf <= hi)
            n_b = int(np.count_nonzero(bin_mask))
            if n_b == 0:
                continue
            acc_b = float(np.mean(g_pred[bin_mask] == g_true[bin_mask]))
            conf_b = float(np.mean(g_conf[bin_mask]))
            ece_acc += (n_b / n_g) * abs(acc_b - conf_b)

        result[_str_key(g)] = float(ece_acc)

    return result


def selective_fairness_auc(
    out: UncertaintyOutput,
    y_true: NDArrayAny,
    sensitive: NDArrayAny,
) -> float:
    """AUC of DP difference as low-confidence samples are abstained on.

    Algorithm
    ---------
    1. Sort samples by ``predictive_entropy`` ascending (most confident first).
    2. For each of 20 coverage thresholds ``c`` linearly spaced in [0.1, 1.0],
       take the top ``ceil(c * n)`` most-confident samples and compute
       demographic-parity difference on that subset.
    3. Integrate the DP-diff curve over coverage using the trapezoidal rule.

    Coverage thresholds where any group has zero samples are skipped.
    Returns ``nan`` if fewer than 2 thresholds are computable.

    Parameters
    ----------
    out:
        Uncertainty output; ``predictive_entropy`` and ``mean_proba`` are used.
    y_true:
        True labels (not used for DP diff itself, kept for API symmetry with
        the other metrics and for potential future extensions).
    sensitive:
        1-D group label array.

    Returns
    -------
    float
        The AUC value.  Lower is better (fairer under abstention).
    """
    n = out.predictive_entropy.shape[0]
    # Sort ascending by entropy: index 0 = most confident.
    sort_idx = np.argsort(out.predictive_entropy, kind="stable")

    coverages_raw = np.linspace(0.1, 1.0, 20)

    coverages_ok: list[float] = []
    dp_diffs: list[float] = []

    for c in coverages_raw:
        k = max(1, math.ceil(c * n))
        top_k_idx = sort_idx[:k]
        mask = np.zeros(n, dtype=bool)
        mask[top_k_idx] = True

        dp = _dp_diff_on_subset(out.mean_proba, mask, sensitive)
        if dp is None:
            continue
        coverages_ok.append(c)
        dp_diffs.append(dp)

    if len(coverages_ok) < 2:
        return float("nan")

    # Trapezoidal integration over coverage.
    auc = float(np.trapz(dp_diffs, coverages_ok))
    return auc
