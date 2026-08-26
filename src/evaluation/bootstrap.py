"""Bootstrap confidence intervals for the metrics reported elsewhere in this
project. A paper about quantifying uncertainty should not report its own
headline numbers as bare point estimates - these functions close that gap
without needing to retrain anything, since bootstrapping resamples an
already-computed (y_true, score) pair, not the training data.

Point estimates are always computed on the full data passed in - `max_n`
(where offered) only controls how many rows each bootstrap *resample* draws
from, to keep runtime bounded on evaluation sets with hundreds of thousands
of rows. It must never change the reported point estimate, only how fast the
CI around it is computed; the code below computes `point` before any
subsampling touches the arrays.
"""

import numpy as np
from scipy.stats import rankdata


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via the rank-sum / Mann-Whitney U identity - mathematically
    identical to sklearn's roc_auc_score for binary labels, but far faster
    under repeated calls (no input validation, no curve construction),
    which matters for a bootstrap loop calling it hundreds of times."""
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(y_score)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _resample_indices(rng, n: int, max_n: int | None) -> np.ndarray:
    size = n if max_n is None else min(n, max_n)
    return rng.integers(0, n, size=size)


def bootstrap_metric(
    y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 1000,
    alpha: float = 0.05, seed: int = 42, max_n: int | None = None,
) -> dict:
    """Percentile bootstrap CI for a single metric on one evaluation set."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_indices(rng, n, max_n)
        boot[i] = metric_fn(y_true[idx], y_score[idx])
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "ci_low": float(low), "ci_high": float(high), "n_boot": n_boot, "n": n}


def bootstrap_paired_diff(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, metric_fn,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 42, max_n: int | None = None,
) -> dict:
    """CI for metric_fn(a) - metric_fn(b) when a and b are two conditions
    evaluated on the *same* rows (e.g. weighted vs. unweighted conformal
    coverage on the same test set) - resamples once per iteration and
    applies the same indices to both, which is more powerful than treating
    them as independent."""
    rng = np.random.default_rng(seed)
    y_true, score_a, score_b = np.asarray(y_true), np.asarray(score_a), np.asarray(score_b)
    n = len(y_true)
    point = float(metric_fn(y_true, score_a) - metric_fn(y_true, score_b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_indices(rng, n, max_n)
        boot[i] = metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx])
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "ci_low": float(low), "ci_high": float(high), "n_boot": n_boot, "n": n}


def bootstrap_independent_diff(
    y_true_a: np.ndarray, y_score_a: np.ndarray, y_true_b: np.ndarray, y_score_b: np.ndarray,
    metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42, max_n: int | None = None,
) -> dict:
    """CI for metric_fn(a) - metric_fn(b) when a and b come from different,
    independent sets of respondents (e.g. age band 18-44 vs. 65+, or
    Northeast vs. South) - resamples each side independently."""
    rng = np.random.default_rng(seed)
    y_true_a, y_score_a = np.asarray(y_true_a), np.asarray(y_score_a)
    y_true_b, y_score_b = np.asarray(y_true_b), np.asarray(y_score_b)
    n_a, n_b = len(y_true_a), len(y_true_b)
    point = float(metric_fn(y_true_a, y_score_a) - metric_fn(y_true_b, y_score_b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = _resample_indices(rng, n_a, max_n)
        idx_b = _resample_indices(rng, n_b, max_n)
        boot[i] = metric_fn(y_true_a[idx_a], y_score_a[idx_a]) - metric_fn(y_true_b[idx_b], y_score_b[idx_b])
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "ci_low": float(low), "ci_high": float(high), "n_boot": n_boot, "n_a": n_a, "n_b": n_b}


def excludes_zero(ci: dict) -> bool:
    """Convenience check: does this difference's CI exclude zero (i.e. is it
    conventionally "significant" at the CI's alpha level)?"""
    return ci["ci_low"] > 0 or ci["ci_high"] < 0
