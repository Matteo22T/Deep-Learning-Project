from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from src.utils.reproducibility import create_rng

logger = logging.getLogger(__name__)

"""
    this module provides functions for computing confidence intervals using two methods:
        - Student-t: resume variance of metrics across folds
        - Patient-level bootstrap: create repeated samples by resampling patients with replacement 
        and estimate the distribution of a metric.
    folds are not independent, so the Student-t interval is descriptive rather than an exact inferential guarantee.
    bootstrap: the interval's endpoints do not necessarily contain the point estimate.

    the differences with metrics.py is that here we try to quantify the uncertainty of the metrics
"""

# data class to hold the confidence interval information
@dataclass(frozen=True)
class ConfidenceInterval:
    """Summary of a point estimate and confidence interval with validity counts.
    
        Attributes:
            point_estimate (float): The point estimate of the metric.
            ci_low (float): The lower bound of the confidence interval.
            ci_high (float): The upper bound of the confidence interval.
            confidence_level (float): The confidence level used for the interval.
            n_valid (int): The number of valid (finite) values used to compute the interval
            n_invalid (int): The number of invalid (non-finite) values excluded from the computation.
            method (str): The method used to compute the confidence interval (e.g., "student_t_descriptive",
            "patient_bootstrap").
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_valid: int
    n_invalid: int
    method: str


# helper function to validate the confidence level input
def _validate_confidence_level(confidence_level: float) -> float:
    if isinstance(confidence_level, bool) or not isinstance(
        confidence_level, (int, float)
    ):
        raise TypeError(
            f"confidence_level must be a float in (0, 1), got {type(confidence_level).__name__}."
        )
    level = float(confidence_level)
    if not np.isfinite(level) or not (0.0 < level < 1.0):
        raise ValueError(f"confidence_level must lie strictly in (0, 1), got {level}.")
    return level


def compute_student_t_ci(
    values: Sequence[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute a Student-t confidence interval across fold-level metric values.

    Methodological note:
    When applied to k-fold cross-validation scores, the Student-t interval is
    supplementary and descriptive rather than an exact inferential guarantee,
    because CV folds share overlapping training sets and their fold metrics are
    not strictly independent observations.

    Non-finite values (`NaN`, `inf`) are excluded and counted in `n_invalid`.
    If fewer than 2 valid values exist, `ci_low` and `ci_high` are returned as
    `NaN` explicitly.

    Parameters:
        values (Sequence[float] | np.ndarray): A 1D array of metric values
            (e.g., fold-level scores).
        confidence_level (float, optional): The desired confidence level for the
            interval, strictly between 0 and 1. Default is 0.95.
    """
    level = _validate_confidence_level(confidence_level)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-dimensional, got shape {arr.shape}.")

    finite_mask = np.isfinite(arr)
    valid_vals = arr[finite_mask]
    n_valid = int(valid_vals.size)
    n_invalid = int(arr.size - n_valid)

    if n_valid == 0:
        logger.warning("No finite values available for Student-t CI; returning NaN.")
        return ConfidenceInterval(
            point_estimate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            confidence_level=level,
            n_valid=0,
            n_invalid=n_invalid,
            method="student_t_descriptive",
        )

    # compute mean, standard error, and Student-t critical value
    mean_val = float(np.mean(valid_vals))
    if n_valid < 2:
        logger.warning(
            "Student-t CI requires at least 2 finite values (found %d); "
            "returning NaN bounds.",
            n_valid,
        )
        return ConfidenceInterval(
            point_estimate=mean_val,
            ci_low=float("nan"),
            ci_high=float("nan"),
            confidence_level=level,
            n_valid=n_valid,
            n_invalid=n_invalid,
            method="student_t_descriptive",
        )

    sample_std = float(np.std(valid_vals, ddof=1))
    se = sample_std / np.sqrt(n_valid)
    alpha = 1.0 - level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_valid - 1))
    margin = t_crit * se

    return ConfidenceInterval(
        point_estimate=mean_val,
        ci_low=float(mean_val - margin),
        ci_high=float(mean_val + margin),
        confidence_level=level,
        n_valid=n_valid,
        n_invalid=n_invalid,
        method="student_t_descriptive",
    )


def compute_patient_bootstrap_ci(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    stratified: bool = True,
    random_state: int = 42,
) -> ConfidenceInterval:
    """Compute a non-parametric percentile bootstrap CI by resampling unique patients.

    Inputs `y_true` and `y_prob` MUST already be aggregated at the unique-patient
    level (1 entry per real patient, never timepoint-level or sequence-level rows).

    When `stratified=True`, bootstrap indices are drawn with replacement within
    each binary class (`0` and `1`), preserving the exact class counts of `y_true`
    in every replicate.

    Note:
    For skewed or discrete small-sample metrics, the original sample point
    estimate is not guaranteed to fall inside the percentile interval
    `[ci_low, ci_high]`; percentile bounds are reported directly without
    artificial clamping around `point_estimate`.
    """
    level = _validate_confidence_level(confidence_level)
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
    ):
        raise ValueError(
            f"n_resamples must be a positive integer, got {n_resamples!r}."
        )

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    if y_true_arr.ndim != 1 or y_prob_arr.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-dimensional arrays.")
    if y_true_arr.size != y_prob_arr.size:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.size} entries, "
            f"y_prob has {y_prob_arr.size}."
        )
    if y_true_arr.size < 2:
        raise ValueError("At least 2 unique patients are required for bootstrap CI.")
    if not np.isfinite(y_prob_arr).all():
        raise ValueError("y_prob contains non-finite values.")
    if np.any((y_prob_arr < 0.0) | (y_prob_arr > 1.0)):
        raise ValueError("y_prob values must lie within [0.0, 1.0].")

    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("y_true must contain only binary values {0, 1}.")
    y_true_bin = y_true_arr.astype(np.int64)

    point_est = float(metric_fn(y_true_bin, y_prob_arr))

    rng = create_rng(random_state)
    n_patients = y_true_bin.size
    pos_indices = np.flatnonzero(y_true_bin == 1)
    neg_indices = np.flatnonzero(y_true_bin == 0)

    valid_scores: list[float] = []
    n_invalid = 0

    for _ in range(n_resamples):
        if stratified and pos_indices.size > 0 and neg_indices.size > 0:
            sampled_pos = rng.choice(pos_indices, size=pos_indices.size, replace=True)
            sampled_neg = rng.choice(neg_indices, size=neg_indices.size, replace=True)
            sample_idx = np.concatenate([sampled_neg, sampled_pos])
        else:
            sample_idx = rng.integers(0, n_patients, size=n_patients)

        score = float(metric_fn(y_true_bin[sample_idx], y_prob_arr[sample_idx]))
        if np.isfinite(score):
            valid_scores.append(score)
        else:
            n_invalid += 1

    n_valid = len(valid_scores)
    method = "stratified_patient_bootstrap" if stratified else "patient_bootstrap"

    if n_valid == 0:
        logger.warning(
            "All %d bootstrap resamples produced non-finite metric values; "
            "returning NaN CI bounds.",
            n_resamples,
        )
        return ConfidenceInterval(
            point_estimate=point_est,
            ci_low=float("nan"),
            ci_high=float("nan"),
            confidence_level=level,
            n_valid=0,
            n_invalid=n_invalid,
            method=method,
        )

    if n_invalid > 0:
        logger.warning(
            "Bootstrap CI encountered %d/%d invalid (non-finite) resamples; "
            "percentiles computed over %d valid resamples.",
            n_invalid,
            n_resamples,
            n_valid,
        )

    alpha = 1.0 - level
    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(
        np.asarray(valid_scores, dtype=np.float64), [low_q, high_q]
    )

    return ConfidenceInterval(
        point_estimate=point_est,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=level,
        n_valid=n_valid,
        n_invalid=n_invalid,
        method=method,
    )
