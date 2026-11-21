from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.data.loader import extract_real_patient_id
from src.evaluation.bootstrap_ci import ConfidenceInterval, compute_student_t_ci

logger = logging.getLogger(__name__)

SCALAR_METRIC_NAMES: tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "f1",
    "accuracy",
    "sensitivity",
    "specificity",
    "brier_score",
)

"""
    in this module we aggregate predictions, calculate metrics, and check clinical consistency.

    in particular:

    - Patient-level aggregation: aggregate timepoint-level predictions to unique real patients, 
    ensuring that each patient has a single label and a single mean probability.
    it doesnt do only mean but checks if ID are consistent and if labels are consistent across timepoints.

    - Evaluation metrics: compute binary classification metrics:
        - ROC AUC: capacity of distinguishing between classes
        - PR AUC/AP: average precision, useful for imbalanced datasets
        - F1 score: balance between precision and recall
        - Accuracy: percentage of correct predictions
        - Sensitivity (Recall): ability to identify positive cases
        - Specificity: ability to identify negative cases
        - Brier score: quality of probabilistic predictions, lower is better
    Not defined metrics return NaN and log a warning instead of returning misleading fallback numbers.

    - Clinical consistency check: check that PFS_1y = 1 implies OS_1y = 1, both in terms of probabilities
    and binary predictions.

    - Confidence intervals: this module delegates fold-level confidence interval
    calculations to `bootstrap_ci.py`.

    - Confidence intervals: 2 methods
        - Student-t: resume variance of metrics across folds
        - Patient-level bootstrap: create repeated samples by resampling patients with replacement 
        and estimate the distribution of a metric.
    folds are not independent, so the Student-t interval is descriptive rather than an exact inferential guarantee.
    bootstrap: the interval's endpoints do not necessarily contain the point estimate.
    """


# this dataclass is used to store aggregated predictions
@dataclass(frozen=True)
class PatientPredictions:
    """Unique-patient evaluation predictions aggregated across timepoints.

    Attributes:
        base_patient_ids: Sorted unique real patient IDs (e.g., `PCNSL041`).
        y_true: Binary clinical outcome per unique patient, aligned with `base_patient_ids`.
        y_prob: Mean predicted probability across each patient's timepoints.
        timepoint_counts: Number of timepoint observations aggregated per patient.
        source_patient_ids: Sorted timepoint-level `patient_ids` (e.g.,
            `("PCNSL041_baseline", "PCNSL041_pre-bx")`) grouped into each patient.
    """

    base_patient_ids: tuple[str, ...]
    y_true: np.ndarray
    # prob for each patient, aggregated across timepoints (mean)
    y_prob: np.ndarray
    # how many timepoints were aggregated for each patient, aligned with `base_patient_ids`
    timepoint_counts: tuple[int, ...]
    # ids of the timepoints that were aggregated for each patient, aligned with `base_patient_ids`
    # to not lose traceability
    source_patient_ids: tuple[tuple[str, ...], ...]


# dataclass to store clinical consistency report
@dataclass(frozen=True)
class ClinicalConsistencyReport:
    """Read-only diagnostic of clinical constraint `PFS_1y = 1 => OS_1y = 1`."""

    n_patients: int
    # number of patients where prob(PFS) > prob(OS)
    n_probability_violations: int
    # number of patients where PFS >= threshold and OS < threshold
    n_binary_violations: int
    # IDs of patients where prob(PFS) > prob(OS)
    probability_violating_patient_ids: tuple[str, ...]
    # IDs of patients where PFS >= threshold and OS < threshold
    binary_violating_patient_ids: tuple[str, ...]
    # maximum margin of probability violation across all patients max(prob(PFS) - prob(OS))
    max_probability_violation_margin: float
    # decision threshold used for binary violation check
    decision_threshold: float

# validation functions before aggregation and metrics computation

# check if y_true is a 1D array of binary values (0 or 1)
def _validate_binary_labels(y_true: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(y_true)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("y_true must be a non-empty 1-dimensional sequence.")
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ValueError("y_true contains non-finite values.")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("y_true must contain only binary values {0, 1}.")
    return arr.astype(np.int64)

# check if y_prob is a 1D array of probabilities in [0, 1]
def _validate_probabilities(
    y_prob: Sequence[float] | np.ndarray, name: str = "y_prob"
) -> np.ndarray:
    arr = np.asarray(y_prob, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-dimensional sequence.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} values must lie within [0.0, 1.0].")
    return arr

# check if threshold is a float in (0, 1)
def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(
            f"threshold must be a float in (0, 1), got {type(threshold).__name__}."
        )
    val = float(threshold)
    if not np.isfinite(val) or not (0.0 < val < 1.0):
        raise ValueError(f"threshold must lie strictly in (0, 1), got {val}.")
    return val


def aggregate_predictions_to_patient_level(
    patient_ids: Sequence[str],
    base_patient_ids: Sequence[str],
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> PatientPredictions:
    """Aggregate timepoint-level predictions to the unique real patient level.

    Args:
        patient_ids: Timepoint-level observation IDs (e.g., `PCNSL041_baseline`,
            `PCNSL041_pre-bx`). Must be unique within this evaluation batch so no
            timepoint is accidentally double-counted.
        base_patient_ids: Normalized unique real patient IDs (e.g., `PCNSL041`)
            corresponding element-wise to `patient_ids`. Validated against
            `extract_real_patient_id(patient_id)`.
        y_true: Binary target labels aligned with `patient_ids`.
        y_prob: Predicted positive-class probabilities in `[0.0, 1.0]`.

    Returns:
        `PatientPredictions` containing exactly one row per unique
        `base_patient_id`, sorted deterministically by `base_patient_id`, with
        `y_prob` equal to the arithmetic mean across each patient's timepoints.
    """
    # convert ids to strings and validate inputs
    tp_ids = [str(pid) for pid in patient_ids]
    base_ids = [str(bid) for bid in base_patient_ids]
    y_true_arr = _validate_binary_labels(y_true)
    y_prob_arr = _validate_probabilities(y_prob, name="y_prob")

    n = len(tp_ids)
    if not (n == len(base_ids) == y_true_arr.size == y_prob_arr.size):
        raise ValueError(
            "All inputs (patient_ids, base_patient_ids, y_true, y_prob) must "
            "have identical lengths."
        )
    if len(set(tp_ids)) != n:
        raise ValueError(
            "Duplicate timepoint-level patient_ids detected; each timepoint "
            "must appear at most once before patient-level aggregation."
        )

    # aggregation

    # dict with key=base_patient_id, value=list of tuples (tp_id, y_true, y_prob)
    grouped: dict[str, list[tuple[str, int, float]]] = {}
    for tp_id, base_id, label, prob in zip(
        tp_ids, base_ids, y_true_arr.tolist(), y_prob_arr.tolist()
    ):
        expected_base_id = extract_real_patient_id(tp_id)
        if expected_base_id != base_id:
            raise ValueError(
                f"Mismatched patient ID mapping: timepoint '{tp_id}' resolves to "
                f"base patient '{expected_base_id}', but received '{base_id}'."
            )
        """
            create a blank list for each base_id if not already present, then append the tuple
            result is like this:
            {
                "PCNSL041": [
                    ("PCNSL041_baseline", 1, 0.60),
                    ("PCNSL041_pre-bx", 1, 0.80),
                ],
                "PCNSL042": [
                    ("PCNSL042_baseline", 0, 0.20),
                ],
            }
        """
        grouped.setdefault(base_id, []).append((tp_id, int(label), float(prob)))

    sorted_base_ids = tuple(sorted(grouped.keys()))
    out_y_true: list[int] = []
    out_y_prob: list[float] = []
    out_counts: list[int] = []
    out_source_ids: list[tuple[str, ...]] = []

    # for each patient we get the observation and order them by timepoint
    for base_id in sorted_base_ids:
        records = sorted(grouped[base_id], key=lambda item: item[0])
        # we check that all the timepoints for the same patient have the same label, otherwise we raise an error
        labels = {rec[1] for rec in records}
        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent y_true labels across timepoints for patient '{base_id}': "
                f"{sorted(labels)}."
            )
        # take the real patient label
        out_y_true.append(records[0][1])
        # compute the mean probability across timepoints for this patient
        out_y_prob.append(float(np.mean([rec[2] for rec in records])))
        # store the number of timepoints aggregated for this patient
        out_counts.append(len(records))
        # store the timepoint-level patient_ids that were aggregated for this patient
        out_source_ids.append(tuple(rec[0] for rec in records))

    return PatientPredictions(
        base_patient_ids=sorted_base_ids,
        y_true=np.asarray(out_y_true, dtype=np.int64),
        y_prob=np.asarray(out_y_prob, dtype=np.float64),
        timepoint_counts=tuple(out_counts),
        source_patient_ids=tuple(out_source_ids),
    )


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics on unique-patient predictions.

    Metric definition note:
    The returned `"pr_auc"` key represents **Average Precision (AP)** computed
    via `sklearn.metrics.average_precision_score` (stepwise precision-recall
    summary), rather than trapezoidal PR-AUC integration.

    Undefined metrics (for example, `roc_auc` or `pr_auc` when only one class
    appears in `y_true`, or `sensitivity`/`specificity`/`f1` with zero
    denominators) are returned explicitly as `NaN` with a logged warning rather
    than returning misleading fallback numbers.

    Parameters:
        y_true: Binary ground-truth labels (0 or 1) for each unique patient
        y_prob: Predicted positive-class probabilities in [0.0, 1.0] for each unique patient
        threshold: Decision threshold for converting probabilities to binary predictions

    Returns:
        Dictionary containing the following keys:
            - "roc_auc": ROC AUC score (float)
            - "pr_auc": Average Precision (AP) score (float)
            - "f1": F1 score (float)
            - "accuracy": Accuracy score (float)
            - "sensitivity": Sensitivity/Recall score (float)
            - "specificity": Specificity score (float)
            - "brier_score": Brier score (float)
            - "confusion_matrix": Dictionary with keys "tp", "tn", "fp", "fn" representing counts of true positives, true negatives, false positives, and false negatives
            - "n_patients": Total number of unique patients (int)
            - "n_positive": Number of positive patients (int)
            - "n_negative": Number of negative patients (int)
    """
    y_true_arr = _validate_binary_labels(y_true)
    y_prob_arr = _validate_probabilities(y_prob, name="y_prob")
    thresh = _validate_threshold(threshold)

    if y_true_arr.size != y_prob_arr.size:
        raise ValueError(
            f"Length mismatch: y_true ({y_true_arr.size}) vs y_prob ({y_prob_arr.size})."
        )

    n_patients = int(y_true_arr.size)
    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))

    y_pred = (y_prob_arr >= thresh).astype(np.int64)
    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "Single-class evaluation batch (n_pos=%d, n_neg=%d); "
            "roc_auc and pr_auc (Average Precision) are undefined and set to NaN.",
            n_pos,
            n_neg,
        )
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        # Average Precision (AP) — stepwise PR summary, not trapezoidal PR-AUC
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))

    if (tp + fn) == 0:
        logger.warning("No positive patients in batch; sensitivity is undefined (NaN).")
        sensitivity = float("nan")
    else:
        sensitivity = float(tp / (tp + fn))

    if (tn + fp) == 0:
        logger.warning("No negative patients in batch; specificity is undefined (NaN).")
        specificity = float("nan")
    else:
        specificity = float(tn / (tn + fp))

    f1_denom = 2 * tp + fp + fn
    if f1_denom == 0:
        logger.warning("F1 denominator is zero (TP=FP=FN=0); f1 is undefined (NaN).")
        f1 = float("nan")
    else:
        f1 = float((2 * tp) / f1_denom)

    accuracy = float((tp + tn) / n_patients)
    brier_score = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": brier_score,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "n_patients": n_patients,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

# TODO: confidence level could be put in config file
def summarize_fold_metrics(
    fold_metrics_list: Sequence[Mapping[str, Any]],
    confidence_level: float = 0.95,
) -> dict[str, ConfidenceInterval]:
    """Summarize scalar metrics across folds using descriptive Student-t CIs.

    Non-finite values (e.g., single-class fold metrics) are excluded from the
    mean/CI calculation and recorded in `ConfidenceInterval.n_invalid`.
    """
    if not fold_metrics_list:
        raise ValueError("fold_metrics_list must not be empty.")

    summary: dict[str, ConfidenceInterval] = {}
    for metric_name in SCALAR_METRIC_NAMES:
        values = [float(fold[metric_name]) for fold in fold_metrics_list]
        summary[metric_name] = compute_student_t_ci(
            values, confidence_level=confidence_level
        )
    return summary


def check_clinical_consistency(
    base_patient_ids: Sequence[str],
    prob_os: Sequence[float] | np.ndarray,
    prob_pfs: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> ClinicalConsistencyReport:
    """Read-only diagnostic checking `PFS_1y = 1 => OS_1y = 1` on original predictions.

    This function NEVER mutates or corrects `prob_os` or `prob_pfs`.
    """
    pids = [str(pid) for pid in base_patient_ids]
    os_arr = _validate_probabilities(prob_os, name="prob_os")
    pfs_arr = _validate_probabilities(prob_pfs, name="prob_pfs")
    thresh = _validate_threshold(threshold)

    n = len(pids)
    if not (n == os_arr.size == pfs_arr.size):
        raise ValueError(
            "base_patient_ids, prob_os, and prob_pfs must have identical lengths."
        )
    if len(set(pids)) != n:
        raise ValueError("base_patient_ids must contain unique patient identifiers.")

    prob_diff = pfs_arr - os_arr
    prob_viol_mask = prob_diff > 0.0
    bin_viol_mask = (pfs_arr >= thresh) & (os_arr < thresh)

    prob_viol_ids = tuple(pid for pid, flag in zip(pids, prob_viol_mask) if flag)
    bin_viol_ids = tuple(pid for pid, flag in zip(pids, bin_viol_mask) if flag)
    max_margin = float(np.max(np.maximum(0.0, prob_diff)))

    return ClinicalConsistencyReport(
        n_patients=n,
        n_probability_violations=len(prob_viol_ids),
        n_binary_violations=len(bin_viol_ids),
        probability_violating_patient_ids=prob_viol_ids,
        binary_violating_patient_ids=bin_viol_ids,
        max_probability_violation_margin=max_margin,
        decision_threshold=thresh,
    )


def apply_exploratory_consistency_correction(
    prob_os: Sequence[float] | np.ndarray,
    prob_pfs: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Optional exploratory post-processing enforcing `p_corr(OS) = max(p(OS), p(PFS))`.

    Methodological note:
    This function is strictly for separate exploratory analysis. It always
    allocates and returns a new NumPy array and must NEVER be used to overwrite
    the original independent OS_1y and PFS_1y predictions in the main pipeline.
    """
    os_arr = _validate_probabilities(prob_os, name="prob_os")
    pfs_arr = _validate_probabilities(prob_pfs, name="prob_pfs")
    if os_arr.size != pfs_arr.size:
        raise ValueError("prob_os and prob_pfs must have identical lengths.")

    return np.maximum(os_arr, pfs_arr).astype(np.float64, copy=True)
