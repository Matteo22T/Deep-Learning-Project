from __future__ import annotations

import inspect
import os
from pathlib import Path
import random
import subprocess
import sys
import ast

import numpy as np
import pytest
from scipy import stats

from src.evaluation.bootstrap_ci import (
    compute_patient_bootstrap_ci,
    compute_student_t_ci,
)
from src.evaluation.metrics import (
    aggregate_predictions_to_patient_level,
    apply_exploratory_consistency_correction,
    check_clinical_consistency,
    compute_binary_metrics,
    summarize_fold_metrics,
)
import src.utils.reproducibility as repro_mod
from src.utils.reproducibility import create_rng, set_global_seed


# ---------------------------------------------------------------------------
# Reproducibility tests (no TensorFlow)
# ---------------------------------------------------------------------------


def test_reproducibility_seed_determinism_and_no_tensorflow() -> None:
    source = inspect.getsource(repro_mod)
    syntax_tree = ast.parse(source)

    forbidden_modules = {"tensorflow", "keras"}

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.Import):
            imported_modules = {alias.name.split(".")[0] for alias in node.names}
            assert not imported_modules.intersection(forbidden_modules)

        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_module = node.module.split(".")[0]
                assert imported_module not in forbidden_modules

    # Verify import isolation in a fresh subprocess.
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import src.utils.reproducibility as r; "
                "r.set_global_seed(42); "
                "assert 'tensorflow' not in sys.modules; "
                "assert 'keras' not in sys.modules"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"

    set_global_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
    py_seq_1 = [random.random() for _ in range(5)]
    np_seq_1 = np.random.rand(5)

    set_global_seed(123)
    py_seq_2 = [random.random() for _ in range(5)]
    np_seq_2 = np.random.rand(5)

    assert py_seq_1 == py_seq_2
    np.testing.assert_allclose(np_seq_1, np_seq_2)

    rng_a = create_rng(42)
    rng_b = create_rng(42)
    np.testing.assert_allclose(rng_a.normal(size=10), rng_b.normal(size=10))

    with pytest.raises(TypeError):
        set_global_seed(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        set_global_seed(-1)


# ---------------------------------------------------------------------------
# Patient-level prediction aggregation tests
# ---------------------------------------------------------------------------


def test_patient_level_aggregation_averages_timepoints_and_orders_deterministically() -> (
    None
):
    # Unsorted input with dual-timepoint patients PCNSL002 and PCNSL000
    patient_ids = [
        "PCNSL002_pre-bx",
        "PCNSL000_baseline",
        "PCNSL001_baseline",
        "PCNSL002_baseline",
        "PCNSL000_pre-bx",
    ]
    base_patient_ids = [
        "PCNSL002",
        "PCNSL000",
        "PCNSL001",
        "PCNSL002",
        "PCNSL000",
    ]
    y_true = [1, 0, 1, 1, 0]
    y_prob = [0.9, 0.2, 0.6, 0.7, 0.4]

    agg = aggregate_predictions_to_patient_level(
        patient_ids=patient_ids,
        base_patient_ids=base_patient_ids,
        y_true=y_true,
        y_prob=y_prob,
    )

    assert agg.base_patient_ids == ("PCNSL000", "PCNSL001", "PCNSL002")
    assert agg.timepoint_counts == (2, 1, 2)
    assert agg.source_patient_ids == (
        ("PCNSL000_baseline", "PCNSL000_pre-bx"),
        ("PCNSL001_baseline",),
        ("PCNSL002_baseline", "PCNSL002_pre-bx"),
    )
    np.testing.assert_array_equal(agg.y_true, np.array([0, 1, 1], dtype=np.int64))
    np.testing.assert_allclose(agg.y_prob, np.array([0.3, 0.6, 0.8], dtype=np.float64))


def test_patient_level_aggregation_rejects_inconsistent_labels_and_duplicate_timepoints() -> (
    None
):
    with pytest.raises(ValueError, match="Inconsistent y_true labels"):
        aggregate_predictions_to_patient_level(
            patient_ids=["PCNSL001_baseline", "PCNSL001_pre-bx"],
            base_patient_ids=["PCNSL001", "PCNSL001"],
            y_true=[0, 1],
            y_prob=[0.2, 0.8],
        )

    with pytest.raises(ValueError, match="Duplicate timepoint-level patient_ids"):
        aggregate_predictions_to_patient_level(
            patient_ids=["PCNSL001_baseline", "PCNSL001_baseline"],
            base_patient_ids=["PCNSL001", "PCNSL001"],
            y_true=[1, 1],
            y_prob=[0.7, 0.8],
        )


def test_patient_level_aggregation_rejects_mismatched_base_patient_id() -> None:
    with pytest.raises(ValueError, match="Mismatched patient ID mapping"):
        aggregate_predictions_to_patient_level(
            patient_ids=["PCNSL001_baseline", "PCNSL002_pre-bx"],
            base_patient_ids=["PCNSL001", "PCNSL001"],
            y_true=[1, 1],
            y_prob=[0.7, 0.8],
        )


# ---------------------------------------------------------------------------
# Binary metrics & fold summary tests
# ---------------------------------------------------------------------------


def test_compute_binary_metrics_exact_values() -> None:
    # 6 unique toy patients: TP=2, FN=1, TN=2, FP=1
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    y_prob = np.array([0.9, 0.7, 0.4, 0.6, 0.2, 0.1], dtype=np.float64)

    res = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert res["confusion_matrix"] == {"tp": 2, "tn": 2, "fp": 1, "fn": 1}
    assert res["n_patients"] == 6
    assert res["n_positive"] == 3
    assert res["n_negative"] == 3
    assert res["accuracy"] == pytest.approx(4 / 6)
    assert res["sensitivity"] == pytest.approx(2 / 3)
    assert res["specificity"] == pytest.approx(2 / 3)
    assert res["f1"] == pytest.approx(2 / 3)
    assert np.isfinite(res["roc_auc"])
    assert np.isfinite(res["pr_auc"])
    assert res["brier_score"] == pytest.approx(np.mean((y_prob - y_true) ** 2))


def test_compute_binary_metrics_single_class_undefined_handling() -> None:
    y_all_neg = np.array([0, 0, 0, 0], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    res = compute_binary_metrics(y_all_neg, y_prob, threshold=0.5)
    assert np.isnan(res["roc_auc"])
    assert np.isnan(res["pr_auc"])
    assert np.isnan(res["sensitivity"])
    assert np.isnan(res["f1"])
    assert res["specificity"] == pytest.approx(1.0)
    assert res["accuracy"] == pytest.approx(1.0)


def test_summarize_fold_metrics_skips_nan_folds() -> None:
    fold_ok_1 = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    fold_ok_2 = compute_binary_metrics([0, 0, 1, 1], [0.2, 0.6, 0.4, 0.8])
    fold_single_class = compute_binary_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])

    summary = summarize_fold_metrics([fold_ok_1, fold_ok_2, fold_single_class])
    assert summary["roc_auc"].n_valid == 2
    assert summary["roc_auc"].n_invalid == 1
    assert np.isfinite(summary["roc_auc"].ci_low)
    assert np.isfinite(summary["roc_auc"].ci_high)
    assert summary["roc_auc"].ci_low <= summary["roc_auc"].ci_high

    assert summary["accuracy"].n_valid == 3
    assert summary["accuracy"].n_invalid == 0


# ---------------------------------------------------------------------------
# Clinical consistency diagnostic & exploratory correction tests
# ---------------------------------------------------------------------------


def test_clinical_consistency_diagnostic_and_non_mutating_correction() -> None:
    pids = ["PCNSL000", "PCNSL001", "PCNSL002", "PCNSL003"]
    prob_os = np.array([0.8, 0.4, 0.6, 0.3], dtype=np.float64)
    prob_pfs = np.array([0.7, 0.7, 0.65, 0.2], dtype=np.float64)

    os_before = prob_os.copy()
    pfs_before = prob_pfs.copy()

    report = check_clinical_consistency(pids, prob_os, prob_pfs, threshold=0.5)
    assert report.n_patients == 4
    assert report.n_probability_violations == 2
    assert report.probability_violating_patient_ids == ("PCNSL001", "PCNSL002")
    assert report.n_binary_violations == 1
    assert report.binary_violating_patient_ids == ("PCNSL001",)
    assert report.max_probability_violation_margin == pytest.approx(0.3)

    np.testing.assert_array_equal(prob_os, os_before)
    np.testing.assert_array_equal(prob_pfs, pfs_before)

    corrected_os = apply_exploratory_consistency_correction(prob_os, prob_pfs)
    np.testing.assert_allclose(corrected_os, np.array([0.8, 0.7, 0.65, 0.3]))
    np.testing.assert_array_equal(prob_os, os_before)
    np.testing.assert_array_equal(prob_pfs, pfs_before)
    assert corrected_os is not prob_os

    post_report = check_clinical_consistency(
        pids, corrected_os, prob_pfs, threshold=0.5
    )
    assert post_report.n_probability_violations == 0
    assert post_report.n_binary_violations == 0


# ---------------------------------------------------------------------------
# Student-t CI and patient-level bootstrap CI tests
# ---------------------------------------------------------------------------


def test_student_t_ci_exact_formula_and_edge_cases() -> None:
    vals = [0.60, 0.70, 0.80, 0.75, 0.65]
    ci = compute_student_t_ci(vals, confidence_level=0.95)

    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1)) / np.sqrt(5)
    t_crit = float(stats.t.ppf(0.975, df=4))

    assert ci.n_valid == 5
    assert ci.n_invalid == 0
    assert ci.method == "student_t_descriptive"
    assert ci.point_estimate == pytest.approx(mean)
    assert ci.ci_low == pytest.approx(mean - t_crit * se)
    assert ci.ci_high == pytest.approx(mean + t_crit * se)
    assert ci.ci_low <= ci.ci_high

    ci_one = compute_student_t_ci([0.8, float("nan")])
    assert ci_one.n_valid == 1
    assert ci_one.n_invalid == 1
    assert ci_one.point_estimate == pytest.approx(0.8)
    assert np.isnan(ci_one.ci_low)
    assert np.isnan(ci_one.ci_high)


def test_stratified_bootstrap_preserves_class_counts_determinism_and_counts() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    y_prob = np.array(
        [0.1, 0.25, 0.45, 0.6, 0.35, 0.55, 0.7, 0.8, 0.85, 0.95], dtype=np.float64
    )

    observed_class_counts: list[tuple[int, int]] = []

    def tracking_roc_auc(yt: np.ndarray, yp: np.ndarray) -> float:
        observed_class_counts.append((int(np.sum(yt == 0)), int(np.sum(yt == 1))))
        return float(compute_binary_metrics(yt, yp)["roc_auc"])

    ci_a = compute_patient_bootstrap_ci(
        y_true=y_true,
        y_prob=y_prob,
        metric_fn=tracking_roc_auc,
        n_resamples=300,
        confidence_level=0.95,
        stratified=True,
        random_state=42,
    )
    assert len(observed_class_counts) == 301
    assert all(counts == (4, 6) for counts in observed_class_counts)

    ci_b = compute_patient_bootstrap_ci(
        y_true=y_true,
        y_prob=y_prob,
        metric_fn=lambda yt, yp: float(compute_binary_metrics(yt, yp)["roc_auc"]),
        n_resamples=300,
        confidence_level=0.95,
        stratified=True,
        random_state=42,
    )

    assert ci_a == ci_b
    assert ci_a.n_valid == 300
    assert ci_a.n_invalid == 0
    assert np.isfinite(ci_a.ci_low) and np.isfinite(ci_a.ci_high)
    assert ci_a.ci_low <= ci_a.ci_high


def test_bootstrap_tracks_invalid_resamples_and_handles_all_invalid() -> None:
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.5, 0.9], dtype=np.float64)

    ci_unstrat = compute_patient_bootstrap_ci(
        y_true=y_true,
        y_prob=y_prob,
        metric_fn=lambda yt, yp: float(compute_binary_metrics(yt, yp)["roc_auc"]),
        n_resamples=400,
        stratified=False,
        random_state=7,
    )
    assert ci_unstrat.n_valid + ci_unstrat.n_invalid == 400
    assert ci_unstrat.n_invalid > 0
    assert ci_unstrat.n_valid > 0
    assert np.isfinite(ci_unstrat.ci_low) and np.isfinite(ci_unstrat.ci_high)
    assert ci_unstrat.ci_low <= ci_unstrat.ci_high

    ci_all_invalid = compute_patient_bootstrap_ci(
        y_true=y_true,
        y_prob=y_prob,
        metric_fn=lambda yt, yp: float("nan"),
        n_resamples=50,
        stratified=True,
        random_state=7,
    )
    assert ci_all_invalid.n_valid == 0
    assert ci_all_invalid.n_invalid == 50
    assert np.isnan(ci_all_invalid.ci_low)
    assert np.isnan(ci_all_invalid.ci_high)


def test_bootstrap_rejects_out_of_range_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    metric_fn = lambda yt, yp: float(compute_binary_metrics(yt, yp)["accuracy"])

    with pytest.raises(ValueError, match=r"within \[0\.0, 1\.0\]"):
        compute_patient_bootstrap_ci(
            y_true=y_true,
            y_prob=np.array([-0.05, 0.2, 0.7, 0.9]),
            metric_fn=metric_fn,
            n_resamples=50,
        )

    with pytest.raises(ValueError, match=r"within \[0\.0, 1\.0\]"):
        compute_patient_bootstrap_ci(
            y_true=y_true,
            y_prob=np.array([0.1, 0.2, 0.7, 1.05]),
            metric_fn=metric_fn,
            n_resamples=50,
        )
