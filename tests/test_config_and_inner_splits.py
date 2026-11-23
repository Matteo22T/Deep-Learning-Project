from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.data.loader import extract_real_patient_id
from src.data.splitter import generate_inner_patient_splits, generate_patient_folds
from src.preprocessing.aggregator import aggregate_sequences
from src.preprocessing.scaler import PerSequenceStandardScaler
from src.utils.config import ConfigLoader


# Note on Option B longitudinal observations:
# Baseline and pre-bx scans are kept as separate observations rather than
# averaged across timepoints. Patients with two timepoints therefore contribute
# more rows to scaler fitting and downstream model training.


def _make_toy_real_df(
    n_patients: int = 50,
    n_rare: int = 2,
    n_features: int = 6,
    n_longitudinal: int = 13,
) -> pd.DataFrame:
    """Create a synthetic toy internal cohort with configurable rare-stratum count."""
    seqs = ("T1", "T1gd", "T2", "Flair")
    n_neg = min(20, max(1, (n_patients - n_rare) // 2))
    rows: list[dict[str, object]] = []
    for idx in range(n_patients):
        base_id = f"PCNSL{idx:03d}"
        if idx < n_neg:
            os_val, pfs_val = 0, 0
        elif idx < n_neg + n_rare:
            os_val, pfs_val = 1, 0
        else:
            os_val, pfs_val = 1, 1

        timepoints = ["baseline", "pre-bx"] if idx < n_longitudinal else ["baseline"]
        for tp in timepoints:
            pid = f"{base_id}_{tp}"
            for s_idx, seq in enumerate(seqs):
                rec: dict[str, object] = {
                    "patient_id": pid,
                    "sequence": seq,
                    "OS_1y": os_val,
                    "PFS_1y": pfs_val,
                }
                for f_idx in range(n_features):
                    # Sequence-dependent offset & scale so scaling before aggregation is distinct
                    rec[f"f{f_idx}"] = float(
                        (s_idx + 1) * 100.0 + idx * (s_idx + 1.5) + 0.25 * f_idx
                    )
                rows.append(rec)
    return pd.DataFrame(rows)


def _filter_by_base_patients(
    df: pd.DataFrame, base_patient_ids: tuple[str, ...]
) -> pd.DataFrame:
    base_ids = df["patient_id"].astype(str).apply(extract_real_patient_id)
    return df.loc[base_ids.isin(base_patient_ids)].copy()


# ---------------------------------------------------------------------------
# Configuration backward compatibility & validation tests
# ---------------------------------------------------------------------------


def test_config_preprocessing_yaml_backward_compatibility() -> None:
    cfg = ConfigLoader("configs/config_preprocessing.yaml").get_config()
    assert cfg.experiment.name == "pcnsl_survival_milestone3"
    assert cfg.experiment.random_seed == 42
    assert cfg.splitting.n_splits == 5
    assert cfg.splitting.synthetic_ratios == [0.0, 0.25, 0.5, 1.0, 2.0]
    assert cfg.splitting.fold_seed_stride == 1
    assert cfg.splitting.quota_integer_tol == pytest.approx(1e-9)
    assert cfg.preprocessing.features.prefix == "f"
    assert cfg.preprocessing.sequences.order == ["T1", "T1gd", "T2", "Flair"]
    assert cfg.preprocessing.scaling.epsilon == pytest.approx(1e-8)
    assert cfg.preprocessing.aggregation.strategy == "mean"
    assert cfg.preprocessing.aggregation.strict_sequences is True
    assert cfg.preprocessing.aggregation.allow_duplicates is False
    assert not hasattr(cfg, "evaluation")
    assert not hasattr(cfg, "tuning")


def test_experiment_config_yaml_loading_and_nested_access() -> None:
    cfg = ConfigLoader("configs/experiment_config.yaml").get_config()
    assert cfg.experiment.random_seed == 42
    assert cfg.experiment.targets == ["OS_1y", "PFS_1y"]
    assert cfg.splitting.n_splits == 5
    assert cfg.splitting.synthetic_ratios == [0.0]
    assert cfg.preprocessing.sequences.order == ["T1", "T1gd", "T2", "Flair"]
    assert cfg.preprocessing.scaling.epsilon == pytest.approx(1e-8)
    assert cfg.evaluation.decision_threshold == pytest.approx(0.5)
    assert cfg.evaluation.patient_aggregation == "mean_prob"
    assert cfg.evaluation.bootstrap_resamples == 2000
    assert cfg.evaluation.bootstrap_stratified is True
    assert cfg.evaluation.confidence_level == pytest.approx(0.95)
    assert cfg.evaluation.check_clinical_consistency is True
    assert cfg.evaluation.run_exploratory_consistency_correction is False
    assert cfg.tuning.protocol == "nested_cv"
    assert cfg.tuning.inner_n_splits == 4
    assert cfg.tuning.inner_seed_stride == 100
    assert cfg.tuning.selection_metric == "roc_auc"
    assert cfg.tuning.tie_breaker_metrics == ["pr_auc", "brier_score"]
    assert cfg.tuning.refit_epoch_strategy == "median_best_epoch"
    assert cfg.tuning.search_space.aggregation_strategies == ["mean", "concat"]
    # Framework-agnostic placeholders remain empty in PR2
    assert cfg.model == SimpleNamespace()
    assert cfg.training == SimpleNamespace()


def test_experiment_config_validation_rejects_invalid_values(tmp_path: Path) -> None:
    base_yaml = Path("configs/experiment_config.yaml").read_text(encoding="utf-8")

    # Invalid decision threshold (>= 1.0)
    bad_thresh = tmp_path / "bad_thresh.yaml"
    bad_thresh.write_text(
        base_yaml.replace("decision_threshold: 0.5", "decision_threshold: 1.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="decision_threshold"):
        ConfigLoader(bad_thresh)

    # Invalid inner_n_splits (< 2)
    bad_inner = tmp_path / "bad_inner.yaml"
    bad_inner.write_text(
        base_yaml.replace("inner_n_splits: 4", "inner_n_splits: 1"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inner_n_splits"):
        ConfigLoader(bad_inner)

    # Disallowed aggregation strategy in tuning search space (per_sequence postponed)
    bad_agg = tmp_path / "bad_agg.yaml"
    bad_agg.write_text(
        base_yaml.replace("      - concat", "      - per_sequence"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aggregation_strategies"):
        ConfigLoader(bad_agg)

    # Invalid target name
    bad_target = tmp_path / "bad_target.yaml"
    bad_target.write_text(
        base_yaml.replace("    - PFS_1y", "    - INVALID_TARGET"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="experiment.targets"):
        ConfigLoader(bad_target)

    # Non-mapping model placeholder
    bad_model = tmp_path / "bad_model.yaml"
    bad_model.write_text(
        base_yaml.replace("model: {}", "model: invalid_string"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config.model must be a mapping"):
        ConfigLoader(bad_model)


# ---------------------------------------------------------------------------
# Inner patient-level splitting & rare-stratum tests
# ---------------------------------------------------------------------------


def test_inner_splits_isolation_longitudinal_grouping_and_determinism() -> None:
    cfg = ConfigLoader("configs/experiment_config.yaml").get_config()
    df_real = _make_toy_real_df(n_patients=50, n_rare=2, n_longitudinal=13)

    outer_folds = generate_patient_folds(df_real, config=cfg, df_synth=None)
    assert len(outer_folds) == 5

    for outer_fold in outer_folds:
        outer_train_set = set(outer_fold.train_real_patient_ids)
        outer_val_set = set(outer_fold.val_real_patient_ids)
        df_outer_train = _filter_by_base_patients(
            df_real, outer_fold.train_real_patient_ids
        )

        inner_a = generate_inner_patient_splits(
            df_outer_train, config=cfg, outer_fold_idx=outer_fold.fold_idx
        )
        inner_b = generate_inner_patient_splits(
            df_outer_train, config=cfg, outer_fold_idx=outer_fold.fold_idx
        )
        assert len(inner_a) == 4
        assert inner_a == inner_b

        all_inner_val_patients: list[str] = []
        for inner_fold in inner_a:
            assert inner_fold.outer_fold_idx == outer_fold.fold_idx
            inner_train_set = set(inner_fold.train_real_patient_ids)
            inner_val_set = set(inner_fold.val_real_patient_ids)

            assert inner_train_set.isdisjoint(inner_val_set)
            assert inner_train_set | inner_val_set == outer_train_set
            assert inner_train_set.isdisjoint(outer_val_set)
            assert inner_val_set.isdisjoint(outer_val_set)

            # Longitudinal timepoints stay strictly inside their patient's inner fold
            for tp_id in inner_fold.train_real_timepoint_ids:
                assert extract_real_patient_id(tp_id) in inner_train_set
            for tp_id in inner_fold.val_real_timepoint_ids:
                assert extract_real_patient_id(tp_id) in inner_val_set

            all_inner_val_patients.extend(inner_fold.val_real_patient_ids)

        # Every outer-train patient appears in inner validation exactly once
        assert sorted(all_inner_val_patients) == sorted(outer_train_set)


@pytest.mark.parametrize("n_rare", [0, 1, 2])
def test_inner_splits_rare_stratum_warnings_and_stratification_preserved(
    n_rare: int, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = ConfigLoader("configs/experiment_config.yaml").get_config()
    df_outer_train = _make_toy_real_df(
        n_patients=40, n_rare=n_rare, n_longitudinal=10
    )

    with caplog.at_level(logging.WARNING, logger="src.data.splitter"):
        inner_folds = generate_inner_patient_splits(
            df_outer_train, config=cfg, outer_fold_idx=0
        )

    assert len(inner_folds) == 4
    # StratifiedKFold must be used even when n_rare == 0, 1, or 2
    assert all(f.used_stratified_split for f in inner_folds)
    assert any("rare stratum (OS=1, PFS=0)" in rec.message for rec in caplog.records)

    all_val_patients = [pid for f in inner_folds for pid in f.val_real_patient_ids]
    assert len(all_val_patients) == 40
    assert len(set(all_val_patients)) == 40
    assert sum(f.val_strata_counts["OS=1, PFS=0"] for f in inner_folds) == n_rare


def test_inner_splits_fallback_when_stratified_kfold_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = ConfigLoader("configs/experiment_config.yaml").get_config()
    df_outer_train = _make_toy_real_df(n_patients=16, n_rare=1, n_longitudinal=4)

    def _raise_value_error(self, X, y, groups=None):
        raise ValueError("Simulated StratifiedKFold failure on degenerate toy data")

    monkeypatch.setattr(StratifiedKFold, "split", _raise_value_error)

    with caplog.at_level(logging.WARNING, logger="src.data.splitter"):
        inner_folds = generate_inner_patient_splits(
            df_outer_train, config=cfg, outer_fold_idx=1
        )

    assert len(inner_folds) == 4
    assert all(not f.used_stratified_split for f in inner_folds)
    assert any(
        "stratification is no longer guaranteed" in rec.message
        for rec in caplog.records
    )

    all_val_patients = [pid for f in inner_folds for pid in f.val_real_patient_ids]
    assert len(all_val_patients) == 16
    assert len(set(all_val_patients)) == 16


# ---------------------------------------------------------------------------
# Inner-fold scaler isolation & pre-aggregation scaling tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["mean", "concat"])
def test_inner_fold_scaler_isolation_and_pre_aggregation_scaling(
    strategy: str,
) -> None:
    cfg = ConfigLoader("configs/experiment_config.yaml").get_config()
    cfg.preprocessing.aggregation.strategy = strategy

    df_real = _make_toy_real_df(n_patients=50, n_rare=2, n_features=6, n_longitudinal=13)
    outer_fold = generate_patient_folds(df_real, config=cfg, df_synth=None)[0]
    df_outer_train = _filter_by_base_patients(df_real, outer_fold.train_real_patient_ids)
    inner_fold = generate_inner_patient_splits(
        df_outer_train, config=cfg, outer_fold_idx=outer_fold.fold_idx
    )[0]

    feat_cols = [f"f{i}" for i in range(6)]
    base_ids_all = df_real["patient_id"].astype(str).apply(extract_real_patient_id)

    # 1. Baseline scaler fitted strictly on unchanged inner_train
    df_inner_train_orig = _filter_by_base_patients(
        df_real, inner_fold.train_real_patient_ids
    )
    baseline_scaler = PerSequenceStandardScaler(cfg).fit(df_inner_train_orig)
    baseline_stats = {
        seq: (st.mean.copy(), st.scale.copy(), st.zero_variance_mask.copy())
        for seq, st in baseline_scaler._stats_by_seq.items()
    }

    # 2. Mutating inner_val must not alter fitted per-sequence means/stds
    df_mutated_inner_val = df_real.copy()
    mask_inner_val = base_ids_all.isin(inner_fold.val_real_patient_ids)
    df_mutated_inner_val.loc[mask_inner_val, feat_cols] = (
        df_mutated_inner_val.loc[mask_inner_val, feat_cols] * 500.0 + 9999.0
    )
    scaler_after_inner_val_mutation = PerSequenceStandardScaler(cfg).fit(
        _filter_by_base_patients(
            df_mutated_inner_val, inner_fold.train_real_patient_ids
        )
    )
    for seq, (mean_b, scale_b, mask_b) in baseline_stats.items():
        st_m = scaler_after_inner_val_mutation._stats_by_seq[seq]
        np.testing.assert_allclose(st_m.mean, mean_b)
        np.testing.assert_allclose(st_m.scale, scale_b)
        np.testing.assert_array_equal(st_m.zero_variance_mask, mask_b)

    # 3. Mutating outer_val must not alter fitted per-sequence means/stds
    df_mutated_outer_val = df_real.copy()
    mask_outer_val = base_ids_all.isin(outer_fold.val_real_patient_ids)
    df_mutated_outer_val.loc[mask_outer_val, feat_cols] = (
        df_mutated_outer_val.loc[mask_outer_val, feat_cols] * -250.0 - 7777.0
    )
    scaler_after_outer_val_mutation = PerSequenceStandardScaler(cfg).fit(
        _filter_by_base_patients(
            df_mutated_outer_val, inner_fold.train_real_patient_ids
        )
    )
    for seq, (mean_b, scale_b, mask_b) in baseline_stats.items():
        st_m = scaler_after_outer_val_mutation._stats_by_seq[seq]
        np.testing.assert_allclose(st_m.mean, mean_b)
        np.testing.assert_allclose(st_m.scale, scale_b)
        np.testing.assert_array_equal(st_m.zero_variance_mask, mask_b)

    # 4. Verify scaling occurs BEFORE sequence aggregation for both mean and concat
    df_inner_val_orig = _filter_by_base_patients(
        df_real, inner_fold.val_real_patient_ids
    )
    df_inner_val_scaled = baseline_scaler.transform(df_inner_val_orig)
    agg_scaled = aggregate_sequences(df_inner_val_scaled, config=cfg)
    agg_unscaled = aggregate_sequences(df_inner_val_orig, config=cfg)

    expected_dim = 6 if strategy == "mean" else 24
    assert agg_scaled.features.shape == (len(inner_fold.val_real_timepoint_ids), expected_dim)
    assert not np.allclose(agg_scaled.features, agg_unscaled.features)

    # Check exact numerical match against manual per-sequence Z-score followed by aggregation
    first_tp = agg_scaled.patient_ids[0]
    tp_rows = df_inner_val_orig[df_inner_val_orig["patient_id"] == first_tp]
    manual_per_seq = []
    for seq in cfg.preprocessing.sequences.order:
        raw_vec = tp_rows.loc[tp_rows["sequence"] == seq, feat_cols].to_numpy(
            dtype=np.float32
        )[0]
        mean_b, scale_b, _ = baseline_stats[seq]
        manual_per_seq.append((raw_vec - mean_b) / scale_b)

    expected_first = (
        np.mean(np.stack(manual_per_seq, axis=0), axis=0)
        if strategy == "mean"
        else np.concatenate(manual_per_seq, axis=0)
    )
    np.testing.assert_allclose(agg_scaled.features[0], expected_first, rtol=1e-5, atol=1e-5)