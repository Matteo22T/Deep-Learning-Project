from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.loader import extract_real_patient_id, get_synthetic_parents
from src.data.splitter import (
    build_clinical_strata,
    generate_patient_folds,
    get_eligible_synthetic_ids,
    select_nested_synthetic_quotas,
)
from src.preprocessing.aggregator import aggregate_sequences
from src.preprocessing.scaler import PerSequenceStandardScaler
from src.utils.config import ConfigLoader


def _make_config(
    *,
    random_seed: int = 42,
    n_splits: int = 5,
    synthetic_ratios: list[float] | None = None,
    fold_seed_stride: int = 1,
    quota_integer_tol: float = 1e-9,
    feature_prefix: str = "f",
    sequence_order: list[str] | None = None,
    epsilon: float = 1e-8,
    strategy: str = "mean",
    strict_sequences: bool = True,
    allow_duplicates: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        experiment=SimpleNamespace(name="test_exp", random_seed=random_seed),
        splitting=SimpleNamespace(
            n_splits=n_splits,
            synthetic_ratios=(
                [0.0, 0.25, 0.5, 1.0, 2.0]
                if synthetic_ratios is None
                else synthetic_ratios
            ),
            fold_seed_stride=fold_seed_stride,
            quota_integer_tol=quota_integer_tol,
        ),
        preprocessing=SimpleNamespace(
            features=SimpleNamespace(prefix=feature_prefix),
            sequences=SimpleNamespace(
                order=["T1", "T1gd", "T2", "Flair"]
                if sequence_order is None
                else sequence_order
            ),
            scaling=SimpleNamespace(epsilon=epsilon),
            aggregation=SimpleNamespace(
                strategy=strategy,
                strict_sequences=strict_sequences,
                allow_duplicates=allow_duplicates,
            ),
        ),
    )


def _make_toy_real_df(
    n_patients: int = 50,
    n_features: int = 8,
    sequence_order: list[str] | None = None,
    feature_prefix: str = "f",
) -> pd.DataFrame:
    """Create a synthetic toy internal cohort with longitudinal scans on first 13 patients."""
    seqs = ["T1", "T1gd", "T2", "Flair"] if sequence_order is None else sequence_order
    rows: list[dict[str, object]] = []
    for idx in range(n_patients):
        base_id = f"PCNSL{idx:03d}"
        if idx < 20:
            os_val, pfs_val = 0, 0
        elif idx < 22:
            os_val, pfs_val = 1, 0
        else:
            os_val, pfs_val = 1, 1

        timepoints = ["baseline", "pre-bx"] if idx < 13 else ["baseline"]
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
                    rec[f"{feature_prefix}{f_idx}"] = float(
                        idx * 10 + s_idx + 0.1 * f_idx
                    )
                rows.append(rec)
    return pd.DataFrame(rows)


def _make_toy_synth_df(
    n_synth: int = 225,
    n_features: int = 8,
    sequence_order: list[str] | None = None,
    feature_prefix: str = "f",
) -> pd.DataFrame:
    """Create `n_synth` distinct toy synthetic IDs `A<id>_baseline__B<id>_baseline`."""
    seqs = ["T1", "T1gd", "T2", "Flair"] if sequence_order is None else sequence_order
    rows: list[dict[str, object]] = []
    for s_i in range(n_synth):
        host_num = s_i % 50
        donor_num = (host_num + 1 + (s_i // 50)) % 50
        if donor_num == host_num:
            donor_num = (donor_num + 1) % 50

        if host_num < 20 and donor_num < 20:
            os_val, pfs_val = 0, 0
        else:
            os_val, pfs_val = 1, 1

        synth_id = f"A{host_num:03d}_baseline__B{donor_num:03d}_baseline"
        for s_idx, seq in enumerate(seqs):
            rec: dict[str, object] = {
                "patient_id": synth_id,
                "sequence": seq,
                "OS_1y": os_val,
                "PFS_1y": pfs_val,
            }
            for f_idx in range(n_features):
                rec[f"{feature_prefix}{f_idx}"] = float(
                    1000 + s_i + s_idx + 0.01 * f_idx
                )
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Config loading and validation tests
# ---------------------------------------------------------------------------


def test_config_loader_yaml_and_nested_access() -> None:
    cfg = ConfigLoader("configs/config_preprocessing.yaml").get_config()
    assert cfg.experiment.random_seed == 42
    assert cfg.splitting.n_splits == 5
    assert cfg.splitting.synthetic_ratios == [0.0, 0.25, 0.5, 1.0, 2.0]
    assert cfg.splitting.fold_seed_stride == 1
    assert cfg.preprocessing.features.prefix == "f"
    assert cfg.preprocessing.sequences.order == ["T1", "T1gd", "T2", "Flair"]
    assert cfg.preprocessing.scaling.epsilon == pytest.approx(1e-8)
    assert cfg.preprocessing.aggregation.strategy == "mean"
    assert cfg.preprocessing.aggregation.strict_sequences is True
    assert cfg.preprocessing.aggregation.allow_duplicates is False


def test_config_validation_rejects_invalid_values(tmp_path: Path) -> None:
    bad_dup_ratio = tmp_path / "dup_ratio.yaml"
    bad_dup_ratio.write_text(
        """
experiment:
  random_seed: 42
splitting:
  n_splits: 5
  synthetic_ratios: [0.0, 0.25, 0.25]
preprocessing:
  features:
    prefix: f
  sequences:
    order: [T1, T1gd, T2, Flair]
  scaling:
    epsilon: 1.0e-8
  aggregation:
    strategy: mean
    strict_sequences: true
    allow_duplicates: false
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate synthetic ratio"):
        ConfigLoader(bad_dup_ratio)

    bad_eps = tmp_path / "bad_eps.yaml"
    bad_eps.write_text(
        """
experiment:
  random_seed: 42
splitting:
  n_splits: 5
  synthetic_ratios: [0.0, 0.25]
preprocessing:
  features:
    prefix: f
  sequences:
    order: [T1, T1gd, T2, Flair]
  scaling:
    epsilon: 0.0
  aggregation:
    strategy: mean
    strict_sequences: true
    allow_duplicates: false
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scaling.epsilon must be a positive number"):
        ConfigLoader(bad_eps)

    missing_section = tmp_path / "missing.yaml"
    missing_section.write_text(
        """
experiment:
  random_seed: 42
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing or invalid 'splitting' section"):
        ConfigLoader(missing_section)


# ---------------------------------------------------------------------------
# Splitter tests & config behavior changes
# ---------------------------------------------------------------------------


def test_splitter_no_overlap_and_longitudinal_grouped() -> None:
    cfg = _make_config(random_seed=42, n_splits=5)
    df_real = _make_toy_real_df(n_patients=50)
    df_synth = _make_toy_synth_df(n_synth=225)
    folds = generate_patient_folds(df_real, config=cfg, df_synth=df_synth)

    assert len(folds) == 5
    all_val_patients: list[str] = []

    for fold in folds:
        train_set = set(fold.train_real_patient_ids)
        val_set = set(fold.val_real_patient_ids)

        assert len(train_set) == 40
        assert len(val_set) == 10
        assert train_set.isdisjoint(val_set)

        for tp_id in fold.train_real_timepoint_ids:
            assert extract_real_patient_id(tp_id) in train_set
        for tp_id in fold.val_real_timepoint_ids:
            assert extract_real_patient_id(tp_id) in val_set

        assert not any(sid.startswith("A") for sid in fold.val_real_timepoint_ids)
        all_val_patients.extend(fold.val_real_patient_ids)

    assert sorted(all_val_patients) == [f"PCNSL{i:03d}" for i in range(50)]


def test_splitter_parent_leakage_and_nested_quotas() -> None:
    cfg = _make_config(
        random_seed=123,
        n_splits=5,
        synthetic_ratios=[0.0, 0.25, 0.5, 1.0, 2.0],
    )
    df_real = _make_toy_real_df(n_patients=50)
    df_synth = _make_toy_synth_df(n_synth=225)
    folds = generate_patient_folds(df_real, config=cfg, df_synth=df_synth)

    expected_counts = {0.0: 0, 0.25: 10, 0.5: 20, 1.0: 40, 2.0: 80}

    for fold in folds:
        val_set = set(fold.val_real_patient_ids)
        for synth_id in fold.eligible_synthetic_ids:
            host_id, donor_id = get_synthetic_parents(synth_id)
            assert host_id not in val_set
            assert donor_id not in val_set

        for ratio, expected_k in expected_counts.items():
            chosen = fold.sampled_synthetic_by_ratio[ratio]
            assert len(chosen) == expected_k
            assert len(set(chosen)) == expected_k
            for sid in chosen:
                host_id, donor_id = get_synthetic_parents(sid)
                assert host_id not in val_set
                assert donor_id not in val_set

        assert set(fold.sampled_synthetic_by_ratio[0.25]).issubset(
            set(fold.sampled_synthetic_by_ratio[0.5])
        )
        assert set(fold.sampled_synthetic_by_ratio[0.5]).issubset(
            set(fold.sampled_synthetic_by_ratio[1.0])
        )
        assert set(fold.sampled_synthetic_by_ratio[1.0]).issubset(
            set(fold.sampled_synthetic_by_ratio[2.0])
        )



def test_splitter_config_changes_folds_ratios_and_determinism() -> None:
    df_real = _make_toy_real_df(n_patients=50)
    df_synth = _make_toy_synth_df(n_synth=225)

    cfg_a = _make_config(
        random_seed=7,
        n_splits=5,
        synthetic_ratios=[0.0, 0.25, 0.5],
    )
    cfg_b = _make_config(
        random_seed=7,
        n_splits=5,
        synthetic_ratios=[0.0, 0.25, 0.5],
    )
    cfg_changed = _make_config(
        random_seed=99,
        n_splits=3,
        synthetic_ratios=[0.0, 0.5, 1.0],
    )

    # Same configuration must produce identical folds and synthetic samples.
    folds_a = generate_patient_folds(df_real, config=cfg_a, df_synth=df_synth)
    folds_b = generate_patient_folds(df_real, config=cfg_b, df_synth=df_synth)

    assert len(folds_a) == len(folds_b)

    for fa, fb in zip(folds_a, folds_b):
        assert fa.train_real_patient_ids == fb.train_real_patient_ids
        assert fa.val_real_patient_ids == fb.val_real_patient_ids
        assert fa.sampled_synthetic_by_ratio == fb.sampled_synthetic_by_ratio

    # Fractional synthetic quotas must raise an error.
    # With 50 patients and 3 folds, a training fold has 33 or 34 patients:
    # 50% of 33 = 16.5, which is not a valid integer quota.
    with pytest.raises(ValueError, match="not an integer count"):
        generate_patient_folds(df_real, config=cfg_changed, df_synth=df_synth)

    # A valid 3-fold configuration must produce three folds.
    cfg_valid_3fold = _make_config(
        random_seed=99,
        n_splits=3,
        synthetic_ratios=[0.0, 1.0],
    )
    folds_3 = generate_patient_folds(
        df_real,
        config=cfg_valid_3fold,
        df_synth=df_synth,
    )

    assert len(folds_3) == 3
    assert set(folds_3[0].sampled_synthetic_by_ratio.keys()) == {0.0, 1.0}
    assert len(folds_3[0].sampled_synthetic_by_ratio[1.0]) == len(
        folds_3[0].train_real_patient_ids
    )

    # Eligibility filtering must exclude synthetic samples with a validation parent.
    eligible = get_eligible_synthetic_ids(
        df_synth,
        val_patient_ids={"PCNSL001"},
    )
    assert len(eligible) > 0

    # Insufficient eligible synthetic samples must raise an error.
    with pytest.raises(
        ValueError,
        match="Insufficient leakage-free synthetic samples",
    ):
        select_nested_synthetic_quotas(
            eligible_synth_ids=["A010__B011", "A012__B013"],
            n_real_train_patients=40,
            ratios=[0.0, 0.25, 0.5, 1.0, 2.0],
            random_state=42,
            quota_integer_tol=1e-9,
        )



def test_splitter_rare_stratum_reported_and_invalid_pair_rejected() -> None:
    cfg = _make_config(random_seed=42, n_splits=5)
    df_real = _make_toy_real_df(n_patients=50)
    strata = build_clinical_strata(df_real)
    assert int((strata == 1).sum()) == 2

    folds = generate_patient_folds(df_real, config=cfg, df_synth=None)
    rare_per_fold = [f.val_strata_counts["OS=1, PFS=0"] for f in folds]
    assert sum(rare_per_fold) == 2
    assert any(c == 0 for c in rare_per_fold)

    bad_df = df_real.copy()
    mask_p0 = bad_df["patient_id"].str.startswith("PCNSL000")
    bad_df.loc[mask_p0, "OS_1y"] = 0
    bad_df.loc[mask_p0, "PFS_1y"] = 1
    with pytest.raises(ValueError, match="Clinically invalid label pair"):
        build_clinical_strata(bad_df)


# ---------------------------------------------------------------------------
# Aggregator tests & config behavior changes
# ---------------------------------------------------------------------------


def test_aggregator_strategies_sequence_order_and_timepoint_preservation() -> None:
    df_real = _make_toy_real_df(n_patients=2, n_features=4)

    cfg_mean = _make_config(strategy="mean")
    mean_res = aggregate_sequences(df_real, config=cfg_mean)
    assert mean_res.features.shape == (4, 4)
    assert mean_res.features.dtype == np.float32
    assert mean_res.patient_ids == [
        "PCNSL000_baseline",
        "PCNSL000_pre-bx",
        "PCNSL001_baseline",
        "PCNSL001_pre-bx",
    ]
    assert mean_res.base_patient_ids == ["PCNSL000", "PCNSL000", "PCNSL001", "PCNSL001"]

    cfg_concat = _make_config(strategy="concat")
    concat_res = aggregate_sequences(df_real, config=cfg_concat)
    assert concat_res.features.shape == (4, 16)
    assert concat_res.features.dtype == np.float32

    cfg_per_seq = _make_config(strategy="per_sequence")
    per_seq_res = aggregate_sequences(df_real, config=cfg_per_seq)
    assert per_seq_res.features.shape == (16, 4)
    assert per_seq_res.sequences == ["T1", "T1gd", "T2", "Flair"] * 4

    # Changing configured sequence order changes concatenation block layout
    custom_order = ["Flair", "T2", "T1gd", "T1"]
    df_custom = _make_toy_real_df(n_patients=1, n_features=2, sequence_order=custom_order)
    cfg_custom_concat = _make_config(strategy="concat", sequence_order=custom_order)
    custom_res = aggregate_sequences(df_custom, config=cfg_custom_concat)
    assert custom_res.sequence_order == tuple(custom_order)
    assert custom_res.features.shape == (2, 8)
    # First 2 features come from "Flair" (s_idx=0 -> base value 0.0 + 0.1*f_idx)
    np.testing.assert_allclose(custom_res.features[0, :2], np.array([0.0, 0.1], dtype=np.float32))


def test_aggregator_missing_and_duplicate_sequences() -> None:
    df_real = _make_toy_real_df(n_patients=1, n_features=4)
    missing_seq_df = df_real[df_real["sequence"] != "Flair"].copy()
    cfg_strict = _make_config(strategy="mean", strict_sequences=True)
    with pytest.raises(ValueError, match="expected exact set"):
        aggregate_sequences(missing_seq_df, config=cfg_strict)

    dup_df = pd.concat([df_real, df_real.iloc[[0]]], ignore_index=True)
    cfg_no_dups = _make_config(strategy="mean", allow_duplicates=False)
    with pytest.raises(ValueError, match="Duplicate sequences"):
        aggregate_sequences(dup_df, config=cfg_no_dups)


# ---------------------------------------------------------------------------
# Scaler tests, epsilon behavior, and unknown sequence validation
# ---------------------------------------------------------------------------


def test_scaler_train_only_fit_epsilon_and_zero_variance_safety() -> None:
    df_train = _make_toy_real_df(n_patients=10, n_features=4)
    df_val = _make_toy_real_df(n_patients=3, n_features=4)

    # Set f3 to constant 5.0 in train, and f2 to tiny spread (1e-5)
    df_train["f3"] = 5.0
    df_val["f3"] = 999.0
    df_train["f2"] = np.linspace(0.0, 1e-5, len(df_train))

    cfg_small_eps = _make_config(epsilon=1e-8)
    cfg_large_eps = _make_config(epsilon=1.0)

    scaler_small = PerSequenceStandardScaler(cfg_small_eps).fit(df_train)
    scaler_large = PerSequenceStandardScaler(cfg_large_eps).fit(df_train)

    # Larger epsilon treats small-std feature f2 as zero-variance -> scaled values become 0.0
    scaled_val_small = scaler_small.transform(df_val)
    scaled_val_large = scaler_large.transform(df_val)
    assert not np.allclose(scaled_val_small["f2"].to_numpy(), 0.0)
    np.testing.assert_allclose(scaled_val_large["f2"].to_numpy(), 0.0)

    stats_before = {
        seq: (st.mean.copy(), st.scale.copy(), st.zero_variance_mask.copy())
        for seq, st in scaler_small._stats_by_seq.items()
    }

    scaled_val_first = scaler_small.transform(df_val)
    scaled_val_second = scaler_small.transform(df_val)
    pd.testing.assert_frame_equal(scaled_val_first, scaled_val_second)

    # Mutating validation data and transforming it must not alter fitted scaler parameters
    df_val_perturbed = df_val.copy()
    df_val_perturbed["f0"] = df_val_perturbed["f0"] * 100.0 + 500.0
    _ = scaler_small.transform(df_val_perturbed)

    for seq, (mean_b, scale_b, mask_b) in stats_before.items():
        st_a = scaler_small._stats_by_seq[seq]
        np.testing.assert_allclose(st_a.mean, mean_b)
        np.testing.assert_allclose(st_a.scale, scale_b)
        np.testing.assert_array_equal(st_a.zero_variance_mask, mask_b)

    feat_cols = ["f0", "f1", "f2", "f3"]
    val_arr = scaled_val_first[feat_cols].to_numpy(dtype=np.float32)
    assert val_arr.dtype == np.float32
    assert np.isfinite(val_arr).all()
    np.testing.assert_allclose(scaled_val_first["f3"].to_numpy(), 0.0)


def test_scaler_rejects_unknown_sequence_values() -> None:
    cfg = _make_config(sequence_order=["T1", "T1gd", "T2", "Flair"])
    df_train = _make_toy_real_df(n_patients=5, n_features=4)
    scaler = PerSequenceStandardScaler(cfg).fit(df_train)

    df_bad_seq = df_train.copy()
    df_bad_seq.loc[0, "sequence"] = "UNKNOWN_SEQ"

    with pytest.raises(ValueError, match="Unknown sequence values"):
        scaler.transform(df_bad_seq)