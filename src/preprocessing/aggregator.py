from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from types import SimpleNamespace

from src.data.loader import extract_real_patient_id


"""
    in our dataset we have 4 MRI sequences (T1, T1gd, T2, Flair) for each patient_id, 
    and we want to aggregate them into a single representation for each patient_id.

    STRUCTURE:
    the file contains 3 ways to aggregate the sequences:
    1. mean: compute the mean of the features across the 4 sequences for each patient_id
    result: 1 embedding of 768 features

    2. concat: concatenate the features of the 4 sequences for each patient_id, 
    resulting in a feature vector 4 times larger than the original
    result: 1 embedding of 3072 features

    3. per_sequence: keep the features of each sequence separate, resulting in a feature vector 4 times 
    larger than the original, but with the sequence information preserved
    result: 4 embeddings of 768 features each
"""

AggregationStrategy = Literal["mean", "concat", "per_sequence"]


@dataclass(frozen=True)
class AggregatedDataset:
    """This class collects the results of the sequence aggregation.

    Attributes:
        features: A 2D NumPy array of shape (n_samples, n_features) containing aggregated features
        patient_ids: A list of unique observation identifiers, so also of single timepoints
        base_patient_ids: A list of unique real patient identifiers, no timepoints suffixes
        sequences: name of the sequence for each row in features, only if strategy is "per_sequence", otherwise None
        sequence_order: order of MRI
        strategy: aggregation strategy used to produce the features
        os_1y: Optional 1-year overall survival labels for each observation, if available
        pfs_1y: Optional 1-year progression-free survival labels for each observation, if available
    """

    features: np.ndarray
    patient_ids: list[str]
    base_patient_ids: list[str]
    sequences: list[str] | None
    sequence_order: tuple[str, ...]
    strategy: AggregationStrategy
    os_1y: np.ndarray | None = None
    pfs_1y: np.ndarray | None = None


def _get_feature_columns(df: pd.DataFrame, feature_prefix: str = "f") -> list[str]:
    """Return feature columns matching `feature_prefix` sorted numerically if possible."""
    cols = [c for c in df.columns if c.startswith(feature_prefix)]
    if not cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'.")

    def _sort_key(col: str) -> tuple[int, str]:
        suffix = col[len(feature_prefix) :]
        return (0, f"{int(suffix):08d}") if suffix.isdigit() else (1, col)

    return sorted(cols, key=_sort_key)


def aggregate_sequences(
    df: pd.DataFrame,
    config: SimpleNamespace,
) -> AggregatedDataset:
    """Aggregate the 4 MRI sequences per timepoint/observation (`patient_id`).

    Methodological note — Option B (Timepoint-level observations):
    - Selected approach: Each distinct `patient_id` (which represents a specific
      timepoint such as `PCNSL041_baseline` or `PCNSL041_pre-bx`, or a synthetic
      sample ID) is aggregated across its four MRI sequences (`T1`, `T1gd`,
      `T2`, `Flair`) into its own representation without averaging across
      different longitudinal timepoints of the same patient. Both the
      timepoint-level `patient_id` and the unique real `base_patient_id` are
      preserved in the output metadata.
    - Advantages & limitations: Preserves timepoint-specific information (e.g.,
      `baseline` vs `pre-bx`) for training while keeping all timepoints of a
      patient within the same CV fold and enabling patient-level prediction
      aggregation at evaluation time. Its main limitation is that patients with
      two timepoints contribute more training observations and therefore
      potentially have greater weight on the training loss; their rows are also
      correlated and must never be treated as independent patients.
    - Alternative considered: Option A averages embeddings across a patient's
      available timepoints before modeling, giving every unique patient equal
      sample count/weight at the cost of blending or losing timepoint-specific
      signal.
    """
    strategy: AggregationStrategy = config.preprocessing.aggregation.strategy  # type: ignore[assignment]
    feature_prefix = str(config.preprocessing.features.prefix)
    strict_sequences = bool(config.preprocessing.aggregation.strict_sequences)
    allow_duplicates = bool(config.preprocessing.aggregation.allow_duplicates)
    sequence_order: tuple[str, ...] = tuple(
        str(s) for s in config.preprocessing.sequences.order
    )
    if strategy not in ("mean", "concat", "per_sequence"):
        raise ValueError(
            f"Unsupported aggregation strategy '{strategy}'. "
            "Expected one of: 'mean', 'concat', 'per_sequence'."
        )

    required_cols = {"patient_id", "sequence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feat_cols = _get_feature_columns(df, feature_prefix=feature_prefix)
    n_features = len(feat_cols)
    expected_set = set(sequence_order)

    has_os = "OS_1y" in df.columns
    has_pfs = "PFS_1y" in df.columns
    has_labels = has_os and has_pfs

    out_features_list: list[np.ndarray] = []
    out_patient_ids: list[str] = []
    out_base_ids: list[str] = []
    out_sequences: list[str] = []
    out_os: list[float] = []
    out_pfs: list[float] = []

    # iterate over each patient_id es. (PCNSL041_baseline)
    for pid, grp in df.groupby("patient_id", sort=False):
        # pid_str is es. "PCNSL041_baseline", base_id is es. "PCNSL041"
        pid_str = str(pid)
        base_id = extract_real_patient_id(pid_str)
        # sequences present in the current group, as a Series of strings (T1, T1gd, T2, Flair)
        seq_series = grp["sequence"].astype(str)

        if not allow_duplicates and seq_series.duplicated().any():
            # control if we have 2 rows with the same sequence for the same patient_id, which is not allowed
            dups = seq_series[seq_series.duplicated()].unique().tolist()
            raise ValueError(
                f"Duplicate sequences found for observation '{pid_str}': {dups}"
            )

        actual_set = set(seq_series.tolist())
        # check if the sequences present in the current group match the expected set of sequences
        if strict_sequences and actual_set != expected_set:
            raise ValueError(
                f"Observation '{pid_str}' has sequences {sorted(actual_set)}, "
                f"expected exact set {sorted(expected_set)}."
            )

        # create a list of present sequences, e.g., ["T1", "T1gd", "T2", "Flair"] if strict_sequences is True, or a subset if False
        present_seqs = [s for s in sequence_order if s in actual_set]
        if not present_seqs:
            raise ValueError(
                f"Observation '{pid_str}' contains none of the expected sequences "
                f"{sequence_order}."
            )

        if has_labels:
            os_unique = grp["OS_1y"].unique()
            pfs_unique = grp["PFS_1y"].unique()
            # check if the labels are consistent across sequences for the same patient_id
            if len(os_unique) != 1 or len(pfs_unique) != 1:
                raise ValueError(
                    f"Inconsistent labels across sequences for observation '{pid_str}'."
                )
            # save them as floats for later use in the output dataset
            obs_os = float(os_unique[0])
            obs_pfs = float(pfs_unique[0])

        # a vec for each sequence, key is the sequence name, value is his embedding (1D array of 768 features)
        seq_to_vec: dict[str, np.ndarray] = {}
        for s in present_seqs:
            # for each sequence, get the feature values for the current sequence from the group DataFrame
            vecs = grp.loc[seq_series == s, feat_cols].to_numpy(dtype=np.float32)
            # if allow_duplicates is True, we compute the mean of the embeddings for that sequence,
            # otherwise we take the first one (since there should be only one row per sequence)
            seq_to_vec[s] = vecs.mean(axis=0) if allow_duplicates else vecs[0]

        if strategy == "mean":
            """
                advantages:
                - reduce dimensionality
                - maintain same dimension of the embedding
                - reduce number of parameters in the model
                disadvantages:
                - mixed information of sequences
                - we cannot distinguish how much each sequence contributes 
                - differences between sequences could be attenuated or lost in the mean
            """
            stacked = np.stack([seq_to_vec[s] for s in present_seqs], axis=0)
            agg_vec = stacked.mean(axis=0).astype(np.float32, copy=False)
            out_features_list.append(agg_vec)
            out_patient_ids.append(pid_str)
            out_base_ids.append(base_id)
            if has_labels:
                out_os.append(obs_os)
                out_pfs.append(obs_pfs)

        elif strategy == "concat":
            """
                advantages:
                - maintain all information of sequences
                - model can learn different relations of sequences
                disadvantages:
                - dimensionality increases 4 times
                - model need to learn to manage more features
                - overfitting risk increases
            """
            if len(present_seqs) != len(sequence_order):
                raise ValueError(
                    f"Concatenation requires all sequences {sequence_order} for "
                    f"observation '{pid_str}', found {present_seqs}."
                )
            concat_vec = np.concatenate(
                [seq_to_vec[s] for s in sequence_order], axis=0
            ).astype(np.float32, copy=False)
            out_features_list.append(concat_vec)
            out_patient_ids.append(pid_str)
            out_base_ids.append(base_id)
            if has_labels:
                out_os.append(obs_os)
                out_pfs.append(obs_pfs)

        else:  # strategy == "per_sequence"
            """
                advantages:
                - maintain all information of sequences
                - model can learn different relations of sequences
                disadvantages:
                - dimensionality increases 4 times
                - model need to learn to manage more features
                - overfitting risk increases
            """
            for s in present_seqs:
                out_features_list.append(seq_to_vec[s].astype(np.float32, copy=False))
                out_patient_ids.append(pid_str)
                out_base_ids.append(base_id)
                out_sequences.append(s)
                if has_labels:
                    out_os.append(obs_os)
                    out_pfs.append(obs_pfs)

    if out_features_list:
        # stack the list of feature vectors into a 2D NumPy array of shape (n_samples, n_features)
        features = np.vstack(out_features_list).astype(np.float32, copy=False)
    else:
        expected_dim = n_features * len(sequence_order) if strategy == "concat" else n_features
        features = np.empty((0, expected_dim), dtype=np.float32)

    return AggregatedDataset(
        features=features,
        patient_ids=out_patient_ids,
        base_patient_ids=out_base_ids,
        sequences=out_sequences if strategy == "per_sequence" else None,
        sequence_order=sequence_order,
        strategy=strategy,
        os_1y=np.asarray(out_os, dtype=np.float32) if has_labels else None,
        pfs_1y=np.asarray(out_pfs, dtype=np.float32) if has_labels else None,
    )