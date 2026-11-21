from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from types import SimpleNamespace


"""
    in the project we have 4 MRI sequences (T1, T1gd, T2, Flair) for each patient_id, and we want to standardize the features 
    per sequence before aggregating them into a single representation for each patient_id.

    we use z-score standardization, which is defined as (x - mean) / std, 
    IMPORTANT, mean and std are computed on the training of each fold, not the all dataset
    and use the same value on the validation and test set, to avoid data leakage.

"""

# used to store the mean and std of each sequence, computed on the training set of each fold
@dataclass
class _SequenceStats:
    """Per-sequence standardization statistics computed strictly on training data."""

    mean: np.ndarray
    scale: np.ndarray
    # which features have zero variance in the training set, to avoid division by zero
    zero_variance_mask: np.ndarray


class PerSequenceStandardScaler:
    """Fold-specific per-sequence Z-score scaler applied before aggregation.

    Methodological note — Per-sequence Z-score before sequence aggregation:
    - Selected approach: Fit a separate mean and standard deviation vector for
      each MRI sequence (`T1`, `T1gd`, `T2`, `Flair`) exclusively on the
      training fold data, transform validation/test data using those fitted
      parameters without refitting, and perform sequence aggregation (`mean` or
      `concat`) afterwards. Features with zero (or near-zero) variance in the
      training fold are safely mapped to `0.0` instead of producing `NaN` or
      `inf`.
    - Advantages & limitations: Each MRI sequence has its own feature
      distribution; scaling per sequence before fusion prevents numerical scale
      differences from disproportionately driving aggregation and enforces
      fold-safe preprocessing. Its limitation is that four separate sets of
      normalization parameters must be stored and applied consistently.
    - Alternative considered: Aggregating the four sequences first and applying
      Z-score afterwards would normalize the already-fused representation rather
      than each modality independently, potentially masking sequence-specific
      distribution differences.
    """

    def __init__(
        self,
        config: SimpleNamespace,
    ) -> None:
        """
            Parameters:
            config: The configuration object containing the settings for the scaler.
        """
        self.sequences: tuple[str, ...] = tuple(
            str(s) for s in config.preprocessing.sequences.order
        )
        self.feature_prefix: str = str(config.preprocessing.features.prefix)
        self.eps: float = float(config.preprocessing.scaling.epsilon)
        if self.eps <= 0.0:
            raise ValueError("config.preprocessing.scaling.epsilon must be positive.")
        if not self.sequences:
            raise ValueError("config.preprocessing.sequences.order must be non-empty.")
        # store the mean and std of each sequence, computed on the training set of each fold 
        self._stats_by_seq: dict[str, _SequenceStats] = {}
        # store names of the feature columns, resolved from the training DataFrame
        self._feature_cols: list[str] = []

    def _resolve_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """
            indentify the feature columns in the DataFrame based on the specified prefix 
            and sort them numerically if possible.
        """
        cols = [c for c in df.columns if c.startswith(self.feature_prefix)]
        if not cols:
            raise ValueError(
                f"No feature columns found with prefix '{self.feature_prefix}'."
            )

        def _sort_key(col: str) -> tuple[int, str]:
            suffix = col[len(self.feature_prefix) :]
            return (0, f"{int(suffix):08d}") if suffix.isdigit() else (1, col)

        return sorted(cols, key=_sort_key)

    def _validate_known_sequences(self, df: pd.DataFrame, context: str) -> pd.Series:
        """
            check if the sequences in the DataFrame are a subset of the ones in the config
        """
        seq_col = df["sequence"].astype(str)
        allowed_set = set(self.sequences)
        unknown_seqs = sorted(set(seq_col.tolist()) - allowed_set)
        if unknown_seqs:
            raise ValueError(
                f"Unknown sequence values in {context}: {unknown_seqs}. "
                f"Expected subset of configured sequences: {list(self.sequences)}."
            )
        return seq_col

    def fit(self, df_train: pd.DataFrame) -> PerSequenceStandardScaler:
        """Fit per-sequence mean and std strictly on training DataFrame rows."""
        if "sequence" not in df_train.columns:
            raise ValueError("Training DataFrame must contain a 'sequence' column.")

        seq_col = self._validate_known_sequences(df_train, context="training data")

        # resolve feature columns and store them for later use and clear any previously stored statistics
        feat_cols = self._resolve_feature_cols(df_train)
        self._feature_cols = feat_cols
        self._stats_by_seq.clear()

        # iterate over each sequence and compute the mean and std for the features of that sequence
        for seq in self.sequences:
            # get the feature values for the current sequence from the training DataFrame
            # if seq="T1", then vals will be a 2D array of shape (n_samples, n_features) 
            # containing the feature values for all rows where sequence=="T1"
            vals = df_train.loc[seq_col == seq, feat_cols].to_numpy(dtype=np.float64)
            if vals.shape[0] == 0:
                raise ValueError(
                    f"Cannot fit scaler for sequence '{seq}': zero training rows found."
                )

            # axis = 0 means we compute the mean and std for each feature column across all rows of that sequence
            # maintainig columns separetely
            # we obtain 1 value for each feature
            mean = np.mean(vals, axis=0, dtype=np.float64)
            # ddof = 0 means we use the population standard deviation formula, which divides by N instead of N-1
            std = np.std(vals, axis=0, ddof=0, dtype=np.float64)
            # identify features with zero variance (std <= eps) to avoid division by zero during scaling
            # apply 1.0 scale for those
            zero_var_mask = (~np.isfinite(std)) | (std <= self.eps)
            safe_scale = np.where(zero_var_mask, 1.0, std)

            # store statistics 
            self._stats_by_seq[seq] = _SequenceStats(
                mean=mean.astype(np.float32),
                scale=safe_scale.astype(np.float32),
                zero_variance_mask=zero_var_mask,
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted training statistics without refitting."""
        if not self._stats_by_seq:
            raise RuntimeError("PerSequenceStandardScaler must be fitted before transform.")
        if "sequence" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'sequence' column.")

        seq_col = self._validate_known_sequences(df, context="input data")
        # verify that the feature columns in the input DataFrame match those used during fitting
        cols = self._resolve_feature_cols(df)
        if cols != self._feature_cols:
            raise ValueError(
                f"Feature columns mismatch: expected {len(self._feature_cols)} columns, "
                f"got {len(cols)}."
            )

        out = df.copy()

        # iterate over each sequence 
        for seq, stats in self._stats_by_seq.items():
            # mask is a boolean array indicating which rows in the DataFrame correspond to the current sequence
            mask = seq_col == seq
            if not mask.any():
                continue

            # get the original values for the features of the current sequence,
            raw = out.loc[mask, cols].to_numpy(dtype=np.float32)
            # apply z-score standardization using the mean and std computed from the training set, 
            scaled = (raw - stats.mean) / stats.scale
            # and handle zero variance features by setting them to 0.0
            scaled[:, stats.zero_variance_mask] = 0.0
            # replace any NaN or infinite values with 0.0 to ensure numerical stability
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )
            # update the DataFrame with the scaled values for the current sequence
            out.loc[mask, cols] = scaled

        return out

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return the transformed training DataFrame."""
        return self.fit(df_train).transform(df_train)