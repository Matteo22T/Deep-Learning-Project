from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from types import SimpleNamespace

from src.data.loader import extract_real_patient_id, get_synthetic_parents

logger = logging.getLogger(__name__)

# Constants for synthetic patient augmentation ratios and clinical state labels
# the percentage is calculated with respect to the number of real patients in the training fold

# States of patients
CLINICAL_STATE_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "OS=0, PFS=0",
    (1, 0): "OS=1, PFS=0",
    (1, 1): "OS=1, PFS=1",
}

# Mapping from integer stratum index to human-readable clinical state label
STRATUM_TO_STATE_LABEL: dict[int, str] = {
    0: CLINICAL_STATE_LABELS[(0, 0)],
    1: CLINICAL_STATE_LABELS[(1, 0)],
    2: CLINICAL_STATE_LABELS[(1, 1)],
}


"""
    In this module we are going to split the patients into training and validation sets, 
    taking care of the fact that we have synthetic patients that are generated from real patients.
    We want to make sure that we don't have any leakage between the training and validation sets, 
    so we will split the real patients first and then select the synthetic patients that are eligible for each fold.

    in particular:

    -SPLIT REAL PATIENTS: patients are split in 5 folds, mantaining if possible the proportion of 
    the 3 classes (OS=0, PFS=0), (OS=1, PFS=0), (OS=1, PFS=1) in each fold.

    -LEAKAGE PREVENTION: all timepoints of the same patient are assigned to the same fold,
    and a synthetic patient cannot be used if it is generated from a real patient that is in the validation set.

    -SYNTHETIC PATIENTS SELECTION: we prepare 5 levels of agumentation from 0% to 200% in a controlled and reproducible way


    GENERAL STRUCTURE:
    1. Prepare the real patients and their clinical strata
    build_clinical_strata()

    2. Split the real patients into folds
    generate_patient_folds()

    3. Syntehtic patients selection
    get_eligible_synthetic_ids()

    4. Select synthetic quantities
    select_nested_synthetic_quotas()

    5. Result
    FoldSplit dataclass contains patients, timepoints and selected synthetic patients for each fold
    
"""


@dataclass(frozen=True)
class FoldSplit:
    """Patient-level cross-validation split and leakage-safe synthetic quotas."""

    fold_idx: int
    train_real_patient_ids: tuple[str, ...]
    val_real_patient_ids: tuple[str, ...]
    train_real_timepoint_ids: tuple[str, ...]
    val_real_timepoint_ids: tuple[str, ...]
    # all eligible synthetic IDs that do not leak into the validation fold
    eligible_synthetic_ids: tuple[str, ...]
    # the ones actually selected
    sampled_synthetic_by_ratio: dict[float, tuple[str, ...]]
    # number of patients in each class in the validation fold
    val_strata_counts: dict[str, int]


def build_clinical_strata(df_real: pd.DataFrame) -> pd.Series:
    """Build a 3-state clinical stratum per unique real patient (`base_patient_id`).

    Valid states are `(0, 0)`, `(1, 0)`, and `(1, 1)`. Rows with `(0, 1)` or
    conflicting labels across timepoints/sequences of the same patient raise ValueError.
    """
    required_cols = {"patient_id", "OS_1y", "PFS_1y"}
    missing = required_cols - set(df_real.columns)
    if missing:
        raise ValueError(f"Missing required columns in real dataset: {sorted(missing)}")

    # copy to avoid modifying the original DataFrame
    work = df_real.copy()
    # we take the real id removing the timepoint suffix
    work["base_patient_id"] = work["patient_id"].astype(str).apply(extract_real_patient_id)

    strata_map: dict[str, int] = {}
    # group rows belonging to same patient
    for base_id, grp in work.groupby("base_patient_id", sort=True):
        os_vals = grp["OS_1y"].unique()
        pfs_vals = grp["PFS_1y"].unique()
        # .unique to check that all timepoints of the same patient have the same label
        if len(os_vals) != 1 or len(pfs_vals) != 1:
            raise ValueError(
                f"Inconsistent OS_1y/PFS_1y labels within real patient {base_id}."
            )
        pair = (int(os_vals[0]), int(pfs_vals[0]))
        # assign a stratum index based on the clinical state pair
        # and then add it to the strata_map dictionary with the base_id as key
        if pair == (0, 0):
            strata_map[base_id] = 0
        elif pair == (1, 0):
            strata_map[base_id] = 1
        elif pair == (1, 1):
            strata_map[base_id] = 2
        else:
            raise ValueError(
                f"Clinically invalid label pair {pair} for real patient {base_id}."
            )

    return pd.Series(strata_map, name="clinical_stratum")


def get_eligible_synthetic_ids(
    df_synth: pd.DataFrame,
    val_patient_ids: set[str] | frozenset[str] | Sequence[str],
) -> list[str]:
    """Filter synthetic IDs so neither Host A nor Donor B is in `val_patient_ids`.

    This function handles eligibility filtering only and does not apply quota
    subsampling.
    """
    if "patient_id" not in df_synth.columns:
        raise ValueError("Synthetic DataFrame must contain a 'patient_id' column.")

    # use frozenset to create a non mutable set of validation patient ids
    val_set = frozenset(str(pid) for pid in val_patient_ids)
    # take all unique synthetic patient ids from the synthetic dataframe and sort them for reproducibility
    unique_synth_ids = sorted(df_synth["patient_id"].astype(str).unique())

    eligible: list[str] = []
    for synth_id in unique_synth_ids:
        # get the host and donor ids for the synthetic patient and check if they are in the validation set, 
        # if not add the synthetic id to the eligible list
        host_id, donor_id = get_synthetic_parents(synth_id)
        if host_id not in val_set and donor_id not in val_set:
            eligible.append(synth_id)

    return eligible


def select_nested_synthetic_quotas(
    eligible_synth_ids: Sequence[str],
    n_real_train_patients: int,
    ratios: Sequence[float],
    random_state: int,
    quota_integer_tol: float,
) -> dict[float, tuple[str, ...]]:
    """Select nested synthetic ID subsets without replacement for each ratio.

    Quotas are computed relative to `n_real_train_patients` (distinct real
    patients in the training fold, e.g., 40), counting distinct synthetic IDs.
    Because subsets are nested (`25% <= 50% <= 100% <= 200%`), a single
    deterministic permutation of `eligible_synth_ids` provides prefix slices.
    """
    if n_real_train_patients <= 0:
        raise ValueError("n_real_train_patients must be positive.")

    # remove duplicates while preserving order, and normalize ratios to floats
    unique_eligible = list(dict.fromkeys(str(sid) for sid in eligible_synth_ids))
    normalized_ratios = [float(r) for r in ratios]
    for r in normalized_ratios:
        if r < 0.0:
            raise ValueError(f"Synthetic ratio cannot be negative: {r}")

    quotas_by_ratio: dict[float, int] = {}
    # calculate the number of synthetic patients to select for each ratio, rounding to the nearest integer
    for r in normalized_ratios:
        raw_count = r * n_real_train_patients
        rounded_count = int(round(raw_count))
        if not np.isclose(raw_count, rounded_count, atol=quota_integer_tol):
            raise ValueError(
                f"Ratio {r} * {n_real_train_patients} = {raw_count} is not an integer count."
            )
        quotas_by_ratio[r] = rounded_count

    # check if we have enough eligible synthetic patients to satisfy the maximum quota
    # if not enough, raise an error
    max_needed = max(quotas_by_ratio.values(), default=0)
    if len(unique_eligible) < max_needed:
        raise ValueError(
            f"Insufficient leakage-free synthetic samples: need {max_needed} for "
            f"max quota, but only {len(unique_eligible)} are eligible."
        )

    # random number generator for reproducibility
    rng = np.random.default_rng(random_state)
    if max_needed > 0:
        # create a random permutation of ids
        perm_indices = rng.permutation(len(unique_eligible))[:max_needed]
        nested_pool = [unique_eligible[idx] for idx in perm_indices]
    else:
        nested_pool = []

    selected: dict[float, tuple[str, ...]] = {}
    # for each ratio select the first k ids from the nested pool, where k is the quota for that ratio
    for r in normalized_ratios:
        k = quotas_by_ratio[r]
        selected[r] = tuple(nested_pool[:k])

    return selected


def generate_patient_folds(
    df_real: pd.DataFrame,
    config: SimpleNamespace,
    df_synth: pd.DataFrame | None = None,
) -> list[FoldSplit]:
    """Generate deterministic, leakage-free patient-level CV folds.

    Methodological note — Patient-level stratification, synthetic ancestry, and
    nested sampling:
    - Selected approach: We stratify on the 50 unique real patients using their
      3-state joint outcome (`(0,0)`, `(1,0)`, `(1,1)`), then map each patient's
      timepoints (`baseline`, `pre-bx`) and MRI sequences to the assigned fold.
      Validation sets contain exclusively real patients. For synthetic data, we
      first isolate samples whose Host A and Donor B are both absent from the
      validation patient set, and then sample distinct synthetic IDs without
      replacement in nested prefixes (`0%`, `25%`, `50%`, `100%`, `200%` of
      unique real training patients).
    - Advantages & limitations: Grouping by real patient prevents intra-patient
      longitudinal leakage and sequence leakage, while checking both parents
      blocks cross-patient synthetic leakage (especially given parenchymal
      dominance in global BrainIAC embeddings). Nested sampling reduces variance
      across augmentation levels. However, with only 2 patients in `(OS=1, PFS=0)`
      and 5 folds, that rare stratum cannot appear in every validation fold, and
      the eligible synthetic pool may shrink in folds whose validation patients
      are frequent hosts or donors.
    - Alternatives considered: Stratifying rows/timepoints instead of unique
      patients would leak longitudinal scans and skew class weights; sampling
      quotas independently per ratio without nesting would add unnecessary Monte
      Carlo noise across ablation steps; silently shrinking quotas when `< 80`
      synthetic IDs are eligible would mask insufficient pool capacity.
    """

    n_splits = int(config.splitting.n_splits)
    ratios = tuple(float(r) for r in config.splitting.synthetic_ratios)
    random_state = int(config.experiment.random_seed)
    fold_seed_stride = int(getattr(config.splitting, "fold_seed_stride", 1))
    quota_integer_tol = float(getattr(config.splitting, "quota_integer_tol", 1e-9))


    # build the strata for real patients
    strata_series = build_clinical_strata(df_real).sort_index()
    # real patients ids
    unique_patients = np.array(list(strata_series.index), dtype=str)
    # clinical class labels for each real patient
    y_strata = strata_series.to_numpy(dtype=int)

    # the class (OS=1, PFS=0) is rare, with only 2 patients in the dataset, 
    # so we log a warning if it cannot appear in every fold
    rare_count = int((y_strata == 1).sum())
    if rare_count < n_splits:
        logger.warning(
            "Rare stratum (OS=1, PFS=0) has only %d patients across %d folds; "
            "it cannot appear in every validation fold. Stratification is kept "
            "as-is without duplicating patients.",
            rare_count,
            n_splits,
        )

    # copy the df and add a column with the base patient id, which is the patient id without the timepoint suffix
    # this will be used to group the timepoints of the same patient together when splitting into folds
    work_real = df_real.copy()
    work_real["base_patient_id"] = work_real["patient_id"].astype(str).apply(extract_real_patient_id)

    # skf object 5 folds, shuffle before splitting, random state for reproducibility
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[FoldSplit] = []

    # for each fold, we get the train and validation indices of the unique patients,
    # not all df so that we can group the timepoints of the same patient together and avoid leakage
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(unique_patients, y_strata)):
        # take the ids of the 2 groups
        train_pids = tuple(sorted(unique_patients[train_idx].tolist()))
        val_pids = tuple(sorted(unique_patients[val_idx].tolist()))
        val_set = set(val_pids)

        # get the timepoint ids of the training and validation patients, so that 
        # they go in the same group
        train_tp_ids = tuple(
            sorted(
                work_real.loc[work_real["base_patient_id"].isin(train_pids), "patient_id"]
                .astype(str)
                .unique()
                .tolist()
            )
        )
        val_tp_ids = tuple(
            sorted(
                work_real.loc[work_real["base_patient_id"].isin(val_pids), "patient_id"]
                .astype(str)
                .unique()
                .tolist()
            )
        )

        # classes of the validation patients, to check that the stratification is correct
        # then we count the number of patients in each class and store it in a dictionary
        val_strata = y_strata[val_idx]
        strata_counts = {
            STRATUM_TO_STATE_LABEL[0]: int((val_strata == 0).sum()),
            STRATUM_TO_STATE_LABEL[1]: int((val_strata == 1).sum()),
            STRATUM_TO_STATE_LABEL[2]: int((val_strata == 2).sum()),
        }

        # get the eligible synthetic ids for this fold, if we have a synthetic dataframe
        if df_synth is not None:
            eligible_ids = get_eligible_synthetic_ids(df_synth, val_set)
            sampled_by_ratio = select_nested_synthetic_quotas(
                eligible_synth_ids=eligible_ids,
                n_real_train_patients=len(train_pids),
                ratios=ratios,
                random_state=random_state + fold_idx * fold_seed_stride,
                quota_integer_tol=quota_integer_tol,
            )
            # random_state is incremented by fold_idx to ensure different permutations for
            # each fold but still deterministic across runs
        else:
            eligible_ids = []
            sampled_by_ratio = {float(r): () for r in ratios}

        # create a FoldSplit dataclass instance for this fold and append it to the list of folds
        folds.append(
            FoldSplit(
                fold_idx=fold_idx,
                train_real_patient_ids=train_pids,
                val_real_patient_ids=val_pids,
                train_real_timepoint_ids=train_tp_ids,
                val_real_timepoint_ids=val_tp_ids,
                eligible_synthetic_ids=tuple(eligible_ids),
                sampled_synthetic_by_ratio=sampled_by_ratio,
                val_strata_counts=strata_counts,
            )
        )

    return folds