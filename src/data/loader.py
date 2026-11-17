from pathlib import Path

import pandas as pd


def extract_real_patient_id(full_id: str) -> str:
    """
    Extracts the real patient ID (e.g., PCNSL041) by removing the timepoint suffix.
    """
    return full_id.split("_")[0]


def get_synthetic_parents(synth_id: str) -> tuple[str, str]:
    """
    Extracts the parent A (Host) and B (Donor) from a synthetic ID.
    Expected format: A<patient>_<tp>__B<patient>_<tp>
    """
    parts = synth_id.split("__")
    if len(parts) != 2:
        raise ValueError(f"Synthetic ID format is invalid: {synth_id}")

    # remove the 'A' and 'B' prefixes and extract the base patient IDs
    host_id = extract_real_patient_id(parts[0][1:])
    donor_id = extract_real_patient_id(parts[1][1:])

    return host_id, donor_id


def validate_dataframe_integrity(
    df: pd.DataFrame, dataset_name: str, expected_rows: int, has_labels: bool = True
) -> None:
    """
    Executes structural and clinical validation assertions on the CSV.
    """
    print(f"\n--- Validation {dataset_name} ---")

    # dimensionality check
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, found {len(df)}"
    print(f"✓ correct number of rows: {len(df)}.")

    # verify MRI sequences and patient_id uniqueness (exactly 4 sequences per patient_id)
    seq_counts = df.groupby("patient_id")["sequence"].nunique()
    assert (seq_counts == 4).all(), "some patients do not have exactly 4 sequences!"

    # verify that the sequences are exactly T1, T1gd, T2, Flair
    expected_seqs = {"T1", "T1gd", "T2", "Flair"}
    actual_seqs = set(df["sequence"].unique())
    assert actual_seqs == expected_seqs, f"Anomalous sequences found: {actual_seqs}"
    print("✓ Sequences T1, T1gd, T2, Flair present for all.")

    # NaN check for features
    feature_cols = [col for col in df.columns if col.startswith("f")]
    assert len(feature_cols) == 768, f"Expected 768 features, found {len(feature_cols)}"
    assert not df[feature_cols].isna().any().any(), (
        "Found NaN values in the embeddings!"
    )
    print("✓ No missing values in the 768 BrainIAC embeddings.")

    # clinical logic checks (only if the dataset has labels)
    if has_labels:
        # A. if PFS_1y == 1, OS_1y MUST be 1
        violations = df[(df["PFS_1y"] == 1) & (df["OS_1y"] == 0)]
        assert violations.empty, (
            f"Clinical violation: Found {len(violations)} cases with PFS=1 and OS=0"
        )

        # B. label must be identical across the 4 sequences of the same timepoint
        label_variance = df.groupby("patient_id")[["OS_1y", "PFS_1y"]].nunique()
        assert bool((label_variance == 1).to_numpy().all()), (
            "Labels are not identical across the 4 sequences of the same timepoint!"
        )
        print("✓ Clinical logic and label consistency verified.")


def load_all_datasets(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """load all datasets."""
    df_synth = pd.read_csv(data_dir / "synthetic_with_labels.csv")
    df_real_in = pd.read_csv(data_dir / "real_with_labels.csv")
    df_real_out = pd.read_csv(data_dir / "real_without_labels.csv")
    df_gt = pd.read_csv(data_dir / "real_without_labels_ground_truth.csv")
    return df_synth, df_real_in, df_real_out, df_gt
