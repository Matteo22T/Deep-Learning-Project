from pathlib import Path

from src.data.loader import (
    extract_real_patient_id,
    get_synthetic_parents,
    load_all_datasets,
    validate_dataframe_integrity,
)


def main() -> None:
    data_dir = Path("data")
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print(
            "Error: insert the CSV files in the 'data' directory before running this script."
        )
        return

    df_synth, df_real_in, df_real_out, df_gt = load_all_datasets(data_dir)

    # structural validation (numbers took from professor's readme)
    validate_dataframe_integrity(
        df_synth, "Synthetic Data", expected_rows=900, has_labels=True
    )
    validate_dataframe_integrity(
        df_real_in, "Real Internal Data", expected_rows=252, has_labels=True
    )
    validate_dataframe_integrity(
        df_real_out, "Real External Data", expected_rows=48, has_labels=False
    )
    validate_dataframe_integrity(
        df_gt, "External Ground Truth", expected_rows=48, has_labels=True
    )
    validate_dataframe_integrity(
        df_gt, "External Ground Truth", expected_rows=48, has_labels=True
    )

    # 100% ID and sequence match with real_without_labels.csv
    unique_ext_patients = set(df_gt["patient_id"].apply(extract_real_patient_id))
    assert len(unique_ext_patients) == 12, (
        f"Expected 12 unique external patients, found {len(unique_ext_patients)}"
    )
    assert (df_gt["patient_id"] == df_real_out["patient_id"]).all(), (
        "Patient ID mismatch with real_without_labels.csv!"
    )
    assert (df_gt["sequence"] == df_real_out["sequence"]).all(), (
        "Sequence alignment mismatch with real_without_labels.csv!"
    )
    print(
        f"✓ 100% ID and sequence alignment with real_without_labels.csv ({len(unique_ext_patients)} unique patients)."
    )

    # Real unique patients analysis (Internal Cohort)
    real_patient_ids = set(df_real_in["patient_id"].apply(extract_real_patient_id))
    print("\n--- Analysis of Real Internal Patients ---")
    print(
        f"Found {len(real_patient_ids)} unique independent patients out of {len(df_real_in) // 4} total timepoints."
    )

    # Synthetic Data Leakage Analysis
    print("\n--- Synthetic Data Leakage Analysis ---")
    synth_unique_ids = df_synth["patient_id"].unique()

    parent_dependencies = set()
    for synth_id in synth_unique_ids:
        host, donor = get_synthetic_parents(synth_id)
        parent_dependencies.add(host)
        parent_dependencies.add(donor)

    print(
        f"the {len(synth_unique_ids)} synthetic timepoints are based on {len(parent_dependencies)} unique real internal patients."
    )

    # parents belong to the internal cohort?
    leaked_from_external = parent_dependencies - real_patient_ids
    assert not leaked_from_external, (
        f"LEAKAGE! synthetic data uses non-internal patients: {leaked_from_external}"
    )
    print(
        "✓ Leakage Security Confirmed: all parent patients A and B are from the internal training set only."
    )


if __name__ == "__main__":
    main()
