from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.data.loader import extract_real_patient_id

# Global plotting style settings for publication
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
FIGURES_DIR = Path("results/figures")


def _save_figure(fig: plt.Figure, filename: str, save: bool = True) -> None:
    """Save the figure as a high-resolution JPEG (>= 300 DPI).

    Args:
        fig: The matplotlib Figure object to save.
        filename: Destination file name (e.g., 'Fig1_Target_Distribution.jpeg').
        save: If True, writes the figure to FIGURES_DIR; otherwise skips saving.
    """
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / filename
        fig.savefig(out_path, format="jpeg", dpi=300, bbox_inches="tight")
        print(f"✓ Figure saved to: {out_path}")


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Extract and return the 768 BrainIAC embedding feature column names (f0 ... f767)."""
    return [col for col in df.columns if col.startswith("f")]


def _map_clinical_state(os_val: int, pfs_val: int) -> str:
    """Map the (OS_1y, PFS_1y) binary outcome pair into one of the 3 valid clinical states."""
    if os_val == 0 and pfs_val == 0:
        return "Deceased/Prog. (OS=0, PFS=0)"
    elif os_val == 1 and pfs_val == 0:
        return "Alive w/ Prog. (OS=1, PFS=0)"
    elif os_val == 1 and pfs_val == 1:
        return "Alive, No Prog. (OS=1, PFS=1)"
    return f"Invalid (OS={os_val}, PFS={pfs_val})"


# ==============================================================================
# FIGURE 1: TARGET DISTRIBUTION AND LONGITUDINAL SCANS
# ==============================================================================
def plot_target_distribution(
    df_real_in: pd.DataFrame, df_synth: pd.DataFrame, save: bool = True
) -> plt.Figure:
    """Generate Figure 1: Joint distribution of predictive targets (OS_1y, PFS_1y).

    Plots the outcomes across the 50 real internal patients (stratified by
    1 Scan vs 2 Scans) and across the 225 synthetic patients.

    Args:
        df_real_in: DataFrame containing real internal cohort scans.
        df_synth: DataFrame containing synthetic cohort scans.
        save: Whether to save the figure to disk.

    Returns:
        The generated matplotlib Figure object.
    """
    state_order = [
        "Deceased/Prog. (OS=0, PFS=0)",
        "Alive w/ Prog. (OS=1, PFS=0)",
        "Alive, No Prog. (OS=1, PFS=1)",
    ]

    # 1. Aggregate at the unique real patient level (N=50)
    df_real = df_real_in.copy()
    df_real["base_id"] = df_real["patient_id"].apply(extract_real_patient_id)

    real_patients = (
        df_real.groupby("base_id")
        .agg(
            OS_1y=("OS_1y", "first"),
            PFS_1y=("PFS_1y", "first"),
            n_timepoints=("patient_id", "nunique"),
        )
        .reset_index()
    )
    real_patients["state"] = [
        _map_clinical_state(o, p)
        for o, p in zip(real_patients["OS_1y"], real_patients["PFS_1y"])
    ]
    real_patients["scan_label"] = real_patients["n_timepoints"].map(
        {1: "1 Scan", 2: "2 Scans"}
    )

    # 2. Aggregate at the synthetic patient level (N=225)
    synth_patients = (
        df_synth.groupby("patient_id")
        .agg(OS_1y=("OS_1y", "first"), PFS_1y=("PFS_1y", "first"))
        .reset_index()
    )
    synth_patients["state"] = [
        _map_clinical_state(o, p)
        for o, p in zip(synth_patients["OS_1y"], synth_patients["PFS_1y"])
    ]

    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Figure 1: Joint Distribution of Predictive Targets and Longitudinal Scans",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # --- Left Panel: Real Internal Patients (N=50) ---
    ax0 = axes[0]
    palette_scans = {"1 Scan": "#5975a4", "2 Scans": "#cc8963"}
    sns.countplot(
        data=real_patients,
        x="state",
        hue="scan_label",
        order=state_order,
        hue_order=["2 Scans", "1 Scan"],
        palette=palette_scans,
        ax=ax0,
    )
    ax0.set_title("Real Internal Patients (N=50)", fontweight="bold", pad=10)
    ax0.set_xlabel("")
    ax0.set_ylabel("Number of Patients", fontweight="bold")
    ax0.tick_params(axis="x", rotation=30)
    # Extra vertical margin (22) so bar labels never collide with the legend
    ax0.set_ylim(0, 22)

    for container in ax0.containers:
        ax0.bar_label(container, fmt="%d", padding=3, fontsize=10)

    # Clean legend placement at the upper center above the lowest bar (n=2)
    ax0.legend(
        title="Available Scans",
        loc="upper center",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    # --- Right Panel: Synthetic Patients (N=225) ---
    ax1 = axes[1]
    sns.countplot(
        data=synth_patients,
        x="state",
        order=state_order,
        color="#5975a4",
        ax=ax1,
    )
    ax1.set_title("Synthetic Augmented Patients (N=225)", fontweight="bold", pad=10)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_ylim(0, 150)

    for container in ax1.containers:
        ax1.bar_label(container, fmt="%d", padding=3, fontsize=10)

    plt.tight_layout()
    _save_figure(fig, "Fig1_Target_Distribution.jpeg", save=save)
    return fig


# ==============================================================================
# FIGURE 2: COSINE SIMILARITY AND HOST A vs DONOR B DOMINANCE
# ==============================================================================
def plot_cosine_similarity_and_dominance(
    df_real_in: pd.DataFrame, df_synth: pd.DataFrame, save: bool = True
) -> tuple[plt.Figure, pd.DataFrame]:
    """Generate Figure 2: Representation collinearity and ancestry dominance.

    - Panel A: Heatmap of mean cosine similarity across the 4 MRI sequences.
    - Panel B: Violin plot of inter-patient collinearity (~0.93 raw vs ~0.00 post z-score).
    - Panel C: Violin plot of similarity between Synthetic and Host A (parenchyma) vs
      Donor B (inpainted tumor) before and after sequence-specific z-score standardization.

    Args:
        df_real_in: DataFrame containing real internal cohort scans.
        df_synth: DataFrame containing synthetic cohort scans.
        save: Whether to save the figure to disk.

    Returns:
        A tuple of (matplotlib Figure, summary statistics DataFrame).
    """
    feat_cols = _get_feature_cols(df_real_in)
    sequences = ["T1", "T1gd", "T2", "Flair"]

    # 1. Compute inter-sequence similarity within the same real timepoint (63 timepoints)
    seq_sim_matrix = np.zeros((4, 4))
    timepoints = df_real_in["patient_id"].unique()
    for tp in timepoints:
        sub = df_real_in[df_real_in["patient_id"] == tp].set_index("sequence")
        mat = sub.loc[sequences, feat_cols].to_numpy()
        seq_sim_matrix += cosine_similarity(mat)
    seq_sim_matrix /= len(timepoints)

    # 2. Sequence-specific Z-Score standardization fitted ONLY on the 50 unique real patients
    df_real = df_real_in.copy()
    df_real["base_id"] = df_real["patient_id"].apply(extract_real_patient_id)

    df_real_z = df_real_in.copy()
    df_synth_z = df_synth.copy()

    for seq in sequences:
        # Average per unique patient on the specific sequence to avoid double-weighting the 13 longitudinal cases
        real_seq_unique = (
            df_real[df_real["sequence"] == seq].groupby("base_id")[feat_cols].mean()
        )
        scaler = StandardScaler().fit(real_seq_unique.to_numpy())

        mask_r = df_real_z["sequence"] == seq
        df_real_z.loc[mask_r, feat_cols] = scaler.transform(
            df_real_z.loc[mask_r, feat_cols].to_numpy()
        )

        mask_s = df_synth_z["sequence"] == seq
        df_synth_z.loc[mask_s, feat_cols] = scaler.transform(
            df_synth_z.loc[mask_s, feat_cols].to_numpy()
        )

    # 3. Mean pooling across unique real patients (Raw vs Z-Score)
    real_vec_raw = df_real.groupby("base_id")[feat_cols].mean()
    df_real_z["base_id"] = df_real["base_id"]
    real_vec_z = df_real_z.groupby("base_id")[feat_cols].mean()

    sim_inter_raw = cosine_similarity(real_vec_raw.to_numpy())
    sim_inter_z = cosine_similarity(real_vec_z.to_numpy())
    triu_idx = np.triu_indices(len(real_vec_raw), k=1)

    inter_patient_raw_vals = sim_inter_raw[triu_idx]
    inter_patient_z_vals = sim_inter_z[triu_idx]

    # 4. Compare Synthetic vs Host A and Donor B (at timepoint level, Mean Pooling)
    tp_real_raw = df_real_in.groupby("patient_id")[feat_cols].mean()
    tp_real_z = df_real_z.groupby("patient_id")[feat_cols].mean()

    tp_synth_raw = df_synth.groupby("patient_id")[feat_cols].mean()
    tp_synth_z = df_synth_z.groupby("patient_id")[feat_cols].mean()

    sim_host_raw, sim_donor_raw = [], []
    sim_host_z, sim_donor_z = [], []

    for synth_id in tp_synth_raw.index:
        parts = synth_id.split("__")
        host_tp = parts[0][1:]  # e.g., PCNSL041_baseline
        donor_tp = parts[1][1:]  # e.g., PCNSL012_pre-bx

        v_s_raw = tp_synth_raw.loc[synth_id].to_numpy().reshape(1, -1)
        v_s_z = tp_synth_z.loc[synth_id].to_numpy().reshape(1, -1)

        # Fallback to base patient if the timepoint suffix differs
        host_key = (
            host_tp
            if host_tp in tp_real_raw.index
            else extract_real_patient_id(host_tp)
        )
        donor_key = (
            donor_tp
            if donor_tp in tp_real_raw.index
            else extract_real_patient_id(donor_tp)
        )

        v_a_raw = (
            tp_real_raw.loc[host_tp].to_numpy().reshape(1, -1)
            if host_tp in tp_real_raw.index
            else real_vec_raw.loc[host_key].to_numpy().reshape(1, -1)
        )
        v_b_raw = (
            tp_real_raw.loc[donor_tp].to_numpy().reshape(1, -1)
            if donor_tp in tp_real_raw.index
            else real_vec_raw.loc[donor_key].to_numpy().reshape(1, -1)
        )

        v_a_z = (
            tp_real_z.loc[host_tp].to_numpy().reshape(1, -1)
            if host_tp in tp_real_z.index
            else real_vec_z.loc[host_key].to_numpy().reshape(1, -1)
        )
        v_b_z = (
            tp_real_z.loc[donor_tp].to_numpy().reshape(1, -1)
            if donor_tp in tp_real_z.index
            else real_vec_z.loc[donor_key].to_numpy().reshape(1, -1)
        )

        sim_host_raw.append(float(cosine_similarity(v_s_raw, v_a_raw)[0, 0]))
        sim_donor_raw.append(float(cosine_similarity(v_s_raw, v_b_raw)[0, 0]))
        sim_host_z.append(float(cosine_similarity(v_s_z, v_a_z)[0, 0]))
        sim_donor_z.append(float(cosine_similarity(v_s_z, v_b_z)[0, 0]))

    # Build the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(
        "Figure 2: Representation Collinearity, Z-Score Effect, and Host A vs Donor B Dominance",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    # Panel A: Intra-patient sequence similarity heatmap
    sns.heatmap(
        seq_sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=sequences,
        yticklabels=sequences,
        vmin=0.85,
        vmax=1.0,
        ax=axes[0],
        cbar_kws={"label": "Mean Cosine Similarity"},
    )
    axes[0].set_title("A. Intra-Patient Sequence Similarity (Raw)", fontweight="bold")

    # Panel B: Inter-patient collinearity (Raw vs Z-Score)
    df_collinearity = pd.DataFrame(
        {
            "Cosine Similarity": np.concatenate(
                [inter_patient_raw_vals, inter_patient_z_vals]
            ),
            "Representation": ["Raw Embeddings"] * len(inter_patient_raw_vals)
            + ["Post Z-Score"] * len(inter_patient_z_vals),
        }
    )
    sns.violinplot(
        data=df_collinearity,
        x="Representation",
        y="Cosine Similarity",
        palette=["#d95f02", "#1b9e77"],
        inner="box",
        cut=0,
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("B. Inter-Patient Collinearity (N=50 Real)", fontweight="bold")
    axes[1].set_xlabel("")

    # Panel C: Synthetic vs Host A and Donor B
    df_dom = pd.DataFrame(
        {
            "Cosine Similarity": sim_host_raw
            + sim_donor_raw
            + sim_host_z
            + sim_donor_z,
            "Setting": (["Raw"] * (2 * len(sim_host_raw)))
            + (["Post Z-Score"] * (2 * len(sim_host_z))),
            "Parent": (
                ["Synth vs Host A\n(Parenchyma)"] * len(sim_host_raw)
                + ["Synth vs Donor B\n(Tumor Inpainted)"] * len(sim_donor_raw)
            )
            * 2,
        }
    )
    sns.violinplot(
        data=df_dom,
        x="Setting",
        y="Cosine Similarity",
        hue="Parent",
        palette=["#4c72b0", "#c44e52"],
        inner="box",
        cut=0,
        ax=axes[2],
    )
    axes[2].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_title("C. Synthetic Ancestry Dominance (N=225)", fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].legend(title="Ancestry Pair", loc="lower left", frameon=True)

    plt.tight_layout()
    _save_figure(fig, "Fig2_Cosine_Similarity_and_Dominance.jpeg", save=save)

    # Summary table for reporting and notebook inspection
    summary_df = pd.DataFrame(
        {
            "Metric / Comparison": [
                "Inter-Patient Real Pairs (Raw)",
                "Inter-Patient Real Pairs (Post Z-Score)",
                "Synthetic vs Host A - Parenchyma (Raw)",
                "Synthetic vs Donor B - Tumor (Raw)",
                "Synthetic vs Host A - Parenchyma (Post Z-Score)",
                "Synthetic vs Donor B - Tumor (Post Z-Score)",
            ],
            "Mean Cosine Sim": [
                np.mean(inter_patient_raw_vals),
                np.mean(inter_patient_z_vals),
                np.mean(sim_host_raw),
                np.mean(sim_donor_raw),
                np.mean(sim_host_z),
                np.mean(sim_donor_z),
            ],
            "Std Dev": [
                np.std(inter_patient_raw_vals),
                np.std(inter_patient_z_vals),
                np.std(sim_host_raw),
                np.std(sim_donor_raw),
                np.std(sim_host_z),
                np.std(sim_donor_z),
            ],
        }
    )
    return fig, summary_df


# ==============================================================================
# FIGURE 3: LATENT SPACE GEOMETRY (PCA & UMAP) AND DOMAIN SHIFT
# ==============================================================================
def plot_latent_space_projections(
    df_real_in: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_real_out: pd.DataFrame,
    df_gt: pd.DataFrame | None = None,
    save: bool = True,
    random_state: int = 42,
) -> plt.Figure:
    """Generate Figure 3: 2D latent space projections and domain shift inspection.

    Computes 2D PCA and UMAP (with t-SNE fallback) post-standardization across
    the Internal Cohort (50), Synthetic Cohort (225), and External Cohort (12).
    The top row is colored by Cohort; the bottom row is colored by Clinical State.

    Args:
        df_real_in: DataFrame containing real internal cohort scans.
        df_synth: DataFrame containing synthetic cohort scans.
        df_real_out: DataFrame containing real external cohort scans.
        df_gt: Optional ground truth DataFrame for external cohort labels.
        save: Whether to save the figure to disk.
        random_state: Seed for reproducible dimensionality reduction.

    Returns:
        The generated matplotlib Figure object.
    """
    feat_cols = _get_feature_cols(df_real_in)
    sequences = ["T1", "T1gd", "T2", "Flair"]

    df_r = df_real_in.copy()
    df_r["base_id"] = df_r["patient_id"].apply(extract_real_patient_id)

    df_s = df_synth.copy()
    df_s["base_id"] = df_s["patient_id"]

    df_e = df_real_out.copy()
    df_e["base_id"] = df_e["patient_id"].apply(extract_real_patient_id)

    # 1. Sequence-specific Z-Score fitted EXCLUSIVELY on the 50 unique real internal patients
    for seq in sequences:
        real_seq_unique = (
            df_r[df_r["sequence"] == seq].groupby("base_id")[feat_cols].mean()
        )
        scaler = StandardScaler().fit(real_seq_unique.to_numpy())

        for df_curr in (df_r, df_s, df_e):
            mask = df_curr["sequence"] == seq
            df_curr.loc[mask, feat_cols] = scaler.transform(
                df_curr.loc[mask, feat_cols].to_numpy()
            )

    # 2. Mean pooling across sequences per unique patient (50 Real, 225 Synthetic, 12 External)
    X_real = df_r.groupby("base_id")[feat_cols].mean()
    y_real = df_r.groupby("base_id")[["OS_1y", "PFS_1y"]].first()

    X_synth = df_s.groupby("base_id")[feat_cols].mean()
    y_synth = df_s.groupby("base_id")[["OS_1y", "PFS_1y"]].first()

    X_ext = df_e.groupby("base_id")[feat_cols].mean()
    if df_gt is not None:
        df_gt_c = df_gt.copy()
        df_gt_c["base_id"] = df_gt_c["patient_id"].apply(extract_real_patient_id)
        y_ext = df_gt_c.groupby("base_id")[["OS_1y", "PFS_1y"]].first().loc[X_ext.index]
        ext_states = [
            _map_clinical_state(o, p) for o, p in zip(y_ext["OS_1y"], y_ext["PFS_1y"])
        ]
    else:
        ext_states = ["External Unlabeled"] * len(X_ext)

    # 3. Fit PCA ONLY on the 50 unique real internal patients and project all cohorts
    pca = PCA(n_components=2, random_state=random_state)
    pca_real = pca.fit_transform(X_real.to_numpy())
    pca_synth = pca.transform(X_synth.to_numpy())
    pca_ext = pca.transform(X_ext.to_numpy())
    var_exp = pca.explained_variance_ratio_ * 100

    # 4. UMAP (or t-SNE fallback) to visualize local manifold topology
    X_all = np.vstack([X_real.to_numpy(), X_synth.to_numpy(), X_ext.to_numpy()])
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=random_state)
        # Joint manifold fit across all cohorts for topological inspection
        emb_2d = reducer.fit_transform(X_all)
        method_name = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2, perplexity=25, random_state=random_state, init="pca"
        )
        emb_2d = reducer.fit_transform(X_all)
        method_name = "t-SNE"

    # Construct plotting DataFrame
    cohorts = (
        ["Real Internal (N=50)"] * len(X_real)
        + ["Synthetic (N=225)"] * len(X_synth)
        + ["Real External (N=12)"] * len(X_ext)
    )
    states = (
        [_map_clinical_state(o, p) for o, p in zip(y_real["OS_1y"], y_real["PFS_1y"])]
        + [
            _map_clinical_state(o, p)
            for o, p in zip(y_synth["OS_1y"], y_synth["PFS_1y"])
        ]
        + ext_states
    )

    df_plot = pd.DataFrame(
        {
            "PCA1": np.concatenate([pca_real[:, 0], pca_synth[:, 0], pca_ext[:, 0]]),
            "PCA2": np.concatenate([pca_real[:, 1], pca_synth[:, 1], pca_ext[:, 1]]),
            "Dim1": emb_2d[:, 0],
            "Dim2": emb_2d[:, 1],
            "Cohort": cohorts,
            "Clinical State": states,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    fig.suptitle(
        f"Figure 3: 768-D BrainIAC Latent Space Geometry (PCA & {method_name} Post Z-Score)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    cohort_palette = {
        "Synthetic (N=225)": "#a6bddb",
        "Real Internal (N=50)": "#1f78b4",
        "Real External (N=12)": "#e31a1c",
    }
    cohort_markers = {
        "Synthetic (N=225)": "o",
        "Real Internal (N=50)": "s",
        "Real External (N=12)": "X",
    }

    state_palette = {
        "Deceased/Prog. (OS=0, PFS=0)": "#d95f02",
        "Alive w/ Prog. (OS=1, PFS=0)": "#e6ab02",
        "Alive, No Prog. (OS=1, PFS=1)": "#1b9e77",
    }

    # Layering order to highlight internal and external real patients over the synthetic cloud
    for ax, (x_col, y_col, title) in zip(
        axes[0],
        [
            (
                "PCA1",
                "PCA2",
                f"A. PCA by Cohort (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)",
            ),
            (
                "Dim1",
                "Dim2",
                f"B. {method_name} Manifold by Cohort (Domain Shift Check)",
            ),
        ],
    ):
        for cohort_name, size, alpha, zord in [
            ("Synthetic (N=225)", 45, 0.55, 2),
            ("Real Internal (N=50)", 75, 0.90, 3),
            ("Real External (N=12)", 110, 1.00, 4),
        ]:
            sub = df_plot[df_plot["Cohort"] == cohort_name]
            ax.scatter(
                sub[x_col],
                sub[y_col],
                label=cohort_name,
                c=cohort_palette[cohort_name],
                marker=cohort_markers[cohort_name],
                s=size,
                alpha=alpha,
                edgecolors="black" if cohort_name != "Synthetic (N=225)" else "none",
                linewidths=0.7,
                zorder=zord,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(title="Cohort", loc="best", frameon=True)

    # Bottom row: colored by Clinical State
    for ax, (x_col, y_col, title) in zip(
        axes[1],
        [
            ("PCA1", "PCA2", "C. PCA Colored by Clinical Outcome (OS_1y, PFS_1y)"),
            ("Dim1", "Dim2", f"D. {method_name} Colored by Clinical Outcome"),
        ],
    ):
        sns.scatterplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue="Clinical State",
            style="Cohort",
            palette=state_palette,
            markers=cohort_markers,
            s=70,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(loc="best", frameon=True, fontsize=9)

    plt.tight_layout()
    _save_figure(fig, "Fig3_Latent_Space_PCA_UMAP.jpeg", save=save)
    return fig
