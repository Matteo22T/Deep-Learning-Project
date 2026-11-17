from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

# Configuration 
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def save_fig(fig: plt.Figure, filename: str, output_dir: str = "results/figures", dpi: int = 300) -> None:
    """
    Saves a matplotlib figure in JPEG format with specified resolution.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    file_path = out_path / f"{filename}.jpeg"
    fig.savefig(file_path, format='jpeg', dpi=dpi, bbox_inches='tight')
    print(f"Graph saved: {file_path} ({dpi} DPI)")

def plot_target_distribution(
    df_real: pd.DataFrame, 
    df_synth: pd.DataFrame, 
    save: bool = True
) -> Optional[plt.Figure]:
    """
    Generates Graph 1: Joint distribution of predictive targets and longitudinal scans.
    - X-axis: Clinical status (OS=0/PFS=0, OS=1/PFS=0, OS=1/PFS=1)
    - Y-axis: Count of real/synthetic patients
    - Hue: Number of available timepoints
    """
    # Pre-processing Real Internal (patient-level)
    real_patients = df_real.groupby('patient_id').first().reset_index()
    real_patients['physical_id'] = real_patients['patient_id'].apply(lambda x: x.split('_')[0])
    
    tp_counts = real_patients.groupby('physical_id').size().reset_index(name='Timepoints')
    labels = real_patients.groupby('physical_id')[['OS_1y', 'PFS_1y']].first().reset_index()
    
    df_real_agg = pd.merge(tp_counts, labels, on='physical_id')
    df_real_agg['Cohort'] = 'Real Internal'
    
    # Pre-processing Synthetic (synthetic patient-level)
    df_synth_agg = df_synth.groupby('patient_id').first().reset_index()
    df_synth_agg['Timepoints'] = 1  # Synthetic patients have only 1 timepoint
    df_synth_agg['Cohort'] = 'Synthetic'
    
    # Concatenation
    df_comb = pd.concat([
        df_real_agg[['Cohort', 'OS_1y', 'PFS_1y', 'Timepoints']],
        df_synth_agg[['Cohort', 'OS_1y', 'PFS_1y', 'Timepoints']]
    ])

    # creation of composite label
    def get_status_label(row):
        if row['OS_1y'] == 0 and row['PFS_1y'] == 0:
            return "Deceased/Prog. (OS=0, PFS=0)"
        elif row['OS_1y'] == 1 and row['PFS_1y'] == 0:
            return "Alive w/ Prog. (OS=1, PFS=0)"
        elif row['OS_1y'] == 1 and row['PFS_1y'] == 1:
            return "Alive, No Prog. (OS=1, PFS=1)"
        return "Invalid"
        
    df_comb['Clinical Status'] = df_comb.apply(get_status_label, axis=1)
    df_comb['Timepoints'] = df_comb['Timepoints'].astype(str) + " Scans"

    # graphical representation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})
    
    # colors
    palette = {"1 Scans": "#4C72B0", "2 Scans": "#DD8452"}
    order = ["Deceased/Prog. (OS=0, PFS=0)", "Alive w/ Prog. (OS=1, PFS=0)", "Alive, No Prog. (OS=1, PFS=1)"]
    
    # real plot
    sns.countplot(
        data=df_comb[df_comb['Cohort'] == 'Real Internal'], 
        x='Clinical Status', hue='Timepoints', 
        ax=axes[0], palette=palette, order=order
    )
    axes[0].set_title("Real Internal Patients (N=50)", fontweight='bold', pad=15)
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title="Available Scans")

    # Plot Synthetic
    sns.countplot(
        data=df_comb[df_comb['Cohort'] == 'Synthetic'], 
        x='Clinical Status', hue='Timepoints', 
        ax=axes[1], palette=palette, order=order
    )
    axes[1].set_title("Synthetic Augmented Patients (N=225)", fontweight='bold', pad=15)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].get_legend().remove()
    
    # numerical annotations on top of bars
    for ax in axes:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=12, xytext=(0, 3), textcoords='offset points')

    plt.suptitle("Figure 1: Joint Distribution of Predictive Targets and Longitudinal Scans", y=1.05, fontsize=18)
    
    if save:
        save_fig(fig, "Fig1_Target_Distribution")
        
    return fig