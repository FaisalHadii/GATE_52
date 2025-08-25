import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import subprocess
import shutil
import os
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
# You can override these via environment variables:
#   TOPAS_EXECUTABLE_PATH, GATE_EXECUTABLE_PATH, TOPAS_BASE_MACRO_FILE, GATE_BASE_MACRO_FILE

TOPAS_EXECUTABLE_PATH = os.environ.get("TOPAS_EXECUTABLE_PATH") or shutil.which("topas") or "/home/fhhadi/topas/bin/topas"
GATE_EXECUTABLE_PATH = os.environ.get("GATE_EXECUTABLE_PATH") or shutil.which("Gate") or shutil.which("gate") or "/usr/local/bin/Gate"

BASE_TOPAS_MACRO_FILE = os.environ.get("TOPAS_BASE_MACRO_FILE") or "compined1.txt"
BASE_GATE_MACRO_FILE = os.environ.get("GATE_BASE_MACRO_FILE") or "gate_template.mac"


# --- TOPAS HELPERS ---
def modify_topas_macro(template_path: Path, output_path: Path, phantom_half_thickness_cm: float, output_dir: Path):
    """Creates a temporary macro for a specific TOPAS run by modifying the phantom's half-thickness and output paths."""
    if not template_path.exists():
        raise FileNotFoundError(f"Base macro file not found: {template_path}")

    content = template_path.read_text()
    content = re.sub(r"(d:Ge/Phantom/HLZ\s*=\s*)[\d\.]+\s*cm", rf"\g<1>{phantom_half_thickness_cm} cm", content)
    content = re.sub(r"(s:Sc/DoseInPhantom/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/phantom_dose"', content)
    content = re.sub(r"(s:Sc/PhaseSpace/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/MyTopasOutput"', content)
    content = re.sub(r"(s:Sc/EnergySpectrum/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/energy_spectrum"', content)
    content = re.sub(r"(s:Sc/PrePhaseSpaceInteractionScorer/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/PrePhaseSpaceInteractions"', content)

    output_path.write_text(content)
    print(f"Created TOPAS macro for HLZ={phantom_half_thickness_cm} cm (thickness {phantom_half_thickness_cm * 2} cm).")


def run_topas_simulation(macro_path: Path) -> bool:
    """Executes a TOPAS simulation."""
    print("-" * 60, f"\nRunning TOPAS for: {macro_path.name}\n", "-" * 60)
    try:
        subprocess.run([TOPAS_EXECUTABLE_PATH, str(macro_path)], check=True, capture_output=True, text=True)
        print("TOPAS simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"TOPAS simulation failed. Error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: TOPAS executable not found at '{TOPAS_EXECUTABLE_PATH}'")
        return False


def read_topas_phasespace(phsp_file: Path) -> pd.DataFrame | None:
    """Reads a TOPAS binary phase space file and returns a DataFrame."""
    if not phsp_file.exists():
        return None
    try:
        dtype = np.dtype([
            ('Position_X', 'f4'),
            ('Position_Y', 'f4'),
            ('Position_Z', 'f4'),
            ('Direction_X', 'f4'),
            ('Direction_Y', 'f4'),
            ('Ekine', 'f4'),
            ('Weight', 'f4'),
            ('Particle_Type', 'i4'),
            ('Direction_Z_Sign', 'b1'),
            ('Is_First_Particle', 'b1'),
            ('Track_ID', 'i4'),
            ('Parent_ID', 'i4'),
            ('Charge', 'f4')
        ])
        data = np.fromfile(phsp_file, dtype=dtype)
        if data.size == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['dZ'] = np.sqrt(1.0 - df['Direction_X']**2 - df['Direction_Y']**2).fillna(0)
        df.rename(columns={
            'Position_X': 'X',
            'Position_Y': 'Y',
            'Direction_X': 'dX',
            'Direction_Y': 'dY'
        }, inplace=True)
        return df
    except Exception:
        return None


def categorize_interactions(df: pd.DataFrame) -> dict:
    """Categorizes interactions exactly like Gate script."""
    df['dZ'] = np.sqrt(1.0 - df['dX']**2 - df['dY']**2).fillna(0)
    cos_theta = np.clip(df['dZ'], -1.0, 1.0)
    df['ScatteringAngle'] = np.degrees(np.arccos(cos_theta))
    return {
        "Unscattered": df[(df['Ekine'] == 0.150) & (df['ScatteringAngle'] < 0.01)],
        "Rayleigh": df[(df['Ekine'] == 0.150) & (df['ScatteringAngle'] >= 0.01)],
        "Compton": df[df['Ekine'] < 0.1499999]
    }


def analyze_interaction_dashboard(output_dir: Path, thickness_cm: float) -> dict:
    """Analyzes transmitted particles and creates the first dashboard."""
    print(f"\nAnalyzing interaction results for thickness {thickness_cm} cm...")
    phasespace_file = output_dir / "MyTopasOutput.phsp"
    dose_file = output_dir / "phantom_dose.csv"
    output_dashboard_file = output_dir / f"dashboard_interactions_{thickness_cm:.4f}cm.png"

    df = read_topas_phasespace(phasespace_file)
    if df is None:
        return {}

    categorized_dfs = categorize_interactions(df)

    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(f'TOPAS Interaction Analysis (Phantom Thickness: {thickness_cm} cm)', fontsize=20, weight='bold')

    # Scatter plots for interactions with log scale
    plot_scatter_on_ax(axs[0, 0], categorized_dfs["Unscattered"], "Unscattered")
    plot_scatter_on_ax(axs[0, 1], categorized_dfs["Rayleigh"], "Rayleigh Scattering")
    plot_scatter_on_ax(axs[1, 0], categorized_dfs["Compton"], "Compton Scattering")

    # The fourth plot (energy deposition) is now also a scatter plot
    plot_edep_scatter_on_ax(axs[1, 1], dose_file, "Energy Deposition", zoom_mm=30.0)

    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.93, bottom=0.05, left=0.07, right=0.95)
    plt.savefig(output_dashboard_file)
    print(f"✅ Interaction dashboard saved to: {output_dashboard_file}")
    plt.close()

    return {"thickness": thickness_cm, "unscattered": len(categorized_dfs["Unscattered"]), "rayleigh": len(categorized_dfs["Rayleigh"]), "compton": len(categorized_dfs["Compton"]), "transmitted": len(df)}


def analyze_source_dose_dashboard(output_dir: Path, thickness_cm: float):
    """Creates the second dashboard showing source and dose distributions."""
    print(f"Analyzing source/dose results for thickness {thickness_cm} cm...")
    source_check_file = output_dir / "PrePhaseSpaceInteractions.csv"
    dose_file = output_dir / "phantom_dose.csv"
    output_dashboard_file = output_dir / f"dashboard_source_dose_{thickness_cm:.4f}cm.png"

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Source and Dose Analysis (Phantom Thickness: {thickness_cm} cm)', fontsize=16, weight='bold')

    # Read source file and handle None case
    source_df = read_source_file(source_check_file)
    plot_scatter_on_ax(axs[0], source_df, "Initial Source Distribution")
    plot_dose_scatter_on_ax(axs[1], dose_file)

    fig.subplots_adjust(wspace=0.3, top=0.88, bottom=0.1, left=0.07, right=0.95)
    plt.savefig(output_dashboard_file)
    print(f"✅ Source/Dose dashboard saved to: {output_dashboard_file}")
    plt.close()


def plot_summary_attenuation(summary_data: list, output_image: Path):
    """Plots the final summary of interactions vs. thickness."""
    if not summary_data:
        return
    df = pd.DataFrame(summary_data).sort_values(by="thickness")
    plt.figure(figsize=(12, 8))
    plt.plot(df['thickness'], df['unscattered'], 'o-', label='Unscattered')
    plt.plot(df['thickness'], df['rayleigh'], 's-', label='Rayleigh')
    plt.plot(df['thickness'], df['compton'], '^-', label='Compton')
    plt.plot(df['thickness'], df['transmitted'], 'd--', label='Total Transmitted', color='black')
    plt.title('Particle Attenuation vs. Phantom Thickness', fontsize=16, weight='bold')
    plt.xlabel('Phantom Thickness (cm)', fontsize=12, weight='bold')
    plt.ylabel('Number of Particles Detected', fontsize=12, weight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.yscale('log')
    plt.savefig(output_image)
    print("-" * 60, f"\n✅ Final attenuation summary plot saved to: {output_image}")
    plt.close()


# --- Helper & Plotting Functions (IDENTICAL to Gate script) ---
def read_source_file(file_path: Path) -> pd.DataFrame | None:
    """Reads source CSV file and converts to Gate-like format."""
    if not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path, skiprows=1)
        if 'Position_X' in df.columns and 'Position_Y' in df.columns:
            df.rename(columns={'Position_X': 'X', 'Position_Y': 'Y'}, inplace=True)
            return df
        return None
    except Exception:
        return None


def plot_scatter_on_ax(ax, df: pd.DataFrame, title: str):
    """Plots a scatter plot of particle positions with a logarithmic scale on the y-axis."""
    # Handle None case first
    if df is None:
        df = pd.DataFrame()

    ax.set_title(f'{title} ({len(df):,} events)', weight='bold', fontsize=14)
    if df.empty:
        ax.text(0.5, 0.5, 'No Events', ha='center', va='center', transform=ax.transAxes)
        return

    # Create the scatter plot with a log-scaled color map
    # We use a histogram-based approach to get the density for coloring the points
    hist, x_edges, y_edges = np.histogram2d(df['X']*10, df['Y']*10, bins=100)

    # Get the bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Map each point to its corresponding bin density
    bin_x = np.searchsorted(x_edges, df['X']*10) - 1
    bin_y = np.searchsorted(y_edges, df['Y']*10) - 1

    # Get the density for each point, adding a small value to avoid log(0)
    densities = hist[bin_x, bin_y] + 1e-6

    # Normalize the colormap using a logarithmic scale
    norm = mcolors.LogNorm(vmin=densities.min(), vmax=densities.max())

    # Plot the scatter plot, coloring by density
    scatter = ax.scatter(df['X']*10, df['Y']*10, s=1, c=densities, cmap='viridis', norm=norm)
    plt.colorbar(scatter, ax=ax, label='Particle Density (log scale)')

    ax.set_xlabel('X position (mm)', weight='bold')
    ax.set_ylabel('Y position (mm)', weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_edep_scatter_on_ax(ax, csv_file_path: Path, title: str, zoom_mm: float):
    """Plots a scatter plot of energy deposition with a logarithmic scale."""
    ax.set_title(f'{title} ({zoom_mm}mm Zoom)', weight='bold', fontsize=14)
    if not csv_file_path.exists():
        ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center', transform=ax.transAxes)
        return
    try:
        df = pd.read_csv(csv_file_path, skiprows=8, header=None)
        if df.shape[1] < 4:
            ax.text(0.5, 0.5, 'No Energy Deposited', ha='center', va='center', transform=ax.transAxes)
            return

        df.columns = ['X_bin', 'Y_bin', 'Z_bin', 'Energy']
        df_filtered = df[df['Energy'] > 0]

        if df_filtered.empty:
            ax.text(0.5, 0.5, 'No Energy Deposited', ha='center', va='center', transform=ax.transAxes)
            return

        # Convert bin indices to mm coordinates
        # 1000 bins of 0.1 cm = 1000 bins of 1 mm
        # Center is at bin 500 (0-indexed: bin 499.5)
        x_coords = (df_filtered['X_bin'] - 500) * 1.0  # 1 mm per bin
        y_coords = (df_filtered['Y_bin'] - 500) * 1.0  # 1 mm per bin
        energies = df_filtered['Energy']

        # Use the energy values for coloring the scatter points
        c = ax.scatter(x_coords, y_coords, c=energies, cmap='hot', s=5, norm=mcolors.LogNorm(vmin=energies.min(), vmax=energies.max()))
        plt.colorbar(c, ax=ax, label='Total Energy Deposited (MeV, log scale)')

        ax.set_xlabel('X position (mm)', weight='bold')
        ax.set_ylabel('Y position (mm)', weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)


def plot_dose_scatter_on_ax(ax, csv_dose_file: Path):
    """Plots a scatter plot of dose deposition with a logarithmic scale."""
    ax.set_title('Dose Deposition', weight='bold', fontsize=14)
    if not csv_dose_file.exists():
        ax.text(0.5, 0.5, 'Dose File Not Found', ha='center', va='center', transform=ax.transAxes)
        return
    try:
        df = pd.read_csv(csv_dose_file, skiprows=8, header=None)
        if df.shape[1] < 4:
            ax.text(0.5, 0.5, 'No Dose Deposited', ha='center', va='center', transform=ax.transAxes)
            return

        df.columns = ['X_bin', 'Y_bin', 'Z_bin', 'Dose']
        df_filtered = df[df['Dose'] > 0]

        if df_filtered.empty:
            ax.text(0.5, 0.5, 'No Dose Deposited', ha='center', va='center', transform=ax.transAxes)
            return

        x_indices = df_filtered['X_bin'].values
        y_indices = df_filtered['Y_bin'].values
        doses = df_filtered['Dose'].values

        im = ax.scatter(x_indices, y_indices, c=doses, cmap='magma', s=5, norm=mcolors.LogNorm(vmin=doses.min(), vmax=doses.max()))
        plt.colorbar(im, ax=ax, label='Dose (Gy, log scale)')

        ax.set_xlabel('X pixel index')
        ax.set_ylabel('Y pixel index')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)


# --- GATE HELPERS ---
def modify_gate_macro(template_path: Path, output_path: Path, phantom_half_thickness_cm: float):
    """Create a temporary Gate macro by substituting half-thickness if a placeholder or known command is present.

    Strategy:
    - If the template contains placeholders like {HLZ_CM} or <HLZ_CM>, replace them with the numeric value in cm
    - Else, attempt regex replacements for common Z half-length commands
    - Leave outputs as defined; run Gate with cwd in run output dir so relative outputs are isolated
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Gate base macro file not found: {template_path}")

    content = template_path.read_text()

    replacements_applied = 0

    # Placeholder replacements
    for placeholder in ("{HLZ_CM}", "<HLZ_CM>", "@HLZ_CM@", "${HLZ_CM}"):
        if placeholder in content:
            content = content.replace(placeholder, f"{phantom_half_thickness_cm} cm")
            replacements_applied += 1

    # Common command patterns for Gate to set Z half-length
    patterns = [
        r"(/gate/volume/\S+/geometry/setZHalfLength\s+)[\d\.]+\s*(mm|cm)",
        r"(/gate/geometry/\S+/setZHalfLength\s+)[\d\.]+\s*(mm|cm)",
        r"(/gate/volume/\S+/geometry/setDimensions\s+[\d\.]+\s*(mm|cm)\s+[\d\.]+\s*(mm|cm)\s+)[\d\.]+\s*(mm|cm)",
    ]

    def to_cm(value_cm: float, unit: str) -> str:
        if unit == 'mm':
            return f"{value_cm * 10.0} mm"
        return f"{value_cm} cm"

    for pat in patterns:
        def repl(match):
            prefix = match.group(1)
            unit = match.group(match.lastindex)
            return f"{prefix}{to_cm(phantom_half_thickness_cm, unit)}"

        new_content, n = re.subn(pat, repl, content)
        if n:
            content = new_content
            replacements_applied += n

    output_path.write_text(content)
    print(f"Created Gate macro for HLZ={phantom_half_thickness_cm} cm (best-effort, changes={replacements_applied}).")


def run_gate_simulation(macro_path: Path, work_dir: Path) -> bool:
    """Executes a Gate simulation using the given macro, with CWD set to work_dir to keep outputs isolated."""
    print("-" * 60, f"\nRunning Gate for: {macro_path.name}\n", "-" * 60)
    try:
        subprocess.run([GATE_EXECUTABLE_PATH, str(macro_path)], cwd=str(work_dir), check=True, capture_output=True, text=True)
        print("Gate simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Gate simulation failed. Error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Gate executable not found at '{GATE_EXECUTABLE_PATH}'")
        return False


def main():
    """Main function to orchestrate the simulation and analysis workflow."""
    thicknesses_cm = [0.0001, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    all_results = []
    base_output_dir = Path("./simulation_runs").resolve()

    base_output_dir.mkdir(parents=True, exist_ok=True)

    for thickness in thicknesses_cm:
        # Calculate the HLZ value (half of the desired thickness)
        half_thickness_cm = thickness / 2.0
        run_output_dir = base_output_dir / f"run_{thickness:.4f}cm"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare macro paths
        temp_topas_macro_path = run_output_dir / "temp_topas.txt"
        temp_gate_macro_path = run_output_dir / "temp_gate.mac"

        # --- Gate ---
        gate_template = Path(BASE_GATE_MACRO_FILE).resolve()
        if gate_template.exists():
            try:
                modify_gate_macro(gate_template, temp_gate_macro_path, half_thickness_cm)
                run_gate_simulation(temp_gate_macro_path, run_output_dir)
            except Exception as e:
                print(f"Skipping Gate run for {thickness} cm due to error: {e}")
        else:
            print(f"Gate base macro not found at {gate_template}; skipping Gate for this thickness.")

        # --- TOPAS ---
        topas_template = Path(BASE_TOPAS_MACRO_FILE).resolve()
        try:
            modify_topas_macro(topas_template, temp_topas_macro_path, half_thickness_cm, run_output_dir)
            if run_topas_simulation(temp_topas_macro_path):
                summary = analyze_interaction_dashboard(run_output_dir, thickness)
                if summary:
                    all_results.append(summary)
                analyze_source_dose_dashboard(run_output_dir, thickness)
            else:
                print(f"Skipping analysis for thickness {thickness} cm due to TOPAS simulation failure.")
        except Exception as e:
            print(f"Skipping TOPAS for {thickness} cm due to error: {e}")

    plot_summary_attenuation(all_results, base_output_dir / "summary_attenuation_plot.png")
    print("\nAutomation script finished.")


if __name__ == "__main__":
    main()