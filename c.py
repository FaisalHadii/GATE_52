import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path
import subprocess
import matplotlib.colors as mcolors
import uproot
import SimpleITK as sitk

# --- CONFIGURATION ---
TOPAS_EXECUTABLE_PATH = "/home/fhhadi/topas/bin/topas"
GATE_EXECUTABLE_PATH = "/home/fhhadi/gate_install/bin/Gate"
TOPAS_BASE_MACRO_FILE = "compined1.txt"
GATE_BASE_MACRO_FILE = "compined1.mac"

# --- GLOBAL VARIABLES FOR PLOTTING SCALE ---
GLOBAL_VMIN = 1e-12
GLOBAL_VMAX = None

# --- CONSTANTS FOR DOSE CONVERSION ---
ALUMINUM_DENSITY_KG_M3 = 2700.0  # Density of Aluminum
MEV_TO_JOULE = 1.602176634e-13


def get_max_dose_value(base_output_dir: Path, thicknesses_cm: list):
    max_val = 0
    for thickness in thicknesses_cm:
        topas_dir = base_output_dir / f"topas_run_{thickness:.4f}cm"
        topas_dose_file = topas_dir / 'phantom_dose.csv'
        if topas_dose_file.exists():
            try:
                df = pd.read_csv(topas_dose_file, comment='#', skiprows=1, header=None, names=['X', 'Y', 'Z', 'Dose'])
                n_particles_topas = 1e6
                voxel_size_cm_x = 0.1
                voxel_size_cm_y = 0.1
                voxel_size_cm_z = 0.1
                voxel_volume_m3 = (voxel_size_cm_x * 0.01) * (voxel_size_cm_y * 0.01) * (voxel_size_cm_z * 0.01)
                voxel_mass_kg = voxel_volume_m3 * ALUMINUM_DENSITY_KG_M3
                df['Dose_Gy_per_primary'] = (df['Dose'] * MEV_TO_JOULE) / (voxel_mass_kg * n_particles_topas)
                if not df.empty and df['Dose_Gy_per_primary'].max() > max_val:
                    max_val = df['Dose_Gy_per_primary'].max()
            except Exception as e:
                print(f"Warning: Could not read TOPAS file for max value calculation: {e}")
        gate_dir = base_output_dir / f"gate_run_{thickness:.4f}cm"
        gate_dose_file = gate_dir / 'energy_deposition-Dose.mhd'
        gate_phsp_file = gate_dir / 'phantom_transmission_TEMPLATE.root'
        if gate_dose_file.exists() and gate_phsp_file.exists():
            try:
                dose_image = sitk.ReadImage(str(gate_dose_file))
                dose_array = sitk.GetArrayFromImage(dose_image)
                with uproot.open(gate_phsp_file) as file:
                    n_particles_gate = file["PhaseSpace"].num_entries
                if n_particles_gate == 0:
                    continue
                normalized_dose = dose_array / n_particles_gate
                if normalized_dose.max() > max_val:
                    max_val = normalized_dose.max()
            except Exception as e:
                print(f"Warning: Could not read GATE file for max value calculation: {e}")
    return max_val


def run_topas_simulation(thickness_cm: float, base_macro_path: Path, output_dir: Path) -> bool:
    print("-" * 60, f"\nRunning TOPAS for: {thickness_cm} cm\n", "-" * 60)
    temp_macro_path = output_dir / "temp_run.txt"
    phantom_half_thickness_cm = thickness_cm / 2.0
    if not base_macro_path.exists():
        print(f"Error: Base macro file not found: {base_macro_path}")
        return False
    content = base_macro_path.read_text()
    content = re.sub(r"(d:Ge/Phantom/HLZ\s*=\s*)[\d\.]+\s*cm", rf"\g<1>{phantom_half_thickness_cm} cm", content)
    content = re.sub(r"(s:Sc/DoseInPhantom/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/phantom_dose"', content)
    content = re.sub(r"(s:Sc/PhaseSpace/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/MyTopasOutput"', content)
    content = re.sub(r"(s:Sc/PrePhaseSpaceInteractionScorer/OutputFile\s*=\s*).*", rf'\g<1>"{output_dir}/PrePhaseSpaceInteractions"', content)
    temp_macro_path.write_text(content)
    try:
        subprocess.run([TOPAS_EXECUTABLE_PATH, str(temp_macro_path)], check=True, capture_output=True, text=True)
        print("TOPAS simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"TOPAS simulation failed. Error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: TOPAS executable not found at '{TOPAS_EXECUTABLE_PATH}'")
        return False


def run_gate_simulation(thickness_cm: float, base_macro_path: Path, output_dir: Path) -> bool:
    print("-" * 60, f"\nRunning GATE for: {thickness_cm} cm\n", "-" * 60)
    temp_macro_path = output_dir / "temp_run.mac"
    if not base_macro_path.exists():
        print(f"Error: Base macro file not found: {base_macro_path}")
        return False
    content = base_macro_path.read_text()
    content = re.sub(r"(/gate/phantom/geometry/setZLength\s+)([\d\.]+)(\s+cm)", rf"\g<1>{thickness_cm}\g<3>", content)
    content = re.sub(r"(/control/alias GateOutputPath\s+)(.*)", rf"\g<1>{output_dir}/", content)
    temp_macro_path.write_text(content)
    try:
        subprocess.run([GATE_EXECUTABLE_PATH, str(temp_macro_path)], check=True, capture_output=True, text=True)
        print("GATE simulation completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"GATE simulation failed. Error:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: GATE executable not found at '{GATE_EXECUTABLE_PATH}'")
        return False


def read_topas_phasespace(phsp_file: Path) -> pd.DataFrame | None:
    if not phsp_file.exists():
        return None
    try:
        dtype = np.dtype([
            ('Position_X', 'f4'), ('Position_Y', 'f4'), ('Position_Z', 'f4'),
            ('Direction_X', 'f4'), ('Direction_Y', 'f4'), ('Ekine', 'f4'),
            ('Weight', 'f4'), ('Particle_Type', 'i4'), ('Direction_Z_Sign', 'b1'),
            ('Is_First_Particle', 'b1'), ('Track_ID', 'i4'), ('Parent_ID', 'i4'),
            ('Charge', 'f4')
        ])
        data = np.fromfile(phsp_file, dtype=dtype)
        df = pd.DataFrame(data)
        df['X_cm'], df['Y_cm'] = df['Position_X'], df['Position_Y']
        df['dX'], df['dY'], df['Ekine'] = df['Direction_X'], df['Direction_Y'], df['Ekine']
        return df
    except Exception as e:
        print(f"Error reading TOPAS phase space file: {e}")
        return None


def read_gate_phasespace(phsp_file: Path) -> pd.DataFrame | None:
    if not phsp_file.exists():
        return None
    try:
        with uproot.open(phsp_file) as file:
            df = file["PhaseSpace"].arrays(library="pd")
            df['X_cm'] = df['X'] / 10.0
            df['Y_cm'] = df['Y'] / 10.0
            df.rename(columns={'Ekine': 'Ekine', 'dX': 'dX', 'dY': 'dY'}, inplace=True)
            return df
    except Exception as e:
        print(f"Error reading GATE phase space file: {e}")
        return None


def filter_gamma_particles(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    masks = []
    if 'PDGEncoding' in df.columns:
        try:
            masks.append(df['PDGEncoding'] == 22)
        except Exception:
            pass
    if 'Particle_Type' in df.columns:
        try:
            masks.append(df['Particle_Type'].isin([22, 1]))
        except Exception:
            pass
    for name_col in ['particle', 'ParticleName', 'particleName']:
        if name_col in df.columns:
            try:
                masks.append(df[name_col].astype(str).str.contains('gamma|photon', case=False, na=False))
            except Exception:
                pass
    if not masks:
        print("Warning: No gamma-identifying column found; using all particles.")
        return df
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return df[mask].copy()


def categorize_interactions(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Unscattered": pd.DataFrame(), "Rayleigh": pd.DataFrame(), "Compton": pd.DataFrame()}
    df['dZ'] = np.sqrt(1.0 - df['dX']**2 - df['dY']**2).fillna(0)
    df['ScatteringAngle'] = np.degrees(np.arccos(np.clip(df['dZ'], -1.0, 1.0)))
    return {
        "Unscattered": df[(df['Ekine'] > 0.149999) & (df['ScatteringAngle'] < 0.01)],
        "Rayleigh": df[(df['Ekine'] > 0.149999) & (df['ScatteringAngle'] >= 0.01)],
        "Compton": df[df['Ekine'] < 0.149999]
    }


def plot_scatter_on_ax(ax, df: pd.DataFrame, title: str):
    if df is None or df.empty:
        ax.set_title(f'{title} (0 events)', weight='bold', fontsize=14)
        ax.text(0.5, 0.5, 'No Events', ha='center', va='center', transform=ax.transAxes)
        return
    ax.set_title(f'{title} ({len(df):,} events)', weight='bold', fontsize=14)
    x_pos = df['X_cm'] * 10
    y_pos = df['Y_cm'] * 10
    hist, x_edges, y_edges = np.histogram2d(x_pos, y_pos, bins=100)
    bin_x = np.clip(np.searchsorted(x_edges, x_pos) - 1, 0, hist.shape[0]-1)
    bin_y = np.clip(np.searchsorted(y_edges, y_pos) - 1, 0, hist.shape[1]-1)
    densities = hist[bin_x, bin_y] + 1e-6
    norm = mcolors.LogNorm(vmin=densities.min(), vmax=densities.max())
    scatter = ax.scatter(x_pos, y_pos, s=1, c=densities, cmap='viridis', norm=norm)
    plt.colorbar(scatter, ax=ax, label='Particle Density (log scale)')
    ax.set_xlabel('X position (mm)', weight='bold')
    ax.set_ylabel('Y position (mm)', weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_energy_deposition(ax, file_path: Path, phsp_file_path: Path, simulation_type: str):
    if not file_path.exists():
        ax.text(0.5, 0.5, 'No Energy Deposition Data', ha='center', va='center', transform=ax.transAxes)
        return
    try:
        norm = mcolors.LogNorm(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)
        if simulation_type.lower() == 'topas':
            df = pd.read_csv(file_path, comment='#', skiprows=1, header=None, names=['X', 'Y', 'Z', 'Dose_MeV'])
            n_particles_topas = 1e6
            voxel_size_cm_x = 0.1
            voxel_size_cm_y = 0.1
            voxel_size_cm_z = 0.1
            voxel_volume_m3 = (voxel_size_cm_x * 0.01) * (voxel_size_cm_y * 0.01) * (voxel_size_cm_z * 0.01)
            voxel_mass_kg = voxel_volume_m3 * ALUMINUM_DENSITY_KG_M3
            df['Dose_normalized'] = (df['Dose_MeV'] * MEV_TO_JOULE) / (voxel_mass_kg * n_particles_topas)
            x_cm = df['X'].values
            y_cm = df['Y'].values
            dose_vals = df['Dose_normalized'].values
            x_unique = np.sort(np.unique(x_cm))
            y_unique = np.sort(np.unique(y_cm))
            if len(x_unique) < 2 or len(y_unique) < 2:
                ax.text(0.5, 0.5, 'TOPAS Dose grid is too small or flat', ha='center', va='center', transform=ax.transAxes)
                return
            x_indices = np.searchsorted(x_unique, x_cm)
            y_indices = np.searchsorted(y_unique, y_cm)
            dose_array = np.zeros((len(y_unique), len(x_unique)))
            np.add.at(dose_array, (y_indices, x_indices), dose_vals)
            x_step = x_unique[1] - x_unique[0]
            y_step = y_unique[1] - y_unique[0]
            half_width = (len(x_unique) * x_step) / 2.0
            half_height = (len(y_unique) * y_step) / 2.0
            extent = [-half_width, half_width, -half_height, half_height]
            im = ax.imshow(dose_array, origin='lower', cmap='OrRd', norm=norm, extent=extent)
            ax.set_xlabel('X position (cm)')
            ax.set_ylabel('Y position (cm)')
        elif simulation_type.lower() == 'gate':
            if not phsp_file_path.exists():
                ax.text(0.5, 0.5, 'GATE Phase Space File Not Found', ha='center', va='center', transform=ax.transAxes)
                return
            dose_image = sitk.ReadImage(str(file_path))
            dose_array = sitk.GetArrayFromImage(dose_image)
            with uproot.open(phsp_file_path) as file:
                n_particles_gate = file["PhaseSpace"].num_entries
            if n_particles_gate == 0:
                ax.text(0.5, 0.5, 'No Particles in GATE Phase Space', ha='center', va='center', transform=ax.transAxes)
                return
            normalized_dose_array = dose_array / n_particles_gate
            heatmap_2d = np.sum(normalized_dose_array, axis=0)
            if np.sum(heatmap_2d) == 0:
                ax.text(0.5, 0.5, 'No Energy Deposited', ha='center', va='center', transform=ax.transAxes)
                return
            size = dose_image.GetSize()
            spacing = dose_image.GetSpacing()
            origin = dose_image.GetOrigin()
            x_min = origin[0] - 0.5 * spacing[0]
            y_min = origin[1] - 0.5 * spacing[1]
            x_max = x_min + size[0] * spacing[0]
            y_max = y_min + size[1] * spacing[1]
            extent = [x_min, x_max, y_min, y_max]
            im = ax.imshow(heatmap_2d.T, origin='lower', cmap='OrRd', norm=norm, extent=extent)
            ax.set_xlabel('X position (mm)')
            ax.set_ylabel('Y position (mm)')
        else:
            ax.text(0.5, 0.5, 'Unsupported Simulation Type', ha='center', va='center', transform=ax.transAxes)
            return
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Absorbed Dose (Gy/primary, log scale)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
    except Exception as e:
        print(f"Error plotting energy deposition from {file_path}: {e}")
        ax.text(0.5, 0.5, f'Error: {str(e)[:40]}...', ha='center', va='center', transform=ax.transAxes)


def plot_interaction_dashboard(df: pd.DataFrame, title: str, output_path: Path, thickness: float, source_dir: Path):
    categorized_dfs = categorize_interactions(df)
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(f'{title} (Phantom Thickness: {thickness} cm)', fontsize=20, weight='bold')
    plot_scatter_on_ax(axs[0, 0], categorized_dfs["Unscattered"], "Unscattered")
    plot_scatter_on_ax(axs[0, 1], categorized_dfs["Rayleigh"], "Rayleigh Scattering")
    plot_scatter_on_ax(axs[1, 0], categorized_dfs["Compton"], "Compton Scattering")
    axs[1, 1].set_title('Energy Deposition', fontsize=14, weight='bold')
    axs[1, 1].set_aspect('equal')
    if "topas" in title.lower():
        plot_energy_deposition(axs[1, 1], source_dir / 'phantom_dose.csv', None, 'topas')
    elif "gate" in title.lower():
        plot_energy_deposition(axs[1, 1], source_dir / 'energy_deposition-Dose.mhd', source_dir / 'phantom_transmission_TEMPLATE.root', 'gate')
    fig.subplots_adjust(hspace=0.25, wspace=0.3, top=0.93, bottom=0.05, left=0.07, right=0.95)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Interaction dashboard saved to: {output_path}")


def plot_summary_attenuation(summary_data: list, output_image: Path):
    if not summary_data:
        return
    df = pd.DataFrame(summary_data).sort_values(by="thickness")
    plt.figure(figsize=(12, 8))
    plt.plot(df['thickness'], df['unscattered'], 'o-', label='Unscattered')
    plt.plot(df['thickness'], df['rayleigh'], 's-', label='Rayleigh')
    plt.plot(df['thickness'], df['compton'], '^-', label='Compton')
    if 'transmitted' in df.columns:
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


def plot_counts_comparison(topas_results: list, gate_results: list, output_image: Path):
    if not topas_results or not gate_results:
        print("Not enough data to create the comparison chart.")
        return
    topas_df = pd.DataFrame(topas_results).sort_values(by="thickness")
    gate_df = pd.DataFrame(gate_results).sort_values(by="thickness")
    comp_df = pd.merge(topas_df, gate_df, on="thickness", suffixes=('_topas', '_gate'))
    thicknesses = comp_df['thickness'].unique()
    interactions = ['unscattered', 'rayleigh', 'compton']
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('Comparison of Particle Interactions: TOPAS vs. GATE', fontsize=16, weight='bold')
    for i, interaction in enumerate(interactions):
        ax = axs[i]
        x = np.arange(len(thicknesses))
        width = 0.35
        ax.bar(x - width/2, comp_df[f'{interaction}_topas'], width, label='TOPAS', color='b', alpha=0.7)
        ax.bar(x + width/2, comp_df[f'{interaction}_gate'], width, label='gAte', color='g', alpha=0.7)
        ax.set_title(interaction.capitalize(), fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(thicknesses, rotation=45, ha='right')
        ax.set_xlabel('Phantom Thickness (cm)')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    axs[0].set_ylabel('Number of Particles')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image)
    print(f"✅ Comparison chart saved to {output_image}")
    plt.close()


def plot_cross_section_comparison(topas_df: pd.DataFrame, gate_df: pd.DataFrame, thickness: float, output_path: Path, y_window_mm=10.0):
    plt.figure(figsize=(10, 7))
    plt.title(f'Cross-Section Particle Count Comparison (Thickness: {thickness} cm)', fontsize=16, weight='bold')
    plt.xlabel('X Position (mm)', fontsize=12)
    plt.ylabel('Particle Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    x_bins = np.linspace(-120, 120, 200)
    if topas_df is not None and not topas_df.empty:
        topas_df = filter_gamma_particles(topas_df)
    if gate_df is not None and not gate_df.empty:
        gate_df = filter_gamma_particles(gate_df)
    if topas_df is not None and not topas_df.empty:
        topas_df['X_mm'] = topas_df['X_cm'] * 10
        topas_df['Y_mm'] = topas_df['Y_cm'] * 10
        topas_cross_section = topas_df[(topas_df['Y_mm'] > -y_window_mm/2) & (topas_df['Y_mm'] < y_window_mm/2)]
        hist, _ = np.histogram(topas_cross_section['X_mm'], bins=x_bins)
        centers = x_bins[:-1] + np.diff(x_bins)/2
        hist = np.where(hist > 0, hist, np.nan)
        plt.plot(centers, hist, '-', linewidth=0.8, label='TOPAS (gamma)', alpha=0.95, color='black')
    else:
        print(f"No TOPAS data to plot for {thickness} cm.")
    if gate_df is not None and not gate_df.empty:
        gate_df['X_mm'] = gate_df['X_cm'] * 10
        gate_df['Y_mm'] = gate_df['Y_cm'] * 10
        gate_cross_section = gate_df[(gate_df['Y_mm'] > -y_window_mm/2) & (gate_df['Y_mm'] < y_window_mm/2)]
        hist, _ = np.histogram(gate_cross_section['X_mm'], bins=x_bins)
        centers = x_bins[:-1] + np.diff(x_bins)/2
        hist = np.where(hist > 0, hist, np.nan)
        plt.plot(centers, hist, '-', linewidth=0.8, label='GATE (gamma)', alpha=0.95, color='magenta')
    else:
        print(f"No GATE data to plot for {thickness} cm.")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Cross-section comparison plot saved to: {output_path}")


def plot_crosssection_sum_comparison(topas_df, gate_df, thickness, output_dir, plot_range_mm=150.0, bin_size_mm=1.0):
    if topas_df is None or gate_df is None or topas_df.empty or gate_df.empty:
        print(f"Skipping cross-section sum comparison for thickness {thickness} cm due to missing data.")
        return
    topas_gamma = filter_gamma_particles(topas_df)
    gate_gamma = filter_gamma_particles(gate_df)
    if topas_gamma.empty and gate_gamma.empty:
        print(f"No gamma particles for thickness {thickness} cm; skipping cross-section sum comparison.")
        return
    for d in (topas_gamma, gate_gamma):
        if d is not None and not d.empty:
            d['X_mm'] = d['X_cm'] * 10.0
            d['Y_mm'] = d['Y_cm'] * 10.0
    # Use aligned fixed bins so TOPAS and GATE are directly comparable
    edges = np.arange(-plot_range_mm, plot_range_mm + bin_size_mm, bin_size_mm)
    hist_range = [[-plot_range_mm, plot_range_mm], [-plot_range_mm, plot_range_mm]]
    # 2D histogram per dataset
    h_topas, _, _ = np.histogram2d(topas_gamma['X_mm'], topas_gamma['Y_mm'], bins=[edges, edges], range=hist_range) if not topas_gamma.empty else (np.zeros((len(edges)-1, len(edges)-1)), edges, edges)
    h_gate,  _, _ = np.histogram2d(gate_gamma['X_mm'],  gate_gamma['Y_mm'],  bins=[edges, edges], range=hist_range) if not gate_gamma.empty else (np.zeros((len(edges)-1, len(edges)-1)), edges, edges)
    # Sum over Y for each X bin (axis=1 because np.histogram2d returns shape (len(x_bins)-1, len(y_bins)-1))
    topas_x_sum = h_topas.sum(axis=1)
    gate_x_sum  = h_gate.sum(axis=1)
    # Sum over X for each Y bin (axis=0)
    topas_y_sum = h_topas.sum(axis=0)
    gate_y_sum  = h_gate.sum(axis=0)
    # Bin centers for plotting
    centers = edges[:-1] + np.diff(edges) / 2.0
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Gamma Cross-Section Totals (Thickness: {thickness} cm)', fontsize=16, weight='bold')
    # X direction totals
    x_top = np.where(topas_x_sum > 0, topas_x_sum, np.nan)
    x_gate = np.where(gate_x_sum > 0,  gate_x_sum,  np.nan)
    axs[0].plot(centers, x_top, '-', linewidth=0.8, label='TOPAS (gamma)', alpha=0.95, color='black')
    axs[0].plot(centers, x_gate, '-', linewidth=0.8, label='GATE (gamma)',  alpha=0.95, color='magenta')
    axs[0].set_xlabel('X Position (mm)')
    axs[0].set_ylabel('Total count over Y')
    axs[0].set_title('Total gamma count per X (sum over Y)')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    # Y direction totals
    y_top = np.where(topas_y_sum > 0, topas_y_sum, np.nan)
    y_gate = np.where(gate_y_sum > 0,  gate_y_sum,  np.nan)
    axs[1].plot(centers, y_top, '-', linewidth=0.8, label='TOPAS (gamma)', alpha=0.95, color='black')
    axs[1].plot(centers, y_gate, '-', linewidth=0.8, label='GATE (gamma)',  alpha=0.95, color='magenta')
    axs[1].set_xlabel('Y Position (mm)')
    axs[1].set_ylabel('Total count over X')
    axs[1].set_title('Total gamma count per Y (sum over X)')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / f"crosssection_sum_comparison_{thickness:.4f}cm.png", dpi=300)
    plt.close()
    print(f"✅ Cross-section sum comparison plot saved for thickness {thickness} cm.")


def plot_all_thicknesses_comparison(topas_results: dict, gate_results: dict, output_path: Path):
    thicknesses = sorted(list(topas_results.keys()))
    n_plots = len(thicknesses)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    fig.suptitle('Cross-Section Particle Distribution Comparison', fontsize=18, weight='bold')
    x_bins = np.linspace(-120, 120, 200)
    y_window_mm = 10.0
    for i, thickness in enumerate(thicknesses):
        ax = axs[i // ncols, i % ncols]
        topas_df = topas_results.get(thickness)
        gate_df = gate_results.get(thickness)
        ax.set_title(f'Thickness: {thickness} cm', fontsize=12)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Particle Count')
        ax.grid(True, linestyle='--', alpha=0.6)
        if topas_df is not None and not topas_df.empty:
            topas_df = filter_gamma_particles(topas_df)
        if gate_df is not None and not gate_df.empty:
            gate_df = filter_gamma_particles(gate_df)
        if topas_df is not None and not topas_df.empty:
            topas_df['X_mm'] = topas_df['X_cm'] * 10
            topas_df['Y_mm'] = topas_df['Y_cm'] * 10
            topas_cross_section = topas_df[(topas_df['Y_mm'] > -y_window_mm/2) & (topas_df['Y_mm'] < y_window_mm/2)]
            hist, _ = np.histogram(topas_cross_section['X_mm'], bins=x_bins)
            ax.plot(x_bins[:-1] + np.diff(x_bins)/2, hist, label='TOPAS (gamma)', alpha=0.7)
        if gate_df is not None and not gate_df.empty:
            gate_df['X_mm'] = gate_df['X_cm'] * 10
            gate_df['Y_mm'] = gate_df['Y_cm'] * 10
            gate_cross_section = gate_df[(gate_df['Y_mm'] > -y_window_mm/2) & (gate_df['Y_mm'] < y_window_mm/2)]
            hist, _ = np.histogram(gate_cross_section['X_mm'], bins=x_bins)
            ax.plot(x_bins[:-1] + np.diff(x_bins)/2, hist, label='GATE (gamma)', linestyle='--', alpha=0.7)
        ax.legend()
    for i in range(n_plots, nrows * ncols):
        fig.delaxes(axs[i // ncols, i % ncols])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ All-thickness cross-section comparison saved to: {output_path}")


def plot_heatmap_percentage_difference(topas_df, gate_df, thickness, output_path, bins=100, plot_range=150):
    if topas_df is None or gate_df is None or topas_df.empty or gate_df.empty:
        print(f"Skipping heatmap difference plot for thickness {thickness} cm due to missing data.")
        return
    topas_df = filter_gamma_particles(topas_df)
    gate_df = filter_gamma_particles(gate_df)
    if topas_df.empty or gate_df.empty:
        print(f"Skipping heatmap difference plot for thickness {thickness} cm due to no gamma events.")
        return
    topas_x = topas_df['X_cm'] * 10
    topas_y = topas_df['Y_cm'] * 10
    gate_x = gate_df['X_cm'] * 10
    gate_y = gate_df['Y_cm'] * 10
    bin_edges = np.linspace(-plot_range, plot_range, bins + 1)
    hist_range = [[-plot_range, plot_range], [-plot_range, plot_range]]
    h_topas, _, _ = np.histogram2d(topas_x, topas_y, bins=bin_edges, range=hist_range)
    h_gate, _, _ = np.histogram2d(gate_x, gate_y, bins=bin_edges, range=hist_range)
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_diff = 100 * (h_gate - h_topas) / (h_topas + 1e-9)
    percentage_diff[np.isnan(percentage_diff)] = 0
    percentage_diff[np.isinf(percentage_diff)] = 200
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = 100
    vmin = -100
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    im = ax.imshow(percentage_diff.T, origin='lower', cmap='coolwarm', norm=norm,
                   extent=[-plot_range, plot_range, -plot_range, plot_range])
    cbar = fig.colorbar(im, ax=ax, extend='both')
    cbar.set_label('Percentage Difference (%)\n[100 * (GATE - TOPAS) / TOPAS]')
    ax.set_title(f'Particle Count Difference: GATE vs. TOPAS (Thickness: {thickness} cm)', fontsize=16, weight='bold')
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✅ Heatmap difference plot saved to: {output_path}")


def main():
    global GLOBAL_VMAX
    thicknesses_cm = [0.0001, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    base_output_dir = Path("./simulation_runs")
    topas_results = []
    gate_results = []
    topas_dfs = {}
    gate_dfs = {}
    for thickness in thicknesses_cm:
        topas_dir = base_output_dir / f"topas_run_{thickness:.4f}cm"
        gate_dir = base_output_dir / f"gate_run_{thickness:.4f}cm"
        topas_dir.mkdir(parents=True, exist_ok=True)
        gate_dir.mkdir(parents=True, exist_ok=True)
        # run_topas_simulation(thickness, Path(TOPAS_BASE_MACRO_FILE), topas_dir)
        # run_gate_simulation(thickness, Path(GATE_BASE_MACRO_FILE), gate_dir)
    GLOBAL_VMAX = get_max_dose_value(base_output_dir, thicknesses_cm)
    if GLOBAL_VMAX == 0:
        print("\nWarning: No dose data found. Check simulation outputs. Setting a default global max value.")
        GLOBAL_VMAX = 2.5e-10
    print(f"\nDetermined global maximum dose for common scale: {GLOBAL_VMAX:.4e} Gy/primary")
    for thickness in thicknesses_cm:
        print(f"\n--- Analyzing thickness: {thickness} cm ---")
        topas_dir = base_output_dir / f"topas_run_{thickness:.4f}cm"
        gate_dir = base_output_dir / f"gate_run_{thickness:.4f}cm"
        topas_df = read_topas_phasespace(topas_dir / "MyTopasOutput.phsp")
        topas_dfs[thickness] = topas_df if topas_df is not None else pd.DataFrame()
        if topas_df is not None and not topas_df.empty:
            counts = categorize_interactions(topas_df)
            topas_results.append({
                "thickness": thickness,
                "unscattered": len(counts["Unscattered"]),
                "rayleigh": len(counts["Rayleigh"]),
                "compton": len(counts["Compton"]),
                "transmitted": len(topas_df)
            })
            plot_interaction_dashboard(
                topas_df, "TOPAS Interaction Analysis",
                topas_dir / f"dashboard_interactions_{thickness:.4f}cm.png",
                thickness, topas_dir
            )
        else:
            print(f"No TOPAS data found for {thickness} cm.")
        gate_df = read_gate_phasespace(gate_dir / "phantom_transmission_TEMPLATE.root")
        gate_dfs[thickness] = gate_df if gate_df is not None else pd.DataFrame()
        if gate_df is not None and not gate_df.empty:
            counts = categorize_interactions(gate_df)
            gate_results.append({
                "thickness": thickness,
                "unscattered": len(counts["Unscattered"]),
                "rayleigh": len(counts["Rayleigh"]),
                "compton": len(counts["Compton"]),
                "transmitted": len(gate_df)
            })
            plot_interaction_dashboard(
                gate_df, "GATE Interaction Analysis",
                gate_dir / f"dashboard_interactions_{thickness:.4f}cm.png",
                thickness, gate_dir
            )
        else:
            print(f"No GATE data found for {thickness} cm.")
        current_topas_df = topas_dfs[thickness]
        current_gate_df = gate_dfs[thickness]
        plot_cross_section_comparison(
            current_topas_df, current_gate_df, thickness,
            base_output_dir / f"cross_section_comparison_{thickness:.4f}cm.png"
        )
        plot_crosssection_sum_comparison(
            current_topas_df, current_gate_df, thickness, base_output_dir
        )
        plot_heatmap_percentage_difference(
            current_topas_df, current_gate_df, thickness,
            base_output_dir / f"heatmap_diff_{thickness:.4f}cm.png"
        )
    plot_summary_attenuation(topas_results, base_output_dir / "topas_summary_attenuation.png")
    plot_summary_attenuation(gate_results, base_output_dir / "gate_summary_attenuation.png")
    plot_counts_comparison(topas_results, gate_results, base_output_dir / "topas_gate_count_comparison.png")
    plot_all_thicknesses_comparison(topas_dfs, gate_dfs, base_output_dir / "all_thicknesses_cross_section_comparison.png")
    print("\nAutomation and comparison script finished.")


if __name__ == "__main__":
    main()

