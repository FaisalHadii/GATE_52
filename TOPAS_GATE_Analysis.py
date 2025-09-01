import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path
import subprocess
import matplotlib.colors as mcolors
import uproot
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION ---
TOPAS_EXECUTABLE_PATH = "/home/fhhadi/topas/bin/topas"
GATE_EXECUTABLE_PATH = "/home/fhhadi/gate_install/bin/Gate"
TOPAS_BASE_MACRO_FILE = "compined1.txt"
GATE_BASE_MACRO_FILE = "compined1.mac"

# --- GLOBAL VARIABLES FOR PLOTTING SCALE ---
GLOBAL_VMIN = 1e-12
GLOBAL_VMAX = None

# --- CONSTANTS FOR DOSE CONVERSION ---
ALUMINUM_DENSITY_KG_M3 = 2700.0 # Density of Aluminum
MEV_TO_JOULE = 1.602176634e-13

def get_max_dose_value(base_output_dir: Path, thicknesses_cm: list):
    """
    Analyzes the results to find the global maximum dose value for scaling.
    This version correctly handles normalization based on the actual number
    of particles in the output phase space.
    """
    max_val = 0
    for thickness in thicknesses_cm:
        # Check TOPAS data
        topas_dir = base_output_dir / f"topas_run_{thickness:.4f}cm"
        topas_dose_file = topas_dir / 'phantom_dose.csv'
        
        if topas_dose_file.exists():
            try:
                df = pd.read_csv(topas_dose_file, comment='#', skiprows=1, header=None, names=['X', 'Y', 'Z', 'Dose'])
                
                # Assume a fixed number of primaries from TOPAS macro
                n_particles_topas = 1e6 # You must adjust this to your macro setting
                
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
        
        # Check GATE data
        gate_dir = base_output_dir / f"gate_run_{thickness:.4f}cm"
        gate_dose_file = gate_dir / 'energy_deposition-Dose.mhd'
        gate_phsp_file = gate_dir / 'phantom_transmission_TEMPLATE.root'
        
        if gate_dose_file.exists() and gate_phsp_file.exists():
            try:
                dose_image = sitk.ReadImage(str(gate_dose_file))
                dose_array = sitk.GetArrayFromImage(dose_image)
                
                # Get the number of particles from the GATE phase space file
                # This is more robust as it accounts for attenuation
                with uproot.open(gate_phsp_file) as file:
                    n_particles_gate = file["PhaseSpace"].num_entries
                
                if n_particles_gate == 0:
                    continue
                
                # Normalize GATE dose by the number of particles that actually deposited energy
                normalized_dose = dose_array / n_particles_gate
                
                if normalized_dose.max() > max_val:
                    max_val = normalized_dose.max()
            except Exception as e:
                print(f"Warning: Could not read GATE file for max value calculation: {e}")
                
    return max_val

def run_topas_simulation(thickness_cm: float, base_macro_path: Path, output_dir: Path) -> bool:
    """Modifies, runs, and checks the status of a TOPAS simulation."""
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
    """Modifies, runs, and checks the status of a GATE simulation."""
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

# --- Data Reading and Analysis Functions ---
def read_topas_phasespace(phsp_file: Path) -> pd.DataFrame | None:
    """Reads a TOPAS binary phase space file and returns a DataFrame."""
    if not phsp_file.exists(): return None
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
    """Reads phasespace data from a ROOT file and returns a DataFrame."""
    if not phsp_file.exists(): return None
    try:
        with uproot.open(phsp_file) as file:
            df = file["PhaseSpace"].arrays(library="pd")
            df.rename(columns={'X': 'X_cm', 'Y': 'Y_cm', 'Ekine': 'Ekine', 'dX': 'dX', 'dY': 'dY'}, inplace=True)
            return df
    except Exception as e:
        print(f"Error reading GATE phase space file: {e}")
        return None

def categorize_interactions(df: pd.DataFrame) -> dict:
    """Categorizes particles by interaction type based on energy and scattering angle."""
    if df.empty: return {"Unscattered": pd.DataFrame(), "Rayleigh": pd.DataFrame(), "Compton": pd.DataFrame()}
    df['dZ'] = np.sqrt(1.0 - df['dX']**2 - df['dY']**2).fillna(0)
    df['ScatteringAngle'] = np.degrees(np.arccos(np.clip(df['dZ'], -1.0, 1.0)))
    return {
        "Unscattered": df[(df['Ekine'] > 0.149999) & (df['ScatteringAngle'] < 0.01)],
        "Rayleigh": df[(df['Ekine'] > 0.149999) & (df['ScatteringAngle'] >= 0.01)],
        "Compton": df[df['Ekine'] < 0.149999]
    }

# --- Plotting Functions ---

def plot_scatter_on_ax(ax, df: pd.DataFrame, title: str):
    """Helper function to plot scatter data on a given axis with correct scaling."""
    # MODIFIED: Define appropriate scale for unscattered particles based on 0.4 micrometer beam
    if "Unscattered" in title:
        # For 0.4 micrometer beam, we need much finer scale
        # 0.4 micrometer = 0.0004 mm = 0.00004 cm
        # Let's use a range that's 100x the beam size for visibility
        x_lim_mm = 0.04  # 0.04 mm = 40 micrometers
        y_lim_mm = 0.04  # 0.04 mm = 40 micrometers
        ax.set_xlim(-x_lim_mm, x_lim_mm)
        ax.set_ylim(-y_lim_mm, y_lim_mm)
    else:
        # For other plots, let the axis scale be determined by the data
        pass

    if df is None or df.empty:
        ax.set_title(f'{title} (0 events)', weight='bold', fontsize=14)
        ax.text(0.5, 0.5, 'No Events', ha='center', va='center', transform=ax.transAxes)
        return
    
    ax.set_title(f'{title} ({len(df):,} events)', weight='bold', fontsize=14)
    
    if 'Position_X' in df.columns:
        x_pos = df['Position_X'] * 10
        y_pos = df['Position_Y'] * 10
    else:
        x_pos = df['X_cm'] * 10  # Convert cm to mm
        y_pos = df['Y_cm'] * 10  # Convert cm to mm
    
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
    """Plots energy deposition data from a TOPAS or GATE output file."""
    if not file_path.exists():
        ax.text(0.5, 0.5, 'No Energy Deposition Data', ha='center', va='center', transform=ax.transAxes)
        return

    try:
        norm = mcolors.LogNorm(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)

        if simulation_type.lower() == 'topas':
            df = pd.read_csv(file_path, comment='#', skiprows=1, header=None, names=['X', 'Y', 'Z', 'Dose_MeV'])
            
            # Assuming a fixed number of primaries from TOPAS macro
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
    """
    Creates a dashboard with scatter plots for different interaction types,
    including a working energy deposition plot.
    """
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
    """Plots the final summary of interactions vs. thickness."""
    if not summary_data: return
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
    """Plots a bar chart comparing TOPAS and GATE particle counts."""
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
    """
    Plots a 2D line graph comparing the total number of particles along a cross-section
    for TOPAS and GATE for a given thickness.
    """
    plt.figure(figsize=(10, 7))
    plt.title(f'Cross-Section Particle Count Comparison (Thickness: {thickness} cm)', fontsize=16, weight='bold')
    plt.xlabel('X Position (mm)', fontsize=12)
    plt.ylabel('Particle Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Use a wider, more appropriate range for the bins, and more of them for higher resolution
    # Based on the plots, a range of +/- 120 mm seems appropriate
    x_bins = np.linspace(-120, 120, 200) # Increased bin count for better resolution
    y_window_cm = y_window_mm / 10.0

    # Process TOPAS data
    if not topas_df.empty:
        # This conversion is correct, but let's confirm the data types
        topas_df['X_mm'] = topas_df['X_cm'] * 10
        topas_df['Y_mm'] = topas_df['Y_cm'] * 10
        
        # Filter for the cross-section slice
        topas_cross_section = topas_df[(topas_df['Y_mm'] > -y_window_mm/2) & (topas_df['Y_mm'] < y_window_mm/2)]
        
        hist, _ = np.histogram(topas_cross_section['X_mm'], bins=x_bins)
        plt.plot(x_bins[:-1] + np.diff(x_bins)/2, hist, 'o-', label='TOPAS', alpha=0.7)
    else:
        print(f"No TOPAS data to plot for {thickness} cm.")

    # Process GATE data
    if not gate_df.empty:
        # Convert GATE data from cm to mm for consistency
        gate_df['X_mm'] = gate_df['X_cm'] * 10
        gate_df['Y_mm'] = gate_df['Y_cm'] * 10
        
        # Filter for the cross-section slice
        gate_cross_section = gate_df[(gate_df['Y_mm'] > -y_window_mm/2) & (gate_df['Y_mm'] < y_window_mm/2)]
        
        hist, _ = np.histogram(gate_cross_section['X_mm'], bins=x_bins)
        plt.plot(x_bins[:-1] + np.diff(x_bins)/2, hist, 's--', label='GATE', alpha=0.7)
    else:
        print(f"No GATE data to plot for {thickness} cm.")
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Cross-section comparison plot saved to: {output_path}")

def plot_all_thicknesses_comparison(topas_results: dict, gate_results: dict, output_path: Path):
    """
    Generates a multi-panel plot comparing TOPAS and GATE particle cross-sections
    for all thicknesses.
    """
    thicknesses = list(topas_results.keys())
    n_plots = len(thicknesses)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    fig.suptitle('Cross-Section Particle Distribution Comparison', fontsize=18, weight='bold')

    # Use a wider, more appropriate range for the bins
    x_bins = np.linspace(-120, 120, 200)
    y_window_mm = 10.0

    for i, thickness in enumerate(thicknesses):
        ax = axs[i // ncols, i % ncols]
        
        topas_df = topas_results[thickness]
        gate_df = gate_results[thickness]
        
        ax.set_title(f'Thickness: {thickness} cm', fontsize=12)
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Particle Count')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Process TOPAS data
        if not topas_df.empty:
            topas_df['X_mm'] = topas_df['X_cm'] * 10
            topas_df['Y_mm'] = topas_df['Y_cm'] * 10
            
            topas_cross_section = topas_df[(topas_df['Y_mm'] > -y_window_mm/2) & (topas_df['Y_mm'] < y_window_mm/2)]
            hist, _ = np.histogram(topas_cross_section['X_mm'], bins=x_bins)
            ax.plot(x_bins[:-1] + np.diff(x_bins)/2, hist, label='TOPAS', alpha=0.7)
        
        # Process GATE data
        if not gate_df.empty:
            gate_df['X_mm'] = gate_df['X_cm'] * 10
            gate_df['Y_mm'] = gate_df['Y_cm'] * 10
            
            gate_cross_section = gate_df[(gate_df['Y_mm'] > -y_window_mm/2) & (gate_df['Y_mm'] < y_window_mm/2)]
            hist, _ = np.histogram(gate_cross_section['X_mm'], bins=x_bins)
            ax.plot(x_bins[:-1] + np.diff(x_bins)/2, hist, label='GATE', linestyle='--', alpha=0.7)

        ax.legend()
        ax.set_yscale('log') # Use log scale for clarity
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ All-thickness cross-section comparison saved to: {output_path}")

def plot_all_energy_distributions_comparison(topas_dfs: dict, gate_dfs: dict, output_path: Path):
    """
    Generates a multi-panel plot comparing TOPAS and GATE particle energy distributions
    for all thicknesses.
    """
    thicknesses = list(topas_dfs.keys())
    n_plots = len(thicknesses)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    fig.suptitle('Particle Energy Distribution Comparison', fontsize=18, weight='bold')

    # Use a consistent energy binning for all plots
    energy_bins = np.linspace(0, 0.150, 100) # Assuming initial energy is 150 keV
    smoothing_sigma = 3 # Adjust this value to change the smoothing amount

    for i, thickness in enumerate(thicknesses):
        ax = axs[i // ncols, i % ncols]
        
        topas_df = topas_dfs[thickness]
        gate_df = gate_dfs[thickness]
        
        ax.set_title(f'Thickness: {thickness} cm', fontsize=12)
        ax.set_xlabel('Kinetic Energy (MeV)')
        ax.set_ylabel('Particle Count')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Process TOPAS data
        if not topas_df.empty:
            hist, _ = np.histogram(topas_df['Ekine'], bins=energy_bins)
            smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=smoothing_sigma)
            ax.plot(energy_bins[:-1] + np.diff(energy_bins)/2, smoothed_hist, label='TOPAS', alpha=0.7)
        
        # Process GATE data
        if not gate_df.empty:
            hist, _ = np.histogram(gate_df['Ekine'], bins=energy_bins)
            smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=smoothing_sigma)
            ax.plot(energy_bins[:-1] + np.diff(energy_bins)/2, smoothed_hist, label='GATE', linestyle='--', alpha=0.7)

        ax.legend()
        ax.set_yscale('log') # Use log scale for clarity
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ All-thickness energy distribution comparison saved to: {output_path}")

def main():
    """Main function to orchestrate the simulation and analysis workflow."""
    global GLOBAL_VMAX

    thicknesses_cm = [0.0001, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    base_output_dir = Path("./simulation_runs")
    
    topas_results = []
    gate_results = []
    
    # Dictionaries to store DataFrames for all-thickness comparison
    topas_dfs = {}
    gate_dfs = {}
    
    for thickness in thicknesses_cm:
        topas_dir = base_output_dir / f"topas_run_{thickness:.4f}cm"
        gate_dir = base_output_dir / f"gate_run_{thickness:.4f}cm"
        topas_dir.mkdir(parents=True, exist_ok=True)
        gate_dir.mkdir(parents=True, exist_ok=True)
        run_topas_simulation(thickness, Path(TOPAS_BASE_MACRO_FILE), topas_dir)
        run_gate_simulation(thickness, Path(GATE_BASE_MACRO_FILE), gate_dir)

    GLOBAL_VMAX = get_max_dose_value(base_output_dir, thicknesses_cm)
    if GLOBAL_VMAX == 0:
        print("\nWarning: No dose data found. Check simulation outputs. Setting a default global max value.")
        GLOBAL_VMAX = 2.5e-10

    print(f"\nDetermined global maximum dose for common scale: {GLOBAL_VMAX:.4e} Gy/primary")

    for thickness in thicknesses_cm:
        topas_dir = base_output_dir / f"topas_run_{thickness:.4f}cm"
        gate_dir = base_output_dir / f"gate_run_{thickness:.4f}cm"

        topas_df = read_topas_phasespace(topas_dir / "MyTopasOutput.phsp")
        topas_dfs[thickness] = topas_df
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
                topas_df,
                "TOPAS Interaction Analysis",
                topas_dir / f"dashboard_interactions_{thickness:.4f}cm.png",
                thickness,
                topas_dir
            )
        else:
            print(f"No TOPAS data found for {thickness} cm.")
            topas_df = pd.DataFrame()

        gate_df = read_gate_phasespace(gate_dir / "phantom_transmission_TEMPLATE.root")
        gate_dfs[thickness] = gate_df
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
                gate_df,
                "GATE Interaction Analysis",
                gate_dir / f"dashboard_interactions_{thickness:.4f}cm.png",
                thickness,
                gate_dir
            )
        else:
            print(f"No GATE data found for {thickness} cm.")
            gate_df = pd.DataFrame()

        # Call the new cross-section plot for each thickness
        plot_cross_section_comparison(
            topas_df,
            gate_df,
            thickness,
            base_output_dir / f"cross_section_comparison_{thickness:.4f}cm.png"
        )
        
    plot_summary_attenuation(topas_results, base_output_dir / "topas_summary_attenuation.png")
    plot_summary_attenuation(gate_results, base_output_dir / "gate_summary_attenuation.png")
    
    plot_counts_comparison(topas_results, gate_results, base_output_dir / "topas_gate_count_comparison.png")
    
    # Call the new function to plot all thicknesses on one graph
    plot_all_thicknesses_comparison(topas_dfs, gate_dfs, base_output_dir / "all_thicknesses_cross_section_comparison.png")
    plot_all_energy_distributions_comparison(topas_dfs, gate_dfs, base_output_dir / "all_thicknesses_energy_distribution_comparison.png")

    print("\nAutomation and comparison script finished.")

if __name__ == "__main__":
    main()