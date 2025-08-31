import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from pathlib import Path
import subprocess
import shutil
from typing import Optional, Tuple

try:
    import uproot
except Exception as import_error:  # noqa: F401
    uproot = None


# --- CONFIGURATION ---
TOPAS_EXECUTABLE_PATH = "/home/fhhadi/topas/bin/topas"
GATE_EXECUTABLE_PATH = "/home/fhhadi/gate_install/bin/Gate"
TOPAS_BASE_MACRO_FILE = "compined1.txt"
GATE_BASE_MACRO_FILE = "compined1.mac"


def _re_sub_or_append(pattern: str, replacement: str, content: str) -> str:
    """
    Replace first match using regex. If none, DO NOT append replacement blindly
    (since replacement may contain backrefs). This helper is now used only where
    replacement does not use backrefs or where a match is guaranteed.
    """
    new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    return new_content


def _set_or_add_line(content: str, key: str, value: str) -> str:
    """
    Set a line by key to a given value.
    - For TOPAS-style keys (not starting with '/'), use 'key = value'.
    - For GATE-style commands (keys starting with '/'), use 'key value'.
    """
    if key.startswith('/'):
        # GATE syntax: "/path/to/cmd value"
        line = f"{key} {value}"
        # Match any existing line that starts with this command
        pattern = rf"^{re.escape(key)}\b.*$"
    else:
        # TOPAS syntax: "key = value"
        line = f"{key} = {value}"
        pattern = rf"^{re.escape(key)}\s*=.*$"

    if re.search(pattern, content, flags=re.MULTILINE):
        return re.sub(pattern, line, content, flags=re.MULTILINE)
    # Append safely
    if not content.endswith("\n"):
        content += "\n"
    return content + line + "\n"


def run_topas_simulation_vacuum(
    thickness_cm: float,
    source_size_cm: float,
    base_macro_path: Path,
    output_dir: Path,
) -> bool:
    """
    Modifies TOPAS macro for a vacuum phantom and runs the simulation.

    Key fixes vs the original:
    - Keep the correct source model keys used in the base macro
    - Set source size using BeamPositionCutoffX/Y in mm (half-length)
    - Force pencil beam by setting AngularCutoffX/Y = 0.0 rad
    - Update output files to the given output_dir
    """
    print("-" * 60, f"\nRunning TOPAS with vacuum for {thickness_cm} cm\n", "-" * 60)

    temp_macro_path = output_dir / "temp_topas_vacuum.txt"
    phantom_half_thickness_cm = thickness_cm / 2.0
    # Convert source half size to mm (TOPAS expects mm for cutoff X/Y in this macro)
    source_half_length_cm = source_size_cm / 2.0
    source_half_length_mm = source_half_length_cm * 10.0

    if not base_macro_path.exists():
        print(f"Error: Base macro file not found: {base_macro_path}")
        return False

    content = base_macro_path.read_text()

    # Phantom thickness (HLZ is half-length)
    content = _set_or_add_line(content, "d:Ge/Phantom/HLZ", f"{phantom_half_thickness_cm} cm")

    # Source position distribution: Flat rectangle with specified half sizes (mm)
    content = _set_or_add_line(content, "s:So/XRaySource/BeamPositionDistribution", '"Flat"')
    content = _set_or_add_line(content, "s:So/XRaySource/BeamPositionCutoffShape", '"Rectangle"')
    content = _set_or_add_line(content, "d:So/XRaySource/BeamPositionCutoffX", f"{source_half_length_mm} mm")
    content = _set_or_add_line(content, "d:So/XRaySource/BeamPositionCutoffY", f"{source_half_length_mm} mm")

    # Force pencil beam (no divergence). Use a tiny non-zero cutoff to avoid TOPAS errors with 0.
    tiny_ang_rad = 1.0e-6
    content = _set_or_add_line(content, "s:So/XRaySource/BeamAngularDistribution", '"Flat"')
    content = _set_or_add_line(content, "d:So/XRaySource/BeamAngularCutoffX", f"{tiny_ang_rad} rad")
    content = _set_or_add_line(content, "d:So/XRaySource/BeamAngularCutoffY", f"{tiny_ang_rad} rad")

    # Outputs
    content = _set_or_add_line(content, "s:Sc/DoseInPhantom/OutputFile", f'"{output_dir}/phantom_dose"')
    content = _set_or_add_line(content, "s:Sc/PhaseSpace/OutputFile", f'"{output_dir}/MyTopasOutput"')

    temp_macro_path.write_text(content)

    try:
        cp = subprocess.run(
            [TOPAS_EXECUTABLE_PATH, str(temp_macro_path)],
            check=True,
            capture_output=True,
            text=True,
            errors="ignore",
        )
        if cp.stdout:
            print(cp.stdout[:5000])
        print("TOPAS simulation with vacuum completed successfully.")
        print(f"TOPAS macro used: {temp_macro_path}")
        return True
    except subprocess.CalledProcessError as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        print("TOPAS simulation failed. See outputs below.")
        print(f"TOPAS macro used: {temp_macro_path}")
        if stdout:
            print("--- STDOUT (first 5000 chars) ---\n" + stdout[:5000])
        if stderr:
            print("--- STDERR (first 5000 chars) ---\n" + stderr[:5000])
    except FileNotFoundError:
        print(f"Error: TOPAS executable not found at '{TOPAS_EXECUTABLE_PATH}'")
    return False


def run_gate_simulation_vacuum(
    thickness_cm: float,
    source_size_cm: float,
    base_macro_path: Path,
    output_dir: Path,
) -> bool:
    """
    Modifies GATE macro for a vacuum phantom and runs the simulation.

    Key fixes vs the original:
    - Update phantom Z length
    - Update output path alias
    - Use GPS Plane+Square with pos/halfx,y in mm (half-length)
    - Force pencil beam by setting ang/type iso with 0–0 deg
    """
    print("-" * 60, f"\nRunning GATE with vacuum for {thickness_cm} cm\n", "-" * 60)

    temp_macro_path = output_dir / "temp_gate_vacuum.mac"
    source_half_length_cm = source_size_cm / 2.0
    source_half_length_mm = source_half_length_cm * 10.0

    if not base_macro_path.exists():
        print(f"Error: Base macro file not found: {base_macro_path}")
        return False

    content = base_macro_path.read_text()

    # Ensure both possible output locations exist to avoid ROOT open errors
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        Path("./output").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Phantom thickness
    content = _set_or_add_line(content, "/gate/phantom/geometry/setZLength", f"{thickness_cm} cm")

    # Output path alias
    content = _set_or_add_line(content, "/control/alias GateOutputPath", f"{output_dir}/")

    # Determine source name
    m = re.search(r"/gate/source/addSource\s+(\w+)", content)
    source_name = m.group(1) if m else "xray_source"

    # GPS position: Plane + Square, half-lengths in mm
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/pos/type", "Plane")
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/pos/shape", "Square")
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/pos/halfx", f"{source_half_length_mm} mm")
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/pos/halfy", f"{source_half_length_mm} mm")

    # GPS angular: pencil beam
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/ang/type", "iso")
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/mintheta", "0. deg")
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/maxtheta", "0. deg")
    # Also set the newer 'ang/mintheta' and 'ang/maxtheta' to avoid deprecation warnings
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/ang/mintheta", "0. deg")
    content = _set_or_add_line(content, f"/gate/source/{source_name}/gps/ang/maxtheta", "0. deg")

    temp_macro_path.write_text(content)

    try:
        cp = subprocess.run(
            [GATE_EXECUTABLE_PATH, str(temp_macro_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        if cp.stdout:
            print(cp.stdout[:5000])
        print("GATE simulation with vacuum completed successfully.")
        print(f"GATE macro used: {temp_macro_path}")

        # If GATE still wrote into ./output, copy expected files back to output_dir
        legacy = Path("./output")
        for fname in [
            "source_check.root",
            "phantom_transmission_TEMPLATE.root",
            "energy_deposition.mhd",
        ]:
            src = legacy / fname
            dst = output_dir / fname
            try:
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            except Exception:
                pass
        return True
    except subprocess.CalledProcessError as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        print("GATE simulation failed. See outputs below.")
        print(f"GATE macro used: {temp_macro_path}")
        if stdout:
            print("--- STDOUT (first 5000 chars) ---\n" + stdout[:5000])
        if stderr:
            print("--- STDERR (first 5000 chars) ---\n" + stderr[:5000])
    except FileNotFoundError:
        print(f"Error: GATE executable not found at '{GATE_EXECUTABLE_PATH}'")
    return False


def read_topas_phasespace(phsp_file: Path) -> Optional[pd.DataFrame]:
    """Reads a TOPAS binary phase space file and returns a DataFrame.

    Assumes positions are stored in mm.
    """
    if not phsp_file.exists():
        print(f"TOPAS phase-space file not found: {phsp_file}")
        return None
    try:
        dtype = np.dtype(
            [
                ("Position_X", "f4"),
                ("Position_Y", "f4"),
                ("Position_Z", "f4"),
                ("Direction_X", "f4"),
                ("Direction_Y", "f4"),
                ("Ekine", "f4"),
                ("Weight", "f4"),
                ("Particle_Type", "i4"),
                ("Direction_Z_Sign", "b1"),
                ("Is_First_Particle", "b1"),
                ("Track_ID", "i4"),
                ("Parent_ID", "i4"),
                ("Charge", "f4"),
            ]
        )
        data = np.fromfile(phsp_file, dtype=dtype)
        if data.size == 0:
            print("TOPAS phase-space has zero records.")
            return None
        df = pd.DataFrame(data)
        df["X_mm"], df["Y_mm"] = df["Position_X"], df["Position_Y"]
        return df
    except Exception as e:
        print(f"Error reading TOPAS phase space file: {e}")
        return None


def read_gate_phasespace(phsp_file: Path) -> Optional[pd.DataFrame]:
    """Reads GATE PhaseSpace ROOT file and returns a DataFrame. Assumes mm units."""
    if not phsp_file.exists():
        print(f"GATE phase-space file not found: {phsp_file}")
        return None
    if uproot is None:
        print("Error: uproot is not installed. Install with 'pip install uproot'.")
        return None
    try:
        with uproot.open(phsp_file) as file:
            if "PhaseSpace" not in file:
                print("PhaseSpace tree not found in ROOT file.")
                return None
            df = file["PhaseSpace"].arrays(library="pd")

        # Determine which columns contain positions (mm)
        if {"X", "Y"}.issubset(df.columns):
            df.rename(columns={"X": "X_mm", "Y": "Y_mm"}, inplace=True)
        elif {"PostPosition_X", "PostPosition_Y"}.issubset(df.columns):
            df.rename(
                columns={"PostPosition_X": "X_mm", "PostPosition_Y": "Y_mm"},
                inplace=True,
            )
        elif {"PrePosition_X", "PrePosition_Y"}.issubset(df.columns):
            df.rename(
                columns={"PrePosition_X": "X_mm", "PrePosition_Y": "Y_mm"},
                inplace=True,
            )
        else:
            print("Could not find X/Y columns in GATE PhaseSpace tree.")
            return None

        return df[["X_mm", "Y_mm"]]
    except Exception as e:
        print(f"Error reading GATE phase space file: {e}")
        return None


def _compute_histogram_bins_um(
    x_um_a: Optional[np.ndarray], x_um_b: Optional[np.ndarray]
) -> Tuple[Tuple[float, float], int]:
    """Compute a sensible symmetric range and number of bins in micrometers."""
    arrays = [arr for arr in [x_um_a, x_um_b] if arr is not None and arr.size > 0]
    if not arrays:
        return (-2000.0, 2000.0), 400  # default: ±2 mm with 10 µm bins

    data = np.concatenate(arrays)
    p1, p99 = np.percentile(data, [0.5, 99.5])
    max_abs = max(abs(p1), abs(p99))
    span = max_abs * 1.2 if max_abs > 0 else 100.0
    lo, hi = -span, span

    # Choose around ~400 bins but not too fine
    desired_bins = 400
    n_bins = max(100, int(desired_bins))
    return (lo, hi), n_bins


def plot_field_profile(
    topas_df: Optional[pd.DataFrame],
    gate_df: Optional[pd.DataFrame],
    output_path: Path,
    thickness: float,
):
    """
    Plots and compares the X and Y field profiles for TOPAS and GATE.
    Positions are assumed in mm; plotted in µm.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        f"Particle Field Profile Comparison (Vacuum, {thickness} cm)",
        fontsize=16,
        weight="bold",
    )

    # Prepare arrays in µm
    topas_x_um = (
        (topas_df["X_mm"].to_numpy() * 1000.0) if (topas_df is not None and not topas_df.empty) else None
    )
    topas_y_um = (
        (topas_df["Y_mm"].to_numpy() * 1000.0) if (topas_df is not None and not topas_df.empty) else None
    )
    gate_x_um = (
        (gate_df["X_mm"].to_numpy() * 1000.0) if (gate_df is not None and not gate_df.empty) else None
    )
    gate_y_um = (
        (gate_df["Y_mm"].to_numpy() * 1000.0) if (gate_df is not None and not gate_df.empty) else None
    )

    bin_range, n_bins = _compute_histogram_bins_um(topas_x_um, gate_x_um)

    # --- X-axis Profile ---
    ax_x = axs[0]
    ax_x.set_title("X-axis Profile", fontsize=14)
    ax_x.set_xlabel(r"X Position (µm)", fontsize=12)
    ax_x.set_ylabel("Particle Count", fontsize=12)
    ax_x.grid(True, linestyle="--", alpha=0.6)

    if topas_x_um is not None:
        hist_x_t, bins = np.histogram(topas_x_um, bins=n_bins, range=bin_range)
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax_x.plot(centers, hist_x_t, "o-", label="TOPAS", alpha=0.7)

    if gate_x_um is not None:
        hist_x_g, bins = np.histogram(gate_x_um, bins=n_bins, range=bin_range)
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax_x.plot(centers, hist_x_g, "s--", label="GATE", alpha=0.7)

    # --- Y-axis Profile --- (use same binning for visual parity)
    ax_y = axs[1]
    ax_y.set_title("Y-axis Profile", fontsize=14)
    ax_y.set_xlabel(r"Y Position (µm)", fontsize=12)
    ax_y.set_ylabel("Particle Count", fontsize=12)
    ax_y.grid(True, linestyle="--", alpha=0.6)

    # Reuse X bins range for Y for a consistent scale
    if topas_y_um is not None:
        hist_y_t, bins = np.histogram(topas_y_um, bins=n_bins, range=bin_range)
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax_y.plot(centers, hist_y_t, "o-", label="TOPAS", alpha=0.7)

    if gate_y_um is not None:
        hist_y_g, bins = np.histogram(gate_y_um, bins=n_bins, range=bin_range)
        centers = 0.5 * (bins[1:] + bins[:-1])
        ax_y.plot(centers, hist_y_g, "s--", label="GATE", alpha=0.7)

    handles, labels = ax_x.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Field profile comparison plot saved to: {output_path}")


def main():
    """Run simulations and plot the vacuum field profile with corrected settings."""
    thickness_cm = 10.0
    # Full source size (not half) in cm. Example: 0.00001 cm = 0.1 µm
    source_size_cm = 0.00001
    base_output_dir = Path("./vacuum_field_profile")

    # Clean output
    if base_output_dir.exists():
        shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    topas_success = run_topas_simulation_vacuum(
        thickness_cm, source_size_cm, Path(TOPAS_BASE_MACRO_FILE), base_output_dir
    )
    gate_success = run_gate_simulation_vacuum(
        thickness_cm, source_size_cm, Path(GATE_BASE_MACRO_FILE), base_output_dir
    )

    topas_df = None
    if topas_success:
        phsp_topas = base_output_dir / "MyTopasOutput.phsp"
        topas_df = read_topas_phasespace(phsp_topas)
        if topas_df is None:
            print("Failed to read TOPAS phase space data.")
            topas_df = pd.DataFrame()

    gate_df = None
    if gate_success:
        phsp_gate = base_output_dir / "phantom_transmission_TEMPLATE.root"
        gate_df = read_gate_phasespace(phsp_gate)
        if gate_df is None:
            print("Failed to read GATE phase space data.")
            gate_df = pd.DataFrame()

    plot_field_profile(
        topas_df,
        gate_df,
        base_output_dir / "topas_gate_field_profile_comparison.png",
        thickness_cm,
    )

    print("\nScript finished. Check the output directory for the results.")
    print(f"Output files are located in: {base_output_dir.resolve()}")


if __name__ == "__main__":
    main()

