"""
torchcor/ecg/kcl_pipeline.py
=============================
Full KCL whole-torso cardiac simulation pipeline:

  KCL anatomy VTU  +  CARP mesh
        │
        ▼  kcl_extract.py  (one-time)
  Simulation-ready CARP sub-meshes
        │
        ▼  Monodomain  (torchcor.simulator)
  Transmembrane potential  Vm(t)  on biventricular mesh
        │
        ▼  ECG Forward solve  (lead field method, torchcor.ecg.test)
  12-lead ECG  +  clinical layout plot

KCL CARP tag convention
-----------------------
  1 – 25 : anatomy
  26 – 33: ICD hardware  (excluded)
  34     : RV wall   (bidomain)
  35     : LV wall   (bidomain)
  24     : RV blood pool  (isotropic)
  25     : LV blood pool  (isotropic)

Usage
-----
    python -m torchcor.ecg.kcl_pipeline
    python -m torchcor.ecg.kcl_pipeline --device cuda:0 --T 600 --dt 0.05
    python -m torchcor.ecg.kcl_pipeline --skip-extract --skip-sim
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from torchcor.ecg.torso_tags import CONDUCTIVITY   # tags 1-25 isotropic (S/m)

# ── Default paths ──────────────────────────────────────────────────────────────
_KCL_DIR        = Path(r"C:\Users\bulli\cardiac_data\kcl_torso\KCL_torso3")
DEFAULT_SIM_DIR = _KCL_DIR / "sim"
DEFAULT_RESULT  = DEFAULT_SIM_DIR / "results"

# Bidomain conductivities for ventricular myocardium (Hooks et al. 2007 / demo values)
MILLI_VENTRICULAR_ENDO_IL = 0.5272   # S/m  intracellular longitudinal
MILLI_VENTRICULAR_ENDO_IT = 0.2076   # S/m  intracellular transverse
MILLI_VENTRICULAR_EPI_EL  = 1.0732   # S/m  extracellular longitudinal
MILLI_VENTRICULAR_EPI_ET  = 0.4227   # S/m  extracellular transverse

# Blood-pool conductivity in bidomain context (isotropic, high)
SIGMA_BLOOD = 0.667


def _device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ==============================================================================
#  Step 1 – Extract simulation-ready CARP meshes
# ==============================================================================

def step_extract(sim_dir: Path, force: bool = False) -> None:
    from torchcor.ecg.kcl_extract import extract, DEFAULT_CARP_PREFIX, DEFAULT_VTU
    extract(
        carp_prefix=DEFAULT_CARP_PREFIX,
        vtu_path=DEFAULT_VTU,
        out_dir=sim_dir,
        force=force,
    )


# ==============================================================================
#  Step 2 – Monodomain simulation on biventricular mesh
# ==============================================================================

def step_simulate(
    sim_dir: Path,
    result_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    T: float = 600.0,
    dt: float = 0.05,
    snapshot_interval: int = 1,
    force: bool = False,
) -> torch.Tensor:
    """
    Run monodomain simulation on the extracted biventricular mesh.

    Returns
    -------
    Vm : (n_snapshots, n_nodes) tensor  (or loaded from cache)
    """
    import torchcor as tc
    from torchcor.simulator import Monodomain
    from torchcor.ionic import TenTusscherPanfilov

    vm_cache = result_dir / "Vm.pt"
    if vm_cache.exists() and not force:
        print(f"Loading cached Vm from {vm_cache}")
        return torch.load(str(vm_cache), map_location=device)

    result_dir.mkdir(parents=True, exist_ok=True)
    vent_dir   = sim_dir / "ventricles"
    pacing_dir = vent_dir / "pacing"

    print("\n" + "=" * 60)
    print(f"[Monodomain] T={T} ms, dt={dt} ms, device={device}")
    print("=" * 60)

    tc.set_device(str(device))

    # Ionic model: Ten Tusscher-Panfilov 2006, ENDO cell type
    im = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)

    simulator = Monodomain(ionic_models=[im], T=T, dt=dt, dtype=dtype)
    simulator.load_mesh(path=str(vent_dir), unit_conversion=1000)  # µm on disk → mm in sim

    # Always use all four conductivity parameters (il, it, el, et).
    # Calling add_conductivity with only il/it sets sigma_il=None internally,
    # which breaks any subsequent bidomain call. Use el=et=il=it for isotropic.
    #
    # Tags 34, 35 = ventricular walls (bidomain — must come first)
    simulator.add_conductivity(
        [34, 35],
        il=MILLI_VENTRICULAR_ENDO_IL,
        it=MILLI_VENTRICULAR_ENDO_IT,
        el=MILLI_VENTRICULAR_EPI_EL,
        et=MILLI_VENTRICULAR_EPI_ET,
    )
    # Tags 24, 25 = blood pools (isotropic — use 4-param form to avoid nullifying sigma_il)
    simulator.add_conductivity(
        [24, 25],
        il=SIGMA_BLOOD, it=SIGMA_BLOOD,
        el=SIGMA_BLOOD, et=SIGMA_BLOOD,
    )

    # Pacing: LV endocardium at t=0, RV endocardium at t=5 ms
    lv_vtx = pacing_dir / "LV_endo.vtx"
    rv_vtx = pacing_dir / "RV_endo.vtx"

    if lv_vtx.exists():
        simulator.add_stimulus(str(lv_vtx), start=0.0, duration=2.0, intensity=80.0)
        print(f"  LV pacing: {lv_vtx.name}")
    else:
        print(f"  WARNING: {lv_vtx} not found — no LV stimulus applied")

    if rv_vtx.exists():
        simulator.add_stimulus(str(rv_vtx), start=5.0, duration=2.0, intensity=80.0)
        print(f"  RV pacing: {rv_vtx.name}")
    else:
        print(f"  WARNING: {rv_vtx} not found — no RV stimulus applied")

    t0 = time.time()
    Vm = simulator.solve(
        a_tol=1e-5,
        r_tol=1e-5,
        max_iter=200,
        snapshot_interval=snapshot_interval,
        verbose=True,
        result_path=str(result_dir / "monodomain"),
    )
    dt_wall = time.time() - t0
    print(f"\nSimulation done in {dt_wall:.1f} s  ({dt_wall/60:.1f} min)")

    # Compute activation / repolarisation maps
    ATs = simulator.compute_activation_map(
        Vm=Vm, snapshot_interval=snapshot_interval, threshold=0)
    RTs = simulator.compute_repolarization_map(
        Vm=Vm, snapshot_interval=snapshot_interval, threshold=-70)
    print(f"  AT  range: {ATs.min().item():.1f} – {ATs.max().item():.1f} ms")
    print(f"  RT  range: {RTs.min().item():.1f} – {RTs.max().item():.1f} ms")

    # Save AT / RT maps for visualisation
    np.save(str(result_dir / "activation_times.npy"),    ATs.cpu().numpy())
    np.save(str(result_dir / "repolarization_times.npy"), RTs.cpu().numpy())
    print(f"  Saved AT/RT maps → {result_dir}")

    # Save Vm snapshot series
    torch.save(Vm, str(vm_cache))
    print(f"  Saved Vm ({Vm.shape}) → {vm_cache}")

    # Write VTK visualisation files (every 10 snapshots)
    try:
        simulator.vm_to_vtk(Vm=Vm, step=10)
    except Exception as e:
        print(f"  (VTK export skipped: {e})")

    return Vm


# ==============================================================================
#  Step 3 – ECG forward solve  (lead field method)
# ==============================================================================

def step_ecg(
    sim_dir: Path,
    result_dir: Path,
    Vm: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    dt_ms: float = 1.0,   # snapshot spacing in ms
    force: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute 12-lead ECG using the reciprocal lead field method.

    Parameters
    ----------
    Vm     : (n_time, n_heart_nodes) — transmembrane potential in mV
    dt_ms  : time between snapshots (for axis labelling)

    Returns
    -------
    ecg : dict { 'I', 'II', …, 'V6' : (n_time,) tensor }
    """
    from torchcor.ecg.test import ECGSolver

    ecg_cache = result_dir / "ecg_12lead.pt"
    if ecg_cache.exists() and not force:
        print(f"Loading cached ECG from {ecg_cache}")
        return torch.load(str(ecg_cache), map_location=device)

    torso_dir = sim_dir / "torso"
    vent_dir  = sim_dir / "ventricles"
    elec_vtx  = sim_dir / "electrodes.vtx"
    h2t_file  = sim_dir / "heart_to_torso.npy"

    for f in (torso_dir, vent_dir, elec_vtx, h2t_file):
        if not Path(str(f)).exists():
            raise FileNotFoundError(
                f"Required file/dir missing: {f}\n"
                "Run step_extract first (or kcl_extract.py)."
            )

    heart_to_torso = np.load(str(h2t_file))
    print(f"\n{'='*60}")
    print(f"[ECG Forward]  torso nodes: (loading…)  heart nodes: {len(heart_to_torso)}")
    print("=" * 60)

    solver = ECGSolver(
        torso_mesh_dir=str(torso_dir),
        heart_mesh_dir=str(vent_dir),
        heart_to_torso_node=heart_to_torso,
        device=device,
        dtype=dtype,
    )

    # ── Torso conductivities (isotropic, S/m) ─────────────────────────
    for tag, g in CONDUCTIVITY.items():
        solver.set_torso_conductivity([tag], g=float(g))

    # ── Heart conductivities (bidomain, S/m) ──────────────────────────
    # Blood pools: high isotropic (il=el=sigma_blood, it=et=sigma_blood)
    solver.set_heart_conductivity(
        [24, 25],
        il=SIGMA_BLOOD, it=SIGMA_BLOOD,
        el=SIGMA_BLOOD, et=SIGMA_BLOOD,
    )
    # Ventricular walls: anisotropic bidomain
    solver.set_heart_conductivity(
        [34, 35],
        il=MILLI_VENTRICULAR_ENDO_IL, it=MILLI_VENTRICULAR_ENDO_IT,
        el=MILLI_VENTRICULAR_EPI_EL,  et=MILLI_VENTRICULAR_EPI_ET,
    )

    # ── Assemble stiffness matrices ────────────────────────────────────
    print("\nAssembling FEM stiffness matrices…")
    t0 = time.time()
    solver.build()
    print(f"  Assembly time: {time.time()-t0:.1f} s")

    # ── Load electrode node indices ────────────────────────────────────
    _load_electrodes_kcl(solver, elec_vtx)

    # ── Precompute lead fields (one CG solve per electrode) ────────────
    print("\nPrecomputing lead fields…")
    t0 = time.time()
    solver.precompute_lead_fields(I=1.0, a_tol=1e-8, r_tol=1e-8, max_iter=5000)
    print(f"  Lead field time: {time.time()-t0:.1f} s")

    # ── Compute 12-lead ECG ────────────────────────────────────────────
    print("\nComputing 12-lead ECG…")
    Vm_dev = Vm.to(device=device, dtype=dtype)
    ecg = solver.compute_12lead(Vm_dev)

    # Save
    torch.save({k: v.cpu() for k, v in ecg.items()}, str(ecg_cache))
    print(f"  Saved ECG → {ecg_cache}")

    # Plot
    _plot_ecg_12lead(ecg, dt_ms=dt_ms,
                     filepath=result_dir / "ecg_12lead.png")
    _save_ecg_csv(ecg, dt_ms=dt_ms,
                  filepath=result_dir / "ecg_12lead.csv")

    return ecg


def _load_electrodes_kcl(solver, elec_vtx: Path) -> None:
    """
    Parse the KCL-format electrodes.vtx file (written by kcl_extract.py)
    and populate solver.electrodes.

    File format:
        10          ← number of electrodes
        1234  # V1
        5678  # V2
        ...
    """
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
    with open(elec_vtx) as f:
        n = int(f.readline().strip())
        node_ids = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_id = int(line.split()[0])
            node_ids.append(node_id)

    if len(node_ids) != len(LABELS):
        raise ValueError(
            f"electrodes.vtx has {len(node_ids)} entries; expected {len(LABELS)}"
        )
    solver.electrodes = dict(zip(LABELS, node_ids))
    print(f"  Loaded {len(solver.electrodes)} electrodes, ground = {solver.ground}")


# ==============================================================================
#  Plotting
# ==============================================================================

def _plot_ecg_12lead(
    ecg: dict[str, torch.Tensor],
    dt_ms: float,
    filepath: Path,
) -> None:
    """Plot 12-lead ECG in standard 3×4 clinical layout."""
    import matplotlib.pyplot as plt

    lead_order = [
        "I",   "aVR", "V1", "V4",
        "II",  "aVL", "V2", "V5",
        "III", "aVF", "V3", "V6",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    fig.patch.set_facecolor("#f8f8f0")

    for ax, lead in zip(axes.flatten(), lead_order):
        if lead not in ecg:
            ax.set_visible(False)
            continue
        sig = ecg[lead].detach().cpu().float().numpy()
        t   = np.arange(len(sig)) * dt_ms
        ax.plot(t, sig, "k-", lw=0.9)
        ax.axhline(0, color="#aaaaaa", lw=0.4, ls="--")
        ax.set_title(lead, fontsize=11, fontweight="bold")
        ax.set_ylabel("mV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f8f8f0")

    for ax in axes[-1]:
        ax.set_xlabel("ms", fontsize=8)

    fig.suptitle("12-Lead ECG  —  KCL torso model  (TenTusscher-Panfilov 2006)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ECG plot → {filepath}")


def _save_ecg_csv(
    ecg: dict[str, torch.Tensor],
    dt_ms: float,
    filepath: Path,
) -> None:
    """Save all ECG leads to a CSV file."""
    import pandas as pd

    n_time = next(iter(ecg.values())).shape[0]
    t = np.arange(n_time) * dt_ms
    df = pd.DataFrame({"time_ms": t})
    for lead, sig in sorted(ecg.items()):
        df[lead] = sig.detach().cpu().float().numpy()
    df.to_csv(str(filepath), index=False)
    print(f"  Saved ECG CSV → {filepath}")


# ==============================================================================
#  Master runner
# ==============================================================================

def run(
    sim_dir: Path = DEFAULT_SIM_DIR,
    result_dir: Path = DEFAULT_RESULT,
    device_str: str = "auto",
    T: float = 600.0,
    dt: float = 0.05,
    snapshot_interval: int = 1,
    dtype_str: str = "float32",
    skip_extract: bool = False,
    skip_sim: bool = False,
    skip_ecg: bool = False,
    force_extract: bool = False,
    force_sim: bool = False,
    force_ecg: bool = False,
) -> None:
    device = _device(device_str)
    dtype  = torch.float32 if dtype_str == "float32" else torch.float64

    print(f"\nKCL Pipeline — device={device}, dtype={dtype}")
    print(f"  sim_dir    : {sim_dir}")
    print(f"  result_dir : {result_dir}")

    # ── Step 1: Extract ─────────────────────────────────────────────────
    if not skip_extract:
        step_extract(sim_dir, force=force_extract)

    # ── Step 2: Monodomain simulation ───────────────────────────────────
    if not skip_sim:
        Vm = step_simulate(
            sim_dir=sim_dir,
            result_dir=result_dir,
            device=device,
            dtype=dtype,
            T=T,
            dt=dt,
            snapshot_interval=snapshot_interval,
            force=force_sim,
        )
    else:
        vm_cache = result_dir / "Vm.pt"
        if not vm_cache.exists():
            raise FileNotFoundError(
                f"--skip-sim specified but no cached Vm found at {vm_cache}"
            )
        print(f"Loading Vm from cache: {vm_cache}")
        Vm = torch.load(str(vm_cache), map_location=device)

    print(f"\nVm shape: {Vm.shape}  (snapshots × heart nodes)")

    # ── Step 3: ECG forward solve ────────────────────────────────────────
    if not skip_ecg:
        # snapshot_interval * dt gives time step in ms between snapshots
        dt_snapshot_ms = snapshot_interval * dt
        ecg = step_ecg(
            sim_dir=sim_dir,
            result_dir=result_dir,
            Vm=Vm,
            device=device,
            dtype=dtype,
            dt_ms=dt_snapshot_ms,
            force=force_ecg,
        )
        print(f"\nECG leads: {sorted(ecg.keys())}")
        for lead, sig in sorted(ecg.items()):
            v = sig.cpu().float()
            print(f"  {lead:>4s}: min={v.min():.3f} mV  max={v.max():.3f} mV")

    print("\nPipeline complete.")


# ==============================================================================
#  CLI
# ==============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="KCL whole-torso cardiac simulation pipeline"
    )
    p.add_argument("--sim-dir",    default=str(DEFAULT_SIM_DIR))
    p.add_argument("--result-dir", default=str(DEFAULT_RESULT))
    p.add_argument("--device",     default="auto",
                   help="e.g. cuda:0, cpu, auto")
    p.add_argument("--dtype",      default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--T",          type=float, default=600.0,
                   help="Simulation duration (ms)")
    p.add_argument("--dt",         type=float, default=0.05,
                   help="Time step (ms)")
    p.add_argument("--snapshot-interval", type=int, default=1,
                   help="Save Vm every N time steps")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-sim",     action="store_true")
    p.add_argument("--skip-ecg",     action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force-sim",     action="store_true")
    p.add_argument("--force-ecg",     action="store_true")
    args = p.parse_args()

    run(
        sim_dir=Path(args.sim_dir),
        result_dir=Path(args.result_dir),
        device_str=args.device,
        T=args.T,
        dt=args.dt,
        snapshot_interval=args.snapshot_interval,
        dtype_str=args.dtype,
        skip_extract=args.skip_extract,
        skip_sim=args.skip_sim,
        skip_ecg=args.skip_ecg,
        force_extract=args.force_extract,
        force_sim=args.force_sim,
        force_ecg=args.force_ecg,
    )


if __name__ == "__main__":
    main()
