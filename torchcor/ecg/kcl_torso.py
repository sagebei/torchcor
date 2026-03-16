"""
KCL Whole-Torso ECG
====================
Adapter for the KCL Whole Torso Computational Models (Zenodo 7253863):

    Qian S., Monaci S., Mendonca-Costa C., et al.
    "KCL Whole Torso Computational Models", Zenodo, 2022.
    https://doi.org/10.5281/zenodo.7253863  (CC BY 4.0)

The dataset provides five healthy human torso models (KCL_torso1–5) as
CARP-format files (.pts / .elem / .lon), each containing the full torso
(skin, bones, lungs, abdominal organs) together with a biventricular heart.
Region tags and conductivities follow the table in KCL_Info.pdf (included).

Region tag table
----------------
  1  Inner body (bath)           0.24725 S/m  (isotropic)
  2  Skin                        0.117   S/m
  3  Bones                       0.05    S/m
  4  Kidneys                     0.1667  S/m
  5  Liver                       0.1667  S/m
  6  Stomach                     0.1     S/m
  7  Spleen                      0.1     S/m
  8  Lungs                       0.0714  S/m
  9  Superior vena cava blood    0.6667  S/m
 10  Superior vena cava          0.6667  S/m
 11  L. pulmonary artery blood   0.6667  S/m
 12  L. pulmonary artery         0.6667  S/m
 13  Aorta                       0.6667  S/m
 14  Aorta blood pool            0.6667  S/m
 15  Superior vena cava plane    0.6667  S/m
 16  Mitral valve                0.6667  S/m
 17  Pulmonary valve             0.6667  S/m
 18  Aortic valve                0.6667  S/m
 19  Tricuspid valve             0.6667  S/m
 20  Left atrial wall            0.25    S/m
 21  Left atrial blood pool      0.6667  S/m
 22  Right atrial blood pool     0.6667  S/m
 23  Right atrial blood wall     0.25    S/m
 24  Right ventricle blood pool  0.6667  S/m
 25  Left ventricle blood pool   0.6667  S/m
 26  Coronary sinus coil         0.24725 S/m
 27  Superior vena cava coil     0.6667  S/m
 28  RV apical coil              0.6667  S/m
 29  RV septal coil              0.6667  S/m
 30  Subcutaneous ICD lead       0.24725 S/m
 31  Left can                    0.24725 S/m
 32  Right can                   0.24725 S/m
 33  Subcutaneous ICD can        0.24725 S/m
 34  Right ventricle wall        anisotropic (uses .lon fibres)
 35  Left ventricle wall         anisotropic (uses .lon fibres)

Usage
-----
    from torchcor.ecg.kcl_torso import KCLTorsoECG
    import torch

    ecg = KCLTorsoECG(
        mesh_dir="C:/Users/.../cardiac_data/kcl_torso/KCL_torso3",
        device=torch.device("cuda"),
    )
    ecg.load()
    ecg.auto_place_electrodes()   # heuristic surface projection
    ecg.assemble()
    ecg.precompute_lead_fields()

    Vm = torch.load("biventricle/Vm.pt")
    leads = ecg.compute_12lead(Vm)
    ecg.plot_ecg(leads, "kcl_ecg.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchcor.core import MeshReader, Matrices3D, Preconditioner, ConjugateGradient

Tensor = torch.Tensor

# ── KCL region tag → bulk isotropic conductivity (S/m) ──────────────────────
# Ventricular walls (34, 35) are anisotropic; handled separately.
KCL_ISOTROPIC: Dict[int, float] = {
    1:  0.24725,   # inner body (bath)
    2:  0.117,     # skin
    3:  0.05,      # bones
    4:  0.1667,    # kidneys
    5:  0.1667,    # liver
    6:  0.1,       # stomach
    7:  0.1,       # spleen
    8:  0.0714,    # lungs
    9:  0.6667,    # SVC blood pool
    10: 0.6667,    # SVC
    11: 0.6667,    # LPA blood pool
    12: 0.6667,    # LPA
    13: 0.6667,    # aorta
    14: 0.6667,    # aorta blood pool
    15: 0.6667,    # SVC plane
    16: 0.6667,    # mitral valve
    17: 0.6667,    # pulmonary valve
    18: 0.6667,    # aortic valve
    19: 0.6667,    # tricuspid valve
    20: 0.25,      # L. atrial wall
    21: 0.6667,    # L. atrial blood pool
    22: 0.6667,    # R. atrial blood pool
    23: 0.25,      # R. atrial blood wall
    24: 0.6667,    # RV blood pool
    25: 0.6667,    # LV blood pool
    26: 0.24725,   # coronary sinus coil
    27: 0.6667,    # SVC coil
    28: 0.6667,    # RV apical coil
    29: 0.6667,    # RV septal coil
    30: 0.24725,   # subcutaneous ICD lead
    31: 0.24725,   # left can
    32: 0.24725,   # right can
    33: 0.24725,   # subcutaneous ICD can
}

# Ventricular myocardium tags
KCL_HEART_TAGS = [34, 35]   # RV wall, LV wall

# Bidomain conductivities for ventricular myocardium
# (Clerc 1976 / LeGrice 1995 values used in TenTusscher benchmarks, S/m)
KCL_HEART_IL = 0.5272    # intracellular longitudinal
KCL_HEART_IT = 0.2076    # intracellular transverse
KCL_HEART_EL = 1.0732    # extracellular longitudinal
KCL_HEART_ET = 0.4227    # extracellular transverse

# IEC 60601-2-25 electrode colour coding
_COLORS: Dict[str, str] = {
    "RA": "#FF0000", "LA": "#FFFF00",
    "LL": "#111111", "RL": "#00AA00",
    "V1": "#7B0000", "V2": "#FF4500",
    "V3": "#FF8C00", "V4": "#DAA520",
    "V5": "#228B22", "V6": "#1E5FA8",
}


class KCLTorsoECG:
    """
    12-lead ECG forward solver using the KCL whole-torso CARP mesh.

    Parameters
    ----------
    mesh_dir : str or Path
        Directory containing KCL_torsoN.pts / .elem / .lon  (or any other
        CARP mesh with the same KCL region-tag convention).
    device : torch.device
        Computation device.
    dtype : torch.dtype
        Floating-point precision.
    unit_conversion : float
        Divide node coordinates by this value (default 1000: µm → mm).
    heart_il, heart_it, heart_el, heart_et : float
        Bidomain conductivities for ventricular myocardium (S/m).
    """

    def __init__(
        self,
        mesh_dir: str | Path,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        unit_conversion: float = 1000.0,
        heart_il: float = KCL_HEART_IL,
        heart_it: float = KCL_HEART_IT,
        heart_el: float = KCL_HEART_EL,
        heart_et: float = KCL_HEART_ET,
    ):
        self.mesh_dir = Path(mesh_dir)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.unit_conversion = unit_conversion
        self.heart_il = heart_il
        self.heart_it = heart_it
        self.heart_el = heart_el
        self.heart_et = heart_et

        # set by load()
        self.nodes: Tensor | None = None
        self.elems: Tensor | None = None
        self.regions: Tensor | None = None
        self.fibres: Tensor | None = None
        self.n_nodes: int = 0

        # heart submesh (extracted from full mesh)
        self.heart_nodes: Tensor | None = None
        self.heart_elems: Tensor | None = None
        self.heart_fibres: Tensor | None = None
        self.heart_to_torso_node: Tensor | None = None  # index map

        # stiffness matrices (built by assemble())
        self.K_torso: Tensor | None = None
        self.K_heart: Tensor | None = None

        # electrodes: name → torso node index
        self.electrodes: Dict[str, int] = {}
        self.ground: str = "RL"

        # lead field vectors (one per non-ground electrode)
        self.q_heart: Dict[str, Tensor] = {}

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load CARP mesh from mesh_dir and extract heart submesh."""
        reader = MeshReader(self.mesh_dir)
        nodes_np, elems_np, regions_np, fibres_np = reader.read(
            unit_conversion=self.unit_conversion
        )

        # elems_np is an Elems object; extract only tetrahedral elements
        tt = elems_np.Tt
        elems_array  = tt.data    # (E, 4) numpy int array
        regions_array = tt.region  # (E,) numpy int array — overrides regions_np

        self.nodes   = torch.from_numpy(nodes_np).to(self.device, self.dtype)
        self.elems   = torch.from_numpy(elems_array).to(self.device, torch.long)
        self.regions = torch.from_numpy(regions_array).to(self.device, torch.long)
        self.fibres  = torch.from_numpy(fibres_np).to(self.device, self.dtype)
        self.n_nodes = self.nodes.shape[0]

        print(f"Loaded mesh: {self.n_nodes:,} nodes,  {self.elems.shape[0]:,} tetrahedra")
        print(f"  x: {self.nodes[:,0].min():.1f} .. {self.nodes[:,0].max():.1f} mm")
        print(f"  y: {self.nodes[:,1].min():.1f} .. {self.nodes[:,1].max():.1f} mm")
        print(f"  z: {self.nodes[:,2].min():.1f} .. {self.nodes[:,2].max():.1f} mm")

        self._extract_heart_submesh()

    def _extract_heart_submesh(self) -> None:
        """Extract ventricular myocardium (tags 34, 35) as an independent submesh."""
        heart_tag_tensor = torch.tensor(
            KCL_HEART_TAGS, device=self.device, dtype=torch.long
        )
        mask = torch.isin(self.regions, heart_tag_tensor)

        heart_elems_global = self.elems[mask]         # (H, 4) global node indices
        heart_fibs_all     = self.fibres[mask]        # (H, 3)

        # Unique torso-mesh node indices that belong to the heart
        self.heart_to_torso_node = torch.unique(
            heart_elems_global.reshape(-1), sorted=True
        )

        # Build reverse map: global node id → local heart node id
        n_torso = self.n_nodes
        reverse = torch.full((n_torso,), -1, device=self.device, dtype=torch.long)
        reverse[self.heart_to_torso_node] = torch.arange(
            len(self.heart_to_torso_node), device=self.device, dtype=torch.long
        )

        self.heart_elems  = reverse[heart_elems_global]   # local indices
        self.heart_nodes  = self.nodes[self.heart_to_torso_node]
        self.heart_fibres = heart_fibs_all

        n_hn = self.heart_nodes.shape[0]
        n_he = self.heart_elems.shape[0]
        print(f"Heart submesh: {n_hn:,} nodes,  {n_he:,} tetrahedra  (tags {KCL_HEART_TAGS})")

    # ── Conductivity & assembly ───────────────────────────────────────────────

    def assemble(self) -> None:
        """Build K_torso and K_heart stiffness matrices."""
        n_elems = self.elems.shape[0]
        print("Assembling conductivity tensors …")

        # ── Torso sigma (isotropic everywhere except ventricular walls) ──────
        sigma_torso = torch.zeros(
            (n_elems, 3, 3), device=self.device, dtype=self.dtype
        )
        I3 = torch.eye(3, device=self.device, dtype=self.dtype)

        for tag, g in KCL_ISOTROPIC.items():
            mask = (self.regions == tag)
            sigma_torso[mask] = g * I3

        # ── Ventricular walls: bidomain sigma_i + sigma_e ────────────────────
        # Compute anisotropic conductivity tensors directly from fibre vectors.
        heart_tag_tensor = torch.tensor(
            KCL_HEART_TAGS, device=self.device, dtype=torch.long
        )
        h_mask = torch.isin(self.regions, heart_tag_tensor)

        f = self.fibres[h_mask]                        # (H, 3) fibre directions
        f_norm = f / (f.norm(dim=1, keepdim=True).clamp(min=1e-12))

        # Outer products: f ⊗ f  and  I - f ⊗ f
        ff  = torch.einsum("ni,nj->nij", f_norm, f_norm)
        Imat = I3.unsqueeze(0).expand(ff.shape[0], -1, -1)
        perp = Imat - ff

        sigma_i = self.heart_il * ff + self.heart_it * perp
        sigma_e = self.heart_el * ff + self.heart_et * perp

        sigma_torso[h_mask] = sigma_i + sigma_e

        # ── Assemble K_torso ─────────────────────────────────────────────────
        print("Assembling K_torso …")
        torso_mats = Matrices3D(
            vertices=self.nodes,
            tetrahedrons=self.elems,
            device=self.device,
            dtype=self.dtype,
        )
        K, _ = torso_mats.assemble_matrices(sigma_torso)
        self.K_torso = K.to_sparse_csr()
        print(f"  K_torso: {self.K_torso.shape}")

        # ── Assemble K_heart (intracellular only) ─────────────────────────────
        print("Assembling K_heart …")
        heart_mats = Matrices3D(
            vertices=self.heart_nodes,
            tetrahedrons=self.heart_elems,
            device=self.device,
            dtype=self.dtype,
        )
        K_h, _ = heart_mats.assemble_matrices(sigma_i)
        self.K_heart = K_h.to_sparse_csr()
        print(f"  K_heart: {self.K_heart.shape}")

    # ── Electrode placement ───────────────────────────────────────────────────

    def load_electrodes(
        self,
        filepath: str | Path,
        names=("V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"),
    ) -> None:
        """
        Load electrode node IDs from a text file (one integer per line,
        first line = count header — same format as neumann.py).
        """
        ids = np.loadtxt(filepath, dtype=np.int64, skiprows=1).tolist()
        if len(ids) != len(names):
            raise ValueError(
                f"Expected {len(names)} electrode IDs, got {len(ids)}"
            )
        self.electrodes = {n: int(i) for n, i in zip(names, ids)}
        print(f"Loaded {len(self.electrodes)} electrodes from {filepath}")

    def auto_place_electrodes(self) -> None:
        """
        Project standard 12-lead electrode target positions onto the nearest
        torso surface node.

        The target positions (in mm, centred on the mesh bounding box) are
        anatomically approximate for a standing adult.  For publication-quality
        results replace with manually verified node IDs via load_electrodes().
        """
        nodes_np = self.nodes.cpu().numpy()  # (N, 3) in mm

        # Bounding box
        xmin, xmax = nodes_np[:, 0].min(), nodes_np[:, 0].max()
        ymin, ymax = nodes_np[:, 1].min(), nodes_np[:, 1].max()
        zmin, zmax = nodes_np[:, 2].min(), nodes_np[:, 2].max()

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        rx = (xmax - xmin) / 2.0    # half-width lateral
        ry = (ymax - ymin) / 2.0    # half-depth AP
        rz = (zmax - zmin) / 2.0    # half-height SI

        print(f"Mesh centre: ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm")
        print(f"Mesh half-extents: rx={rx:.0f}  ry={ry:.0f}  rz={rz:.0f} mm")

        # Target electrode positions as fractions of bounding-box half-extents,
        # relative to centre.  Based on Kligfield et al. JACC 2007 anatomy:
        #   x: +right / -left (patient)
        #   y: anterior surface = ymax  (the mesh has all-negative y, anterior = least negative)
        #   z: superior = zmax
        ant_y = ymax - 3.0    # anterior chest-wall surface (3 mm inset)
        targets = {
            # Precordial: anterior chest, sternal / midclavicular / axillary
            "V1": (cx + 0.12 * rx, ant_y, cz + 0.30 * rz),   # 4th ICS, right sternal
            "V2": (cx - 0.12 * rx, ant_y, cz + 0.30 * rz),   # 4th ICS, left sternal
            "V3": (cx - 0.30 * rx, ant_y, cz + 0.18 * rz),   # between V2 and V4
            "V4": (cx - 0.50 * rx, ant_y, cz + 0.07 * rz),   # 5th ICS, midclavicular
            "V5": (cx - 0.65 * rx, ant_y, cz + 0.07 * rz),   # anterior axillary
            "V6": (cx - 0.80 * rx, ant_y, cz + 0.07 * rz),   # midaxillary
            # Limb: proximal positions on torso surface
            "RA": (cx + 0.85 * rx, cy + 0.30 * ry, cz + 0.70 * rz),  # right shoulder
            "LA": (cx - 0.85 * rx, cy + 0.30 * ry, cz + 0.70 * rz),  # left shoulder
            "RL": (cx + 0.50 * rx, cy,               cz - 0.80 * rz),  # right lower abd
            "LL": (cx - 0.50 * rx, cy,               cz - 0.80 * rz),  # left lower abd
        }

        from scipy.spatial import cKDTree
        tree = cKDTree(nodes_np)
        print("\nElectrode placement (nearest-node projection):")
        for name, target in targets.items():
            dist, idx = tree.query(target)
            self.electrodes[name] = int(idx)
            pos = nodes_np[idx]
            print(f"  {name:3s}: node {idx:7d}  pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm  dist={dist:.1f} mm")

    # ── Lead-field computation ────────────────────────────────────────────────

    def _default_alpha(self) -> float:
        K = self.K_torso.to_sparse_coo().coalesce()
        idx = K.indices()
        val = K.values()
        diag = val[idx[0] == idx[1]]
        return float(diag.abs().mean().item()) * 1e8 + 1.0

    def _solve_neumann(
        self, source_node: int, ground_node: int,
        I: float, a_tol: float, r_tol: float, max_iter: int
    ) -> Tensor:
        b = torch.zeros(self.n_nodes, device=self.device, dtype=self.dtype)
        b[source_node]  =  I
        b[ground_node]  = -I

        alpha = self._default_alpha()
        indices = torch.tensor(
            [[ground_node], [ground_node]], device=self.device, dtype=torch.long
        )
        values = torch.tensor([alpha], device=self.device, dtype=self.dtype)
        D = torch.sparse_coo_tensor(
            indices, values,
            size=(self.n_nodes, self.n_nodes),
            device=self.device, dtype=self.dtype,
        ).to_sparse_csr()

        A = self.K_torso + D
        pcd = Preconditioner()
        pcd.create_Jocobi(A.to_sparse_coo())
        cg = ConjugateGradient(pcd, A, dtype=self.dtype)
        cg.initialize(
            x=torch.zeros(self.n_nodes, device=self.device, dtype=self.dtype),
            linear_guess=False,
        )
        phi, iters = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        print(f"    CG converged in {iters} iterations")

        if torch.isnan(phi).any():
            raise RuntimeError("Neumann solve diverged — check conductivities and mesh.")

        phi -= phi.mean()
        return phi

    def _phi_to_heart(self, phi_torso: Tensor) -> Tensor:
        return phi_torso[self.heart_to_torso_node]

    def precompute_lead_fields(
        self,
        I: float = 1.0,
        a_tol: float = 1e-8,
        r_tol: float = 1e-8,
        max_iter: int = 10000,
    ) -> None:
        """Compute lead-field vector q = K_heart @ Z_heart for every electrode."""
        if not self.electrodes:
            raise RuntimeError("No electrodes defined — call auto_place_electrodes() or load_electrodes() first.")
        ground_node = self.electrodes[self.ground]
        for name, node in self.electrodes.items():
            if name == self.ground:
                continue
            print(f"  Lead field [{name}] …")
            phi_torso = self._solve_neumann(
                node, ground_node, I, a_tol, r_tol, max_iter
            )
            phi_heart = self._phi_to_heart(phi_torso)
            self.q_heart[name] = self.K_heart @ phi_heart

    # ── ECG computation ───────────────────────────────────────────────────────

    def unipolar(self, Vm: Tensor, electrode: str) -> Tensor:
        """
        Compute unipolar electrogram for one electrode.

        Parameters
        ----------
        Vm : (T, N_heart) tensor  — transmembrane potential time series
        electrode : electrode name

        Returns
        -------
        (T,) tensor — voltage waveform
        """
        return Vm @ self.q_heart[electrode]

    def compute_12lead(self, Vm: Tensor) -> Dict[str, Tensor]:
        """
        Compute all 12 standard ECG leads from transmembrane potential Vm.

        Parameters
        ----------
        Vm : (T, N_heart) tensor  — rows = time steps, cols = heart nodes.
             Heart node ordering must match self.heart_nodes (i.e., the order
             produced by _extract_heart_submesh).

        Returns
        -------
        dict : lead name → (T,) tensor
        """
        Vm = Vm.to(self.device, self.dtype)
        required = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
        missing = [k for k in required if k not in self.q_heart]
        if missing:
            raise RuntimeError(f"Missing lead fields for: {missing}")

        ra = self.unipolar(Vm, "RA")
        la = self.unipolar(Vm, "LA")
        ll = self.unipolar(Vm, "LL")

        wct = (ra + la + ll) / 3.0

        return {
            "I":    la - ra,
            "II":   ll - ra,
            "III":  ll - la,
            "aVR":  ra - 0.5 * (la + ll),
            "aVL":  la - 0.5 * (ra + ll),
            "aVF":  ll - 0.5 * (ra + la),
            "V1":   self.unipolar(Vm, "V1") - wct,
            "V2":   self.unipolar(Vm, "V2") - wct,
            "V3":   self.unipolar(Vm, "V3") - wct,
            "V4":   self.unipolar(Vm, "V4") - wct,
            "V5":   self.unipolar(Vm, "V5") - wct,
            "V6":   self.unipolar(Vm, "V6") - wct,
        }

    def plot_ecg(self, ecg_dict: Dict[str, Tensor], filename: str = "kcl_ecg.png") -> None:
        order = ["I", "aVR", "V1", "V4",
                 "II", "aVL", "V2", "V5",
                 "III", "aVF", "V3", "V6"]
        fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True)
        for ax, lead in zip(axes.flatten(), order):
            sig = ecg_dict[lead].detach().cpu().numpy()
            ax.plot(sig, lw=1.2)
            ax.set_title(lead, fontsize=10)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.supylabel("Potential (a.u.)", fontsize=11)
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Saved: {filename}")

    # ── Convenience: save electrode list ─────────────────────────────────────

    def save_electrodes(self, filepath: str | Path) -> None:
        """Write electrode node IDs to a .vtx-style text file."""
        names = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
        with open(filepath, "w") as f:
            f.write(f"{len(names)}\n")
            for n in names:
                f.write(f"{self.electrodes[n]}\n")
        print(f"Saved electrode IDs to {filepath}")


# ── Download helper ───────────────────────────────────────────────────────────

def download_kcl_torso(
    torso_number: int = 3,
    out_dir: str | Path = ".",
    extract: bool = True,
) -> Path:
    """
    Download one KCL torso model from Zenodo (doi:10.5281/zenodo.7253863).

    Parameters
    ----------
    torso_number : int  (1–5)
    out_dir : destination directory
    extract : unzip after download

    Returns
    -------
    Path to the extracted mesh directory (or zip file if extract=False)
    """
    import requests
    import zipfile

    if not 1 <= torso_number <= 5:
        raise ValueError("torso_number must be 1–5")

    name = f"KCL_torso{torso_number}"
    url = f"https://zenodo.org/api/records/7253863/files/{name}.zip/content"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"

    if not zip_path.exists():
        print(f"Downloading {name}.zip from Zenodo …")
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
        print()
    else:
        print(f"Already downloaded: {zip_path}")

    if extract:
        mesh_dir = out_dir / name
        if not mesh_dir.exists():
            print(f"Extracting to {mesh_dir} …")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out_dir)
        return mesh_dir

    return zip_path


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mesh_dir = Path("C:/Users/bulli/cardiac_data/kcl_torso/KCL_torso3")
    if not mesh_dir.exists():
        print("Mesh not found locally — downloading from Zenodo …")
        mesh_dir = download_kcl_torso(3, out_dir="C:/Users/bulli/cardiac_data/kcl_torso")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    ecg = KCLTorsoECG(mesh_dir=mesh_dir, device=device)
    ecg.load()
    ecg.auto_place_electrodes()
    ecg.save_electrodes(mesh_dir / "electrodes_auto.vtx")

    print("\nAssembling stiffness matrices …")
    ecg.assemble()

    print("\nPrecomputing lead fields …")
    ecg.precompute_lead_fields(a_tol=1e-8, r_tol=1e-8, max_iter=10000)

    # Load Vm from a prior ventricular simulation and project to heart submesh
    vm_path = Path("./biventricle/Vm.pt")
    if vm_path.exists():
        print("\nLoading Vm …")
        Vm_full = torch.load(vm_path, map_location=device)
        # Vm_full is (T, N_heart_sim) — if from the KCL heart, it should
        # align with ecg.heart_nodes.  Otherwise provide your own Vm.
        leads = ecg.compute_12lead(Vm_full)
        ecg.plot_ecg(leads, "kcl_ecg.png")
    else:
        print("\nNo Vm.pt found — skipping ECG computation.")
        print("Run a monodomain simulation first and save Vm, then re-run.")
