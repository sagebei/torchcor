"""
Synthetic Torso ECG
===================
Generates a synthetic male torso mesh (ellipsoidal approximation) with
anatomically placed 12-lead ECG electrodes, for use when patient-specific
torso meshes are unavailable.

Torso geometry  : prolate ellipsoid  44 cm wide × 24 cm deep × 60 cm tall
Heart geometry  : smaller ellipsoid, offset left / slightly posterior / upper chest
Fibre direction : uniform longitudinal (apex–base) for the synthetic heart

Electrode placement follows:
  Kligfield et al., JACC 2007;49:1109-1127

ECG forward theory follows the lead-field (reciprocal) method:
  Potse M., Front Physiol 2018;9:370
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Arc
import warnings

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

from torchcor.core import Matrices3D, Preconditioner, ConjugateGradient

Tensor = torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Anatomical constants  (centimetres, origin = geometric torso centre)
# ─────────────────────────────────────────────────────────────────────────────

# Torso: prolate ellipsoid semi-axes
_TA, _TB, _TC = 22.0, 12.0, 30.0   # lateral, AP, SI

# Heart: smaller ellipsoid — slightly left, slightly posterior, upper chest
_HA, _HB, _HC = 5.0, 4.0, 7.0
_HCX, _HCY, _HCZ = -2.0, -2.0, 3.0

# Conductivities (S/m)
_SIG_TORSO    = 0.22      # bulk thoracic tissue (isotropic)
_SIG_HEART_IL = 0.5272    # intracellular longitudinal
_SIG_HEART_IT = 0.2076    # intracellular transverse
_SIG_HEART_EL = 1.0732    # extracellular longitudinal
_SIG_HEART_ET = 0.4227    # extracellular transverse

# Standard 12-lead electrode positions  (3-D, cm from torso centre)
# Precordial (V1–V6): anterior chest wall
# Limb (RA, LA, RL, LL): proximal limb approximations on torso surface
_ELECTRODES_3D: Dict[str, Tuple[float, float, float]] = {
    "V1": ( 2.5,  11.5,  9.0),   # 4th ICS, right sternal border
    "V2": (-2.5,  11.5,  9.0),   # 4th ICS, left sternal border
    "V3": (-6.5,  11.0,  6.0),   # between V2 and V4
    "V4": (-11.0, 10.0,  3.0),   # 5th ICS, midclavicular line
    "V5": (-15.0,  8.5,  3.0),   # anterior axillary line
    "V6": (-18.5,  6.0,  3.0),   # midaxillary line
    "RA": ( 19.0,  5.0, 22.0),   # right shoulder
    "LA": (-19.0,  5.0, 22.0),   # left shoulder
    "RL": ( 11.0,  2.0,-26.0),   # right lower abdomen  (ground)
    "LL": (-11.0,  2.0,-26.0),   # left lower abdomen
}

# International (IEC 60601-2-25) electrode colour coding
_COLORS: Dict[str, str] = {
    "RA": "#FF0000", "LA": "#FFFF00",
    "LL": "#111111", "RL": "#00AA00",
    "V1": "#7B0000", "V2": "#FF4500",
    "V3": "#FF8C00", "V4": "#DAA520",
    "V5": "#228B22", "V6": "#1E5FA8",
}


# ─────────────────────────────────────────────────────────────────────────────
# Mesh generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _in_ellipsoid(pts: np.ndarray, cx, cy, cz, ax, ay, az) -> np.ndarray:
    rel = pts - np.array([cx, cy, cz])
    return (rel[:, 0]/ax)**2 + (rel[:, 1]/ay)**2 + (rel[:, 2]/az)**2 <= 1.0


def _surface_pts(n: int, cx, cy, cz, ax, ay, az) -> np.ndarray:
    sq = int(np.ceil(np.sqrt(n)))
    phi   = np.linspace(1e-4, np.pi - 1e-4, sq)
    theta = np.linspace(0, 2*np.pi, sq, endpoint=False)
    phi, theta = np.meshgrid(phi, theta)
    phi, theta = phi.ravel(), theta.ravel()
    return np.column_stack([
        cx + ax * np.sin(phi) * np.cos(theta),
        cy + ay * np.sin(phi) * np.sin(theta),
        cz + az * np.cos(phi),
    ])


def _generate_mesh(
    n_interior: int = 2500,
    n_surface:  int = 350,
    seed:       int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a conformal tetrahedral mesh of the torso with a heart sub-region.

    Returns
    -------
    nodes   : (N, 3) float64   node coordinates (cm)
    tets    : (M, 4) int64     tetrahedral connectivity
    regions : (M,)  int64      1 = torso tissue,  2 = heart tissue
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)

    # --- Interior nodes (rejection sampling inside torso ellipsoid) ---
    chunks = []
    while sum(len(c) for c in chunks) < n_interior:
        batch = rng.uniform(-1, 1, (n_interior * 5, 3)) * np.array([_TA, _TB, _TC])
        chunks.append(batch[_in_ellipsoid(batch, 0, 0, 0, _TA, _TB, _TC)])
    interior = np.vstack(chunks)[:n_interior]

    # --- Surface nodes (ellipsoid parameterisation) ---
    surface = _surface_pts(n_surface, 0, 0, 0, _TA, _TB, _TC)

    nodes = np.vstack([interior, surface])

    # --- Delaunay triangulation in 3-D ---
    tri  = Delaunay(nodes)
    tets = tri.simplices.astype(np.int64)

    # --- Keep only tets whose centroid is inside the torso ---
    centroids = nodes[tets].mean(axis=1)
    inside    = _in_ellipsoid(centroids, 0, 0, 0, _TA, _TB, _TC)
    tets      = tets[inside]
    centroids = centroids[inside]

    # --- Tag heart elements ---
    in_heart = _in_ellipsoid(centroids, _HCX, _HCY, _HCZ, _HA, _HB, _HC)
    regions  = np.ones(len(tets), dtype=np.int64)
    regions[in_heart] = 2

    return nodes, tets, regions


def _electrode_nodes(nodes: np.ndarray) -> Dict[str, int]:
    """
    For each electrode, return the index of the nearest node on/near
    the outer surface of the torso ellipsoid.
    """
    r = (nodes[:, 0]/_TA)**2 + (nodes[:, 1]/_TB)**2 + (nodes[:, 2]/_TC)**2
    surf_idx = np.where(r > 0.80)[0]
    surf_pts = nodes[surf_idx]

    elec_nodes: Dict[str, int] = {}
    for name, (ex, ey, ez) in _ELECTRODES_3D.items():
        pos   = np.array([ex, ey, ez])
        dists = np.linalg.norm(surf_pts - pos, axis=1)
        elec_nodes[name] = int(surf_idx[np.argmin(dists)])
    return elec_nodes


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic transmembrane potential  (plane-wave activation)
# ─────────────────────────────────────────────────────────────────────────────

def _ap_waveform(t: np.ndarray) -> np.ndarray:
    """Empirical action-potential shape (mV) as a function of time since
    local activation (ms).  Resting ~ −85 mV, peak ~ +30 mV."""
    Vm = np.full_like(t, -85.0)
    up = (t >= 0) & (t < 2);    Vm[up] = -85 + 115 * (t[up] / 2)
    pl = (t >= 2) & (t < 200);  Vm[pl] =   5 - 25 * (t[pl] - 2) / 198
    rp = (t >= 200) & (t < 260); Vm[rp] = 5 - 90 * (t[rp] - 200) / 60
    return Vm


def generate_synthetic_Vm(
    heart_nodes: np.ndarray,
    n_timesteps: int   = 500,
    dt_ms:       float = 1.0,
    cv_cm_ms:    float = 0.07,
) -> Tensor:
    """
    Generate a synthetic Vm array on the heart nodes using a plane-wave
    activation front spreading from the apex.

    Parameters
    ----------
    heart_nodes : (N_heart, 3) node coordinates (cm)
    n_timesteps : number of time steps
    dt_ms       : time step (ms)
    cv_cm_ms    : conduction velocity (cm / ms)

    Returns
    -------
    Vm : (n_timesteps, N_heart) float32 tensor (mV)
    """
    apex   = np.array([_HCX, _HCY, _HCZ - _HC * 0.9])
    dists  = np.linalg.norm(heart_nodes - apex, axis=1)   # (N_heart,)
    AT     = dists / cv_cm_ms                              # activation times (ms)

    t          = np.arange(n_timesteps) * dt_ms            # (T,)
    t_since    = t[:, None] - AT[None, :]                  # (T, N_heart)
    Vm         = _ap_waveform(t_since)
    Vm[t_since < 0] = -85.0

    return torch.from_numpy(Vm.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# CG solver with penalty grounding
# ─────────────────────────────────────────────────────────────────────────────

def _solve(
    K_csr:    Tensor,
    b:        Tensor,
    gnd:      int,
    device:   torch.device,
    dtype:    torch.dtype,
    a_tol:    float = 1e-8,
    r_tol:    float = 1e-8,
    max_iter: int   = 10000,
) -> Tensor:
    N = b.shape[0]
    K_coo = K_csr.to_sparse_coo().coalesce()
    idx, val = K_coo.indices(), K_coo.values()
    alpha = val[idx[0] == idx[1]].abs().mean().item() * 1e6

    pen = torch.sparse_coo_tensor(
        torch.tensor([[gnd], [gnd]], device=device, dtype=torch.long),
        torch.tensor([alpha], device=device, dtype=dtype),
        size=(N, N), device=device, dtype=dtype,
    ).to_sparse_csr()

    A   = K_csr + pen
    pcd = Preconditioner()
    pcd.create_Jocobi(A.to_sparse_coo())
    cg  = ConjugateGradient(pcd, A, dtype=dtype)
    cg.initialize(torch.zeros(N, device=device, dtype=dtype), linear_guess=False)
    x, iters = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
    if torch.isnan(x).any():
        raise RuntimeError(f"CG diverged ({iters} iters). Check mesh quality.")
    return x


# ─────────────────────────────────────────────────────────────────────────────
# SyntheticTorsoECG
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticTorsoECG:
    """
    12-lead ECG forward solver using a synthetic ellipsoidal torso mesh.
    No external mesh files required.

    Useful for:
    - Testing the ECG pipeline end-to-end without patient-specific geometry
    - Approximate ECGs when only a ventricular mesh is available
    - Validating lead field implementation

    Usage
    -----
    >>> solver = SyntheticTorsoECG(device=torch.device("cpu"))
    >>> solver.build()
    >>> solver.precompute_lead_fields()
    >>> Vm = generate_synthetic_Vm(solver.heart_nodes)
    >>> ecg = solver.compute_12lead(Vm)
    >>> SyntheticTorsoECG.plot_ecg(ecg, filename="ecg_synthetic.png")
    """

    TORSO_TAG = 1
    HEART_TAG = 2

    def __init__(
        self,
        device:      torch.device = torch.device("cpu"),
        dtype:       torch.dtype  = torch.float32,
        n_interior:  int = 2500,
        n_surface:   int = 350,
        seed:        int = 42,
    ):
        self.device     = device
        self.dtype      = dtype
        self._n_int     = n_interior
        self._n_sur     = n_surface
        self._seed      = seed
        self.ground     = "RL"

        self._nodes_np:  Optional[np.ndarray] = None
        self._heart_idx: Optional[np.ndarray] = None
        self._elec_map:  Dict[str, int]       = {}

        self.K_torso: Optional[Tensor] = None
        self.K_heart: Optional[Tensor] = None
        self._q:      Dict[str, Tensor] = {}

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def heart_nodes(self) -> np.ndarray:
        """(N_heart, 3) coordinates of heart nodes (cm)."""
        if self._heart_idx is None:
            raise RuntimeError("Call build() first.")
        return self._nodes_np[self._heart_idx]

    @property
    def n_heart(self) -> int:
        return int(self._heart_idx.shape[0])

    @property
    def n_torso(self) -> int:
        return int(self._nodes_np.shape[0])

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self):
        """Generate mesh and assemble stiffness matrices."""
        print("Generating synthetic torso mesh …", flush=True)
        nodes, tets, regions = _generate_mesh(
            n_interior=self._n_int,
            n_surface =self._n_sur,
            seed      =self._seed,
        )
        self._nodes_np = nodes

        # Heart node indices  (all nodes appearing in heart elements)
        heart_elem_mask   = (regions == self.HEART_TAG)
        self._heart_idx   = np.unique(tets[heart_elem_mask].ravel())

        print(f"  Nodes: {len(nodes)} | Tets: {len(tets)} | "
              f"Heart nodes: {len(self._heart_idx)} | "
              f"Heart tets: {heart_elem_mask.sum()}", flush=True)

        self._elec_map = _electrode_nodes(nodes)

        # Tensors
        nodes_t  = torch.from_numpy(nodes).to(self.device, self.dtype)
        tets_t   = torch.from_numpy(tets).to(self.device, torch.long)
        regs_t   = torch.from_numpy(regions).to(self.device, torch.long)

        I3 = torch.eye(3, device=self.device, dtype=self.dtype)

        # --- Per-element conductivity tensors ---
        heart_mask_e  = (regs_t == self.HEART_TAG)

        # Isotropic averages (no fibre data for the synthetic heart)
        sig_i_iso  = (_SIG_HEART_IL + 2*_SIG_HEART_IT) / 3
        sig_e_iso  = (_SIG_HEART_EL + 2*_SIG_HEART_ET) / 3
        sig_ie_iso = sig_i_iso + sig_e_iso

        sigma_full = torch.zeros(len(tets), 3, 3, device=self.device, dtype=self.dtype)
        sigma_full[regs_t == self.TORSO_TAG] = _SIG_TORSO * I3
        sigma_full[heart_mask_e]             = sig_ie_iso * I3

        sigma_i_full = torch.zeros_like(sigma_full)
        sigma_i_full[heart_mask_e] = sig_i_iso * I3

        # --- K_torso (full mesh, sigma_ie on heart / sigma_T elsewhere) ---
        print("Assembling K_torso …", flush=True)
        mats = Matrices3D(vertices=nodes_t, tetrahedrons=tets_t,
                          device=self.device, dtype=self.dtype)
        K, _ = mats.assemble_matrices(sigma_full)
        self.K_torso = K.to_sparse_csr()

        # --- K_heart (heart-local numbering, sigma_i only) ---
        print("Assembling K_heart …", flush=True)
        h2g = torch.from_numpy(self._heart_idx).to(self.device, torch.long)
        g2h = torch.full((len(nodes),), -1, device=self.device, dtype=torch.long)
        g2h[h2g] = torch.arange(len(self._heart_idx),
                                  device=self.device, dtype=torch.long)

        heart_elem_idx   = torch.where(heart_mask_e)[0]
        heart_tets_local = g2h[tets_t[heart_elem_idx]]   # (Mh, 4) local indices
        heart_nodes_t    = nodes_t[h2g]
        sigma_i_heart    = sigma_i_full[heart_elem_idx]

        mats_h = Matrices3D(vertices=heart_nodes_t, tetrahedrons=heart_tets_local,
                            device=self.device, dtype=self.dtype)
        Kh, _ = mats_h.assemble_matrices(sigma_i_heart)
        self.K_heart = Kh.to_sparse_csr()

        self._h2g = h2g
        print("Build complete.", flush=True)

    # ── Lead field precomputation ─────────────────────────────────────────────

    def precompute_lead_fields(
        self,
        I:        float = 1.0,
        a_tol:    float = 1e-8,
        r_tol:    float = 1e-8,
        max_iter: int   = 10000,
    ):
        """Compute lead field vector q for each non-ground electrode."""
        gnd = self._elec_map[self.ground]
        for name, node_id in self._elec_map.items():
            if name == self.ground:
                continue
            print(f"  Lead field {name} …", end=" ", flush=True)
            b = torch.zeros(self.n_torso, device=self.device, dtype=self.dtype)
            b[node_id] =  I
            b[gnd]     = -I
            Z_torso = _solve(self.K_torso, b, gnd,
                             self.device, self.dtype, a_tol, r_tol, max_iter)
            Z_heart = Z_torso[self._h2g]
            self._q[name] = self.K_heart @ Z_heart
            print("done")

    # ── ECG ──────────────────────────────────────────────────────────────────

    def compute_12lead(self, Vm: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        Vm : (n_time, n_heart_nodes) transmembrane potential (mV)

        Returns
        -------
        dict  lead_name → (n_time,) tensor
        """
        Vm = Vm.to(self.device, self.dtype)
        if Vm.shape[1] != self.n_heart:
            raise ValueError(f"Vm has {Vm.shape[1]} columns; "
                             f"heart has {self.n_heart} nodes.")
        missing = [k for k in self._elec_map
                   if k != self.ground and k not in self._q]
        if missing:
            raise RuntimeError(f"Run precompute_lead_fields() first. Missing: {missing}")

        phi: Dict[str, Tensor] = {}
        for name in self._elec_map:
            if name == self.ground:
                phi[name] = torch.zeros(Vm.shape[0],
                                        device=self.device, dtype=self.dtype)
            else:
                phi[name] = Vm @ self._q[name]

        ra, la, ll = phi["RA"], phi["LA"], phi["LL"]
        wct = (ra + la + ll) / 3.0
        return {
            "I":   la - ra,
            "II":  ll - ra,
            "III": ll - la,
            "aVR": ra - 0.5*(la + ll),
            "aVL": la - 0.5*(ra + ll),
            "aVF": ll - 0.5*(ra + la),
            "V1":  phi["V1"] - wct,
            "V2":  phi["V2"] - wct,
            "V3":  phi["V3"] - wct,
            "V4":  phi["V4"] - wct,
            "V5":  phi["V5"] - wct,
            "V6":  phi["V6"] - wct,
        }

    @staticmethod
    def plot_ecg(
        ecg_dict: Dict[str, Tensor],
        dt_ms:    float = 1.0,
        filename: str   = "ecg_synthetic.png",
    ):
        """Plot 12-lead ECG in standard 3×4 clinical layout."""
        order = ["I","aVR","V1","V4","II","aVL","V2","V5","III","aVF","V3","V6"]
        fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True)
        for ax, lead in zip(axes.flatten(), order):
            sig = ecg_dict[lead].detach().cpu().numpy()
            t   = np.arange(len(sig)) * dt_ms
            ax.plot(t, sig, "k-", lw=0.8)
            ax.set_title(lead, fontsize=10, fontweight="bold")
            ax.set_ylabel("mV", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("ms", fontsize=8)
        fig.suptitle("12-Lead ECG — Synthetic Torso", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Cartoon lead-placement visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_torso_leads(filename: str = "ecg_lead_placement.png") -> None:
    """
    Render a labelled clinical schematic (anterior view) of the male torso
    showing standard 12-lead ECG electrode placement and anatomical heart
    position.

    Follows standard medical imaging convention:
      Patient's RIGHT is on the viewer's LEFT.
    """
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.set_aspect("equal")
    # Flip x-axis: positive x = patient right = viewer's left
    ax.set_xlim(32, -32)
    ax.set_ylim(-40, 46)
    ax.axis("off")

    # ── Torso silhouette ─────────────────────────────────────────────────────
    torso_pts = np.array([
        [  0,  30],                            # top-centre (base of neck)
        [  8,  28], [ 16,  25], [ 22,  21],   # right shoulder
        [ 24,  16], [ 22,   8],               # right armpit
        [ 21,   0], [ 20, -12], [ 17, -27],   # right side
        [ 11, -35], [  0, -37],               # lower right → centre
        [-11, -35],
        [-17, -27], [-20, -12], [-21,   0],
        [-22,   8], [-24,  16],
        [-22,  21], [-16,  25], [ -8,  28],
        [  0,  30],
    ])
    ax.add_patch(plt.Polygon(torso_pts, closed=True,
                             facecolor="#F5CBA7", edgecolor="#935116",
                             linewidth=2.0, alpha=0.85, zorder=1))

    # Neck
    ax.add_patch(plt.Polygon([[-4,30],[4,30],[4,33],[-4,33]],
                              facecolor="#F5CBA7", edgecolor="none", zorder=1))

    # Head
    ax.add_patch(Ellipse((0, 38.5), 13, 15,
                          facecolor="#F5CBA7", edgecolor="#935116",
                          linewidth=1.5, alpha=0.85, zorder=2))

    # ── Rib-cage arcs (simplified) ───────────────────────────────────────────
    for yc, hw in [(18,17),(14,18),(10,19),(6,19),(2,18),(-2,17)]:
        ax.add_patch(Arc((0, yc), 2*hw, 9,
                         theta1=0, theta2=180,
                         color="#A04000", linewidth=0.6, alpha=0.20, zorder=2))

    # Sternum (dashed midline)
    ax.plot([0, 0], [28, 0], color="#7D6608", linewidth=1.3,
            linestyle="--", alpha=0.45, zorder=2, label="Sternum")

    # ── Heart (anatomically placed) ──────────────────────────────────────────
    # Apex points left–inferior (angle ≈ −15°)
    ax.add_patch(Ellipse((-2, 8), 12, 16, angle=-15,
                          facecolor="#E74C3C", edgecolor="#922B21",
                          linewidth=1.8, alpha=0.60, zorder=3))
    ax.text(-2, 8, "Heart", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#641E16", zorder=4)

    # ── Electrode dots ────────────────────────────────────────────────────────
    # 2-D positions in the same coordinate system as the silhouette
    # (positive x = patient right, but x-axis is FLIPPED on screen)
    elec_2d: Dict[str, Tuple[float, float]] = {
        "V1": ( 2.5, 14.0),   # right sternal border, 4th ICS
        "V2": (-2.5, 14.0),   # left  sternal border, 4th ICS
        "V3": (-7.0, 10.0),   # between V2 and V4
        "V4": (-11.5,  7.0),  # midclavicular, 5th ICS
        "V5": (-15.5,  7.0),  # anterior axillary line
        "V6": (-19.5,  7.0),  # midaxillary line
        "RA": ( 20.0, 21.0),  # right shoulder
        "LA": (-20.0, 21.0),  # left shoulder
        "RL": ( 13.0,-31.0),  # right lower abdomen
        "LL": (-13.0,-31.0),  # left lower abdomen
    }

    # Label offsets & alignment (screen coords with flipped x)
    label_cfg: Dict[str, Tuple[float, float, str]] = {
        "V1": ( 3.0, 0.8, "left"),   "V2": (-3.0, 0.8, "right"),
        "V3": (-3.0, 0.8, "right"),  "V4": (-3.0, 0.8, "right"),
        "V5": (-3.0, 0.8, "right"),  "V6": (-3.5, 0.8, "right"),
        "RA": ( 2.5, 0.8, "left"),   "LA": (-2.5, 0.8, "right"),
        "RL": ( 2.5,-2.5, "left"),   "LL": (-2.5,-2.5, "right"),
    }

    for name, (ex, ey) in elec_2d.items():
        col = _COLORS[name]
        ax.plot(ex, ey, "o", markersize=15, color=col,
                markeredgecolor="black", markeredgewidth=1.6, zorder=6)
        ox, oy, ha = label_cfg[name]
        ax.text(ex + ox, ey + oy, name,
                fontsize=9, fontweight="bold", ha=ha, va="bottom", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.78))

    # ── Clinical reference annotations ───────────────────────────────────────
    ax.annotate("4th ICS →\nRight sternal border",
                xy=(2.5, 14), xytext=(13, 17.5),
                fontsize=7.5, color="#444444", ha="left",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

    ax.annotate("← 5th ICS\n   midclavicular",
                xy=(-11.5, 7), xytext=(-30, 11),
                fontsize=7.5, color="#444444", ha="left",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

    ax.annotate("← Midaxillary\n   line",
                xy=(-19.5, 7), xytext=(-30, 3.5),
                fontsize=7.5, color="#444444", ha="left",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

    # ── Right / Left orientation ─────────────────────────────────────────────
    # x-axis is flipped: patient's right appears on viewer's LEFT
    ax.text( 29, -37, "Patient's\nRight", fontsize=8.5, ha="center",
             color="gray", style="italic")
    ax.text(-29, -37, "Patient's\nLeft",  fontsize=8.5, ha="center",
             color="gray", style="italic")
    ax.annotate("", xy=(23,-39), xytext=(-23,-39),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.1))

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#E74C3C", edgecolor="#922B21", label="Heart"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#FF0000",
                   markeredgecolor="k", markersize=11, label="RA — Red"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#FFFF00",
                   markeredgecolor="k", markersize=11, label="LA — Yellow"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#111111",
                   markeredgecolor="gray", markersize=11, label="LL — Black"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#00AA00",
                   markeredgecolor="k", markersize=11, label="RL — Green (GND)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#1E5FA8",
                   markeredgecolor="k", markersize=11, label="V1–V6  Precordial"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9,
              framealpha=0.93, title="Electrode Colours (IEC 60601)",
              title_fontsize=9, borderpad=0.9)

    ax.set_title(
        "Standard 12-Lead ECG Electrode Placement\n"
        "Anterior View — Male (patient's right on viewer's left)",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Cartoon diagram (no GPU or mesh required)
    print("Rendering lead placement diagram …")
    visualize_torso_leads("ecg_lead_placement.png")

    # 2. Full synthetic ECG pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    solver = SyntheticTorsoECG(device=device, n_interior=2500, n_surface=350)
    solver.build()
    solver.precompute_lead_fields(a_tol=1e-6, r_tol=1e-6, max_iter=5000)

    print("\nGenerating synthetic Vm (plane-wave activation) …")
    Vm = generate_synthetic_Vm(solver.heart_nodes, n_timesteps=500, dt_ms=1.0)

    print("Computing 12-lead ECG …")
    ecg = solver.compute_12lead(Vm.to(device))
    SyntheticTorsoECG.plot_ecg(ecg, dt_ms=1.0, filename="ecg_synthetic.png")

    print("\nOutputs: ecg_lead_placement.png  ecg_synthetic.png")
