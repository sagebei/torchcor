"""
ECG Lead Field Solver -- First-Principles Reciprocal Method
===========================================================

Computes body-surface ECG signals from cardiac transmembrane potentials
(Vm) on a combined heart-torso tetrahedral FE mesh.  This implementation
follows the openCARP / Potse (2018) lead-field algorithm using rigorous
Dirichlet grounding (no penalty term) and full float64 numerics.

Physics
-------
Starting from the bidomain equations in the heart,

    div(sigma_i  grad(phi_i))  = + I_m          (intracellular)
    div(sigma_e  grad(phi_e))  = - I_m          (extracellular / heart)

with  Vm = phi_i - phi_e.  Adding and eliminating I_m gives

    div( (sigma_i + sigma_e) grad(phi_e) )  =  -div( sigma_i grad(Vm) )
                                                                (1)

In the passive torso, current is solenoidal:

    div( sigma_T grad(u) ) = 0                                  (2)

Heart-torso coupling (Krassowska & Neu 1994):

    phi_e = u                  on  Gamma_heart  (continuity)
    (sigma_e grad phi_e) . n   =  (sigma_T grad u) . n

Body-surface boundary (insulating air):

    (sigma_T grad u) . n  =  0  on  Gamma_body                  (3)

Unified formulation on Omega = heart ∪ torso
--------------------------------------------
Let G be the bulk conductivity tensor, defined piecewise as

    G(x) = sigma_i(x) + sigma_e(x)   for x in heart
    G(x) = sigma_T(x)                for x in torso

and let  sigma_i_ext(x) = sigma_i(x)  in the heart and  0 in the torso.
Then (1)-(3) reduce to the single elliptic PDE

    -div( G grad u ) = div( sigma_i_ext grad Vm )   in Omega    (4)

with the body-surface Neumann condition  (G grad u).n = 0.  FEM
discretisation yields the symmetric positive semi-definite system

    K_bulk  u(t)  =  - K_i  Vm(t)                               (5)

where  K_bulk  is assembled with G  over the whole torso mesh and  K_i
is assembled with  sigma_i  over the heart elements only (all other
entries are structurally zero).

The pure-Neumann operator has a 1-D null space (constants), so the
solution is unique only up to an additive constant.  We fix that
constant by a physical ground choice: the Right Leg (RL) electrode is
grounded, u(g) = 0, via symmetric Dirichlet row/column elimination.
This is the same gauge used by openCARP.

Reciprocity (lead field)
------------------------
For each measurement electrode e (ground g), the ECG is

    V_e(t) = u(e, t) = e_e^T u(t) = - e_e^T K_bulk^{-1} K_i Vm(t)

Define the (Dirichlet-grounded) adjoint solution

    K_bulk  Z_e  =  e_e,     Z_e(g) = 0                         (6)

Then

    V_e(t) = - Z_e^T K_i Vm(t) = Vm(t)^T q_e

where the lead field vector is

    q_e  =  - K_i Z_e                                           (7)

K_i is zero on non-heart rows, so only the heart-node entries of q_e
contribute.  We store  q_heart = q_e[heart nodes]  so the final ECG is
a simple dense  (T, N_heart) @ (N_heart,) matrix-vector product.

Implementation notes
--------------------
* Everything in the solve runs in float64 (matching PETSc defaults).
* The ground is enforced by symmetric Dirichlet elimination -- row g
  and column g of K_bulk are zeroed and K_bulk[g,g] is set to 1.  The
  RHS is e_e with b[g] = 0 (b[e] = 1).  This removes the null space
  exactly, so CG converges in far fewer iterations than a penalty.
* K_bulk and K_i are both assembled on the torso mesh, using the
  torso-mesh fibre field.  The standalone heart mesh is only loaded to
  verify the node correspondence.
* The 12-lead ECG (I, II, III, aVR, aVL, aVF, V1..V6) is computed with
  the Wilson central terminal as reference for the precordial leads.
  Einthoven's law (II = I + III) is checked as an end-to-end sanity
  test.

References
----------
- Potse M., 2018, "Scalable and accurate ECG simulation for
  reaction-diffusion models of the human heart", Front. Physiol. 9:370.
- Bishop M. J., Plank G., 2011, "Bidomain ECG simulations using an
  augmented monodomain model for the cardiac source", IEEE TBME 58(8).
- Krassowska W., Neu J. C., 1994, "Effective boundary conditions for
  syncytial tissues", IEEE TBME 41(2).
- openCARP: https://opencarp.org/  (ecg.cc / IGBReader).
"""

from typing import Dict, Optional
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from torchcor.core import (
    MeshReader,
    Matrices3D,
    Preconditioner,
    ConjugateGradient,
    Conductivity,
)

import warnings
warnings.filterwarnings(
    "ignore", message="Sparse CSR tensor support is in beta state"
)

Tensor = torch.Tensor

# Internal compute precision for the elliptic ECG solve.
# openCARP uses double precision (PETSc default); lower precision
# degrades lead-field accuracy dramatically, especially near the ground
# electrode where the solution has a strong gradient.
_F64 = torch.float64


# ====================================================================== #
#  LeadField
# ====================================================================== #
class LeadField:
    """
    First-principles ECG lead-field solver (reciprocity method).

    All stiffness matrices live on the torso mesh.  The heart mesh is
    only used to verify the node correspondence and to report sizes.

    Typical usage
    -------------
    >>> lf = LeadField(torso_dir, heart_dir, device, dtype)
    >>> lf.add_torso_conductivity([10, 11], g=0.6667)
    >>> lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076,
    ...                                       el=1.0732, et=0.4227)
    >>> lf.build()
    >>> lf.load_electrodes("lf_src.vtx")
    >>> lf.precompute_all()
    >>> ecg = lf.compute_12lead(Vm)          # Vm: (T, N_heart)
    >>> lf.plot_ecg(ecg, "ecg.png", dt=1.0)
    """

    def __init__(self, torso_mesh_dir, heart_mesh_dir, device, dtype):
        self.device = device
        self.dtype = dtype                       # Vm / storage dtype

        # ---- meshes ----
        self._load_torso_mesh(torso_mesh_dir)
        self._load_heart_mesh(heart_mesh_dir)

        self.n_torso = int(self.torso_nodes.shape[0])

        # ---- per-element sigma for the torso mesh (3x3 tensors) ----
        self.torso_sigma = torch.zeros(
            (self.torso_regions.shape[0], 3, 3),
            device=device, dtype=_F64,           # keep in float64
        )
        self._torso_set = torch.zeros(
            self.torso_regions.shape[0],
            device=device, dtype=torch.bool,
        )

        # ---- heart conductivity bookkeeping ----
        self._heart_cond_params = []
        self.heart_tags = []

        # ---- electrodes ----
        self.electrodes: Dict[str, int] = {}
        self.ground = "RL"

        # ---- matrices (filled by build()) ----
        self.K_bulk_csr = None      # float64 CSR: (sigma_i+sigma_e) in heart, sigma_T in torso
        self.K_i_csr    = None      # float64 CSR: sigma_i on heart elems only

        # ---- mapping (filled by build()) ----
        self.heart_to_torso = None  # (N_heart,) torso node ids of heart nodes

        # ---- lead fields (filled by precompute_*) ----
        # q_heart[name] is float64, length = N_heart
        self.q_heart: Dict[str, Tensor] = {}

        # ---- solver state (filled by _prepare_solver()) ----
        self._A_csr = None          # K_bulk with symmetric Dirichlet at ground
        self._pcd   = None          # Jacobi preconditioner (float64)

    # ================================================================== #
    #  Mesh I/O
    # ================================================================== #
    def _load_torso_mesh(self, mesh_dir, unit_conversion=1000):
        reader = MeshReader(mesh_dir)
        nodes, elems, _, fibres = reader.read(unit_conversion=unit_conversion)

        self.torso_nodes   = torch.from_numpy(nodes).to(self.device, _F64)
        self.torso_elems   = torch.from_numpy(elems.Tt.data).to(self.device, torch.long)
        self.torso_regions = torch.from_numpy(elems.Tt.region).to(self.device, torch.long)
        self.torso_fibres  = torch.from_numpy(
            fibres[elems.Tt.idx]
        ).to(self.device, _F64)

    def _load_heart_mesh(self, mesh_dir, unit_conversion=1000):
        reader = MeshReader(mesh_dir)
        nodes, elems, _, fibres = reader.read(unit_conversion=unit_conversion)

        self.heart_nodes   = torch.from_numpy(nodes).to(self.device, _F64)
        self.heart_elems   = torch.from_numpy(elems.Tt.data).to(self.device, torch.long)
        self.heart_regions = torch.from_numpy(elems.Tt.region).to(self.device, torch.long)
        self.heart_fibres  = torch.from_numpy(
            fibres[elems.Tt.idx]
        ).to(self.device, _F64)

    # ================================================================== #
    #  Conductivity registration
    # ================================================================== #
    def add_torso_conductivity(self, tags, g):
        """
        Isotropic scalar conductivity `g` (S/m) for the given torso region
        tags.  Must be called for every non-heart region present in the
        torso mesh; unset elements would contribute a singular stiffness.
        """
        I3 = torch.eye(3, device=self.device, dtype=_F64)
        for tag in tags:
            mask = self.torso_regions == int(tag)
            if not mask.any():
                continue
            self.torso_sigma[mask] = float(g) * I3
            self._torso_set[mask] = True

    def add_heart_conductivity(self, region_ids, il, it, el=None, et=None):
        """
        Anisotropic bidomain conductivity (S/m) for the given heart
        region tags.  il/it are intracellular longitudinal/transverse,
        el/et are extracellular.  The bulk (sigma_i + sigma_e) is used in
        K_bulk and sigma_i alone is used in K_i.
        """
        if el is None or et is None:
            raise ValueError(
                "ECG lead-field requires bidomain conductivities (il, it, el, et)"
            )
        self.heart_tags.extend(int(r) for r in region_ids)
        self._heart_cond_params.append((region_ids, il, it, el, et))

    # ================================================================== #
    #  FEM assembly
    # ================================================================== #
    def build(self):
        """
        Assemble K_bulk and K_i on the torso mesh.

        K_bulk   (N_torso, N_torso)
            Full elliptic operator:
                G = sigma_i + sigma_e    on heart elements
                G = sigma_T              on extracardiac elements

        K_i      (N_torso, N_torso), very sparse
            Source operator:
                sigma_i    on heart elements
                0          elsewhere
        """
        if not self.heart_tags:
            raise RuntimeError("No heart conductivity registered.")

        tag_t = torch.tensor(self.heart_tags, device=self.device, dtype=torch.long)
        heart_mask = torch.isin(self.torso_regions, tag_t)
        if not heart_mask.any():
            raise RuntimeError(
                "heart_tags do not match any region in the torso mesh."
            )

        # Conductivity on torso heart elements (using TORSO-mesh fibres) --
        torso_heart_regions = self.torso_regions[heart_mask]
        torso_heart_fibres  = self.torso_fibres[heart_mask]

        cond = Conductivity(torso_heart_regions, dtype=_F64)
        for region_ids, il, it, el, et in self._heart_cond_params:
            cond.add(region_ids, il, it, el, et)
        sigma_i, sigma_e, _ = cond.calculate_sigma(torso_heart_fibres)

        # Validate torso conductivity assignments
        self._torso_set[heart_mask] = True  # heart covered via bulk below
        unset = (~self._torso_set).nonzero(as_tuple=False).flatten()
        if unset.numel() > 0:
            missing_tags = torch.unique(self.torso_regions[unset]).tolist()
            raise RuntimeError(
                f"Torso regions with no conductivity assigned: {missing_tags}.  "
                f"Call add_torso_conductivity() for every non-heart tag."
            )

        # K_bulk:  G = sigma_i + sigma_e on heart,  sigma_T elsewhere
        self.torso_sigma[heart_mask] = sigma_i + sigma_e

        torso_mats = Matrices3D(
            vertices=self.torso_nodes,
            tetrahedrons=self.torso_elems,
            device=self.device, dtype=_F64,
        )
        K_bulk_coo, _ = torso_mats.assemble_matrices(self.torso_sigma)
        K_bulk_coo = K_bulk_coo.coalesce()
        self.K_bulk_csr = K_bulk_coo.to_sparse_csr()

        # K_i:  sigma_i on heart elements, torso node numbering
        heart_elems_torso = self.torso_elems[heart_mask]
        heart_mats = Matrices3D(
            vertices=self.torso_nodes,
            tetrahedrons=heart_elems_torso,
            device=self.device, dtype=_F64,
        )
        K_i_coo, _ = heart_mats.assemble_matrices(sigma_i)
        K_i_coo = K_i_coo.coalesce()
        self.K_i_csr = K_i_coo.to_sparse_csr()

        # Heart-to-torso node mapping
        self.heart_to_torso = torch.unique(
            heart_elems_torso.reshape(-1), sorted=True
        )
        self._verify_node_mapping()

        n_heart  = self.heart_to_torso.shape[0]
        n_helems = int(heart_mask.sum())
        print(f"  Torso nodes : {self.n_torso:,}")
        print(f"  Heart nodes : {n_heart:,}")
        print(f"  Heart elems : {n_helems:,}")
        print(f"  K_bulk nnz  : {K_bulk_coo.values().numel():,}")
        print(f"  K_i    nnz  : {K_i_coo.values().numel():,}")

    # ------------------------------------------------------------------ #
    def _verify_node_mapping(self):
        """
        Verify the standalone heart mesh is the same sub-domain of the
        torso mesh we just extracted.  Both must share node ordering
        (guaranteed by torchcor.electrocardiogram.torso_heart.TorsoHeartMesh) and the
        positions must match to floating-point precision.
        """
        n_heart  = self.heart_nodes.shape[0]
        n_mapped = self.heart_to_torso.shape[0]
        if n_heart != n_mapped:
            raise RuntimeError(
                f"Node count mismatch: heart mesh has {n_heart} nodes, "
                f"torso sub-domain has {n_mapped}.  Check heart_tags and "
                f"ensure the heart mesh was extracted from this torso."
            )
        torso_sub = self.torso_nodes[self.heart_to_torso]
        diff = (self.heart_nodes - torso_sub).abs().max().item()
        if diff > 1e-3:
            raise RuntimeError(
                f"Heart and torso node positions do not match "
                f"(max diff = {diff:.3e} m).  The heart mesh must be "
                f"extracted from this torso via TorsoHeartMesh."
            )
        print(f"  Node mapping verified (max coord diff = {diff:.2e} m)")

    # ================================================================== #
    #  Electrodes
    # ================================================================== #
    def load_electrodes(self, filepath, names=None):
        """
        Load electrode torso-node indices from a CARP-style .vtx file.

        Default electrode ordering (openCARP "lf_src.vtx"):
            V1, V2, V3, V4, V5, V6, RA, LA, RL, LL
        """
        if names is None:
            names = ["V1","V2","V3","V4","V5","V6","RA","LA","RL","LL"]

        # CARP .vtx format:  line 1 = count, line 2 = "extra"/"intra",
        # remaining lines = node indices.  Some tools omit the "extra"
        # line, so be tolerant.
        with open(filepath, "r") as f:
            # first line:  "<count>" possibly followed by comments
            first = f.readline().strip().split()
            n_expected = int(first[0])
            pos = f.tell()
            peek_raw = f.readline().strip()
            peek = peek_raw.split()[0] if peek_raw else ""
            try:
                int(peek)
                f.seek(pos)                  # second line is an index
            except ValueError:
                pass                         # second line was a tag; skip
            ids = []
            for line in f:
                s = line.strip()
                if not s:
                    continue
                ids.append(int(s.split()[0]))

        if len(ids) != n_expected:
            raise ValueError(
                f"vtx file header says {n_expected} nodes but found {len(ids)}"
            )
        if len(ids) != len(names):
            raise ValueError(
                f"Expected {len(names)} electrodes, got {len(ids)}"
            )
        self.electrodes = dict(zip(names, ids))

        # Range-check
        for name, idx in self.electrodes.items():
            if not (0 <= idx < self.n_torso):
                raise ValueError(
                    f"Electrode {name} index {idx} is out of range "
                    f"(n_torso = {self.n_torso})."
                )
        if self.ground not in self.electrodes:
            raise ValueError(
                f"Ground electrode '{self.ground}' not present in loaded "
                f"electrode set {list(self.electrodes)}."
            )

    # ================================================================== #
    #  Solver preparation
    # ================================================================== #
    def _apply_symmetric_dirichlet(self, K_coo, g_idx):
        """
        Build A = K with row and column `g_idx` zeroed and A[g,g] = 1.

        This enforces u(g) = 0 exactly for any RHS b with b[g] = 0.  The
        resulting A is symmetric positive-definite (the null space of the
        original Neumann K has been removed), so CG converges rapidly.
        """
        idx = K_coo.indices()
        val = K_coo.values()

        mask = (idx[0] != g_idx) & (idx[1] != g_idx)
        new_idx = idx[:, mask]
        new_val = val[mask]

        # Add A[g, g] = 1
        extra_idx = torch.tensor(
            [[g_idx], [g_idx]], device=idx.device, dtype=torch.long
        )
        extra_val = torch.tensor(
            [1.0], device=val.device, dtype=val.dtype
        )
        new_idx = torch.cat([new_idx, extra_idx], dim=1)
        new_val = torch.cat([new_val, extra_val])

        A_coo = torch.sparse_coo_tensor(
            new_idx, new_val, K_coo.size(),
            device=K_coo.device, dtype=K_coo.dtype,
        ).coalesce()
        return A_coo

    def _prepare_solver(self):
        """
        Build the grounded system matrix A and its Jacobi preconditioner.
        Only depends on the ground electrode, so this is done once.
        """
        if self._A_csr is not None:
            return

        ground_idx = int(self.electrodes[self.ground])

        K_coo = self.K_bulk_csr.to_sparse_coo().coalesce()
        A_coo = self._apply_symmetric_dirichlet(K_coo, ground_idx)
        self._A_csr = A_coo.to_sparse_csr()

        self._pcd = Preconditioner()
        self._pcd.create_Jocobi(A_coo)

        print(
            f"  Solver ready  (symmetric Dirichlet, "
            f"ground = {self.ground} @ node {ground_idx}, dtype = float64)"
        )

    # ================================================================== #
    #  Reciprocal solve
    # ================================================================== #
    def _solve_reciprocal(self, electrode_name, a_tol, r_tol, max_iter):
        """
        Solve  A Z = e_electrode   in float64,
        where A = K_bulk with symmetric Dirichlet at the ground node.

        The RHS is a unit delta at the electrode with b[g] = 0; this is
        consistent with u(g) = 0 and produces the reciprocity adjoint.
        """
        e_idx = int(self.electrodes[electrode_name])
        g_idx = int(self.electrodes[self.ground])

        b = torch.zeros(self.n_torso, device=self.device, dtype=_F64)
        b[e_idx] = 1.0
        # b[g_idx] is already 0 (enforced Dirichlet); do NOT add -1 here.
        # If the user supplied e_idx == g_idx, that degenerate case would
        # give Z = 0 and a zero lead field -- catch it:
        if e_idx == g_idx:
            raise ValueError(
                f"Electrode '{electrode_name}' coincides with the ground."
            )
        b[g_idx] = 0.0

        cg = ConjugateGradient(self._pcd, self._A_csr, dtype=_F64)
        cg.initialize(
            x=torch.zeros(self.n_torso, device=self.device, dtype=_F64),
            linear_guess=False,
        )
        Z, n_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

        if torch.isnan(Z).any():
            raise RuntimeError(
                f"CG produced NaNs for electrode {electrode_name}"
            )

        # Residual check
        r = b - (self._A_csr @ Z)
        res = torch.linalg.vector_norm(r).item()
        b_norm = torch.linalg.vector_norm(b).item() + 1e-300

        Z_heart = Z[self.heart_to_torso]
        print(
            f"    {electrode_name:>3s}:  CG iters = {n_iter:5d}   "
            f"|res|/|b| = {res/b_norm:.2e}   "
            f"Z_heart in [{Z_heart.min().item():+.3e}, "
            f"{Z_heart.max().item():+.3e}]"
        )
        return Z

    # ================================================================== #
    #  Lead-field precomputation
    # ================================================================== #
    def precompute_electrode(self, name, a_tol=1e-10, r_tol=1e-10,
                             max_iter=20000):
        """
        Precompute the lead-field vector for one electrode.

            q_e      = - K_i Z_e            (N_torso,)
            q_heart  = q_e[heart nodes]     (N_heart,)

        Since K_i has support only on heart elements, q_e vanishes on
        all extracardiac rows; extracting q_heart is lossless and saves
        memory / compute at evaluation time.
        """
        self._prepare_solver()
        Z = self._solve_reciprocal(name, a_tol, r_tol, max_iter)

        q_full  = -(self.K_i_csr @ Z)                     # (N_torso,)
        q_heart = q_full[self.heart_to_torso].contiguous() # (N_heart,)
        self.q_heart[name] = q_heart                       # keep float64

    def precompute_all(self, a_tol=1e-10, r_tol=1e-10, max_iter=20000):
        """
        Precompute lead fields for every electrode except the ground.
        The ground electrode has q = 0 by construction.
        """
        if not self.electrodes:
            raise RuntimeError(
                "No electrodes loaded.  Call load_electrodes() first."
            )
        self._prepare_solver()
        for name in self.electrodes:
            if name == self.ground:
                # Zero lead field for the grounded electrode
                self.q_heart[name] = torch.zeros(
                    self.heart_to_torso.shape[0],
                    device=self.device, dtype=_F64,
                )
                continue
            self.precompute_electrode(name, a_tol, r_tol, max_iter)

    # ================================================================== #
    #  ECG computation
    # ================================================================== #
    def unipolar(self, Vm: Tensor, electrode: str) -> Tensor:
        """
        Unipolar signal at one electrode relative to the ground.

            U_e(t) = Vm(t) . q_heart[e]

        Vm may be (T, N_heart) or (N_heart,).  Computation is done in
        float64 for accuracy and the result is returned in `self.dtype`.
        """
        if electrode not in self.q_heart:
            raise KeyError(
                f"Lead field for '{electrode}' not precomputed. "
                f"Available: {list(self.q_heart)}"
            )
        # Compute on whichever device Vm currently lives on.  This lets
        # the caller keep a large Vm tensor on CPU to avoid GPU OOM on
        # big meshes, while still getting float64 precision.
        Vm64 = Vm.to(_F64)
        q = self.q_heart[electrode].to(Vm64.device)        # float64
        sig = Vm64 @ q
        return sig.to(self.dtype)

    # ================================================================== #
    #  Direct forward solve (ground-truth cross-check of the lead field)
    # ================================================================== #
    def unipolar_direct(self, Vm: Tensor, frames=None,
                        a_tol=1e-10, r_tol=1e-10, max_iter=20000) -> Dict[str, Tensor]:
        """
        Compute the unipolar electrode potentials by solving the FULL
        forward problem directly, instead of via the precomputed lead
        fields.  For each requested time frame t this solves

            K_bulk  phi(t) = - K_i Vm(t),    phi(ground) = 0

        (the same grounded operator used for the reciprocal solve) and
        reads phi at every electrode node.  By reciprocity this MUST equal
        ``unipolar()`` for every electrode; comparing the two is a
        ground-truth check of the whole lead-field pipeline on the real
        mesh, independent of any reciprocity assumption.

        Parameters
        ----------
        Vm     : (T, N_heart) or (N_heart,) transmembrane potential.
        frames : iterable of frame indices to solve (default: all).  Each
                 frame is one large elliptic solve, so pass a handful of
                 informative frames (e.g. the QRS peak and a T-wave frame)
                 rather than the whole trace.

        Returns
        -------
        dict {electrode_name: (len(frames),) potential relative to ground}.
        """
        self._prepare_solver()

        if Vm.dim() == 1:
            Vm = Vm.unsqueeze(0)
        if frames is None:
            frames = range(Vm.shape[0])
        frames = list(frames)

        g_idx = int(self.electrodes[self.ground])
        names = list(self.electrodes)
        out = {n: torch.zeros(len(frames), dtype=self.dtype) for n in names}

        for fi, t in enumerate(frames):
            vm_heart = Vm[t].to(self.device, _F64)
            vm_full = torch.zeros(self.n_torso, device=self.device, dtype=_F64)
            vm_full[self.heart_to_torso] = vm_heart

            rhs = -(self.K_i_csr @ vm_full)
            rhs[g_idx] = 0.0                      # consistent with phi(g)=0

            cg = ConjugateGradient(self._pcd, self._A_csr, dtype=_F64)
            cg.initialize(
                x=torch.zeros(self.n_torso, device=self.device, dtype=_F64),
                linear_guess=False,
            )
            phi, n_iter = cg.solve(rhs, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            phi = phi - phi[g_idx]                # enforce exact ground gauge

            for n in names:
                out[n][fi] = phi[int(self.electrodes[n])].to(self.dtype).cpu()
            print(f"    direct frame {t:5d}: CG iters = {n_iter}")

        return out

    def validate_direct(self, Vm: Tensor, frames, **solve_kw) -> float:
        """
        Cross-check the reciprocity lead field against a direct forward
        solve at the given frames.  Prints a per-electrode comparison and
        returns the maximum absolute difference.  A value at solver
        tolerance (<~1e-6 of the signal range) confirms the lead field is
        a correct realisation of the forward problem on this mesh; if the
        ECG still looks wrong, the cause is the Vm input, not the lead
        field.
        """
        direct = self.unipolar_direct(Vm, frames=frames, **solve_kw)
        frames = list(frames)
        Vm2 = Vm.unsqueeze(0) if Vm.dim() == 1 else Vm
        max_err = 0.0
        print("  electrode |   reciprocity        direct          |err|")
        for n in self.electrodes:
            rec = self.unipolar(Vm2[frames], n).to(self.dtype).cpu()
            dir_ = direct[n]
            err = (rec - dir_).abs().max().item()
            max_err = max(max_err, err)
            print(f"  {n:>3s}: {rec.tolist()}  vs  {dir_.tolist()}  |err|={err:.2e}")
        print(f"  MAX |reciprocity - direct| = {max_err:.3e}")
        return max_err

    def compute_12lead(self, Vm: Tensor) -> Dict[str, Tensor]:
        """
        Standard clinical 12-lead ECG from the transmembrane potential.

        Limb leads (Einthoven):
            I   = LA - RA
            II  = LL - RA
            III = LL - LA
        Augmented (Goldberger):
            aVR = RA - (LA + LL)/2
            aVL = LA - (RA + LL)/2
            aVF = LL - (RA + LA)/2
        Precordial (Wilson Central Terminal):
            WCT = (RA + LA + LL)/3
            Vi  = unipolar(Vi) - WCT       for i = 1..6

        Einthoven's law (II = I + III) is checked as an end-to-end test.
        """
        ra = self.unipolar(Vm, "RA")
        la = self.unipolar(Vm, "LA")
        ll = self.unipolar(Vm, "LL")

        I   = la - ra
        II  = ll - ra
        III = ll - la

        aVR = ra - 0.5 * (la + ll)
        aVL = la - 0.5 * (ra + ll)
        aVF = ll - 0.5 * (ra + la)

        wct = (ra + la + ll) / 3.0

        ecg = {
            "I": I, "II": II, "III": III,
            "aVR": aVR, "aVL": aVL, "aVF": aVF,
        }
        for v in ("V1", "V2", "V3", "V4", "V5", "V6"):
            ecg[v] = self.unipolar(Vm, v) - wct

        # Einthoven consistency check (pure algebra -- any deviation is
        # float round-off only).
        err = (II - (I + III)).abs().max().item()
        rng = II.abs().max().item() + 1e-300
        print(f"  Einthoven check  max|II-(I+III)| = {err:.2e} "
              f"(rel {err/rng:.2e})")

        return ecg

    # ================================================================== #
    #  Plotting
    # ================================================================== #
    def plot_ecg(self, ecg_dict, filename="ecg_12lead.png",
                 dt: Optional[float] = None, smooth_sigma: float = 0.0):
        """
        Render the 12-lead ECG in the conventional 3 x 4 clinical layout.

        Parameters
        ----------
        ecg_dict     : dict returned by compute_12lead()
        filename     : output PNG
        dt           : sample period (ms); if None the x axis is samples
        smooth_sigma : Gaussian sigma in samples (0 = no smoothing).
                       Smoothing is for display only and is off by
                       default so the raw waveform is visible.
        """
        order = [
            "I",   "aVR", "V1", "V4",
            "II",  "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6",
        ]
        fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True)
        for ax, lead in zip(axes.flat, order):
            sig = ecg_dict[lead].detach().cpu().numpy()
            if smooth_sigma > 0:
                sig = gaussian_filter1d(sig, sigma=smooth_sigma)
            if dt is not None:
                t = np.arange(len(sig)) * dt
                ax.plot(t, sig, "k", lw=0.8)
                ax.set_xlabel("ms", fontsize=8)
            else:
                ax.plot(sig, "k", lw=0.8)
            ax.set_title(lead, fontsize=10, fontweight="bold")
            ax.set_ylim(-5, 4)            # fixed, consistent scale (~ -4..4 mV)
            ax.set_yticks([-4, -2, 0, 2, 4])
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.supylabel("Potential (a.u.)", fontsize=11)
        fig.suptitle("12-Lead ECG (lead-field reciprocity, Dirichlet gauge)",
                     fontsize=13)
        plt.tight_layout(rect=[0.03, 0.0, 1, 0.96])
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"  Saved {filename}")
