"""Global assembly for :mod:`torchcor.mechanics`.

Turns element-level constitutive response into the global force imbalance
``R(u) = f_int(u) - f_ext(u)`` and its consistent tangent.

Two pieces:

* :class:`SparseAssembler` owns the sparsity pattern.  It is built once from
  the connectivity; assembly thereafter is a single scatter-add into the value
  array, which is the only shape this operation has that a GPU likes.
* :class:`FiniteStrainProblem` evaluates the constitutive response.  Every
  kernel is a batched ``einsum`` over a chunk of elements -- there is no Python
  loop over elements or nodes anywhere in the hot path; the chunk loop exists
  only to bound peak memory and is a handful of iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from torchcor.mechanics.boundary import (
    DirichletBC, DirichletConstraints, FollowerPressure,
)
from torchcor.mechanics.elements import LagrangeHex, LagrangeQuad
from torchcor.mechanics.material import (
    AugmentedLagrangian, HyperelasticMaterial, MaterialAxes, OrientedMaterial,
)
from torchcor.mechanics.mesh import HexMesh

@dataclass
class AssemblyInfo:
    """Scalar diagnostics of one assembly, held **on device**.

    Keeping them as tensors lets the caller choose when to synchronise, and how
    much to carry across in one go.  :meth:`resolve` performs exactly one host
    transfer for every scalar an assembly produced, optionally folding in extra
    device scalars the caller needs at the same moment (a line-search
    directional derivative, say), so a Newton iteration costs one stall rather
    than five.
    """

    min_jacobian: torch.Tensor
    internal_norm: torch.Tensor
    external_norm: torch.Tensor
    residual_norm: torch.Tensor

    def resolve(self, *extra: torch.Tensor) -> Dict[str, float]:
        """One host transfer; extra values come back under ``"extra"``."""
        values = torch.stack([self.min_jacobian, self.internal_norm,
                              self.external_norm, self.residual_norm,
                              *extra]).tolist()
        info = {"min_J": values[0], "fint_norm": values[1],
                "fext_norm": values[2], "residual_norm": values[3]}
        if extra:
            info["extra"] = values[4:]
        return info


class SparseAssembler:
    """Fixed-pattern CSR assembler for one or more connectivity blocks.

    The (row, col) pattern of every element/face block is built once; assembly
    then reduces to a single ``index_add_`` into the value array, which is the
    only GPU-friendly way to do repeated FE assembly.

    Parameters
    ----------
    n_dofs:
        Global number of degrees of freedom.
    connectivities:
        One ``(n_blocks, n_nodes)`` node table per contribution (e.g. volume
        elements, then pressure faces).  ``scatter(i)`` returns the map for the
        ``i``-th table.
    """

    def __init__(self, n_dofs: int, connectivities: Sequence[torch.Tensor],
                 device: torch.device) -> None:
        self.n_dofs = int(n_dofs)
        self.device = device

        keys: List[torch.Tensor] = []
        sizes: List[int] = []
        self.element_dofs: List[torch.Tensor] = []

        for conn in connectivities:
            conn = torch.as_tensor(conn, dtype=torch.long, device=device)
            comps = torch.arange(3, device=device)
            edofs = (3 * conn[:, :, None] + comps[None, None, :]).reshape(conn.shape[0], -1)
            self.element_dofs.append(edofs.contiguous())

            nd = edofs.shape[1]
            rows = edofs[:, :, None].expand(-1, -1, nd).reshape(-1)
            cols = edofs[:, None, :].expand(-1, nd, -1).reshape(-1)
            keys.append(rows * self.n_dofs + cols)
            sizes.append(rows.numel())

        all_keys = torch.cat(keys) if len(keys) > 1 else keys[0]
        uniq, inverse = torch.unique(all_keys, sorted=True, return_inverse=True)
        del all_keys, keys
        self._keys = uniq
        self._transpose_perm: Optional[torch.Tensor] = None

        self.rows = torch.div(uniq, self.n_dofs, rounding_mode="floor")
        self.cols = uniq - self.rows * self.n_dofs
        self.nnz = int(uniq.numel())
        self.scatter_maps = list(torch.split(inverse, sizes))

        counts = torch.bincount(self.rows, minlength=self.n_dofs)
        self.crow_indices = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])

        # Nodal 3x3 block structure, for the block-Jacobi preconditioner.
        node_r = torch.div(self.rows, 3, rounding_mode="floor")
        node_c = torch.div(self.cols, 3, rounding_mode="floor")
        blk = node_r == node_c
        self._blk_pos = torch.nonzero(blk, as_tuple=False).reshape(-1)
        self._blk_node = node_r[blk]
        self._blk_i = self.rows[blk] - 3 * self._blk_node
        self._blk_j = self.cols[blk] - 3 * self._blk_node

        self._diag_pos = torch.nonzero(self.rows == self.cols, as_tuple=False).reshape(-1)
        self._constraint_ready = False

    # ---------------------------------------------------------------- assembly
    def scatter(self, index: int) -> torch.Tensor:
        return self.scatter_maps[index]

    def new_values(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.nnz, dtype=dtype, device=self.device)

    def accumulate(self, values: torch.Tensor, block: int,
                   local: torch.Tensor) -> torch.Tensor:
        """Scatter-add local matrices ``(n_blocks, nd, nd)`` into ``values``."""
        values.index_add_(0, self.scatter_maps[block], local.reshape(-1))
        return values

    def to_csr(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(
            self.crow_indices, self.cols, values,
            size=(self.n_dofs, self.n_dofs), check_invariants=False)

    # ------------------------------------------------------------- constraints
    def set_constrained(self, constrained: torch.Tensor) -> None:
        """Register the constrained-dof mask used by :meth:`apply_constraints`."""
        self._off_mask = constrained[self.rows] | constrained[self.cols]
        self._diag_constrained = self._diag_pos[constrained[self.rows[self._diag_pos]]]
        self._constraint_ready = True

    def apply_constraints(self, values: torch.Tensor) -> torch.Tensor:
        """Zero constrained rows/columns and put 1 on their diagonal (in place).

        ``masked_fill_`` rather than ``values[mask] = 0``: boolean *indexing*
        has to learn the size of its result, which forces a device-to-host
        transfer on every Newton step.
        """
        if not self._constraint_ready:
            raise RuntimeError("call set_constrained() before apply_constraints()")
        values.masked_fill_(self._off_mask, 0.0)
        values[self._diag_constrained] = 1.0
        return values

    @property
    def transpose_perm(self) -> torch.Tensor:
        """Permutation ``P`` with ``values[P]`` the value array of ``A^T``.

        Finite-element patterns are structurally symmetric, so the transposed
        key of every stored entry is itself stored; the permutation is found
        once by a binary search and reused by every Newton step.
        """
        if self._transpose_perm is None:
            key_t = self.cols * self.n_dofs + self.rows
            perm = torch.searchsorted(self._keys, key_t)
            if not bool((self._keys[perm] == key_t).all()):
                raise RuntimeError("sparsity pattern is not structurally symmetric")
            self._transpose_perm = perm
        return self._transpose_perm

    # ---------------------------------------------------------------- extracts
    def diagonal(self, values: torch.Tensor) -> torch.Tensor:
        d = torch.zeros(self.n_dofs, dtype=values.dtype, device=self.device)
        d[self.rows[self._diag_pos]] = values[self._diag_pos]
        return d

    def nodal_blocks(self, values: torch.Tensor) -> torch.Tensor:
        """Extract the ``(n_nodes, 3, 3)`` diagonal nodal blocks."""
        n_nodes = self.n_dofs // 3
        B = torch.zeros(n_nodes, 3, 3, dtype=values.dtype, device=self.device)
        B[self._blk_node, self._blk_i, self._blk_j] = values[self._blk_pos]
        return B


class FiniteStrainProblem:
    """Total-Lagrangian quasi-static finite-strain problem on a hexahedral mesh.

    Assembles the residual

    .. math::  R(u) = f^{\\mathrm{int}}(u) - f^{\\mathrm{ext}}(u)

    and its consistent tangent.  Incompressibility is imposed through an
    mixed pressure space of complete degree p-1, integrated with the material
    on the same full quadrature rule. Local pressure and dilatation elimination
    gives a displacement system at each augmented-Lagrangian iteration.

    Parameters
    ----------
    mesh:
        Hexahedral mesh; also fixes the element order.
    material:
        Constitutive law, evaluated in the local material frame.
    axes:
        Material (fibre) frame.  ``None`` means the frame coincides with the
        global axes, and the rotation is skipped entirely.
    bulk_modulus:
        Augmented Lagrangian penalty ``kappa``.  Only affects conditioning and
        the convergence rate of the multiplier update, not the converged answer,
        because the multiplier field absorbs the remaining pressure.
    dirichlet:
        Prescribed-displacement conditions.
    pressures:
        Follower pressure loads.
    chunk_size:
        Number of elements processed per batched kernel; caps peak memory of the
        tangent evaluation.
    """

    def __init__(
        self,
        mesh: HexMesh,
        material: HyperelasticMaterial,
        axes: Optional[MaterialAxes] = None,
        bulk_modulus: float = 1.0e2,
        dirichlet: Sequence[DirichletBC] = (),
        pressures: Sequence[FollowerPressure] = (),
        chunk_size: int = 2048,
        bc_atol: float = 1e-12,
        quadrature_order: Optional[int] = None,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.chunk_size = int(chunk_size)
        self.bc_atol = float(bc_atol)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        if not mesh.dtype.is_floating_point:
            raise TypeError(f"mesh dtype must be a floating type, got {mesh.dtype}")

        # Everything the problem holds is moved onto the mesh's device once, at
        # construction.  Boundary data built on another device would otherwise
        # force a transfer on every assembly, silently, inside the hot loop.
        self.dirichlet = list(dirichlet)
        self.pressures = [
            bc if bc.face_set.connectivity.device == mesh.device
            else replace(bc, face_set=bc.face_set.to(mesh.device))
            for bc in pressures
        ]

        self.dtype = mesh.dtype
        self.device = mesh.device
        self.n_dofs = mesh.n_dofs

        order = mesh.order
        # Includes the pressure mass matrix on curved isoparametric elements.
        # Q2 needs four points per axis, not the usual three for affine cells.
        n_gauss = max(order + 1, (5*order - 1)//2) if quadrature_order is None else int(quadrature_order)
        if n_gauss < order + 1:
            raise ValueError("quadrature_order must be at least mesh.order + 1")
        self.elem_full = LagrangeHex(order, n_gauss, self.dtype, self.device)
        self.face_elem = LagrangeQuad(order, order + 1, self.dtype, self.device)

        self.X = mesh.points
        self.cells = mesh.cells

        self.dNdX_full, self.w_full = self._reference_geometry(self.elem_full)

        # The constitutive behaviour is composed, not implemented here: the
        # material knows which frame it is written in, and incompressibility is
        # its own model with its own multiplier field.
        self.oriented = OrientedMaterial(
            material, axes, self.dtype, self.device,
            batch_shape=(mesh.n_cells, self.elem_full.N.shape[0]),
            reference_points=(torch.einsum("qa,eai->eqi", self.elem_full.N, self.X[self.cells])
                              if axes is not None and any(callable(v) for v in
                                  (axes.f, axes.s, axes.n)) else None))
        self.volumetric = AugmentedLagrangian(
            bulk_modulus, self.elem_full.points, self.w_full, order - 1)
        self.axes = axes

        self._setup_constraints()
        self._setup_assembler()
        self._eye3 = torch.eye(3, dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"FiniteStrainProblem({self.mesh.n_cells} cells, "
                f"{self.n_dofs} DOF, {self.material!r}, "
                f"device={self.device}, dtype={self.dtype})")

    # -------------------------------------------------------------- geometry
    def _reference_geometry(self, elem: LagrangeHex) -> Tuple[torch.Tensor, torch.Tensor]:
        Xe = self.X[self.cells]                                   # (ne, nen, 3)
        J0 = torch.einsum("eaI,qaj->eqIj", Xe, elem.dN)           # dX_I / dxi_j
        detJ0 = torch.linalg.det(J0)
        if not bool((torch.isfinite(detJ0) & (detJ0 > 0)).all()):
            raise ValueError("mesh contains inverted or degenerate elements "
                             f"(min |J| = {float(detJ0.min()):.3e})")
        invJ0 = torch.linalg.inv(J0)
        dNdX = torch.einsum("qaj,eqjI->eqaI", elem.dN, invJ0)
        w = elem.weights[None, :] * detJ0
        return dNdX.contiguous(), w.contiguous()

    def reference_volume(self) -> float:
        return float(self.w_full.sum())

    # -------------------------------------------------------- material frame
    # ------------------------------------------------------------ constraints
    def _setup_constraints(self) -> None:
        self.constraints = DirichletConstraints(
            self.dirichlet, self.n_dofs, self.dtype, self.device, self.bc_atol)
        # Kept as attributes for convenience; the constraint object owns them.
        self.constrained_dofs = self.constraints.dofs
        self.constrained_values = self.constraints.values
        self.constrained_mask = self.constraints.mask
        self.free_mask = self.constraints.free_mask

    def apply_dirichlet(self, u: torch.Tensor, load_factor: float = 1.0) -> torch.Tensor:
        return self.constraints.apply(u, load_factor)

    # -------------------------------------------------------------- assembler
    def _setup_assembler(self) -> None:
        blocks = [self.cells] + [bc.face_set.connectivity for bc in self.pressures]
        self.assembler = SparseAssembler(self.n_dofs, blocks, self.device)
        self.assembler.set_constrained(self.constrained_mask)

    # --------------------------------------------------------------- kernels
    def _contribution(self, F, S, D9, dNdX, w, tangent):
        """Element force (and stiffness) for one stress measure and quadrature rule."""
        P = torch.einsum("eqiI,eqIJ->eqiJ", F, S)
        fe = torch.einsum("eq,eqiJ,eqaJ->eai", w, P, dNdX)
        if not tangent:
            return fe, None

        ne, nq, nen = dNdX.shape[0], dNdX.shape[1], dNdX.shape[2]
        M = torch.einsum("eqiI,eqaJ->eqaiIJ", F, dNdX).reshape(ne, nq, nen * 3, 9)
        T = torch.einsum("eqAm,eqmn->eqAn", M, D9)
        K = torch.einsum("eq,eqAn,eqBn->eAB", w, T, M)

        G = torch.einsum("eqaI,eqIJ,eqbJ->eqab", dNdX, S, dNdX)
        Kg = torch.einsum("eq,eqab->eab", w, G)
        K.view(ne, nen, 3, nen, 3).add_(
            Kg[:, :, None, :, None] * self._eye3[None, None, :, None, :])
        return fe, K

    def _deformation_gradient(self, xe, dNdX):
        return torch.einsum("eai,eqaI->eqiI", xe, dNdX)

    # ---------------------------------------------------------------- assembly
    def evaluate(self, u: torch.Tensor, load_factor: float = 1.0, tangent: bool = True
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor], AssemblyInfo]:
        """Assemble the residual (and tangent values) at displacement ``u``.

        Returns ``(R, values, info)``; ``values`` is ``None`` when
        ``tangent=False``, and every diagnostic in ``info`` is still a device
        tensor -- see :class:`AssemblyInfo`.  Nothing in this method transfers
        to the host.
        """
        xc = self.X + u.reshape(-1, 3)
        R = torch.zeros(self.n_dofs, dtype=self.dtype, device=self.device)
        values = self.assembler.new_values(self.dtype) if tangent else None

        min_J: List[torch.Tensor] = []
        edofs = self.assembler.element_dofs[0]
        scatter = self.assembler.scatter(0)
        nd = edofs.shape[1]

        for start in range(0, self.mesh.n_cells, self.chunk_size):
            stop = min(start + self.chunk_size, self.mesh.n_cells)
            sl = slice(start, stop)
            xe = xc[self.cells[sl]]

            # ---- isochoric response, full quadrature -------------------------
            F = self._deformation_gradient(xe, self.dNdX_full[sl])
            E = 0.5 * (torch.einsum("eqiI,eqiJ->eqIJ", F, F) - self._eye3)
            S, D9 = self.oriented.response(E, sl, tangent, load_factor=load_factor)
            Sv, Dv9, J = self.volumetric.response(F, sl, tangent)
            fe, Ke = self._contribution(
                F, S + Sv, D9 + Dv9 if tangent else None,
                self.dNdX_full[sl], self.w_full[sl], tangent)
            min_J.append(J.amin())

            R.index_add_(0, edofs[sl].reshape(-1), fe.reshape(-1))
            if tangent:
                Ke = Ke + self.volumetric.stiffness(F, self.dNdX_full[sl], sl)
                values.index_add_(0, scatter[start * nd * nd: stop * nd * nd],
                                  Ke.reshape(-1))

        # Scale of the internal force *before* the external load is subtracted.
        # This, not the applied load, is the natural normaliser for the residual:
        # a bending beam develops internal stresses far larger than the surface
        # pressure that drives it.
        fint_norm = torch.linalg.vector_norm(R)

        fext_norm = self._add_pressure(xc, R, values, load_factor, tangent)

        return R, values, AssemblyInfo(
            min_jacobian=torch.stack(min_J).amin(),
            internal_norm=fint_norm,
            external_norm=fext_norm,
            residual_norm=self.constraints.free_norm(R),
        )

    def _add_pressure(self, xc, R, values, load_factor, tangent) -> torch.Tensor:
        """Scatter each surface load's own contribution into the system."""
        external = torch.zeros((), dtype=self.dtype, device=self.device)
        for block, bc in enumerate(self.pressures, start=1):
            force, stiffness = bc.contribution(xc, self.face_elem, load_factor,
                                               tangent)
            external = external + (force ** 2).sum()
            R.index_add_(0, self.assembler.element_dofs[block].reshape(-1),
                         -force.reshape(-1))
            if tangent:
                values.index_add_(0, self.assembler.scatter(block),
                                  -stiffness.reshape(-1))
        return external.sqrt()

    # ------------------------------------------------- incompressibility state
    def jacobians(self, u: torch.Tensor) -> torch.Tensor:
        """J = det F at the integration points."""
        dNdX = self.dNdX_full
        xc = self.X + u.reshape(-1, 3)
        out = []
        for start in range(0, self.mesh.n_cells, self.chunk_size):
            sl = slice(start, min(start + self.chunk_size, self.mesh.n_cells))
            F = self._deformation_gradient(xc[self.cells[sl]], dNdX[sl])
            out.append(torch.linalg.det(F))
        return torch.cat(out, dim=0)

    def sample_jacobians(self, u: torch.Tensor,
                         n_gauss: Optional[int] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """J and physical weights at integration or independent sample points.

        A denser ``n_gauss`` checks local error between assembly points.
        """
        if n_gauss is None:
            return self.jacobians(u), self.w_full
        elem = LagrangeHex(self.mesh.order, n_gauss, self.dtype, self.device)
        dNdX, w = self._reference_geometry(elem)
        xc = self.X + u.reshape(-1, 3)
        out = []
        for start in range(0, self.mesh.n_cells, self.chunk_size):
            sl = slice(start, min(start + self.chunk_size, self.mesh.n_cells))
            out.append(torch.linalg.det(
                self._deformation_gradient(xc[self.cells[sl]], dNdX[sl])))
        return torch.cat(out, dim=0), w

    def volume_error_per_cell(self, u: torch.Tensor,
                              n_gauss: Optional[int] = None) -> torch.Tensor:
        """Volume-weighted RMS ``|J - 1|`` in each element.

        Where the error sits matters as much as its size: a maximum carried by
        a single distorted element near a clamped face is a different finding
        from one spread through the body.
        """
        J, w = self.sample_jacobians(u, n_gauss)
        return (((J - 1.0) ** 2 * w).sum(dim=1) / w.sum(dim=1)).sqrt()

    def volume_error(self, u: torch.Tensor) -> float:
        """Maximum projected volume residual of the mixed constraint."""
        return float(self.volumetric.residual(self.jacobians(u)).abs().max())

    def volume_diagnostics(self, u: torch.Tensor,
                           n_gauss: Optional[int] = None) -> Dict[str, float]:
        """Projected constraint and independently sampled local volume errors.

        ``constraint_error`` measures the mixed moment residual at assembly
        points. ``max_error`` and volume-weighted ``rms_error`` measure J-1 at
        the requested sampling rule; ``volume_change`` is total dV/V.
        ``min_jacobian`` is the smallest sampled J. Sampling does not certify
        positivity or bound the error everywhere inside an element.
        """
        Jr = self.jacobians(u)
        Jf, w = self.sample_jacobians(u, n_gauss)
        v0 = w.sum()
        stats = torch.stack([
            self.volumetric.residual(Jr).abs().amax(),
            (Jf - 1.0).abs().amax(),
            (((Jf - 1.0) ** 2 * w).sum() / v0).sqrt(),
            (Jf * w).sum() / v0 - 1.0,
            Jf.amin(),
        ]).tolist()
        return dict(zip(("constraint_error", "max_error", "rms_error",
                         "volume_change", "min_jacobian"), stats))

    @property
    def p_bar(self) -> torch.Tensor:
        """The incompressibility multiplier field (owned by :attr:`volumetric`)."""
        return self.volumetric.multiplier

    def update_multipliers(self, u: torch.Tensor) -> float:
        """Uzawa update of the multiplier; returns ``max |J - 1|``."""
        J = self.jacobians(u)
        self.volumetric.update(J)
        return float(self.volumetric.residual(J).abs().max())

    def reset_multipliers(self) -> None:
        self.volumetric.reset()

    # ---------------------------------------------------------------- helpers
    def deformed_points(self, u: torch.Tensor) -> torch.Tensor:
        return self.X + u.reshape(-1, 3)
