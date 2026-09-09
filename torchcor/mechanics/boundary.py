"""Boundary conditions for :mod:`torchcor.mechanics`.

Prescribed displacements (:class:`DirichletBC`) and applied loads
(:class:`FollowerPressure`), plus :class:`DirichletConstraints`, which resolves
a list of possibly overlapping Dirichlet conditions into the single constrained
degree-of-freedom set the assembler and solver work with.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import List, Sequence, Tuple

import torch

from torchcor.mechanics.mesh import FaceSet


__all__ = ["DirichletBC", "FollowerPressure", "DirichletConstraints"]


def _index_vector(values, name: str, device: torch.device) -> torch.Tensor:
    """Accept one-dimensional integer indices; never truncate floating values."""
    if isinstance(values, (list, tuple)):
        if not values:
            return torch.empty(0, dtype=torch.long, device=device)
        # A mixed list would otherwise promote booleans to integers.
        if any(isinstance(v, bool) or (isinstance(v, torch.Tensor)
                                      and v.dtype == torch.bool) for v in values):
            raise ValueError(f"{name} must contain integer indices, not booleans")
    try:
        indices = torch.as_tensor(values, device=device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must contain integer indices within int64 range") from error
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of indices")
    if indices.dtype not in (torch.uint8, torch.uint16, torch.uint32,
                              torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must have integer dtype (uint64 is unsupported)")
    indices = indices.to(dtype=torch.long)
    if bool((indices < 0).any()):
        raise ValueError(f"{name} must contain nonnegative indices")
    return indices


@dataclass
class DirichletBC:
    """Prescribed displacement on a set of nodes.

    Parameters
    ----------
    nodes:
        ``(n,)`` integer node indices. Floating-point and boolean indices are
        rejected rather than converted.
    components:
        Integer indices of the constrained displacement components: 0, 1 or 2.
    values:
        Finite prescribed displacement, broadcastable to
        ``(n, len(components))``: a scalar, one value per component, a column
        ``(n, 1)``, or a full array. Values are scaled by the load factor during
        continuation, so a non-zero datum is ramped alongside the loads.
    """

    nodes: torch.Tensor
    components: Sequence[int] = (0, 1, 2)
    values: float | torch.Tensor = 0.0
    name: str = ""

    @classmethod
    def on_surface(cls, mesh, surface: str, components: Sequence[int] = (0, 1, 2),
                   value=0.0) -> "DirichletBC":
        """Constrain a named surface of ``mesh``.

        ``surface`` is a name the mesh registered, e.g. ``"x-"`` for a box or
        ``"base"`` for a ventricle.
        """
        if surface in mesh.node_sets:
            nodes = mesh.node_set(surface)
        elif surface in mesh.face_sets:
            nodes = mesh.face_set(surface).nodes()
        else:
            raise KeyError(f"unknown surface {surface!r}; this mesh has "
                           f"{sorted(mesh.node_sets)}")
        return cls(nodes, components, value, name=surface)

    def dofs_and_values(self, dtype: torch.dtype, device: torch.device,
                        *, n_nodes: int | None = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = _index_vector(self.nodes, "Dirichlet nodes", device)
        comps = _index_vector(self.components, "Dirichlet components", device)
        if bool((comps > 2).any()):
            raise ValueError("Dirichlet components must be 0, 1 or 2")
        # Bound nodes before multiplying by three; a wrapped DOF could alias a
        # valid node and silently constrain the wrong part of the mesh.
        max_node = (torch.iinfo(torch.long).max - 2) // 3
        if bool((nodes > max_node).any()):
            raise ValueError("Dirichlet node index is too large to form displacement DOFs")
        if n_nodes is not None and bool((nodes >= n_nodes).any()):
            raise ValueError(f"Dirichlet node index out of range for a mesh with {n_nodes} nodes")
        dofs = (3 * nodes[:, None] + comps[None, :]).reshape(-1)

        vals = torch.as_tensor(self.values, dtype=dtype, device=device)
        if not bool(torch.isfinite(vals).all()):
            raise ValueError("prescribed displacements must be finite")
        shape = (nodes.numel(), comps.numel())
        try:
            vals = vals.expand(shape).reshape(-1)
        except RuntimeError as error:
            raise ValueError(f"prescribed displacements with shape {tuple(vals.shape)} "
                             f"must broadcast to {shape} (nodes, components)") from error
        return dofs, vals.contiguous()


_LEVI_CIVITA: dict = {}


def _levi_civita(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, str(device))
    if key not in _LEVI_CIVITA:
        e = torch.zeros(3, 3, 3, dtype=dtype, device=device)
        e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1.0
        e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1.0
        _LEVI_CIVITA[key] = e
    return _LEVI_CIVITA[key]


@dataclass
class FollowerPressure:
    """Pressure acting on the *deformed* configuration of a surface.

    The traction is ``t = -p n`` with ``n`` the outward unit normal of the
    deformed surface and the load scaling with the deformed area, i.e. a true
    follower load.  A positive ``pressure`` therefore pushes the body inwards
    through that face.

    The load knows how to compute itself: :meth:`contribution` returns the nodal
    forces and the load stiffness, so assembly only has to scatter them.  That
    keeps "what a pressure does" in one place, and is what lets a new kind of
    surface load be added without touching the assembler.
    """

    face_set: FaceSet
    pressure: float
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.pressure, torch.Tensor) and self.pressure.ndim != 0:
            raise ValueError("pressure must be a finite scalar")
        try:
            self.pressure = float(self.pressure)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("pressure must be a finite scalar") from error
        if not math.isfinite(self.pressure):
            raise ValueError("pressure must be a finite scalar")

    @classmethod
    def on_surface(cls, mesh, surface: str, pressure: float) -> "FollowerPressure":
        """Load a named surface of ``mesh``, e.g. ``"z-"`` or ``"endo"``."""
        if surface not in mesh.face_sets:
            raise KeyError(f"unknown surface {surface!r}; this mesh has "
                           f"{sorted(mesh.face_sets)}")
        return cls(mesh.face_set(surface), pressure, name=surface)

    def contribution(self, x: torch.Tensor, quad, load_factor: float = 1.0,
                     tangent: bool = True):
        """Nodal force -- and consistent load stiffness -- on deformed coords ``x``.

        The deformed area-weighted normal is just ``d_xi1 x  x  d_xi2`` of the
        deformed surface, so neither a deformation gradient nor an explicit
        Nanson push-forward is needed.  Because that normal turns with the body,
        the stiffness is non-symmetric.

        Returns ``(f, K)`` shaped ``(n_faces, n_nodes, 3)`` and
        ``(n_faces, 3 n_nodes, 3 n_nodes)``; ``K`` is ``None`` when
        ``tangent`` is false.
        """
        conn = self.face_set.connectivity
        n_faces, n_nodes = conn.shape
        xf = x[conn]

        w, N = quad.weights, quad.N
        dN1, dN2 = quad.dN[..., 0], quad.dN[..., 1]
        g1 = torch.einsum("fai,qa->fqi", xf, dN1)
        g2 = torch.einsum("fai,qa->fqi", xf, dN2)
        nda = torch.linalg.cross(g1, g2, dim=-1)

        p = load_factor * self.pressure
        force = -p * torch.einsum("q,qa,fqi->fai", w, N, nda)
        if not tangent:
            return force, None

        eps = _levi_civita(x.dtype, x.device)
        t1 = torch.einsum("ijm,qb,fqm->fqijb", eps, dN1, g2)
        t2 = torch.einsum("imj,qb,fqm->fqijb", eps, dN2, g1)
        K = -p * torch.einsum("q,qa,fqijb->faibj", w, N, t1 + t2)
        return force, K.reshape(n_faces, n_nodes * 3, n_nodes * 3)


class DirichletConstraints:
    """The resolved set of prescribed degrees of freedom.

    Collapses a list of possibly overlapping :class:`DirichletBC` objects into
    one sorted DOF index array, its prescribed values, and the boolean/float
    masks the assembler and solver need.

    Duplicated DOFs are accepted only when they agree.  A plain indexed write
    would silently let the last condition win, and on a GPU "last" is not even
    well defined, so the extreme values written to each DOF are compared
    instead and a genuine conflict is an error.
    """

    def __init__(self, conditions: Sequence[DirichletBC], n_dofs: int,
                 dtype: torch.dtype, device: torch.device,
                 atol: float = 1e-12) -> None:
        if isinstance(n_dofs, bool) or (isinstance(n_dofs, torch.Tensor)
                                       and (n_dofs.dtype == torch.bool or n_dofs.ndim != 0)):
            raise ValueError("n_dofs must be a nonnegative integer multiple of 3")
        try:
            self.n_dofs = operator.index(n_dofs)
        except TypeError as error:
            raise ValueError("n_dofs must be a nonnegative integer multiple of 3") from error
        if self.n_dofs < 0 or self.n_dofs % 3 or self.n_dofs > torch.iinfo(torch.long).max:
            raise ValueError("n_dofs must be a nonnegative integer multiple of 3 within int64 range")
        try:
            atol = float(atol)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Dirichlet conflict tolerance must be finite and nonnegative") from error
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("Dirichlet conflict tolerance must be finite and nonnegative")
        self.dtype = dtype
        self.device = device

        dofs: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        for bc in conditions:
            d, v = bc.dofs_and_values(dtype, device, n_nodes=self.n_dofs // 3)
            dofs.append(d)
            vals.append(v)

        if dofs:
            all_dofs = torch.cat(dofs)
            all_vals = torch.cat(vals)
            uniq, inv = torch.unique(all_dofs, return_inverse=True)

            hi = torch.full_like(uniq, -float("inf"), dtype=dtype)
            lo = torch.full_like(uniq, float("inf"), dtype=dtype)
            hi.scatter_reduce_(0, inv, all_vals, reduce="amax", include_self=False)
            lo.scatter_reduce_(0, inv, all_vals, reduce="amin", include_self=False)
            spread = (hi - lo).abs()
            if bool((spread > atol).any()):
                bad = uniq[spread > atol]
                raise ValueError(
                    f"conflicting Dirichlet data on {bad.numel()} degree(s) of "
                    f"freedom, e.g. DOF {int(bad[0])} "
                    f"(node {int(bad[0]) // 3}, component {int(bad[0]) % 3}) is "
                    f"prescribed both {float(lo[spread > atol][0])} and "
                    f"{float(hi[spread > atol][0])}")
            self.dofs, self.values = uniq, hi
        else:
            self.dofs = torch.zeros(0, dtype=torch.long, device=device)
            self.values = torch.zeros(0, dtype=dtype, device=device)

        self.mask = torch.zeros(self.n_dofs, dtype=torch.bool, device=device)
        self.mask[self.dofs] = True
        self.free_mask = ~self.mask
        # Float form of the free mask: multiplying by it is equivalent to
        # boolean indexing but, unlike indexing, needs no host round trip to
        # discover the size of the result.
        self.free_scale = self.free_mask.to(dtype)

    def apply(self, u: torch.Tensor, load_factor: float = 1.0) -> torch.Tensor:
        """Write the prescribed values, scaled by ``load_factor``, into ``u``."""
        u[self.dofs] = load_factor * self.values
        return u

    def zero_constrained(self, v: torch.Tensor) -> torch.Tensor:
        """Zero the constrained entries of ``v`` in place, without syncing."""
        return v.masked_fill_(self.mask, 0.0)

    def free_norm(self, v: torch.Tensor) -> torch.Tensor:
        """``|v|`` over the free DOFs, as a device tensor (no host transfer)."""
        return torch.linalg.vector_norm(v * self.free_scale)
