"""Mesh generation and geometry utilities for :mod:`torchcor.mechanics`.

The mechanics module works with *tensor-product* (Lagrange) hexahedral meshes of
arbitrary polynomial order ``p``.  Hexahedra are used rather than tetrahedra
because a tensor-product basis keeps every kernel in the solver a batched
``einsum`` over regular arrays, which is what a GPU wants, and because
high-order hexahedra are the standard remedy for the volumetric locking that
plagues nearly incompressible cardiac tissue.

Node ordering
-------------
All entities use a *lexicographic* tensor-product ordering, which makes the
shape functions separable::

    hex  local node (l, m, n)  ->  (l * (p + 1) + m) * (p + 1) + n
    quad local node (a, b)     ->  a * (p + 1) + b

with reference coordinates :math:`\\xi_l = -1 + 2l/p` on :math:`[-1, 1]`.  The
solver relies on this convention; see :func:`lagrange_basis_1d` in
``torchcor.mechanics.solver``.

Real meshes
-----------
:class:`HexMesh` takes nodes and connectivity directly, so a reader for a
stored mesh only has to produce those two arrays plus the element order.  The
one extra thing the rest of the package needs is *named surfaces*: register
them with :meth:`HexMesh.add_face_set` (for surface loads) and
:meth:`HexMesh.add_node_set` (for constraints), the way
:class:`StructuredBoxMesh` registers its six sides, and everything downstream
refers to them by name without knowing where the mesh came from.

Face orientation
----------------
:class:`FaceSet` connectivities are ordered so that

.. math::  \\partial_{\\xi_1} x \\times \\partial_{\\xi_2} x

points **out of** the body.  Follower (pressure) loads rely on this, since the
deformed area-weighted normal is exactly that cross product evaluated on the
deformed coordinates -- no explicit deformation gradient or Nanson push-forward
is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "as_device",
    "FaceSet",
    "HexMesh",
    "StructuredBoxMesh",
    "TruncatedEllipsoidMesh",
    "hex_face_local_nodes",
    "HEX_FACE_NAMES",
]


def as_device(spec: Optional[str | torch.device] = None) -> torch.device:
    """Normalise a device specification.

    The mesh is the first thing built and everything else follows it, so this is
    where the simulation's device gets decided.  ``None`` picks CUDA when it is
    available, honouring a preceding ``torchcor.set_device("cuda:1")`` because
    that sets the current device.  A bare ``"cuda"`` is resolved to that current
    index: an index-less device compares unequal to the ``cuda:0`` a tensor
    actually reports, which would silently defeat same-device checks.
    """
    if spec is None:
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {spec!r} requested but CUDA is not available")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


# The six faces of the reference hexahedron.  ``axis`` is the local index
# (l, m, n) that is held fixed, ``side`` is the value it is held at (0 or p) and
# ``param`` names the two free local indices in the order (xi_1, xi_2) that makes
# d_xi1 x d_xi2 point outwards.
_HEX_FACES = {
    "x-": dict(axis=0, side="lo", param=(2, 1)),  # (zeta, eta) : e3 x e2 = -e1
    "x+": dict(axis=0, side="hi", param=(1, 2)),  # (eta, zeta) : e2 x e3 = +e1
    "y-": dict(axis=1, side="lo", param=(0, 2)),  # (xi,  zeta) : e1 x e3 = -e2
    "y+": dict(axis=1, side="hi", param=(2, 0)),  # (zeta, xi ) : e3 x e1 = +e2
    "z-": dict(axis=2, side="lo", param=(1, 0)),  # (eta, xi  ) : e2 x e1 = -e3
    "z+": dict(axis=2, side="hi", param=(0, 1)),  # (xi,  eta ) : e1 x e2 = +e3
}

HEX_FACE_NAMES: Tuple[str, ...] = tuple(_HEX_FACES)


def hex_face_local_nodes(order: int, face: str) -> np.ndarray:
    """Local node indices of one face of a reference hexahedron of order ``p``.

    Returns a flat array of ``(p + 1) ** 2`` indices into the element node list,
    laid out lexicographically in the face parametrisation ``(xi_1, xi_2)`` that
    yields an outward normal.
    """
    if face not in _HEX_FACES:
        raise KeyError(f"unknown hex face {face!r}; expected one of {HEX_FACE_NAMES}")

    spec = _HEX_FACES[face]
    p = order
    fixed_axis, fixed_val = spec["axis"], (0 if spec["side"] == "lo" else p)
    d1, d2 = spec["param"]

    lmn = np.empty((p + 1, p + 1, 3), dtype=np.int64)
    lmn[..., fixed_axis] = fixed_val
    lmn[..., d1] = np.arange(p + 1)[:, None]
    lmn[..., d2] = np.arange(p + 1)[None, :]

    l, m, n = lmn[..., 0], lmn[..., 1], lmn[..., 2]
    return ((l * (p + 1) + m) * (p + 1) + n).reshape(-1)


@dataclass
class FaceSet:
    """A named set of element faces, used to carry surface boundary conditions.

    Attributes
    ----------
    name:
        Identifier, e.g. ``"z-"``.
    connectivity:
        ``(n_faces, (p + 1) ** 2)`` node indices, lexicographic in the face
        parametrisation, oriented so the outward normal is
        ``d_xi1 x d_xi2``.
    order:
        Polynomial order of the surface element.
    """

    name: str
    connectivity: torch.Tensor
    order: int

    @property
    def n_faces(self) -> int:
        return int(self.connectivity.shape[0])

    @property
    def nodes_per_face(self) -> int:
        return int(self.connectivity.shape[1])

    def nodes(self) -> torch.Tensor:
        """Sorted unique node indices touched by this face set."""
        return torch.unique(self.connectivity.reshape(-1))

    def to(self, device: torch.device) -> "FaceSet":
        return FaceSet(self.name, self.connectivity.to(device), self.order)


class HexMesh:
    """A tensor-product hexahedral mesh living on a torch device.

    Parameters
    ----------
    points:
        ``(n_points, 3)`` nodal coordinates.
    cells:
        ``(n_cells, (p + 1) ** 3)`` connectivity in lexicographic order.
    order:
        Polynomial order ``p`` of the element.
    """

    def __init__(
        self,
        points: torch.Tensor,
        cells: torch.Tensor,
        order: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        # Follow the coordinates when no device is asked for, so a mesh built
        # from CUDA tensors does not silently land on the host.
        if device is None and isinstance(points, torch.Tensor):
            device = points.device
        device = as_device(device)

        self.order = int(order)
        self.dtype = dtype
        self.device = device

        self.points = torch.as_tensor(points, dtype=dtype, device=device).contiguous()
        self.cells = torch.as_tensor(cells, dtype=torch.long, device=device).contiguous()

        expected = (self.order + 1) ** 3
        if self.cells.shape[1] != expected:
            raise ValueError(
                f"order-{self.order} hexahedra need {expected} nodes per cell, "
                f"got {self.cells.shape[1]}"
            )

        self.node_sets: Dict[str, torch.Tensor] = {}
        self.face_sets: Dict[str, FaceSet] = {}

    # ------------------------------------------------------------------ sizes
    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def nodes_per_cell(self) -> int:
        return int(self.cells.shape[1])

    @property
    def n_dofs(self) -> int:
        """Number of displacement degrees of freedom (3 per node)."""
        return 3 * self.n_points

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(order={self.order}, n_points={self.n_points}, "
            f"n_cells={self.n_cells}, device={self.device})"
        )

    # ----------------------------------------------------------------- device
    def to(self, device: torch.device) -> "HexMesh":
        """Move the mesh (and all its sets) to ``device``, in place."""
        device = torch.device(device)
        self.device = device
        self.points = self.points.to(device)
        self.cells = self.cells.to(device)
        self.node_sets = {k: v.to(device) for k, v in self.node_sets.items()}
        self.face_sets = {k: v.to(device) for k, v in self.face_sets.items()}
        return self

    # ------------------------------------------------------------------- sets
    def add_node_set(self, name: str, nodes: torch.Tensor) -> torch.Tensor:
        nodes = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        self.node_sets[name] = nodes
        return nodes

    def add_face_set(self, face_set: FaceSet) -> FaceSet:
        self.face_sets[face_set.name] = face_set
        return face_set

    def node_set(self, name: str) -> torch.Tensor:
        return self.node_sets[name]

    def face_set(self, name: str) -> FaceSet:
        return self.face_sets[name]

    def faces_of_cells(self, name: str, cell_ids: torch.Tensor, face: str) -> FaceSet:
        """Build a :class:`FaceSet` from ``face`` of the given cells."""
        local = torch.as_tensor(
            hex_face_local_nodes(self.order, face), dtype=torch.long, device=self.device
        )
        cell_ids = torch.as_tensor(cell_ids, dtype=torch.long, device=self.device)
        return FaceSet(name, self.cells[cell_ids][:, local].contiguous(), self.order)

    # -------------------------------------------------------------- selectors
    def nodes_where(self, predicate: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Node indices whose coordinates satisfy ``predicate(points) -> bool mask``."""
        return torch.nonzero(predicate(self.points), as_tuple=False).reshape(-1)

    def nodes_on_plane(self, axis: int, value: float, atol: float = 1e-9) -> torch.Tensor:
        """Node indices on the plane ``x[axis] == value``."""
        return self.nodes_where(lambda X: (X[:, axis] - value).abs() <= atol)

    def bounding_box(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.points.amin(dim=0), self.points.amax(dim=0)

    # ------------------------------------------------------------ interpolate
    def interpolate_local(self, nodal_field: torch.Tensor, cells: torch.Tensor,
                          xi: torch.Tensor) -> torch.Tensor:
        """Evaluate a nodal field at reference coordinates inside given cells.

        ``cells`` is ``(n,)`` cell indices and ``xi`` ``(n, 3)`` coordinates in
        ``[-1, 1]^3``.  Locating the points is the caller's job; this is the
        part that does not depend on how the mesh was built.
        """
        from torchcor.mechanics.elements import lagrange_basis_1d  # local: avoid cycle

        field = torch.as_tensor(nodal_field, dtype=self.dtype, device=self.device)
        if xi.shape[0] == 0:
            return field.new_empty((0,) + field.shape[1:])
        # Separable basis: N_{lmn}(xi) = L_l(xi_0) L_m(xi_1) L_n(xi_2).
        L = [lagrange_basis_1d(self.order, xi[:, d])[0] for d in range(3)]
        N = (L[0][:, :, None, None] * L[1][:, None, :, None]
             * L[2][:, None, None, :]).reshape(xi.shape[0], -1)

        values = field[self.cells[cells]]                    # (n, n_en, ...)
        extra = values.dim() - 2
        return (N.reshape(N.shape + (1,) * extra) * values).sum(dim=1)

    def locate(self, points: torch.Tensor, tolerance: float = 1e-8,
               outside: str = "raise") -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the cell and reference coordinates of physical ``points``.

        This is the one thing a mesh must implement to support Cartesian
        probing; :meth:`interpolate` and hence ``Mechanics.probe`` are built on
        it.  Returns ``(cell_ids, xi)`` with ``xi`` in ``[-1, 1]^3``.
        ``tolerance`` is a distance in mesh units.  ``outside='raise'`` rejects
        points that cannot be located; ``'clamp'`` permits a local projection.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement locate(), so it cannot be "
            "probed at Cartesian points")

    def _probe_points(self, points, tolerance: float, outside: str) -> torch.Tensor:
        """Normalise and validate the shared Cartesian probing arguments."""
        if outside not in ("raise", "clamp"):
            raise ValueError(f"outside must be 'raise' or 'clamp', got {outside!r}")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        points = torch.as_tensor(points, dtype=self.dtype, device=self.device)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        points = points.reshape(-1, 3)
        if not bool(torch.isfinite(points).all()):
            raise ValueError("points must be finite")
        return points

    @staticmethod
    def _check_location(distance: torch.Tensor, tolerance: float, outside: str) -> None:
        if not bool(torch.isfinite(distance).all()):
            raise ValueError("point location produced a non-finite residual")
        missed = distance > tolerance
        if outside == "raise" and bool(missed.any()):
            raise ValueError(
                f"could not locate {int(missed.sum())} point(s) within the mesh; "
                f"largest residual distance is {float(distance.max()):.3e}. "
                "Use outside='clamp' to allow a local projection.")

    def refine_local(self, points: torch.Tensor, cells: torch.Tensor,
                     xi: torch.Tensor, tolerance: float = 1e-10,
                     iterations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Newton-correct reference coordinates against the actual FE geometry.

        An analytic inverse of the ideal shape only locates a point on the
        *ideal* surface; the mesh interpolates that surface, so the two differ
        by the geometry error of the element.  This solves
        ``x(xi) = point`` on the real isoparametric map, starting from ``xi``.

        A small, Jacobian-scaled regularisation makes the local least-squares
        step defined even on collapsed edges.  Where several reference
        coordinates map to the same physical point, any of them is valid.

        Returns the corrected coordinates and the remaining residual distance.
        A nonzero residual may require another cell or an outside-point policy.
        """
        from torchcor.mechanics.elements import lagrange_basis_1d  # local: avoid cycle

        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        target = torch.as_tensor(points, dtype=self.dtype,
                                 device=self.device).reshape(-1, 3)
        if target.shape[0] == 0:
            return xi.clone(), target.new_empty(0)
        nodes = self.points[self.cells[cells]]                  # (n, n_en, 3)
        xi = xi.clone()
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        regularisation = max(1e-12, 10*torch.finfo(self.dtype).eps)
        for _ in range(iterations):
            bases = [lagrange_basis_1d(self.order, xi[:, d]) for d in range(3)]
            N = (bases[0][0][:, :, None, None]*bases[1][0][:, None, :, None]
                 * bases[2][0][:, None, None, :]).reshape(xi.shape[0], -1)
            dN = torch.stack([
                (bases[0][1][:, :, None, None]*bases[1][0][:, None, :, None]
                 * bases[2][0][:, None, None, :]).reshape(xi.shape[0], -1),
                (bases[0][0][:, :, None, None]*bases[1][1][:, None, :, None]
                 * bases[2][0][:, None, None, :]).reshape(xi.shape[0], -1),
                (bases[0][0][:, :, None, None]*bases[1][0][:, None, :, None]
                 * bases[2][1][:, None, None, :]).reshape(xi.shape[0], -1),
            ], dim=-1)                                          # (n, n_en, 3)

            residual = torch.einsum("na,nai->ni", N, nodes) - target
            if float(residual.norm(dim=1).max()) <= tolerance:
                break
            jacobian = torch.einsum("naj,nai->nij", dN, nodes)  # dx_i/dxi_j
            normal = jacobian.mT @ jacobian
            damping = regularisation * torch.diagonal(
                normal, dim1=-2, dim2=-1).amax(dim=-1).clamp(min=1e-30)
            normal = normal + damping[:, None, None]*eye
            step, info = torch.linalg.solve_ex(
                normal, jacobian.mT @ residual.unsqueeze(-1), check_errors=False)
            step = step.squeeze(-1)
            usable = (info == 0) & torch.isfinite(step).all(dim=-1)
            xi = torch.where(usable[:, None], (xi - step).clamp(-1.0, 1.0), xi)

        bases = [lagrange_basis_1d(self.order, xi[:, d])[0] for d in range(3)]
        N = (bases[0][:, :, None, None]*bases[1][:, None, :, None]
             * bases[2][:, None, None, :]).reshape(xi.shape[0], -1)
        distance = (torch.einsum("na,nai->ni", N, nodes) - target).norm(dim=1)
        return xi, distance

    def interpolate(self, nodal_field: torch.Tensor, points: torch.Tensor,
                    **locate) -> torch.Tensor:
        """Evaluate a nodal field at arbitrary physical ``points``.

        Keyword arguments are passed to :meth:`locate`, so a caller can widen
        the residual tolerance or choose what happens outside the mesh.
        """
        return self.interpolate_local(nodal_field, *self.locate(points, **locate))

    # ------------------------------------------------------------------- i/o
    def linear_cells(self) -> np.ndarray:
        """Split every order-``p`` hex into ``p ** 3`` 8-node hexes.

        Used for VTK output: linear sub-cells render in every ParaView version
        and resolve the curved high-order solution instead of flattening it.
        """
        p = self.order

        def lid(l, m, n):
            return (l * (p + 1) + m) * (p + 1) + n

        # VTK_HEXAHEDRON node order.
        corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        local = np.array(
            [
                [lid(l + dl, m + dm, n + dn) for (dl, dm, dn) in corners]
                for l in range(p)
                for m in range(p)
                for n in range(p)
            ],
            dtype=np.int64,
        )
        cells = self.cells.detach().cpu().numpy()
        return cells[:, local].reshape(-1, 8)

    def write_vtu(
        self,
        path: str | Path,
        point_data: Optional[Dict[str, torch.Tensor]] = None,
        displacement: Optional[torch.Tensor] = None,
    ) -> Path:
        """Write this mesh to a ``.vtu`` file, subdividing high-order cells.

        With ``displacement`` the mesh is written in its deformed configuration.
        Thin wrapper over :mod:`torchcor.mechanics.visualisation`, which holds
        all VTK output for the mechanics module.
        """
        from torchcor.mechanics import visualisation  # local: avoid cycle

        if displacement is None:
            return visualisation.write_mesh(path, self, point_data=point_data)
        return visualisation.write_deformed(path, self, displacement,
                                            point_data=point_data)


class StructuredBoxMesh(HexMesh):
    """A structured order-``p`` hexahedral mesh of an axis-aligned box.

    The regular grid gives exact, closed-form point location, so nodal fields can
    be probed at arbitrary physical points (needed to report benchmark
    quantities at prescribed coordinates).

    The six boundary sides are registered automatically as face sets *and* node
    sets under the names ``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, ``"z-"``,
    ``"z+"``.
    """

    def __init__(
        self,
        origin: Sequence[float],
        lengths: Sequence[float],
        divisions: Sequence[int],
        order: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        origin = tuple(float(v) for v in origin)
        lengths = tuple(float(v) for v in lengths)
        divisions = tuple(int(v) for v in divisions)
        if len(origin) != 3 or len(lengths) != 3 or len(divisions) != 3:
            raise ValueError("origin, lengths and divisions must each have 3 entries")
        if min(divisions) < 1:
            raise ValueError(f"divisions must be >= 1, got {divisions}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        p = int(order)
        nx, ny, nz = divisions
        shape = (p * nx + 1, p * ny + 1, p * nz + 1)

        device = as_device(device)

        # ---- nodes: lexicographic (i, j, k) with k fastest -------------------
        axes = [
            torch.linspace(origin[d], origin[d] + lengths[d], shape[d],
                           dtype=dtype, device=device)
            for d in range(3)
        ]
        grid = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack([g.reshape(-1) for g in grid], dim=1)

        # ---- cells -----------------------------------------------------------
        node_id = torch.arange(points.shape[0], device=device).reshape(shape)
        ex = torch.arange(nx, device=device)
        ey = torch.arange(ny, device=device)
        ez = torch.arange(nz, device=device)
        off = torch.arange(p + 1, device=device)

        i = (p * ex[:, None, None, None, None, None] + off[None, None, None, :, None, None])
        j = (p * ey[None, :, None, None, None, None] + off[None, None, None, None, :, None])
        k = (p * ez[None, None, :, None, None, None] + off[None, None, None, None, None, :])
        cells = node_id[i, j, k].reshape(nx * ny * nz, (p + 1) ** 3)

        super().__init__(points, cells, order=p, device=device, dtype=dtype)

        self.origin = origin
        self.lengths = lengths
        self.divisions = divisions
        self.node_shape = shape

        self._register_sides()

    # ------------------------------------------------------------------ sides
    def _register_sides(self) -> None:
        nx, ny, nz = self.divisions
        ex, ey, ez = torch.meshgrid(
            torch.arange(nx, device=self.device),
            torch.arange(ny, device=self.device),
            torch.arange(nz, device=self.device),
            indexing="ij",
        )
        cell_id = (ex * ny + ey) * nz + ez

        selectors = {
            "x-": cell_id[0, :, :], "x+": cell_id[-1, :, :],
            "y-": cell_id[:, 0, :], "y+": cell_id[:, -1, :],
            "z-": cell_id[:, :, 0], "z+": cell_id[:, :, -1],
        }
        for name, ids in selectors.items():
            fs = self.faces_of_cells(name, ids.reshape(-1), name)
            self.add_face_set(fs)
            self.add_node_set(name, fs.nodes())

    # ------------------------------------------------------------ point probe
    def locate(self, points: torch.Tensor, tolerance: float = 1e-8,
               outside: str = "raise") -> Tuple[torch.Tensor, torch.Tensor]:
        """Locate physical ``points`` in the grid.

        Returns ``(cell_ids, xi)`` with ``xi`` the reference coordinates in
        ``[-1, 1]^3``.  Points outside the box by more than ``tolerance`` are
        rejected unless ``outside='clamp'`` requests their nearest box point.
        """
        pts = self._probe_points(points, tolerance, outside)
        div = torch.tensor(self.divisions, dtype=self.dtype, device=self.device)
        org = torch.tensor(self.origin, dtype=self.dtype, device=self.device)
        lng = torch.tensor(self.lengths, dtype=self.dtype, device=self.device)

        # Position in "element units": t in [0, n_elem] per direction.
        t = (pts - org) / lng * div
        t = t.clamp(torch.zeros_like(div), div)
        distance = (org + t/div*lng - pts).norm(dim=-1)
        self._check_location(distance, tolerance, outside)

        idx = t.floor().clamp(max=div - 1).to(torch.long)
        local = t - idx.to(self.dtype)  # in [0, 1]
        xi = 2.0 * local - 1.0

        nx, ny, nz = self.divisions
        cell_ids = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
        return cell_ids, xi


class TruncatedEllipsoidMesh(HexMesh):
    """A truncated-ellipsoid ventricle, as used by benchmark problems 2 and 3.

    The wall is parametrised by ``(t, s, v)``:

    ==========  ==============================================================
    ``t``       transmural, 0 on the endocardium and 1 on the epicardium
    ``s``       0 at the apex, 1 on the base plane
    ``v``       around the long axis, periodic
    ==========  ==============================================================

    with radii interpolated linearly through the wall and the surface point

    .. math::

        x = r_s \sin u \cos v, \quad y = r_s \sin u \sin v, \quad z = r_l \cos u

    where ``u = -pi + s (u_base(t) + pi)`` and
    ``u_base(t) = -arccos(z_base / r_l(t))``.  Because ``u_base`` is taken per
    ``t``, ``z = z_base`` exactly on the whole ``s = 1`` face, so the base is
    planar by construction rather than by approximation.

    Two topological joins are made, and both matter physically:

    * the seam at ``v = +-pi`` shares nodes, so the wall is a closed ring and
      not a slit that can open;
    * every node at the apex ``s = 0`` is *one* node per ``t``, so the apex
      cannot tear apart.  The elements touching it are hexahedra with one edge
      collapsed; their Jacobian stays positive at the quadrature points, which
      the assembler verifies.

    Registers the surfaces ``"endo"``, ``"epi"`` and ``"base"``.
    """

    def __init__(
        self,
        divisions: Sequence[int] = (1, 8, 16),
        order: int = 2,
        endo: Tuple[float, float] = (7.0, 17.0),
        epi: Tuple[float, float] = (10.0, 20.0),
        base_z: float = 5.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        nt, nu, nv = (int(v) for v in divisions)
        if min(nt, nu) < 1 or nv < 3:
            raise ValueError(f"divisions must be >= (1, 1, 3), got {tuple(divisions)}")
        for name, radii in (("endo", endo), ("epi", epi)):
            if len(radii) != 2 or not all(np.isfinite(r) and r > 0 for r in radii):
                raise ValueError(f"{name} radii must be two finite positive "
                                 f"numbers, got {tuple(radii)}")
        if not (epi[0] > endo[0] and epi[1] > endo[1]):
            raise ValueError(f"the epicardium must enclose the endocardium, got "
                             f"endo={tuple(endo)}, epi={tuple(epi)}")
        if not np.isfinite(base_z):
            raise ValueError(f"base_z must be finite, got {base_z}")
        # The base plane has to cut both ellipsoids, otherwise arccos is undefined
        # and the mesh silently fills with NaN.
        if not (-endo[1] < base_z < endo[1] and -epi[1] < base_z < epi[1]):
            raise ValueError(f"the base plane z={base_z} must lie strictly inside "
                             f"both ellipsoids (|z| < {min(endo[1], epi[1])})")
        p = int(order)
        device = as_device(device)

        n1, n2, n3 = p*nt + 1, p*nu + 1, p*nv + 1
        t = torch.linspace(0., 1., n1, dtype=dtype, device=device)
        s_ = torch.linspace(0., 1., n2, dtype=dtype, device=device)
        # v runs +pi -> -pi so that (t, s, v) is right-handed and elements are
        # not inverted; the assembler rejects the other orientation outright.
        v = torch.linspace(torch.pi, -torch.pi, n3, dtype=dtype, device=device)

        rs = endo[0] + t*(epi[0] - endo[0])
        rl = endo[1] + t*(epi[1] - endo[1])
        u_base = -torch.arccos(base_z/rl)
        u = -torch.pi + s_[None, :]*(u_base[:, None] + torch.pi)      # (n1, n2)

        ring = rs[:, None]*torch.sin(u)                               # (n1, n2)
        points = torch.stack([
            ring[:, :, None]*torch.cos(v)[None, None, :],
            ring[:, :, None]*torch.sin(v)[None, None, :],
            (rl[:, None]*torch.cos(u))[:, :, None].expand(n1, n2, n3),
        ], dim=-1).reshape(-1, 3)

        ids = torch.arange(n1*n2*n3, device=device).reshape(n1, n2, n3)
        ids[:, :, -1] = ids[:, :, 0]              # close the circumferential seam
        ids[:, 0, :] = ids[:, 0, :1]              # collapse the apex to one node
        keep, inverse = torch.unique(ids.reshape(-1), return_inverse=True)
        points = points[keep]
        ids = inverse.reshape(n1, n2, n3)

        off = torch.arange(p + 1, device=device)
        i = p*torch.arange(nt, device=device)[:, None, None, None, None, None] + off[None, None, None, :, None, None]
        j = p*torch.arange(nu, device=device)[None, :, None, None, None, None] + off[None, None, None, None, :, None]
        k = p*torch.arange(nv, device=device)[None, None, :, None, None, None] + off[None, None, None, None, None, :]
        cells = ids[i, j, k].reshape(nt*nu*nv, (p + 1)**3)

        super().__init__(points, cells, order=p, device=device, dtype=dtype)

        self.divisions = (nt, nu, nv)
        self.endo, self.epi, self.base_z = tuple(endo), tuple(epi), float(base_z)

        cell_id = torch.arange(nt*nu*nv, device=device).reshape(nt, nu, nv)
        for name, ids_, face in (("endo", cell_id[0], "x-"), ("epi", cell_id[-1], "x+"),
                                 ("base", cell_id[:, -1], "y+")):
            fs = self.faces_of_cells(name, ids_.reshape(-1), face)
            self.add_face_set(fs)
            self.add_node_set(name, fs.nodes())

    # ---------------------------------------------------------------- probing
    def radii(self, t) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(rs, rl)`` at transmural position ``t``."""
        t = torch.as_tensor(t, dtype=self.dtype, device=self.device)
        return (self.endo[0] + t*(self.epi[0] - self.endo[0]),
                self.endo[1] + t*(self.epi[1] - self.endo[1]))

    def points_at(self, t, s, v) -> torch.Tensor:
        """Undeformed coordinates at parametric ``(t, s, v)``, shape ``(n, 3)``."""
        t, s, v = (torch.as_tensor(a, dtype=self.dtype, device=self.device)
                   for a in torch.broadcast_tensors(
                       *[torch.as_tensor(a, dtype=self.dtype, device=self.device)
                         for a in (t, s, v)]))
        rs, rl = self.radii(t)
        u = -torch.pi + s*(-torch.arccos(self.base_z/rl) + torch.pi)
        return torch.stack([rs*torch.sin(u)*torch.cos(v),
                            rs*torch.sin(u)*torch.sin(v),
                            rl*torch.cos(u)], dim=-1)

    def locate_parametric(self, t, s, v) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cell indices and reference coordinates for parametric ``(t, s, v)``.

        The mesh is a structured grid in these coordinates, so this is exact
        rather than an iterative point search.
        """
        t, s, v = torch.broadcast_tensors(
            *[torch.as_tensor(a, dtype=self.dtype, device=self.device)
              for a in (t, s, v)])
        nt, nu, nv = self.divisions
        t, s = t.reshape(-1), s.reshape(-1)
        if not all(bool(torch.isfinite(a).all()) for a in (t, s, v)):
            raise ValueError("parametric coordinates must be finite")
        for name, a in (("t", t), ("s", s)):
            if bool(((a < -1e-12) | (a > 1 + 1e-12)).any()):
                raise ValueError(f"{name} must lie in [0, 1]; got "
                                 f"[{float(a.min())}, {float(a.max())}]")
        # v is periodic, so wrap it rather than clamping: clamping would map
        # v + 2*pi to the seam and silently return a different location.
        angle = torch.remainder((torch.pi - v.reshape(-1))/(2*torch.pi)*nv, nv)
        fracs = torch.stack([t.clamp(0., 1.)*nt, s.clamp(0., 1.)*nu, angle], dim=1)
        counts = torch.tensor([nt, nu, nv], dtype=self.dtype, device=self.device)
        index = fracs.floor().clamp(max=counts - 1).to(torch.long)
        xi = 2.0*(fracs - index.to(self.dtype)) - 1.0
        return (index[:, 0]*nu + index[:, 1])*nv + index[:, 2], xi

    def interpolate_at(self, nodal_field: torch.Tensor, t, s, v) -> torch.Tensor:
        """Evaluate a nodal field at parametric ``(t, s, v)``."""
        return self.interpolate_local(nodal_field, *self.locate_parametric(t, s, v))

    def to_parametric(self, points: torch.Tensor, iterations: int = 60):
        """Convert Cartesian points to ``(t, s, v)``.

        The azimuth follows directly from ``atan2``.  The transmural coordinate
        is the root of ``(r/rs(t))^2 + (z/rl(t))^2 = 1``, which is monotone in
        ``t``, so a fixed number of bisections finds it to machine precision
        without an iteration-count guess.  ``u`` and then ``s`` follow in closed
        form.
        """
        q = torch.as_tensor(points, dtype=self.dtype, device=self.device).reshape(-1, 3)
        r = q[:, :2].norm(dim=1)
        # The mesh lays out x = rs sin(u) cos(v) with sin(u) <= 0, so the
        # geometric azimuth is v + pi.
        v = torch.atan2(q[:, 1], q[:, 0]) - torch.pi

        lo = torch.zeros_like(r)
        hi = torch.ones_like(r)
        for _ in range(iterations):
            mid = 0.5*(lo + hi)
            rs, rl = self.radii(mid)
            outside = (r/rs)**2 + (q[:, 2]/rl)**2 > 1.0
            lo = torch.where(outside, mid, lo)
            hi = torch.where(outside, hi, mid)
        t = 0.5*(lo + hi)

        rs, rl = self.radii(t)
        u = -torch.atan2(r/rs, q[:, 2]/rl)                  # in [-pi, 0]
        s = (u + torch.pi)/(-torch.arccos(self.base_z/rl) + torch.pi)
        return t, s.clamp(0.0, 1.0), v

    def locate(self, points: torch.Tensor, tolerance: float = 1e-8,
               outside: str = "raise"):
        """Cell and reference coordinates of Cartesian ``points``.

        The analytic inversion of the ideal ellipsoid is only a starting guess;
        it is then Newton-corrected against the real element geometry, so the
        returned coordinates reproduce the requested point to ``tolerance``.
        Failed guesses try adjacent cells, moving through the grid while the
        physical residual improves.

        ``outside`` decides what happens to points the mesh does not contain:
        ``"raise"`` reports unresolved points.  ``"clamp"`` accepts the best
        local projection found in the searched cells; this is not guaranteed
        to be the globally nearest boundary point.
        """
        points = self._probe_points(points, tolerance, outside)
        cells, xi = self.locate_parametric(*self.to_parametric(points))
        xi, distance = self.refine_local(points, cells, xi, tolerance=tolerance)
        if bool((distance > tolerance).any()):
            cells, xi, distance = self._locate_neighbors(
                points, cells, xi, distance, tolerance)
        self._check_location(distance, tolerance, outside)
        return cells, xi

    def _locate_neighbors(self, points, cells, xi, distance, tolerance):
        """Retry only unresolved points in batched adjacent-cell candidates."""
        nt, nu, nv = self.divisions
        offsets = torch.cartesian_prod(*[
            torch.arange(-1, 2, device=self.device) for _ in range(3)])
        pending = distance > tolerance
        for _ in range(nt + nu + nv):
            rows = pending.nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                break
            ids = cells[rows]
            index = torch.stack([ids//(nu*nv), (ids//nv) % nu, ids % nv], dim=-1)
            candidates = index[:, None, :] + offsets[None, :, :]
            valid = ((candidates[..., 0] >= 0) & (candidates[..., 0] < nt)
                     & (candidates[..., 1] >= 0) & (candidates[..., 1] < nu))
            candidates[..., 0].clamp_(0, nt - 1)
            candidates[..., 1].clamp_(0, nu - 1)
            candidates[..., 2].remainder_(nv)
            candidate_cells = ((candidates[..., 0]*nu + candidates[..., 1])*nv
                               + candidates[..., 2])
            guesses = (xi[rows, None, :] - 2*offsets).clamp(-1., 1.)
            refined, residual = self.refine_local(
                points[rows, None, :].expand(-1, 27, -1).reshape(-1, 3),
                candidate_cells.reshape(-1), guesses.reshape(-1, 3),
                tolerance=tolerance)
            residual = residual.reshape(-1, 27).masked_fill(~valid, float("inf"))
            best_distance, best = residual.min(dim=1)
            improved = best_distance < distance[rows]
            selected = torch.arange(rows.numel(), device=self.device)
            best_xi = refined.reshape(-1, 27, 3)[selected, best]
            best_cells = candidate_cells[selected, best]
            update = rows[improved]
            cells[update] = best_cells[improved]
            xi[update] = best_xi[improved]
            distance[update] = best_distance[improved]
            pending[rows] = improved & (best_distance > tolerance)
        return cells, xi, distance


def _self_test() -> None:  # pragma: no cover - developer sanity check
    """Check face orientation: outward normals of the reference box."""
    from torchcor.mechanics.elements import LagrangeQuad

    for order in (1, 2, 3):
        mesh = StructuredBoxMesh((0.0, 0.0, 0.0), (2.0, 3.0, 4.0), (2, 2, 2), order=order)
        expected = {
            "x-": (0, -1.0), "x+": (0, 1.0),
            "y-": (1, -1.0), "y+": (1, 1.0),
            "z-": (2, -1.0), "z+": (2, 1.0),
        }
        quad = LagrangeQuad(order, dtype=mesh.dtype, device=mesh.device)
        for name, (axis, sign) in expected.items():
            fs = mesh.face_set(name)
            xf = mesh.points[fs.connectivity]
            g1 = torch.einsum("fai,qa->fqi", xf, quad.dN[..., 0])
            g2 = torch.einsum("fai,qa->fqi", xf, quad.dN[..., 1])
            nrm = torch.cross(g1, g2, dim=-1)
            nrm = nrm / nrm.norm(dim=-1, keepdim=True)
            ref = torch.zeros(3, dtype=mesh.dtype)
            ref[axis] = sign
            err = (nrm - ref).abs().max().item()
            assert err < 1e-12, f"order {order} face {name}: normal error {err}"
        # total volume
        print(f"order {order}: face orientation OK, {mesh.n_cells} cells, "
              f"{mesh.n_points} nodes")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
