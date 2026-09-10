"""Biventricular geometry and fibres of the Arostica et al. benchmark.

Benchmark 2 distributes its mesh *and* its fibre field, so neither is
constructed here: the mesh is read as it was given, and the fibres are the
LDRB field the participants were handed rather than a rule evaluated again.
Everything else -- connectivity, face sets, promotion to quadratic
displacement, interpolation -- comes from :class:`TetMesh` unchanged.

Section 4 of the paper fixes the geometry; the surfaces carry the generator's
own markers, and the two cavities are separate so they can take separate
pressures.
"""

from pathlib import Path

import h5py
import numpy as np
import torch

from torchcor.mechanics.elements import LagrangeTet
from torchcor.mechanics.material import MaterialAxes
from torchcor.mechanics.mesh import (
    FaceSet, TetMesh, as_device, tet_face_local_nodes)

#: Where the distributed meshes live, one directory per refinement level.
#: Section 4.4 asks for both; Table 6 gives their sizes.
REFERENCE = Path(__file__).with_name("reference")
RESOLUTIONS = ("coarse", "fine")
#: Its facet markers, from ``cardiac_benchmark``'s ``BiVGeometry``.
SURFACES = {10: "base", 20: "rv", 30: "lv", 40: "epi"}
#: How DOLFIN numbers the edges of a tetrahedron, which is the order the
#: distributed fibre checkpoint stores its midside values in.  Verified rather
#: than assumed: under any other order the field is discontinuous across
#: shared edges, and under this one it agrees exactly.
UFC_EDGES = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))


def _read_checkpoint(path: Path, name: str, cells: np.ndarray, n_vertex: int):
    """Nodal values of one quadratic vector field written by DOLFIN.

    The file stores, per cell, the thirty degrees of freedom of a quadratic
    vector element grouped by component.  The field is continuous, so each
    value belongs to a mesh entity -- a vertex or an edge -- and is recovered
    here by that entity rather than by degree-of-freedom index, which makes it
    independent of how either code happens to number its cells.
    """
    with h5py.File(path, "r") as handle:
        dofs = handle[f"{name}/cell_dofs"][:].reshape(-1, 30)
        values = handle[f"{name}/vector_0"][:]
    per_cell = values[dofs].reshape(-1, 3, 10).transpose(0, 2, 1)

    vertex = np.zeros((n_vertex, 3))
    vertex[cells] = per_cell[:, :4, :]
    edge = {}
    for k, (a, b) in enumerate(UFC_EDGES):
        ends = np.sort(cells[:, [a, b]], axis=1)
        for (lo, hi), value in zip(ends, per_cell[:, 4 + k, :]):
            edge[(lo, hi)] = value
    return vertex, edge


class BiventricleMesh(TetMesh):
    """The biventricular mesh distributed with benchmark 2.

    Registers the surfaces ``"lv"``, ``"rv"``, ``"epi"`` and ``"base"``, and
    carries the supplied fibre and sheet fields as quadratic nodal values.
    """

    #: Names the refinement level in reports and result filenames; set by
    #: :meth:`load` to the one that was read.
    label = "coarse"

    @classmethod
    def load(cls, resolution: str = "coarse", device=None,
             dtype: torch.dtype = torch.float64) -> "BiventricleMesh":
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}, "
                             f"got {resolution!r}")
        path = REFERENCE/resolution/"bi_ventricular.h5"
        device = as_device(device)
        with h5py.File(path, "r") as handle:
            points = handle["Mesh/mesh/geometry"][:]
            cells = handle["Mesh/mesh/topology"][:].astype(np.int64)
            triangles = handle["MeshFunction/0/mesh/topology"][:].astype(np.int64)
            markers = handle["MeshFunction/0/values"][:].reshape(-1)

        # The fibres are read against the topology as stored, before any
        # reordering here, because that is the numbering they were written in.
        fibre = _read_checkpoint(path.with_name("bi_ventricular_fiber.h5"),
                                 "fiber", cells, len(points))
        sheet = _read_checkpoint(path.with_name("bi_ventricular_sheet.h5"),
                                 "sheet", cells, len(points))

        coordinates = torch.as_tensor(points, dtype=dtype, device=device)
        connectivity = torch.as_tensor(cells, device=device)
        # The assembly needs positively oriented cells; swapping two vertices
        # is the only change that makes, and it leaves every edge intact.
        span = coordinates[connectivity][:, 1:] - coordinates[connectivity][:, :1]
        flip = torch.linalg.det(span) < 0
        connectivity[flip] = connectivity[flip][:, [0, 2, 1, 3]]

        linear = cls(coordinates, connectivity, 1, device, dtype)
        linear._register_surfaces(triangles, markers)
        mesh = linear.promote()
        for name in SURFACES.values():
            mesh.add_node_set(name, mesh.face_set(name).nodes())
        mesh.fibre = mesh._nodal_field(fibre)
        mesh.sheet = mesh._nodal_field(sheet)
        mesh.label = resolution
        return mesh

    def _register_surfaces(self, triangles, markers) -> None:
        """Orient each stored boundary triangle by the cell face it belongs to."""
        faces = torch.cat([
            self.cells[:, torch.as_tensor(tet_face_local_nodes(1, f),
                                          device=self.device)] for f in range(4)])
        n = self.n_points
        code = lambda t: (t.sort(dim=1).values.to(torch.int64)
                          * torch.tensor([n*n, n, 1], device=t.device)).sum(dim=1)
        keys = code(faces)
        order = keys.argsort()
        for value, name in SURFACES.items():
            stored = torch.as_tensor(triangles[markers == value], device=self.device)
            want = code(stored)
            index = torch.searchsorted(keys[order], want).clamp(max=keys.numel() - 1)
            if not bool((keys[order][index] == want).all()):
                raise ValueError(f"surface {name!r} has triangles that are not "
                                 "faces of the stored tetrahedra")
            self.add_face_set(FaceSet(name, faces[order[index]].contiguous(), 1))

    def _nodal_field(self, field) -> torch.Tensor:
        """Spread a (vertex, edge) field onto this promoted mesh's nodes."""
        vertex, edge = field
        out = torch.zeros((self.n_points, 3), dtype=self.dtype, device=self.device)
        out[:len(vertex)] = torch.as_tensor(vertex, dtype=self.dtype,
                                            device=self.device)
        corners = self.cells[:, :4].cpu().numpy()
        midside = self.cells[:, 4:].cpu().numpy()
        for k, (a, b) in enumerate(LagrangeTet.EDGES):
            ends = np.sort(corners[:, [a, b]], axis=1)
            values = np.array([edge[(lo, hi)] for lo, hi in ends])
            out[torch.as_tensor(midside[:, k], device=self.device)] = \
                torch.as_tensor(values, dtype=self.dtype, device=self.device)
        return out

    def fibre_axes(self, quadrature_order: int) -> MaterialAxes:
        """The supplied material frame at the cells' own quadrature points.

        The distributed field is quadratic on this mesh, so it is evaluated
        the way the reference implementation evaluates it: interpolated with
        the element's own basis, at the points the assembly integrates on.
        """
        elem = self.cell_element(self.order, quadrature_order, self.dtype,
                                 self.device)
        sample = lambda field: torch.einsum("qa,eai->eqi", elem.N, field[self.cells])
        return MaterialAxes(f=sample(self.fibre), s=sample(self.sheet))
