"""Carrying fields between the discretisations electromechanics couples.

The contraction model lives on nodes, the material wants a tension per cell,
and the stretch it produces comes back from a gradient evaluated inside each
cell.  Those are three different places, and this module is the only one that
knows it -- everything else works with whichever field it owns.

    nodes  --NodeToCell-->  cells  --(mechanics)-->  displacement
    nodes  <--CellToNode--  cells  <--CellDeformation, fibre_stretch--

Which nodes take part is the caller's choice.  When the contraction model runs
on the mechanics mesh, pass the full connectivity; when it runs on a coarser
electrophysiology mesh whose nodes are the cell vertices, pass just those
columns.  Getting that wrong is silent: nodes the connectivity never mentions
keep whatever they were initialised with.
"""

import torch


class NodeToCell:
    """Average a per-node field onto cells.

    ``connectivity`` is ``(n_cells, n_per_cell)``.  With ``keepdim`` the result
    is ``(n_cells, 1)``, which is the shape a material expects for one value
    per cell.
    """

    def __init__(self, connectivity):
        self.connectivity = connectivity

    def __call__(self, nodal, keepdim=True):
        cell = nodal[self.connectivity].mean(dim=1)
        return cell.unsqueeze(1) if keepdim else cell


class CellToNode:
    """Average a per-cell field back onto nodes.

    Each node takes the mean of the cells that contain it.  Nodes absent from
    ``connectivity`` receive nothing and stay zero, so the connectivity has to
    cover every node the result is read at.
    """

    def __init__(self, connectivity, n_nodes):
        self.connectivity = connectivity
        self.n_nodes = int(n_nodes)
        flat = connectivity.reshape(-1)
        counts = torch.zeros(self.n_nodes, device=connectivity.device)
        counts.index_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        self.count = counts.clamp_(min=1.0)
        self.covered = int((counts > 0).sum())

    def __call__(self, cell):
        flat = self.connectivity.reshape(-1)
        values = cell.reshape(-1, 1).expand(-1, self.connectivity.shape[1]).reshape(-1)
        total = torch.zeros(self.n_nodes, device=cell.device, dtype=cell.dtype)
        total.index_add_(0, flat, values)
        return total / self.count.to(cell.dtype)


class CellDeformation:
    """Deformation gradient at chosen points of every cell, from *all* nodes.

    ``F = I + Grad u`` evaluated with the mesh's own shape functions, so a
    quadratic displacement is differentiated as a quadratic one.  Using only
    the corner nodes of a quadratic tetrahedron discards the midside values,
    and they carry real strain: a displacement that is zero at every vertex and
    0.1 at one edge node has a true fibre stretch of 1.08 at an interior point,
    which a vertex-only gradient reports as 1.00.

    The reference gradients depend only on the undeformed mesh, so they are
    formed once and reused.  ``points`` are evaluation points in reference
    coordinates; the default is the cell centroid.
    """

    def __init__(self, mesh, points=None):
        if points is None:
            points = torch.full((1, 3), 0.25, device=mesh.device, dtype=mesh.dtype)
        points = torch.as_tensor(points, device=mesh.device, dtype=mesh.dtype)
        _, dN = mesh.cell_element.basis(mesh.order, points)
        jacobian = torch.einsum("eai,qaj->eqij", mesh.points[mesh.cells], dN)
        self.grad = torch.einsum("qaj,eqjI->eqaI", dN, torch.linalg.inv(jacobian))
        self.cells = mesh.cells
        self.eye = torch.eye(3, device=mesh.device, dtype=mesh.dtype)

    def __call__(self, displacement):
        """``(n_cells, n_points, 3, 3)`` from a ``(n_nodes, 3)`` displacement."""
        return self.eye + torch.einsum("eai,eqaI->eqiI",
                                       displacement[self.cells], self.grad)


def fibre_stretch(deformation_gradient, fibre):
    """``lambda = |F f|``, the stretch along the fibre.

    ``deformation_gradient`` is ``(..., 3, 3)`` and ``fibre`` ``(..., 3)``, and
    the leading axes have to line up: a ``(cells, points, 3, 3)`` gradient needs
    a ``(cells, 1, 3)`` fibre, not ``(cells, 3)``, which would broadcast the
    cell axis against the point axis instead.
    """
    return torch.linalg.vector_norm(
        (deformation_gradient @ fibre.unsqueeze(-1)).squeeze(-1), dim=-1)
