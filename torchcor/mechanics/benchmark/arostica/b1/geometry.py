"""Monoventricular geometry and fibres of the Arostica et al. benchmark.

Both are specific to that paper -- a long axis along ``x``, endocardium and
epicardium truncated at their own angles, and a rule-based fibre field written
in the ellipsoid's own coordinates -- so they live here rather than in
``torchcor.mechanics``.  Everything else (connectivity, face sets,
interpolation, point location) comes from the library meshes unchanged.

Two discretisations of the same domain are provided, and they share everything
that is prescribed by the paper: :class:`MonoventricleMesh` builds a structured
hexahedral mesh from the equations of section 3.1, and :class:`ReferenceMesh`
reads the tetrahedral mesh the participants were given.  Comparing them
separates the benchmark's physics from the geometry it is discretised on.

Section 3.1 fixes the surfaces, equations (12)-(16) the fibres.
"""

from pathlib import Path

import numpy as np
import torch

from torchcor.mechanics.linear import ConjugateGradient, JacobiPreconditioner
from torchcor.mechanics.material import MaterialAxes
from torchcor.mechanics.mesh import (
    FaceSet, HexMesh, TetMesh, as_device, tet_face_local_nodes)

#: Endocardial and epicardial ``(r_short, r_long)`` in metres, equations (10), (11).
ENDO = (2.5e-2, 9.0e-2)
EPI = (3.5e-2, 9.7e-2)
#: Truncation angles of the two surfaces: ``mu`` runs from the apex to these.
MU_ENDO = -np.arccos(5.0/17.0)
MU_EPI = -np.arccos(5.0/20.0)
#: Fibre helix angle on each surface, equation (16).
#:
#: Table 3 lists -60 on the endocardium and +60 on the epicardium, but that is
#: relative to the paper's circumferential direction, whose sign convention its
#: own footnote 5 describes as swapped against the usual one.  Taken literally
#: against ``e_theta`` as defined here it reverses the twist: the passive case
#: then gives u_z = +17.6 mm at both tracked points where figure 9 shows
#: -17.5 mm, with u_x and u_y bit-identical either way.  The sense below is the
#: one that reproduces the published sign, and is also the standard
#: physiological helix -- positive at the endocardium, negative at the
#: epicardium.
ALPHA_ENDO, ALPHA_EPI = 60.0, -60.0


class _Ventricle:
    """What the paper prescribes about the domain, for any element family.

    Both meshes below discretise the same truncated ellipsoid, so the radii,
    the transmural Laplace problem of equation (13) and the fibre rule of
    equations (12)-(16) are written once here and inherited by both.  They use
    only the mesh interface -- ``cell_element``, ``locate``,
    ``interpolate_local`` and the named surfaces -- never the way the mesh was
    built.
    """

    # ------------------------------------------------------------ parameters
    @staticmethod
    def radii(t):
        """``(r_short, r_long)`` at transmural coordinate ``t`` in ``[0, 1]``."""
        return (ENDO[0] + t*(EPI[0] - ENDO[0]), ENDO[1] + t*(EPI[1] - ENDO[1]))

    def transmural_field(self) -> torch.Tensor:
        """Nodal solution of the Laplace problem of equation (13).

        ``0`` on the endocardium, ``1`` on the epicardium, no flux through the
        base.  This is the transmural coordinate the fibre rule is written in;
        it is not the same as the geometric parameter the mesh is built from,
        which is why it is solved for rather than read off.
        """
        if getattr(self, "_transmural", None) is not None:
            return self._transmural

        elem = self.cell_element(self.order, self.order + 1, self.dtype, self.device)
        Xe = self.points[self.cells]
        jacobian = torch.einsum("eaI,qaj->eqIj", Xe, elem.dN)
        dNdX = torch.einsum("qaj,eqjI->eqaI", elem.dN,
                            torch.linalg.inv(jacobian))
        w = elem.weights[None, :]*torch.linalg.det(jacobian)
        Ke = torch.einsum("eq,eqaI,eqbI->eab", w, dNdX, dNdX)

        n = self.n_points
        rows = self.cells[:, :, None].expand_as(Ke).reshape(-1)
        cols = self.cells[:, None, :].expand_as(Ke).reshape(-1)

        fixed = torch.zeros(n, dtype=torch.bool, device=self.device)
        value = torch.zeros(n, dtype=self.dtype, device=self.device)
        for name, level in (("endo", 0.0), ("epi", 1.0)):
            fixed[self.node_sets[name]] = True
            value[self.node_sets[name]] = level

        entries = Ke.reshape(-1)
        raw = torch.sparse_coo_tensor(torch.stack([rows, cols]), entries,
                                      (n, n)).coalesce().to_sparse_csr()
        rhs = torch.where(fixed, value, -(raw @ value))

        # Dirichlet elimination: drop the rows and columns of the fixed nodes
        # and put one on their diagonal, so the system stays symmetric positive
        # definite and the fixed values come back unchanged.
        keep = ~(fixed[rows] | fixed[cols])
        eye = torch.arange(n, device=self.device)
        matrix = torch.sparse_coo_tensor(
            torch.cat([torch.stack([rows[keep], cols[keep]]),
                       torch.stack([eye[fixed], eye[fixed]])], dim=1),
            torch.cat([entries[keep],
                       torch.ones(int(fixed.sum()), dtype=self.dtype,
                                  device=self.device)]),
            (n, n)).coalesce().to_sparse_csr()

        # The library's conjugate gradients, not another recurrence: it brings
        # the true-residual acceptance and breakdown tests with it.
        jacobi = JacobiPreconditioner()
        diagonal = torch.zeros(n, dtype=self.dtype, device=self.device)
        on_diagonal = keep & (rows == cols)
        diagonal.index_add_(0, rows[on_diagonal], entries[on_diagonal])
        diagonal = torch.where(fixed, torch.ones_like(diagonal), diagonal)
        jacobi.inv_diag = torch.where(diagonal.abs() > 0, diagonal,
                                      torch.ones_like(diagonal)).reciprocal()
        result = ConjugateGradient(jacobi, rtol=1e-12, max_iter=5*n).solve(matrix, rhs)
        if not result.converged:
            raise RuntimeError("the transmural Laplace solve did not converge "
                               f"(relative residual {result.residual:.2e})")
        self._transmural = result.x.clamp(0.0, 1.0)
        return self._transmural

    def fibre_axes(self, quadrature_order: int) -> MaterialAxes:
        """The material frame at the cells' own quadrature points.

        Equivalent to handing the assembly a callable of position, except that
        these points are known to be the quadrature points of known cells, so
        their transmural coordinate is one interpolation rather than a search
        of the mesh -- which is what the reference implementation also does,
        interpolating its fibre field into a space on the same cells.
        ``quadrature_order`` must be the rule the assembly will use.
        """
        elem = self.cell_element(self.order, quadrature_order, self.dtype,
                                 self.device)
        t = torch.einsum("qa,ea->eq", elem.N, self.transmural_field()[self.cells])
        x = torch.einsum("qa,eai->eqi", elem.N, self.points[self.cells])
        f, s = self._frame_at(x, t.clamp(0.0, 1.0))
        return MaterialAxes(f=f, s=s)

    def fibre_frame(self, points):
        """``(f, s)`` of equation (15) at arbitrary physical ``points``.

        The helix angle follows the Laplace transmural coordinate of equation
        (13); ``n`` is the surface normal of the ellipsoid family and ``s``
        completes the frame.  Locating the points is the cost here, so the
        assembly uses :meth:`fibre_axes` instead.
        """
        x = torch.as_tensor(points, dtype=self.dtype, device=self.device)
        # t-bar from the Laplace problem, interpolated at these points.
        t = self.interpolate_local(self.transmural_field()[:, None],
                                   *self.locate(x.reshape(-1, 3)))[:, 0]
        return self._frame_at(x, t.clamp(0.0, 1.0).reshape(x.shape[:-1]))

    def _frame_at(self, x, t):
        """``(f, s)`` where the transmural coordinate is already known.

        Equation (12): mu and theta follow from the radii ``t`` selects.
        """
        shape = x.shape
        flat = x.reshape(-1, 3)
        t = t.reshape(-1)
        rs, rl = self.radii(t)
        rho = torch.linalg.vector_norm(flat[:, 1:], dim=1)
        mu = torch.atan2(-rho/rs, flat[:, 0]/rl)
        v = torch.atan2(-flat[:, 2], -flat[:, 1])
        sin_mu, cos_mu = torch.sin(mu), torch.cos(mu)
        sin_th, cos_th = torch.sin(v), torch.cos(v)

        unit = lambda a: a/torch.linalg.vector_norm(a, dim=-1, keepdim=True)
        e_mu = unit(torch.stack([-rl*sin_mu, rs*cos_mu*cos_th, rs*cos_mu*sin_th], -1))
        # d(x)/d(theta) carries a factor sin(mu) that vanishes at the apex;
        # cancelling it analytically leaves a unit vector everywhere.
        zero = torch.zeros_like(sin_mu)
        orient = torch.where(sin_mu > 0, 1.0, -1.0)[:, None]
        e_th = orient*torch.stack([zero, -sin_th, cos_th], -1)
        alpha = torch.deg2rad(ALPHA_ENDO + t*(ALPHA_EPI - ALPHA_ENDO))
        f = unit(torch.sin(alpha)[:, None]*e_mu + torch.cos(alpha)[:, None]*e_th)
        n = unit(torch.linalg.cross(e_mu, e_th, dim=-1))
        s = unit(torch.linalg.cross(f, n, dim=-1))
        return f.reshape(shape), s.reshape(shape)


class MonoventricleMesh(_Ventricle, HexMesh):
    r"""Truncated ellipsoid of section 3.1, as a structured hexahedral mesh.

    Parametrised by ``(t, s, v)``: ``t`` transmural from endocardium to
    epicardium, ``s`` from apex to base, ``v`` around the long axis.  The
    surface point follows equation (9) with the long axis along ``x``,

    .. math::  x = r_l\cos\mu, \quad y = r_s\sin\mu\cos\theta,
               \quad z = r_s\sin\mu\sin\theta

    and radii interpolated linearly through the wall.  The two surfaces are
    truncated at *different* angles -- ``arccos(5/17)`` and ``arccos(5/20)``
    against long radii of 90 and 97 mm -- so unlike the Land geometry the base
    is not a plane; ``mu`` at the base is interpolated through the wall, which
    makes it the ruled surface joining the two rings.

    The circumferential seam shares nodes and the apex collapses to one node
    per transmural layer, so neither can tear open.

    Registers the surfaces ``"endo"``, ``"epi"`` and ``"base"``.
    """

    def __init__(self, divisions=(1, 8, 16), order: int = 2, device=None,
                 dtype: torch.dtype = torch.float64, grading: float = 1.0,
                 geometry_order: int = None) -> None:
        nt, nu, nv = (int(v) for v in divisions)
        p = int(order)
        device = as_device(device)
        n1, n2, n3 = p*nt + 1, p*nu + 1, p*nv + 1

        t = torch.linspace(0., 1., n1, dtype=dtype, device=device)
        s = torch.linspace(0., 1., n2, dtype=dtype, device=device)
        # ``grading`` above one shortens the bands nearest the apex, where the
        # elements collapse to a point; it changes only how the same surfaces
        # are divided, never the surfaces themselves.
        self.grading = float(grading)
        if self.grading != 1.0:
            s = s**self.grading
        # v decreasing keeps (t, s, v) right-handed, as the assembler requires.
        v = torch.linspace(torch.pi, -torch.pi, n3, dtype=dtype, device=device)

        rs, rl = self.radii(t)
        mu = -torch.pi + s[None, :]*(self.base_angle(t)[:, None] + torch.pi)
        ring = rs[:, None]*torch.sin(mu)
        points = torch.stack([
            (rl[:, None]*torch.cos(mu))[:, :, None].expand(n1, n2, n3),
            ring[:, :, None]*torch.cos(v)[None, None, :],
            ring[:, :, None]*torch.sin(v)[None, None, :],
        ], dim=-1).reshape(-1, 3)

        ids = torch.arange(n1*n2*n3, device=device).reshape(n1, n2, n3)
        ids[:, :, -1] = ids[:, :, 0]
        ids[:, 0, :] = ids[:, 0, :1]
        keep, inverse = torch.unique(ids.reshape(-1), return_inverse=True)
        points, ids = points[keep], inverse.reshape(n1, n2, n3)

        off = torch.arange(p + 1, device=device)
        i = p*torch.arange(nt, device=device)[:, None, None, None, None, None] + off[None, None, None, :, None, None]
        j = p*torch.arange(nu, device=device)[None, :, None, None, None, None] + off[None, None, None, None, :, None]
        k = p*torch.arange(nv, device=device)[None, None, :, None, None, None] + off[None, None, None, None, None, :]
        cells = ids[i, j, k].reshape(nt*nu*nv, (p + 1)**3)

        self.geometry_order = p if geometry_order is None else int(geometry_order)
        if self.geometry_order == 1 and p > 1:
            points = self._straighten(points, cells, p)

        super().__init__(points, cells, order=p, device=device, dtype=dtype)
        self.divisions = (nt, nu, nv)
        self.label = "x".join(str(d) for d in self.divisions)

        cell_id = torch.arange(nt*nu*nv, device=device).reshape(nt, nu, nv)
        for name, ids_, face in (("endo", cell_id[0], "x-"),
                                 ("epi", cell_id[-1], "x+"),
                                 ("base", cell_id[:, -1], "y+")):
            fs = self.faces_of_cells(name, ids_.reshape(-1), face)
            self.add_face_set(fs)
            self.add_node_set(name, fs.nodes())

    @staticmethod
    def _straighten(points, cells, p):
        """Move every non-corner node onto the multilinear map of its corners.

        The reference implementation runs quadratic *displacement* on a
        *linear* mesh, so its boundary is faceted and its facet normals are
        piecewise constant.  Displacement order and geometry order are
        independent, and a Lagrange basis reproduces degree-one polynomials
        exactly, so placing the interior and edge nodes on the multilinear
        interpolation of the eight corners makes the geometry map exactly
        trilinear while the displacement stays quadratic.  Shared nodes get
        the same position from every element that owns them, so the result is
        conforming.
        """
        n = p + 1
        axis = torch.linspace(0.0, 1.0, n, dtype=points.dtype, device=points.device)
        w = torch.stack([1.0 - axis, axis], dim=-1)                  # (n, 2)
        # corner ordering matches the lexicographic (i, j, k) node numbering
        corner = cells.reshape(-1, n, n, n)[:, ::p, ::p, ::p].reshape(-1, 8)
        blend = torch.einsum("ia,jb,kc->ijkabc", w, w, w).reshape(n**3, 8)
        straight = torch.einsum("qa,ea i->eqi".replace(" ", ""), blend,
                                points[corner])
        flat = cells.reshape(-1)
        out = points.clone()
        out[flat] = straight.reshape(-1, 3)
        return out

    # ------------------------------------------------------------ parameters
    @classmethod
    def base_angle(cls, t):
        """Truncation angle at transmural position ``t``.

        The base joins the two rings with a *straight* segment revolved about
        the axis, which is what meshing the two surfaces and capping them
        produces.  Interpolating the angle instead bows that segment out by up
        to 0.077 mm, so the angle is found from where each intermediate
        ellipsoid crosses the straight line.
        """
        t = torch.as_tensor(t)
        ends = [(rl*np.cos(mu), rs*abs(np.sin(mu)))
                for (rs, rl), mu in ((ENDO, MU_ENDO), (EPI, MU_EPI))]
        (xe, re), (xp, rp) = ends
        rs, rl = cls.radii(t)
        lo = torch.zeros_like(t)
        hi = torch.ones_like(t)
        for _ in range(60):                       # bisect along the segment
            mid = 0.5*(lo + hi)
            x, r = xe + mid*(xp - xe), re + mid*(rp - re)
            inside = (x/rl).square() + (r/rs).square() < 1.0
            lo = torch.where(inside, mid, lo)
            hi = torch.where(inside, hi, mid)
        lam = 0.5*(lo + hi)
        x, r = xe + lam*(xp - xe), re + lam*(rp - re)
        return torch.atan2(-r/rs, x/rl)

    def to_parametric(self, points):
        """``(t, mu, v)`` of physical ``points``, the mesh's own parameters.

        ``t`` is the transmural coordinate of the ellipsoid family the mesh is
        built from, found by bisection on
        ``(x/r_l)^2 + (rho/r_s)^2 = 1``.  The paper instead solves a Laplace
        problem for it, which agrees on both surfaces and differs slightly
        inside; this one is exact for this geometry.
        """
        x = torch.as_tensor(points, dtype=self.dtype, device=self.device).reshape(-1, 3)
        rho = torch.linalg.vector_norm(x[:, 1:], dim=1)
        lo = torch.zeros_like(rho)
        hi = torch.ones_like(rho)
        for _ in range(60):
            mid = 0.5*(lo + hi)
            rs, rl = self.radii(mid)
            outside = (x[:, 0]/rl).square() + (rho/rs).square() > 1.0
            lo = torch.where(outside, mid, lo)
            hi = torch.where(outside, hi, mid)
        t = 0.5*(lo + hi)
        rs, rl = self.radii(t)
        # sin(mu) is negative over [-pi, 0], so r_s sin(mu) = -rho and the
        # azimuth carries the resulting sign flip.
        mu = torch.atan2(-rho/rs, x[:, 0]/rl)
        v = torch.atan2(-x[:, 2], -x[:, 1])
        return t, mu, v

    def locate_parametric(self, t, s, v):
        """Cell and reference coordinates of parametric points.

        The mesh is a uniform structured grid in ``(t, s, v)``, so this is
        exact rather than a search.
        """
        nt, nu, nv = self.divisions
        def split(a, n):
            a = (a*n).clamp(0.0, float(n))
            i = a.floor().clamp(max=n - 1)
            return i.long(), 2.0*(a - i) - 1.0
        i, xi = split(torch.as_tensor(t, dtype=self.dtype, device=self.device), nt)
        j, eta = split(torch.as_tensor(s, dtype=self.dtype, device=self.device), nu)
        w = (torch.pi - torch.as_tensor(v, dtype=self.dtype, device=self.device))/(2*torch.pi)
        k, zeta = split(torch.remainder(w, 1.0), nv)
        return (i*nu + j)*nv + k, torch.stack([xi, eta, zeta], dim=-1)

    def locate(self, points, tolerance: float = None, outside: str = "raise"):
        """Cell and reference coordinates of Cartesian ``points``.

        The analytic inversion only starts the search: it is then Newton
        corrected against the actual element geometry, so the coordinates
        reproduce the requested point rather than the ideal surface.

        The default tolerance is a micrometre because that is the scale of the
        geometry itself: a point on an element interface -- mid-wall on an even
        number of transmural elements, say -- lies a few tens of nanometres off
        the interpolated surface, and is inside the mesh either way.

        On a faceted mesh the analytic inverse can name a neighbouring cell,
        and the point then lies outside the cell it was given.  Those points are
        retried in the adjacent cells rather than accepted under a wider
        tolerance, which would hide the miss instead of correcting it.
        """
        if tolerance is None:
            tolerance = 1e-6
        t, mu, theta = self.to_parametric(points)
        s = (mu + torch.pi)/(self.base_angle(t) + torch.pi)
        # The grid is uniform in the pre-graded coordinate, so invert the
        # grading before indexing into it.
        cells, xi = self.locate_parametric(t, s.pow(1.0/self.grading), theta)
        xi, distance = self.refine_local(points, cells, xi, tolerance=tolerance)
        if bool((distance > tolerance).any()):
            cells, xi, distance = self._locate_neighbors(
                torch.as_tensor(points, dtype=self.dtype, device=self.device
                                ).reshape(-1, 3), cells, xi, distance, tolerance)
        worst = float(distance.max())
        if worst > tolerance and outside == "raise":
            raise ValueError(f"{int((distance > tolerance).sum())} of "
                             f"{distance.numel()} points are not inside the mesh; "
                             f"the worst is {worst:.3e} m away")
        return cells, xi


#: The tetrahedral mesh the participants were given, converted to arrays.
REFERENCE_MESH = Path(__file__).with_name("reference")/"ellipsoid_0.005.npz"
#: Its facet markers, from ``cardiac_benchmark_toolkit``'s mesh generator.
REFERENCE_SURFACES = {1: "endo", 2: "epi", 3: "base"}


class ReferenceMesh(_Ventricle, TetMesh):
    """The tetrahedral mesh distributed with the benchmark.

    The participants were given one mesh, generated at 5 mm by
    ``cardiac_benchmark_toolkit`` and stored as four-node tetrahedra with
    three-node boundary triangles.  Reading it removes the geometry as a
    variable in a comparison: everything the paper prescribes is then shared
    with the participants, and only the solver differs.

    ``order=2`` adds a node on every edge, which is the quadratic
    *displacement* on linear geometry that the reference implementations use;
    the boundary and its facet normals stay exactly as the file has them.

    """

    #: Names the discretisation in reports, as ``divisions`` does for the
    #: structured mesh; :meth:`load` replaces it with the file's own name.
    label = "reference"

    @classmethod
    def load(cls, path=REFERENCE_MESH, order: int = 2, device=None,
             dtype: torch.dtype = torch.float64) -> "ReferenceMesh":
        device = as_device(device)
        stored = np.load(str(path))
        points = torch.as_tensor(stored["points"], dtype=dtype, device=device)
        cells = torch.as_tensor(stored["tets"], dtype=torch.long, device=device)
        # Cells are stored with either orientation; the assembly needs positive
        # ones, and swapping two vertices is the only change that makes.
        corner = points[cells]
        flip = torch.linalg.det(corner[:, 1:] - corner[:, :1]) < 0
        cells[flip] = cells[flip][:, [0, 2, 1, 3]]

        mesh = cls(points, cells, order=1, device=device, dtype=dtype)
        triangles = torch.as_tensor(stored["triangles"], dtype=torch.long, device=device)
        markers = torch.as_tensor(stored["markers"], dtype=torch.long, device=device)
        for marker, name in REFERENCE_SURFACES.items():
            mesh.add_face_set(FaceSet(name, mesh._outward(triangles[markers == marker]), 1))

        if int(order) == 2:
            mesh = mesh.promote()
        for name in REFERENCE_SURFACES.values():
            mesh.add_node_set(name, mesh.face_set(name).nodes())

        stem = "reference" if Path(path) == REFERENCE_MESH else Path(path).stem
        # The displacement order is part of the discretisation, so it belongs
        # in the name whenever it is not the quadratic one the paper implies.
        mesh.label = stem if int(order) == 2 else f"{stem}-p{order}"
        return mesh

    def _outward(self, triangles: torch.Tensor) -> torch.Tensor:
        """Reorder stored boundary triangles to the orientation of their cell.

        A face set promises an outward ``d_xi1 x d_xi2``, and a stored triangle
        carries no orientation.  Each one is matched to the cell face with the
        same three nodes, which the cell numbering already orients outwards.
        """
        code = lambda t: ((t.sort(dim=1).values.to(torch.int64)
                           * torch.tensor([self.n_points**2, self.n_points, 1],
                                          device=t.device)).sum(dim=1))
        faces = torch.cat([self.cells[:, torch.as_tensor(tet_face_local_nodes(1, f),
                                                         device=self.device)]
                           for f in range(4)])
        keys = code(faces)
        order = keys.argsort()
        want = code(triangles)
        found = torch.searchsorted(keys[order], want).clamp(max=keys.numel() - 1)
        if not bool((keys[order][found] == want).all()):
            raise ValueError("the stored boundary triangles are not faces of "
                             "the stored tetrahedra")
        return faces[order[found]].contiguous()
