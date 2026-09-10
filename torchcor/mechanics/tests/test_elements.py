"""Reference elements and the meshes built from them.

The tetrahedral family exists so the benchmark can be run on the mesh its
participants were given, and these checks are the ones that would catch a
wrong node order, a wrong quadrature rule or a wrongly oriented face -- the
mistakes that show up as a plausible but wrong answer rather than as a crash.
"""
import itertools
import math
import os
import unittest

import numpy as np
import torch

from torchcor.mechanics.assembly import (
    FiniteStrainProblem, volume_quadrature_order,
)
from torchcor.mechanics.elements import (
    LagrangeHex, LagrangeQuad, LagrangeTet, LagrangeTri,
)
from torchcor.mechanics.material import NeoHookeanMaterial
from torchcor.mechanics.mesh import (
    FaceSet, StructuredBoxMesh, TetMesh, as_device, tet_face_local_nodes,
)

DEV = as_device(os.environ.get("TORCHCOR_TEST_DEVICE"))
DT = torch.float64
SIMPLICES = (LagrangeTet, LagrangeTri)
FAMILIES = (LagrangeHex, LagrangeQuad) + SIMPLICES


def reference_nodes(cls, order):
    """Coordinates of the element's own nodes, in its own node order."""
    if cls in SIMPLICES:
        eye = np.eye(cls.DIM)
        vertices = np.vstack([np.zeros(cls.DIM), eye])
        if order == 1:
            return vertices
        return np.vstack([vertices] + [0.5*(vertices[a] + vertices[b])
                                       for a, b in cls.EDGES])
    axis = np.linspace(-1.0, 1.0, order + 1)
    return np.array(list(itertools.product(*([axis]*cls.DIM))))


def cube_tets(n, device=DEV, dtype=DT):
    """Kuhn decomposition of an n-by-n-by-n grid of cubes into six tets each."""
    grid = np.arange(n + 1)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)/n
    cells = []
    for base in itertools.product(range(n), repeat=3):
        corner = {b: int(np.ravel_multi_index(np.add(base, b), (n + 1,)*3))
                  for b in itertools.product((0, 1), repeat=3)}
        for permutation in itertools.permutations(range(3)):
            walk, nodes = [0, 0, 0], [corner[(0, 0, 0)]]
            for direction in permutation:
                walk[direction] = 1
                nodes.append(corner[tuple(walk)])
            span = points[nodes[1:]] - points[nodes[0]]
            if np.linalg.det(span) < 0:                    # keep every cell positive
                nodes[1], nodes[2] = nodes[2], nodes[1]
            cells.append(nodes)
    mesh = TetMesh(torch.as_tensor(points, dtype=dtype, device=device),
                   torch.as_tensor(np.array(cells), dtype=torch.long, device=device),
                   1, device, dtype)
    faces = torch.cat([mesh.cells[:, torch.as_tensor(tet_face_local_nodes(1, f),
                                                     device=device)]
                       for f in range(4)])
    _, inverse, count = torch.unique(faces.sort(dim=1).values, dim=0,
                                     return_inverse=True, return_counts=True)
    mesh.add_face_set(FaceSet("surface", faces[count[inverse] == 1].contiguous(), 1))
    return mesh


def surface_integral(mesh, name, field):
    """``integral(field(x) . n dA)`` over a named surface, with ``n`` from the face set."""
    faces = mesh.face_set(name)
    X = mesh.points[faces.connectivity]
    quad = mesh.face_element(faces.order, 4, mesh.dtype, mesh.device)
    normal = torch.linalg.cross(torch.einsum("fai,qa->fqi", X, quad.dN[..., 0]),
                                torch.einsum("fai,qa->fqi", X, quad.dN[..., 1]), dim=-1)
    return float((quad.weights*(field(torch.einsum("fai,qa->fqi", X, quad.N))
                                * normal).sum(-1)).sum())


class Quadrature(unittest.TestCase):
    def test_simplex_rules_integrate_monomials_exactly(self):
        """The Duffy rule is exact to the degree its Gauss order promises."""
        for cls in SIMPLICES:
            for n in (2, 3, 4):
                elem = cls(1, n, DT, DEV)
                degree = 2*n - 3                       # the collapse costs two orders
                for powers in itertools.product(range(degree + 1), repeat=cls.DIM):
                    if sum(powers) > degree:
                        continue
                    exact = (math.prod(math.factorial(p) for p in powers)
                             / math.factorial(sum(powers) + cls.DIM))
                    got = float((elem.weights*math.prod(
                        elem.points[:, d]**p for d, p in enumerate(powers))).sum())
                    with self.subTest(element=cls.__name__, n=n, powers=powers):
                        self.assertAlmostEqual(got/exact, 1.0, delta=1e-12)

    def test_tensor_product_rules_integrate_monomials_exactly(self):
        for cls in (LagrangeHex, LagrangeQuad):
            for n in (2, 3, 4):
                elem = cls(1, n, DT, DEV)
                for powers in itertools.product(range(2*n), repeat=cls.DIM):
                    exact = math.prod(0.0 if p % 2 else 2.0/(p + 1) for p in powers)
                    got = float((elem.weights*math.prod(
                        elem.points[:, d]**p for d, p in enumerate(powers))).sum())
                    with self.subTest(element=cls.__name__, n=n, powers=powers):
                        self.assertAlmostEqual(got - exact, 0.0, delta=1e-13)


class Basis(unittest.TestCase):
    def test_shape_functions_are_nodal(self):
        """``N_a`` is one at node ``a`` and zero at the others, in this node order."""
        for cls in FAMILIES:
            for order in ((1, 2) if cls in SIMPLICES else (1, 2, 3)):
                nodes = torch.as_tensor(reference_nodes(cls, order), dtype=DT, device=DEV)
                N, _ = cls.basis(order, nodes)
                with self.subTest(element=cls.__name__, order=order):
                    self.assertEqual(N.shape, (cls.n_nodes(order),)*2)
                    self.assertLess(float((N - torch.eye(N.shape[0], dtype=DT,
                                                         device=DEV)).abs().max()), 1e-13)

    def test_basis_is_a_partition_of_unity(self):
        for cls in FAMILIES:
            for order in ((1, 2) if cls in SIMPLICES else (1, 2, 3)):
                elem = cls(order, 3, DT, DEV)
                with self.subTest(element=cls.__name__, order=order):
                    self.assertEqual(elem.n_en, cls.n_nodes(order))
                    self.assertLess(float((elem.N.sum(-1) - 1.0).abs().max()), 1e-13)
                    self.assertLess(float(elem.dN.sum(1).abs().max()), 1e-13)

    def test_gradients_match_finite_differences(self):
        for cls in FAMILIES:
            for order in ((1, 2) if cls in SIMPLICES else (1, 2, 3)):
                torch.manual_seed(0)
                # Well inside the reference cell, so a central difference stays there.
                xi = 0.2*torch.rand(16, cls.DIM, dtype=DT, device=DEV)
                _, dN = cls.basis(order, xi)
                h = 1e-6
                for d in range(cls.DIM):
                    step = torch.zeros_like(xi)
                    step[:, d] = h
                    fd = (cls.basis(order, xi + step)[0]
                          - cls.basis(order, xi - step)[0])/(2*h)
                    with self.subTest(element=cls.__name__, order=order, axis=d):
                        self.assertLess(float((fd - dN[..., d]).abs().max()), 1e-7)


class Tetrahedra(unittest.TestCase):
    def test_geometry_is_exact_and_survives_promotion(self):
        """A unit cube split into tets has volume one and surface area six."""
        linear = cube_tets(2)
        for mesh in (linear, linear.promote()):
            problem = FiniteStrainProblem(mesh, NeoHookeanMaterial(1.0),
                                          density=0.0, viscosity=0.0)
            with self.subTest(order=mesh.order):
                self.assertAlmostEqual(problem.reference_volume(), 1.0, places=12)
                # Divergence theorem: the flux of x is three times the volume,
                # which only holds if every face normal points out of the body.
                self.assertAlmostEqual(surface_integral(mesh, "surface", lambda x: x),
                                       3.0, places=12)

    def test_promotion_keeps_the_nodes_it_already_had(self):
        linear = cube_tets(2)
        quadratic = linear.promote()
        self.assertEqual(quadratic.order, 2)
        self.assertEqual(quadratic.nodes_per_cell, 10)
        self.assertLess(float((quadratic.points[:linear.n_points]
                               - linear.points).abs().max()), 1e-15)
        # A midside node is the midpoint of its edge, so the geometry is unmoved.
        edges = torch.as_tensor(LagrangeTet.EDGES, dtype=torch.long, device=DEV)
        ends = quadratic.points[quadratic.cells[:, :4]][:, edges]
        self.assertLess(float((quadratic.points[quadratic.cells[:, 4:]]
                               - ends.mean(dim=2)).abs().max()), 1e-15)

    def test_red_refinement_leaves_the_boundary_where_it_was(self):
        """Eight cells for one, with the surface and its normals unchanged."""
        coarse = cube_tets(1)
        fine = coarse.refine()
        self.assertEqual(fine.n_cells, 8*coarse.n_cells)
        corner = fine.points[fine.cells]
        self.assertTrue(bool((torch.linalg.det(
            corner[:, 1:] - corner[:, :1]) > 0).all()))
        for mesh, problem in ((coarse, None), (fine, None)):
            problem = FiniteStrainProblem(mesh, NeoHookeanMaterial(1.0),
                                          density=0.0, viscosity=0.0)
            with self.subTest(cells=mesh.n_cells):
                self.assertAlmostEqual(problem.reference_volume(), 1.0, places=12)
                # Same surface, subdivided: same flux of x, hence same normals.
                self.assertAlmostEqual(
                    surface_integral(mesh, "surface", lambda x: x), 3.0, places=12)

    def test_the_default_rule_integrates_the_mass_matrix_exactly(self):
        """The rule a family gets by default must at least be exact where it can be.

        The collapsed simplex rule reaches degree ``2n - 3``, not the cube's
        ``2n - 1``, so reusing the tensor-product point count leaves a linear
        tetrahedron integrating a degree-two mass integrand with a degree-one
        rule -- a 17% error in a matrix that sets the acceleration.
        """
        volume = 1.0/6.0                      # the reference tetrahedron
        exact = volume*(torch.ones(4, 4, dtype=DT, device=DEV)
                        + torch.eye(4, dtype=DT, device=DEV))/20.0
        elem = LagrangeTet(1, volume_quadrature_order(1, LagrangeTet), DT, DEV)
        mass = torch.einsum("q,qa,qb->ab", elem.weights, elem.N, elem.N)
        self.assertLess(float((mass - exact).abs().max()/exact.abs().max()), 1e-13)
        # And the tensor-product default is unchanged by the family split.
        for order, points in ((1, 2), (2, 4), (3, 7)):
            with self.subTest(order=order):
                self.assertEqual(volume_quadrature_order(order, LagrangeHex), points)

    def test_default_rules_reach_the_degree_they_are_chosen_for(self):
        """Each family's default must integrate the degree it targets.

        The two families reach different degrees from the same point count --
        ``2n - 1`` on a cube, ``2n - 3`` on a simplex, because the Duffy
        Jacobian costs two orders -- so one shared formula cannot serve both.
        """
        reached = {LagrangeHex: lambda n: 2*n - 1, LagrangeQuad: lambda n: 2*n - 1,
                   LagrangeTet: lambda n: 2*n - 3, LagrangeTri: lambda n: 2*n - 3}
        for cls, degree_of in reached.items():
            orders = (1, 2) if cls in SIMPLICES else (1, 2, 3)
            for order in orders:
                with self.subTest(element=cls.__name__, order=order):
                    # Volume terms: at least the mass matrix, degree 2p.
                    self.assertGreaterEqual(
                        degree_of(cls.volume_quadrature(order)), 2*order)
                    # Surface terms: a follower pressure reaches degree 3p-1.
                    self.assertGreaterEqual(
                        degree_of(cls.surface_quadrature(order)), 3*order - 1)

    def test_the_mass_matrix_is_integrated_on_a_rule_that_covers_it(self):
        """A curved cell's mass integrand outruns the rule chosen for the tangent.

        Two shape functions against the reference-map determinant reach degree
        ``5p-1`` on a curved Qp cell and total degree ``5p-3`` on a curved Pp
        one. The mass matrix is constant and built once, so it takes its own
        rule rather than raising the cost of every Newton iteration; on the
        affine cells the benchmarks use, the two rules agree.
        """
        def mass(mesh, n):
            problem = FiniteStrainProblem(mesh, NeoHookeanMaterial(1.0),
                                          density=1000.0, bulk_modulus=None)
            elem = mesh.cell_element(mesh.order, n, DT, DEV)
            _, weights = problem._reference_geometry(elem)
            return torch.einsum("eq,qa,qb->eab", weights, elem.N, elem.N)

        torch.manual_seed(1)
        corners = torch.tensor([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               dtype=DT, device=DEV)
        curved_tet = TetMesh(corners, torch.tensor([[0, 1, 2, 3]], device=DEV),
                             1, DEV, DT).promote()
        curved_tet.points[4:] += 0.1*torch.randn(6, 3, dtype=DT, device=DEV)

        curved_hex = StructuredBoxMesh((0, 0, 0), (1, 1, 1), (1, 1, 1), order=2,
                                       device=DEV, dtype=DT)
        curved_hex.points[4] += torch.tensor([0.0, 0.10, 0.08], dtype=DT, device=DEV)
        curved_hex.points[13] += torch.tensor([0.09, 0.0, -0.07], dtype=DT, device=DEV)

        for name, mesh in (("curved tetrahedron", curved_tet),
                           ("curved hexahedron", curved_hex)):
            family = mesh.cell_element
            dense = mass(mesh, 9)
            scale = float(dense.abs().max())
            with self.subTest(cell=name):
                # The rule chosen for the nonlinear terms does not cover it ...
                self.assertGreater(
                    float((mass(mesh, family.volume_quadrature(2)) - dense).abs().max())
                    / scale, 1e-6)
                # ... and the mass rule does.
                self.assertLess(
                    float((mass(mesh, family.mass_quadrature(2)) - dense).abs().max())
                    / scale, 1e-12)

        # An affine cell is unaffected: the benchmarks' results do not move.
        flat = TetMesh(corners, torch.tensor([[0, 1, 2, 3]], device=DEV),
                       1, DEV, DT).promote()
        self.assertLess(float((mass(flat, LagrangeTet.volume_quadrature(2))
                               - mass(flat, LagrangeTet.mass_quadrature(2))
                               ).abs().max()), 1e-16)

    def test_promotion_grows_the_node_sets_that_name_a_surface(self):
        """A set used to constrain a surface must still cover it afterwards."""
        linear = cube_tets(2)
        linear.add_node_set("surface", linear.face_set("surface").nodes())
        corner = linear.nodes_where(lambda X: (X.abs() < 1e-12).all(dim=1))
        linear.add_node_set("one corner", corner)
        quadratic = linear.promote()
        self.assertEqual(sorted(quadratic.node_set("surface").tolist()),
                         sorted(quadratic.face_set("surface").nodes().tolist()))
        self.assertGreater(quadratic.node_set("surface").numel(),
                           linear.node_set("surface").numel())
        # A set that names no surface is a list of chosen vertices: untouched.
        self.assertEqual(quadratic.node_set("one corner").tolist(), corner.tolist())

    def test_locate_recovers_the_point_it_was_given(self):
        torch.manual_seed(0)
        probes = torch.rand(64, 3, dtype=DT, device=DEV)
        linear = cube_tets(2)
        for mesh in (linear, linear.promote()):
            cells, xi = mesh.locate(probes)
            with self.subTest(order=mesh.order):
                self.assertLess(float((mesh.interpolate_local(mesh.points, cells, xi)
                                       - probes).abs().max()), 1e-12)
                self.assertTrue(bool((xi >= -1e-12).all()))
                self.assertTrue(bool((xi.sum(dim=1) <= 1 + 1e-12).all()))

    def test_a_wrongly_oriented_cell_is_rejected(self):
        mesh = cube_tets(1)
        mesh.cells[0] = mesh.cells[0][[0, 2, 1, 3]]
        with self.assertRaises(ValueError):
            FiniteStrainProblem(mesh, NeoHookeanMaterial(1.0), density=0.0, viscosity=0.0)


class PatchTest(unittest.TestCase):
    """An affine displacement is an exact equilibrium state of any element."""

    def test_interior_residual_vanishes_for_every_family(self):
        gradient = torch.tensor([[0.10, 0.03, -0.02], [0.00, -0.05, 0.04],
                                 [0.01, 0.02, 0.20]], dtype=DT, device=DEV)
        linear = cube_tets(2)
        meshes = {"tet P1": linear, "tet P2": linear.promote(),
                  "hex Q2": StructuredBoxMesh((0, 0, 0), (1, 1, 1), (2, 2, 2),
                                              order=2, device=DEV, dtype=DT)}
        for name, mesh in meshes.items():
            problem = FiniteStrainProblem(mesh, NeoHookeanMaterial(1.0),
                                          density=0.0, viscosity=0.0)
            R, _, info = problem.evaluate((mesh.points @ gradient.T).reshape(-1),
                                          1.0, tangent=False)
            edge = ((mesh.points.abs() < 1e-12)
                    | ((mesh.points - 1.0).abs() < 1e-12)).any(dim=1)
            with self.subTest(mesh=name):
                self.assertGreater(int((~edge).sum()), 0)
                self.assertLess(float(R.reshape(-1, 3)[~edge].norm(dim=1).max()), 1e-11)
                # The same uniform stretch, so the same volume change.
                self.assertAlmostEqual(float(info.min_jacobian),
                                       float(torch.linalg.det(
                                           torch.eye(3, dtype=DT, device=DEV) + gradient)),
                                       places=12)


if __name__ == "__main__":
    unittest.main()
