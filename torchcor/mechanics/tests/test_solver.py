"""Solver and boundary behaviour, including every defect a review has found.

Each test here pins a failure that was once real, so it stays.
"""
import math
import os
import unittest

import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.assembly import FiniteStrainProblem
from torchcor.mechanics.boundary import DirichletBC, FollowerPressure, RobinBC
from torchcor.mechanics.elements import LagrangeQuad
from torchcor.mechanics.material import (
    ActiveStressMaterial, GuccioneMaterial, HolzapfelOgdenMaterial,
    IsochoricMaterial, MaterialAxes,
)
from torchcor.mechanics.mesh import StructuredBoxMesh, as_device
from torchcor.mechanics.solver import DynamicSolver, QuasiStaticSolver

DEV = as_device(os.environ.get("TORCHCOR_TEST_DEVICE"))
DT = torch.float64
SOFT = dict(a=59.0, b=8.023, a_f=18472.0, b_f=16.026, a_s=2481.0, b_s=11.12,
            a_fs=216.0, b_fs=11.436)


def ActiveIsochoric():
    """A law whose viscous coupling makes the tangent lose symmetry mid-solve."""
    return ActiveStressMaterial(
        IsochoricMaterial(HolzapfelOgdenMaterial(**SOFT, bulk_modulus=1e6)),
        lambda t: 2500.0*t)


def box(divisions=(2, 2, 2), lengths=(1.0, 1.0, 1.0), order=2):
    return StructuredBoxMesh((0, 0, 0), lengths, divisions, order=order,
                             device=DEV, dtype=DT)


class Boundaries(unittest.TestCase):
    def test_pressure_resultant_and_tangent(self):
        """A flat face carries p*A, and the follower tangent is nonsymmetric."""
        m = box(lengths=(2.0, 3.0, 1.5))
        load = FollowerPressure.on_surface(m, "z-", 7.0)
        quad = LagrangeQuad(m.order, m.order + 1, DT, torch.device(DEV))
        force, K = load.contribution(m.points, quad)
        torch.testing.assert_close(
            force.reshape(-1, 3).sum(0),
            torch.tensor([0.0, 0.0, 7.0*2.0*3.0], dtype=DT, device=DEV),
            rtol=1e-12, atol=1e-9)
        self.assertGreater(float((K - K.mT).abs().max()), 1e-6)

    def test_pressure_quadrature_is_converged_at_every_order(self):
        """The follower integrand is cubic in the shape functions, not linear."""
        for order in (2, 3):
            with self.subTest(order=order):
                m = box((1, 1, 1), order=order)
                p = FiniteStrainProblem(
                    m, GuccioneMaterial(10, 1, 1, 1),
                    dirichlet=[DirichletBC.on_surface(m, "x-")],
                    pressures=[FollowerPressure.on_surface(m, "z-", 1.0)])
                torch.manual_seed(0)
                x = m.points + 0.08*torch.randn_like(m.points)
                bc = p.pressures[0]
                dense = LagrangeQuad(order, 8, DT, torch.device(DEV))
                want = bc.contribution(x, dense)
                got = bc.contribution(x, p.face_elem)
                for a, b in zip(got, want):
                    self.assertLess(float((a - b).abs().max()/b.abs().max()), 1e-12)

    def test_robin_support_is_linear_and_directional(self):
        m = box(lengths=(2.0, 1.0, 1.0))
        common = dict(bulk_modulus=1e3, dirichlet=[DirichletBC.on_surface(m, "x-")])
        plain = FiniteStrainProblem(m, GuccioneMaterial(10, 1, 1, 1), **common)
        torch.manual_seed(0)
        u = 0.01*torch.randn(plain.n_dofs, dtype=DT, device=DEV)
        v = 0.02*torch.randn(plain.n_dofs, dtype=DT, device=DEV)
        base = plain.evaluate(u, tangent=False)[0]
        for normal_only in (False, True):
            with self.subTest(normal_only=normal_only):
                r = FiniteStrainProblem(m, GuccioneMaterial(10, 1, 1, 1), robin=[
                    RobinBC.on_surface(m, "x+", stiffness=1e3, damping=7.0,
                                       normal_only=normal_only)], **common)
                K = r.assembler.to_csr(r.stiffness_support)
                C = r.assembler.to_csr(r.damping_support)
                elastic = r.evaluate(u, tangent=False)[0]
                both = r.evaluate(u, tangent=False, velocity=v)[0]
                self.assertLess(float((elastic - base - K @ u).abs().max()), 1e-12)
                self.assertLess(float((both - elastic - C @ v).abs().max()), 1e-12)
                probe = torch.zeros_like(u); probe[1::3] = 1.0
                across = float((K @ probe)[1::3].sum())
                self.assertAlmostEqual(across, 0.0 if normal_only else 1e3, delta=1e-6)

    def test_scalar_pressure_does_not_ramp_with_the_clock(self):
        """Physical time is not a continuation factor."""
        m = box()
        quad = LagrangeQuad(m.order, m.order + 1, DT, torch.device(DEV))
        const = FollowerPressure.on_surface(m, "z-", 2.0)
        sched = FollowerPressure.on_surface(m, "z-", lambda t: 2.0)
        for time in (0.1, 2.0):
            a = const.contribution(m.points, quad, 1.0, False, time)[0]
            b = sched.contribution(m.points, quad, 1.0, False, time)[0]
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        ramped = const.contribution(m.points, quad, 0.5, False, 0.0)[0]
        torch.testing.assert_close(
            ramped, 0.5*const.contribution(m.points, quad, 1.0, False, 0.0)[0],
            rtol=1e-12, atol=1e-12)


class Dynamics(unittest.TestCase):
    def test_integrator_is_second_order(self):
        """Against the modal solution of the same semi-discrete system."""
        mat = HolzapfelOgdenMaterial(1e5, 2.0, 4e5, 4.0, 1e5, 4.0, 5e4, 4.0,
                                     bulk_modulus=1e6)
        m = box()
        sim = Mechanics(m, mat, bulk_modulus=None, density=1e3,
                        boundary=[RobinBC.on_surface(m, "x-", stiffness=1e5),
                                  RobinBC.on_surface(m, "x+", stiffness=1e5)])
        p = sim._build()
        zero = torch.zeros(p.n_dofs, dtype=DT, device=DEV)
        K = p.assembler.to_csr(p.evaluate(zero, tangent=True)[1]).to_dense()
        M = p.assembler.to_csr(p.mass).to_dense()
        K, M = 0.5*(K + K.T), 0.5*(M + M.T)
        L = torch.linalg.cholesky(M)
        A = torch.linalg.solve_triangular(
            L, torch.linalg.solve_triangular(L, K, upper=False).T, upper=False)
        w2, Q = torch.linalg.eigh(0.5*(A + A.T))
        omega = w2.clamp(min=0).sqrt()
        torch.manual_seed(0)
        u0 = 1e-8*torch.randn(p.n_dofs, dtype=DT, device=DEV)
        c = Q.T @ (L.T @ u0)
        exact = lambda t: torch.linalg.solve_triangular(
            L.T, (Q @ (c*torch.cos(omega*t)))[:, None], upper=True).squeeze(1)

        errors = []
        for dt in (2e-4, 1e-4):
            hist = []
            sim.solve_dynamic(6e-3, dt=dt, verbose=False, rho_infinity=1.0, u0=u0,
                              raise_on_failure=False,
                              observer=lambda t, u, v, a: hist.append((t, u.clone())))
            errors.append(max(float((u - exact(t)).abs().max()) for t, u in hist))
        self.assertGreater(errors[0]/errors[1], 3.5)          # second order

    @staticmethod
    def _endpoint_inversion_fixture():
        """A step whose stage state is valid and whose end state is inverted.

        A nearly stress-free cube given a uniform compression rate moves
        ballistically, so with ``rho_infinity=0.5`` the stage sits at two
        thirds of the step: stretch ``1 - 0.8 = 0.2`` where the end state is
        ``1 - 1.2 = -0.2``.  Generalized-alpha solves the stage, so nothing in
        the correction itself sees the inversion.
        """
        m = StructuredBoxMesh((0, 0, 0), (1, 1, 1), (1, 1, 1), order=1,
                              device=DEV, dtype=DT)
        soft = HolzapfelOgdenMaterial(a=1e-6, b=1.0, a_f=0.0, b_f=1.0, a_s=0.0,
                                      b_s=1.0, a_fs=0.0, b_fs=1.0,
                                      bulk_modulus=1e-6)
        sim = Mechanics(m, soft, bulk_modulus=None, density=1.0)
        v0 = torch.zeros(m.n_dofs, dtype=DT, device=DEV)
        v0[0::3] = -12.0*m.points[:, 0]
        return sim, v0

    def test_inverted_end_state_is_rejected(self):
        """The guard must fire on this fixture, not merely survive it."""
        sim, v0 = self._endpoint_inversion_fixture()
        before = sim._build(restrained=False).mass.clone()
        sim.solve_dynamic(0.1, dt=0.1, verbose=False, raise_on_failure=False, v0=v0)
        report = sim.report
        reasons = " ".join(report.cutbacks)
        self.assertIn("inverted end state", reasons,
                      "the endpoint guard did not reject an inverted end state")
        self.assertGreater(report.min_jacobian, 0.0)
        J = sim.problem.jacobians(sim.u)
        self.assertGreater(float(J.min()), 0.0,
                           "an accepted state must not be inverted")
        # the physical mass matrix must survive the rejected step untouched
        torch.testing.assert_close(sim.problem.mass, before, rtol=0, atol=0)

    def test_inverted_end_state_regression_detects_a_missing_guard(self):
        """Remove the guard and the fixture must be accepted -- so the test bites."""
        original = DynamicSolver._advance

        def unguarded(self, t, dt):
            self._dt = dt
            self._time = t + (1.0 - self.alpha_f)*dt
            ok, u, info = self._solve_step(self._u.clone(), 1.0)
            if ok:
                v, a = self._states(u)
                self._u, self._v, self._a = u, v, a
            return ok, info

        DynamicSolver._advance = unguarded
        try:
            sim, v0 = self._endpoint_inversion_fixture()
            sim.solve_dynamic(0.1, dt=0.1, verbose=False, raise_on_failure=False,
                              v0=v0)
            self.assertTrue(sim.report.converged)
            self.assertLess(float(sim.problem.jacobians(sim.u).min()), 0.0)
            self.assertNotIn("inverted end state", " ".join(sim.report.cutbacks))
        finally:
            DynamicSolver._advance = original

    def test_initial_data_must_agree_with_a_fixed_boundary(self):
        m = box((1, 1, 1), order=1)
        sim = Mechanics(m, HolzapfelOgdenMaterial(**SOFT, bulk_modulus=1e6),
                        bulk_modulus=None, density=1e3,
                        boundary=[DirichletBC.on_surface(m, "x-")])
        fixed = sim._build().constraints.mask
        v0 = torch.zeros(m.n_dofs, dtype=DT, device=DEV)
        v0[fixed] = 1.0
        with self.assertRaises(ValueError):
            sim.solve_dynamic(0.01, dt=1e-2, verbose=False, v0=v0)
        u0 = torch.zeros(m.n_dofs, dtype=DT, device=DEV)
        u0[fixed] = 0.5
        with self.assertRaises(ValueError):
            sim.solve_dynamic(0.01, dt=1e-2, verbose=False, u0=u0)
        sim.problem = None
        sim.solve_dynamic(0.01, dt=1e-2, verbose=False, raise_on_failure=False)
        self.assertTrue(sim.report.converged)

    def test_krylov_choice_is_reassessed_every_correction(self):
        """A tangent that loses symmetry mid-solve must change the method."""
        m = box((1, 1, 1), order=1)
        sim = Mechanics(m, ActiveIsochoric(), bulk_modulus=None, density=1e3,
                        viscosity=100.0,
                        boundary=[DirichletBC.on_surface(m, "x-")])
        seen = []
        original = DynamicSolver._ensure_solver

        def record(self, values):
            original(self, values)
            seen.append(type(self.linear_solver).__name__)

        DynamicSolver._ensure_solver = record
        try:
            sim.solve_dynamic(0.02, dt=0.02, verbose=False, raise_on_failure=False)
        finally:
            DynamicSolver._ensure_solver = original
        self.assertGreater(len(seen), 1, "the method was chosen only once")

    def test_initial_acceleration_balances_the_free_equations(self):
        """Constraints must be eliminated before the mass solve, not after."""
        m = box()
        sim = Mechanics(m, GuccioneMaterial(10, 8, 2, 4), bulk_modulus=1e3,
                        density=1e3, boundary=[DirichletBC.on_surface(m, "x-")])
        p = sim._build()
        torch.manual_seed(0)
        u0 = 0.01*torch.randn(p.n_dofs, dtype=DT, device=DEV)
        u0 = p.constraints.zero_constrained(u0)
        solver = DynamicSolver(p, dt=1e-3, verbose=False)
        solver._u, solver._v = u0, torch.zeros_like(u0)
        a = solver._initial_acceleration()
        R = p.evaluate(u0, 1.0, tangent=False, velocity=solver._v, time=0.0)[0]
        free = p.constraints.zero_constrained
        residual = free(p.assembler.to_csr(p.mass) @ a + R)
        self.assertLess(float(residual.norm()/free(R).norm()), 1e-8)

    def test_worst_jacobian_includes_the_state_it_started_from(self):
        """A body that starts compressed is at its worst before step one.

        Reporting only the accepted steps misses that, and the reported
        minimum then describes a state the body had already left.  It is
        expanding here, so every step is above the initial value and only the
        initial state can supply the minimum.
        """
        mesh = box((2, 2, 2), order=2)
        problem = FiniteStrainProblem(
            mesh, HolzapfelOgdenMaterial(**SOFT, bulk_modulus=1.0e6),
            bulk_modulus=None, density=1.0e3)
        squeezed = (-0.05*mesh.points).reshape(-1)
        outward = (50.0*mesh.points).reshape(-1)
        start = float(problem.jacobians(squeezed).amin())
        solver = DynamicSolver(problem, dt=1.0e-4, verbose=False)
        _, report = solver.solve(4.0e-4, u=squeezed, velocity=outward)
        self.assertTrue(report.converged)
        self.assertAlmostEqual(report.worst_jacobian, start, places=12)
        self.assertEqual(report.worst_jacobian_time, 0.0)
        self.assertGreater(report.min_jacobian, start)   # every step is above it

    def test_free_body_translates_at_constant_velocity(self):
        """Nothing holds it, and nothing needs to: mass and v0 define the motion."""
        m = box()
        sim = Mechanics(m, HolzapfelOgdenMaterial(**SOFT), bulk_modulus=None,
                        density=1e3)
        v0 = torch.zeros(m.n_dofs, dtype=DT, device=DEV)
        v0[0::3] = 0.5
        sim.solve_dynamic(0.01, dt=1e-3, verbose=False, v0=v0, raise_on_failure=False)
        moved = sim.displacement
        self.assertTrue(sim.report.converged)
        torch.testing.assert_close(moved[:, 0], torch.full_like(moved[:, 0], 0.005),
                                   rtol=1e-8, atol=1e-10)
        self.assertLess(float(moved[:, 1:].abs().max()), 1e-10)
        self.assertLess(abs(sim.report.residual), 1e-6)


class Continuation(unittest.TestCase):
    def test_initial_increment_respects_the_ceiling(self):
        m = box()
        p = FiniteStrainProblem(m, GuccioneMaterial(10, 8, 2, 4), bulk_modulus=1e3,
                                dirichlet=[DirichletBC.on_surface(m, "x-")],
                                pressures=[FollowerPressure.on_surface(m, "z-", 0.01)])
        solver = QuasiStaticSolver(p, load_steps=1, max_load_step=0.1, verbose=False)
        _, report = solver.solve()
        self.assertTrue(report.converged)
        self.assertGreaterEqual(report.load_steps, 10)


if __name__ == "__main__":
    unittest.main()
