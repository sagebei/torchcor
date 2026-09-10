"""Benchmark scoring and reference handling.

The reference data is the published participant set; these tests guard the
comparison itself, so a run cannot be scored against the wrong case or over an
interval it never covered.
"""
import json
import os
import unittest

import numpy as np
import torch

from torchcor.mechanics.benchmark.arostica.b1 import reference
from torchcor.mechanics.mesh import as_device

DEV = as_device(os.environ.get("TORCHCOR_TEST_DEVICE"))


class Reference(unittest.TestCase):
    def test_red_reproduces_the_published_table(self):
        """Our RED must mean what the paper's does, so it is checked against it.

        Table 11 of Arostica et al. (2025), read for the three groups whose
        rows are unambiguous.
        """
        paper = {"carpentry": {("A", "p0"): 0.134, ("A", "p1"): 0.229,
                               ("B", "p0"): 0.360, ("B", "p1"): 0.301,
                               ("C", "p0"): 0.060, ("C", "p1"): 0.118},
                 "4C": {("A", "p0"): 0.080, ("A", "p1"): 0.115,
                        ("B", "p0"): 0.171, ("B", "p1"): 0.136,
                        ("C", "p0"): 0.059, ("C", "p1"): 0.059}}
        for case in "ABC":
            teams = reference.participants(case)
            anchor = teams[reference.TEAMS[0]]
            scored = reference.compare(
                case, anchor["time"], {p: anchor[p] for p in reference.PROBES})
            for team, expected in paper.items():
                for probe in reference.PROBES:
                    got = scored[probe]["participant_red"][team]
                    self.assertAlmostEqual(got, expected[(case, probe)], places=3)

    def test_every_case_has_the_published_population(self):
        for case in reference.CASES:
            with self.subTest(case=case):
                teams = reference.participants(case)
                self.assertEqual(sorted(teams), sorted(reference.TEAMS))
                grid = teams[reference.TEAMS[0]]["time"]
                self.assertEqual(len(grid), 101)
                for name, data in teams.items():
                    # The published grids differ by at most one 10 ms sample at
                    # the ends -- ambit starts at 0.001 s, simula stops at
                    # 0.999 -- and the reference implementation stacks them as
                    # they are, which is what makes our RED comparable to its
                    # table.  Anything larger would be a different sampling.
                    self.assertEqual(len(data["time"]), 101)
                    self.assertLess(float(np.abs(data["time"] - grid).max()), 1.01e-3)

    def test_partial_or_invalid_histories_are_not_scored(self):
        teams = reference.participants("A")
        anchor = teams[reference.TEAMS[0]]
        grid, history = anchor["time"], {p: anchor[p] for p in reference.PROBES}
        self.assertIsNotNone(reference.compare("A", grid, history))
        short = grid <= 0.25
        self.assertIsNone(reference.compare(
            "A", grid[short], {p: history[p][short] for p in reference.PROBES}))
        broken = {p: history[p].copy() for p in reference.PROBES}
        broken["p0"][5, 0] = np.nan
        self.assertIsNone(reference.compare("A", grid, broken))
        self.assertIsNone(reference.compare("A", grid, history, complete=False))


class Drivers(unittest.TestCase):
    def test_schedules_match_the_published_peaks_and_relaxation(self):
        """Equations (5)-(8) against every value the paper states for them."""
        from torchcor.mechanics.benchmark.arostica.b1 import b1
        tension, pressure = b1.active_schedule(), b1.pressure_schedule()
        self.assertAlmostEqual(tension.peak, 118817.07, delta=0.01*118817.07)
        self.assertAlmostEqual(pressure.peak, 16117.52, delta=0.01*16117.52)
        # both must relax essentially to zero by the end of the beat
        self.assertLess(pressure(1.0), 1.0)
        self.assertLess(tension(1.0), 1.0)
        self.assertAlmostEqual(pressure(0.0), 0.0, places=9)

    def test_step2_dispatch_scores_its_own_reference(self):
        """Exercise the runner, not the table: the defect was in dispatch."""
        from unittest import mock
        from torchcor.mechanics.benchmark.arostica.b1 import b1

        for case in ("A", "B", "C"):
            with self.subTest(case=case):
                seen = {}

                def fake_solve(case=None, step2=None, **kw):
                    seen["physics"] = (case, step2)
                    sim = mock.MagicMock()
                    sim.report.converged = True
                    sim.mesh.label = "reference"
                    sim.mesh.n_dofs = 1
                    grid = reference.participants("A")[reference.TEAMS[0]]["time"]
                    return sim, {"time": grid,
                                 **{p: np.zeros((len(grid), 3)) for p in reference.PROBES}}

                def fake_compare(case_id, *a, **k):
                    seen["scored_against"] = case_id
                    return None

                with mock.patch.object(b1, "solve_b1", fake_solve), \
                        mock.patch.object(b1.reference, "compare", fake_compare), \
                        mock.patch.object(b1, "write_outputs", lambda *a, **k: []), \
                        mock.patch.object(b1, "report", lambda *a, **k: None):
                    b1.main(["--case", case])
                self.assertEqual(seen["physics"], ("both", case))
                self.assertEqual(seen["scored_against"], case)

    def test_the_exit_code_reflects_agreement_and_not_just_completion(self):
        """A case can converge, be scored, and still disagree. That is a failure.

        The exit code once answered only "was every case scored?", so a run
        whose own table printed FAIL still reported success to the caller.
        """
        from unittest import mock
        from torchcor.mechanics.benchmark.arostica.b1 import b1

        grid = reference.participants("passive")[reference.TEAMS[0]]["time"]

        def scored(red):
            """A scored result whose RED sits inside or outside the range."""
            band = {"someone": 0.1, "someone else": 0.2}
            return {"case": "passive", "n_participants": 2, "time": grid,
                    "participants": sorted(band),
                    **{p: {"red": red, "participant_red": band,
                           "mean": np.zeros((len(grid), 3)),
                           "std": np.zeros((len(grid), 3)),
                           "ours": np.zeros((len(grid), 3))} for p in reference.PROBES}}

        def fake_solve(**kw):
            sim = mock.MagicMock()
            sim.report.converged = True
            sim.mesh.label, sim.mesh.n_dofs = "reference", 1
            return sim, {"time": grid,
                         **{p: np.zeros((len(grid), 3)) for p in reference.PROBES}}

        for red, agrees, expected in ((0.15, True, 0), (0.9, False, 1)):
            with self.subTest(red=red):
                with mock.patch.object(b1, "solve_b1", fake_solve), \
                        mock.patch.object(b1.reference, "compare",
                                          lambda *a, **k: scored(red)), \
                        mock.patch.object(b1, "write_outputs", lambda *a, **k: []), \
                        mock.patch.object(b1, "write_csv", lambda *a, **k: "csv"), \
                        mock.patch.object(b1, "report", lambda *a, **k: None):
                    code = b1.main(["--case", "passive"])
                self.assertEqual(b1.agrees(scored(red)), agrees)
                self.assertEqual(code, expected)

    def test_step2_rejects_a_split_case(self):
        """Solving a split and scoring it as the combined case is different physics."""
        from torchcor.mechanics.benchmark.arostica.b1 import b1
        for case in ("active", "passive"):
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    b1.solve_b1(case=case, step2="A", verbose=False)


class Validation(unittest.TestCase):
    """The scorer must refuse a history that would make its answer meaningless."""

    def history(self):
        grid = reference.participants("passive")[reference.TEAMS[0]]["time"]
        return grid, {p: 1e-3*np.column_stack([np.sin(grid), np.cos(grid), grid])
                      for p in reference.PROBES}

    def test_the_scorer_refuses_non_finite_times(self):
        """``[-inf, ..., +inf]`` is increasing, and would span any interval."""
        grid, displacement = self.history()
        for name, bad in (("infinite endpoints", np.r_[-np.inf, grid[1:-1], np.inf]),
                          ("a NaN sample", np.r_[grid[:-1], np.nan])):
            with self.subTest(times=name):
                with self.assertRaises(ValueError):
                    reference.compare("passive", bad, displacement)
        self.assertIsNotNone(reference.compare("passive", grid, displacement))

    def test_a_short_or_failed_run_is_not_scored(self):
        """Interpolation would hold the last value and score that as agreement."""
        grid, displacement = self.history()
        half = len(grid)//2
        self.assertIsNone(reference.compare(
            "passive", grid[:half], {p: u[:half] for p, u in displacement.items()}))
        self.assertIsNone(reference.compare("passive", grid, displacement,
                                            complete=False))


class Benchmark2(unittest.TestCase):
    """Benchmark 2's loading is its own, not benchmark 1's."""

    def test_activation_follows_table_8(self):
        """The biventricular timings differ from benchmark 1's.

        Table 8 gives t_sys = 0.163 s and t_dias = 0.5 s, against 0.16 and
        0.484 in benchmark 1. The reference implementation's shared defaults
        are benchmark 1's, so copying them silently halves the tension through
        the second half of the beat while leaving the peak nearly unchanged --
        which the comparison scores do not catch.
        """
        from torchcor.mechanics.benchmark.arostica.b2 import b2
        self.assertEqual(b2.ACTIVE["t_sys"], 0.163)
        self.assertEqual(b2.ACTIVE["t_dias"], 0.5)
        tau = b2.active_schedule()
        # Fig. 6's peak, to the accuracy the tabulated ODE reproduces it.
        self.assertAlmostEqual(tau.peak/b2.ACTIVE_PEAK, 1.0, delta=0.01)
        # The discriminating sample: benchmark 1's timing gives 75.7 kPa here.
        self.assertAlmostEqual(tau(0.5)/1e3, 117.2, delta=0.5)

    def test_pressures_follow_table_9(self):
        """Both cavities, against the maxima the table states."""
        from torchcor.mechanics.benchmark.arostica.b2 import b2
        for name, parameters, peak in (("lv", b2.LV_PRESSURE, 16491.14),
                                       ("rv", b2.RV_PRESSURE, 4166.66)):
            with self.subTest(cavity=name):
                self.assertEqual(parameters["t_sys"], 0.17)
                self.assertEqual(parameters["t_dias"], 0.484)
                self.assertAlmostEqual(
                    b2.pressure_schedule(parameters).peak/peak, 1.0, delta=0.01)

    def test_the_table_keeps_every_refinement_level(self):
        """A second level must add rows, not replace the first level's.

        The per-run result files carry the resolution in their names, but the
        combined table does not -- it is one file with a ``mesh`` column -- so
        writing it from the run that just finished silently discards whichever
        level ran first.
        """
        import csv
        import json
        import tempfile
        from collections import Counter
        from pathlib import Path
        from torchcor.mechanics.benchmark.arostica.b2 import b2

        grid = [0.0, 0.5, 1.0]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for level, dofs in (("coarse", 227364), ("fine", 574728)):
                (out/f"b2_{level}_dt0.002.json").write_text(json.dumps({
                    "mesh": level, "dofs": dofs, "dt_s": 2e-3, "time_s": grid,
                    "displacement_m": {p: [[0.0, 0.0, 0.0]]*len(grid)
                                       for p in b2.PROBES}}))
            rows = list(csv.reader(b2.write_csv(out).open()))
        counts = Counter(row[0] for row in rows[1:])
        self.assertEqual(counts, {"coarse": len(grid), "fine": len(grid)})

    def test_each_refinement_level_is_scored_against_its_own_population(self):
        """Resolution must select mesh, filename and participants together."""
        from torchcor.mechanics.benchmark.arostica.b2 import b2, geometry
        from torchcor.mechanics.benchmark.arostica.b2 import reference as b2ref
        self.assertEqual(tuple(b2ref.CASES), geometry.RESOLUTIONS)
        for level in geometry.RESOLUTIONS:
            with self.subTest(resolution=level):
                self.assertEqual(len(b2ref.participants(level)), len(b2ref.TEAMS))
        with self.assertRaises(ValueError):
            geometry.BiventricleMesh.load("medium", device=DEV)


class PublishedMesh(unittest.TestCase):
    """The tetrahedral mesh the participants were given, as loaded."""

    #: Facet counts of ``ellipsoid_0.005.xdmf``, by its generator's markers.
    SURFACES = {"endo": 2500, "epi": 3474, "base": 256}

    @classmethod
    def setUpClass(cls):
        from torchcor.mechanics.benchmark.arostica.b1.geometry import ReferenceMesh
        cls.mesh = {order: ReferenceMesh.load(order=order, device=DEV)
                    for order in (1, 2)}

    def surface(self, mesh, name, field):
        """``integral(field(x) . n dA)`` over a named surface of ``mesh``."""
        faces = mesh.face_set(name)
        X = mesh.points[faces.connectivity]
        quad = mesh.face_element(faces.order, 4, mesh.dtype, mesh.device)
        normal = torch.linalg.cross(
            torch.einsum("fai,qa->fqi", X, quad.dN[..., 0]),
            torch.einsum("fai,qa->fqi", X, quad.dN[..., 1]), dim=-1)
        x = torch.einsum("fai,qa->fqi", X, quad.N)
        return float((quad.weights*(field(x)*normal).sum(-1)).sum())

    def test_the_stored_surfaces_are_read_and_oriented_outwards(self):
        for order, mesh in self.mesh.items():
            for name, count in self.SURFACES.items():
                with self.subTest(order=order, surface=name):
                    self.assertEqual(mesh.face_set(name).n_faces, count)
            # Divergence theorem over the closed boundary: the flux of x is
            # three times the enclosed volume, and each surface contributes
            # with the sign its own outward normal gives.
            flux = sum(self.surface(mesh, name, lambda x: x) for name in self.SURFACES)
            with self.subTest(order=order):
                self.assertAlmostEqual(flux/3.0, 1.7769e-4, delta=1e-8)
                # The cavity is inside the body, so its normal points inwards.
                self.assertLess(self.surface(mesh, "endo", lambda x: x), 0.0)
                self.assertGreater(self.surface(mesh, "epi", lambda x: x), 0.0)

    def test_quadratic_displacement_does_not_move_the_boundary(self):
        """The published geometry is linear; promoting it must not curve it."""
        area = lambda m, n: self.surface(m, n, lambda x: x/torch.linalg.vector_norm(
            x, dim=-1, keepdim=True).clamp(min=1e-30))
        for name in self.SURFACES:
            with self.subTest(surface=name):
                self.assertAlmostEqual(area(self.mesh[1], name)/area(self.mesh[2], name),
                                       1.0, places=12)

    def test_the_fibre_frame_is_orthonormal_everywhere(self):
        for order, mesh in self.mesh.items():
            f, s = mesh.fibre_frame(mesh.points)
            with self.subTest(order=order):
                self.assertLess(float((f.norm(dim=-1) - 1).abs().max()), 1e-12)
                self.assertLess(float((s.norm(dim=-1) - 1).abs().max()), 1e-12)
                self.assertLess(float((f*s).sum(-1).abs().max()), 1e-12)

    def test_each_run_gets_its_own_result_file(self):
        """Two runs that differ in anything must not overwrite each other."""
        from torchcor.mechanics.benchmark.arostica.b1 import b1, geometry
        coarse = geometry.REFERENCE_MESH.with_name("ellipsoid_0.01.npz")
        meshes = [m for m in (geometry.REFERENCE_MESH, coarse) if m.exists()]
        settings = [(mesh, order, case, dt)
                    for mesh in meshes for order in (1, 2)
                    for case in ("passive", "both") for dt in (1e-3, 2e-3)]
        stems = set()
        for mesh, order, case, dt in settings:
            loaded = geometry.ReferenceMesh.load(mesh, order=order, device=DEV)
            stems.add(b1.result_stem(case, loaded.label, dt))
        self.assertEqual(len(stems), len(settings))
        self.assertEqual(b1.result_stem("passive", "reference", 2e-3),
                         "b1_passive_reference_dt0.002")

    def test_a_faceted_structured_mesh_still_locates_its_own_points(self):
        """A wrong candidate cell must be searched for, not tolerated.

        With linear geometry the mesh no longer passes through the analytic
        surface, so the parametric inverse can name a neighbouring cell. Those
        points must be found in the neighbours; widening the tolerance until
        the miss disappears would leave the probe reading the wrong element.
        """
        from torchcor.mechanics.benchmark.arostica.b1.geometry import MonoventricleMesh
        for geometry_order in (1, 2):
            # This mesh and sampling rate are the ones that actually expose it:
            # coarser grids happen to pick the right cell every time.
            mesh = MonoventricleMesh((2, 12, 24), order=2, device=DEV,
                                     geometry_order=geometry_order)
            elem = mesh.cell_element(mesh.order, 4, mesh.dtype, mesh.device)
            points = torch.einsum("qa,eai->eqi", elem.N,
                                  mesh.points[mesh.cells]).reshape(-1, 3)
            cells, xi = mesh.locate(points, tolerance=1e-6)
            residual = (mesh.interpolate_local(mesh.points, cells, xi)
                        - points).norm(dim=1)
            with self.subTest(geometry_order=geometry_order):
                self.assertLess(float(residual.max()), 1e-6)

    def test_a_surface_triangle_that_is_not_a_cell_face_is_rejected(self):
        mesh = self.mesh[1]
        stray = mesh.face_set("base").connectivity.clone()
        stray[0, 0] = mesh.cells[0, 0]
        with self.assertRaises(ValueError):
            mesh._outward(stray)


if __name__ == "__main__":
    unittest.main()
