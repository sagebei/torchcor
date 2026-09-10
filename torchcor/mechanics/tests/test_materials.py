"""Constitutive laws: stress against energy, tangent against finite differences.

Every law here is differentiated twice by independent routes, so a change that
breaks the tangent cannot pass by also breaking the stress.
"""
import os
import unittest

import torch
from torch.func import jacfwd, vmap

from torchcor.mechanics.material import (
    ActiveStressMaterial, GuccioneMaterial, HolzapfelOgdenMaterial,
    IsochoricMaterial, NeoHookeanMaterial,
)
from torchcor.mechanics.mesh import as_device

DEV = as_device(os.environ.get("TORCHCOR_TEST_DEVICE"))
DT = torch.float64
HO = dict(a=59.0, b=8.023, a_f=18472.0, b_f=16.026, a_s=2481.0, b_s=11.12,
          a_fs=216.0, b_fs=11.436)


def sym(a):
    return 0.5*(a + a.mT)


def strains(n=192, scale=0.12, seed=0):
    """Admissible Green-Lagrange strains: built from F, so C is positive definite."""
    torch.manual_seed(seed)
    eye = torch.eye(3, dtype=DT, device=DEV)
    F = eye + scale*torch.randn(n, 3, 3, dtype=DT, device=DEV)
    return 0.5*(F.mT @ F - eye)


class Tangents(unittest.TestCase):
    laws = {
        "guccione": GuccioneMaterial(10.0, 8.0, 2.0, 4.0),
        "isochoric guccione": IsochoricMaterial(GuccioneMaterial(10.0, 8.0, 2.0, 4.0)),
        "holzapfel-ogden": HolzapfelOgdenMaterial(**HO, bulk_modulus=1.0e6),
        "active": ActiveStressMaterial(NeoHookeanMaterial(2.0), 60.0),
    }

    def test_tangent_matches_finite_differences(self):
        E = strains()
        dE = sym(torch.randn_like(E))
        for name, law in self.laws.items():
            with self.subTest(law=name):
                S, D = law.stress_and_tangent(E)
                h = 1e-6
                fd = (law.stress(E + h*dE) - law.stress(E - h*dE))/(2*h)
                an = torch.einsum("...ijkl,...kl->...ij", D, dE)
                # Scaled by the stress, so a law whose tangent is identically
                # zero -- a prescribed active stress on a linear matrix -- is
                # still a meaningful comparison rather than 0/0.
                scale = max(float(an.abs().max()), float(S.abs().max()), 1.0)
                self.assertLess(float((fd - an).abs().max())/scale, 1e-7)

    def test_each_holzapfel_ogden_term_has_its_own_tangent(self):
        """One term at a time, so a large term cannot hide a wrong small one.

        Checking the assembled law only compares against its largest entries:
        the fibre exponential is orders of magnitude above the shear term, and
        a shear tangent wrong by a factor of two passes unnoticed. Each term is
        therefore switched on alone and scaled by its own magnitude.
        """
        off = dict(a=0.0, b=1.0, a_f=0.0, b_f=1.0, a_s=0.0, b_s=1.0,
                   a_fs=0.0, b_fs=1.0)
        terms = {
            "isotropic": dict(off, a=HO["a"], b=HO["b"]),
            "fibre": dict(off, a_f=HO["a_f"], b_f=HO["b_f"]),
            "sheet": dict(off, a_s=HO["a_s"], b_s=HO["b_s"]),
            "shear": dict(off, a_fs=HO["a_fs"], b_fs=HO["b_fs"]),
            "volumetric": dict(off, bulk_modulus=1.0e6),
        }
        # Small strains: the point is the tangent's structure, not the
        # exponentials, and a modest strain keeps every term comparable.
        E = strains(scale=0.03)
        for name, parameters in terms.items():
            law = HolzapfelOgdenMaterial(**parameters)
            with self.subTest(term=name):
                D = law.stress_and_tangent(E)[1]
                want = vmap(jacfwd(law.stress))(E)
                want = 0.5*(want + want.transpose(-2, -1))
                scale = max(float(want.abs().max()), 1e-30)
                self.assertLess(float((D - want).abs().max())/scale, 1e-12)

    def test_stress_is_the_energy_derivative(self):
        """Holzapfel-Ogden publishes an energy; the stress must be its gradient."""
        law = self.laws["holzapfel-ogden"]
        E = strains()
        want = sym(vmap(jacfwd(law.energy))(E))
        got = law.stress(E)
        self.assertLess(float((got - want).abs().max()/got.abs().max()), 1e-12)

    def test_tangent_symmetries(self):
        for name, law in self.laws.items():
            with self.subTest(law=name):
                D = law.stress_and_tangent(strains())[1]
                scale = D.abs().max().clamp(min=1e-30)
                self.assertLess(float((D - D.transpose(-2, -1)).abs().max()/scale), 1e-12)
                self.assertLess(float((D - D.permute(0, 3, 4, 1, 2)).abs().max()/scale), 1e-10)

    def test_fibre_compression_switch(self):
        """The fibre term carries tension and not compression.

        Only the fibre term is switched on, so the comparison is not swamped by
        the volumetric penalty, which is large and symmetric in stretch.
        """
        law = HolzapfelOgdenMaterial(a=0.0, b=1.0, a_f=18472.0, b_f=16.026,
                                     a_s=0.0, b_s=1.0, a_fs=0.0, b_fs=1.0)
        E = torch.zeros(2, 3, 3, dtype=DT, device=DEV)
        E[0, 0, 0], E[1, 0, 0] = 0.10, -0.10        # I4f above and below one
        S = law.stress(E)
        self.assertGreater(float(S[0, 0, 0]), 1.0e3)
        self.assertLess(abs(float(S[1, 0, 0])), 1.0e-3*float(S[0, 0, 0]))

    def test_active_stress_composition_order(self):
        """Problem 3's order: the split must not project the prescribed tension."""
        passive, Ta = GuccioneMaterial(10.0, 1.0, 1.0, 1.0), 60.0
        E = torch.zeros(1, 3, 3, dtype=DT, device=DEV)
        base = IsochoricMaterial(passive).stress(E)
        intended = ActiveStressMaterial(IsochoricMaterial(passive), Ta).stress(E) - base
        torch.testing.assert_close(
            intended, torch.diag(torch.tensor([Ta, 0.0, 0.0], dtype=DT, device=DEV))[None],
            rtol=1e-12, atol=1e-9)

    def test_schedules_follow_time_not_the_load_factor(self):
        """A callable tension is a function of time; a constant scales with load."""
        schedule = ActiveStressMaterial(NeoHookeanMaterial(2.0), lambda t: 100.0*t)
        ramped = ActiveStressMaterial(NeoHookeanMaterial(2.0), 100.0, ramp=True)
        E = torch.zeros(1, 3, 3, dtype=DT, device=DEV)
        prepared = schedule.to(DT, torch.device(DEV), batch_shape=(1, 1))
        fibre = lambda m, f, t: float(m.for_cells(slice(None)).at_load(f, t).stress(E)[0, 0, 0])
        self.assertAlmostEqual(fibre(prepared, 1.0, 0.25) - 2.0, 25.0, places=9)
        self.assertAlmostEqual(fibre(prepared, 0.5, 0.25) - 2.0, 25.0, places=9)
        prepared = ramped.to(DT, torch.device(DEV), batch_shape=(1, 1))
        self.assertAlmostEqual(fibre(prepared, 0.5, 99.0) - 2.0, 50.0, places=9)


if __name__ == "__main__":
    unittest.main()
