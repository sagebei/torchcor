"""Shared geometry and measurements for the Land P2 and P3 ventricle benchmarks."""

from dataclasses import dataclass, field

import numpy as np
import torch

from torchcor.mechanics import Mechanics
from torchcor.mechanics.mesh import TruncatedEllipsoidMesh
from torchcor.mechanics.solver import SolveReport


ENDO = (7.0, 17.0)                                # (rs, rl) mm
EPI = (10.0, 20.0)
BASE_Z = 5.0                                      # mm

#: Transmural positions of the three strain layers, and the sampling stations
#: along each, both from section 3 of the paper.
LAYERS = dict(endo=0.1, mid=0.5, epi=0.9)
STATIONS = 0.95*(np.arange(10) + 1)/10            # s = 0.095 .. 0.95
CIRC_ANGLE = np.pi/10                             # partner point for CIRC strain


@dataclass
class VentricleSolution:
    """One solved ventricle and the shared measurements for Land P2 and P3."""

    mesh: TruncatedEllipsoidMesh
    u: torch.Tensor
    report: SolveReport
    divisions: tuple
    order: int
    simulator: Mechanics | None = None
    volume: dict = field(default_factory=dict)

    @property
    def n_dofs(self) -> int:
        return self.mesh.n_dofs

    def at(self, t, s, v) -> torch.Tensor:
        """Deformed position of the paper's parametrically specified point."""
        return self._positions(t, s, v)[1]

    def reference_at(self, t, s, v) -> torch.Tensor:
        """Undeformed position of the same material point.

        The analytic Cartesian point is located in the FE reference mesh.
        Reference and deformed coordinates use that same cell and local point.
        """
        return self._positions(t, s, v)[0]

    def _positions(self, t, s, v):
        points = self.mesh.points_at(t, s, v).reshape(-1, 3)
        cells, xi = self.mesh.locate(points, tolerance=1e-10)
        reference = self.mesh.interpolate_local(self.mesh.points, cells, xi)
        current = self.mesh.interpolate_local(
            self.mesh.points + self.u.reshape(-1, 3), cells, xi)
        return reference, current

    def apex_z(self) -> dict:
        """Deformed z of the endocardial and epicardial apex (figures 6 and 9)."""
        return {"endo": float(self.at(0.0, 0.0, 0.0)[0, 2]),
                "epi": float(self.at(1.0, 0.0, 0.0)[0, 2])}

    def midline(self, n: int = 101) -> torch.Tensor:
        """Deformed midwall line from apex to base at ``v = 0`` (figures 7, 10 and 11)."""
        s = np.linspace(0.0, 1.0, n)
        return self.at(np.full(n, 0.5), s, np.zeros(n))

    def strains(self) -> dict:
        """Longitudinal, circumferential and radial strain (figures 8 and 12).

        Land et al. equation (3.1): the percentage change in distance between
        pairs of material points.  Longitudinal pairs neighbouring stations
        along a layer, circumferential pairs a station with its image rotated
        by ``pi/10``, and radial pairs layers across the wall -- endo-mid at the
        endocardium, mid-epi at the epicardium and endo-epi at the midwall.
        """
        def strain(a, b):
            reference = a[0] - b[0]
            current = a[1] - b[1]
            return ((current.norm(dim=-1)/reference.norm(dim=-1) - 1.0)*100.0
                    ).cpu().numpy()

        zero, ones = np.zeros_like(STATIONS), np.ones_like(STATIONS)
        line = {name: self._positions(t*ones, STATIONS, zero)
                for name, t in LAYERS.items()}
        out = {}
        for name, positions in line.items():
            out[(name, "long")] = strain(tuple(a[1:] for a in positions),
                                         tuple(a[:-1] for a in positions))
            rotated = self._positions(LAYERS[name]*ones, STATIONS, zero + CIRC_ANGLE)
            out[(name, "circ")] = strain(positions, rotated)
        for name, (inner, outer) in (("endo", ("endo", "mid")),
                                     ("epi", ("mid", "epi")),
                                     ("mid", ("endo", "epi"))):
            out[(name, "trans")] = strain(line[inner], line[outer])
        return out

