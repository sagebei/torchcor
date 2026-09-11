"""The stabilized active stress, as a material through the mechanics interfaces.

Regazzoni and Quarteroni, CMAME 373 (2021) 113506, eq. (19): the active tension
handed to the mechanics problem is not the contraction model's ``Ta`` alone but

    Ta + Ka (lambda - lambda_ref),

with ``Ka`` the active stiffness (their eq. 44) and ``lambda_ref`` the stretch
the contraction model was evaluated at -- the previous coupling step's value.
The extra term vanishes as the coupling interval shrinks, so the scheme stays
consistent, while making the mechanics solve aware to first order of how the
tension will answer the stretch it is about to produce.

Unlike a prescribed tension, this active stress *depends on the strain*, so it
contributes to the material tangent as well as the residual.  In the local
fibre frame the fibre is ``e_1``, so ``lambda = sqrt(C_11) = sqrt(2 E_11 + 1)``
and

    S_11 = S_11^passive + Ta + Ka (lambda - lambda_ref),
    dS_11/dE_11 = dS_11^passive/dE_11 + Ka / lambda,

which is the only entry the active part touches.  Nothing in
``torchcor.mechanics`` is changed; this is an ordinary
:class:`~torchcor.mechanics.material.HyperelasticMaterial` composed with a
passive law, in the same way as
:class:`~torchcor.mechanics.material.ActiveStressMaterial`.
"""

import torch

from torchcor.mechanics.material import HyperelasticMaterial


class StabilizedActiveStressMaterial(HyperelasticMaterial):
    """A passive law plus a stretch-dependent stabilized active fibre stress.

    ``tension``, ``stiffness`` and ``reference_stretch`` are per-cell fields,
    ``(n_cells, 1)``, written in place between coupling steps.  With
    ``stiffness`` zero this reduces exactly to a prescribed active tension.

    ``ramp`` scales the whole active contribution by the load factor during
    continuation, tension and stabilization together, so the converged
    full-load problem is unchanged.
    """

    def __init__(self, passive, tension, stiffness, reference_stretch, *,
                 ramp: bool = False):
        self.passive = passive
        self.tension = tension
        self.stiffness = stiffness
        self.reference_stretch = reference_stretch
        self.ramp = bool(ramp)
        self.scale = 1.0

    def __repr__(self):                       # pragma: no cover - cosmetic
        return (f"StabilizedActiveStressMaterial({self.passive!r}, "
                f"ramp={self.ramp})")

    def _like(self, passive, tension, stiffness, reference):
        new = StabilizedActiveStressMaterial(passive, tension, stiffness,
                                             reference, ramp=self.ramp)
        new.scale = self.scale
        return new

    def to(self, dtype, device, *, batch_shape=None):
        fields = []
        for name, value in (("tension", self.tension), ("stiffness", self.stiffness),
                            ("reference stretch", self.reference_stretch)):
            field = torch.as_tensor(value, dtype=dtype, device=device)
            if field.ndim > 2 or not bool(torch.isfinite(field).all()):
                raise ValueError(f"active {name} must be finite with at most two dimensions")
            if batch_shape is not None:
                try:
                    ok = torch.broadcast_shapes(field.shape, batch_shape) == batch_shape
                except RuntimeError:
                    ok = False
                if not ok:
                    raise ValueError(
                        f"active {name} shape {tuple(field.shape)} does not broadcast to "
                        f"the cell/quadrature shape {tuple(batch_shape)}; use (n_cells, 1)")
            fields.append(field)
        return self._like(self.passive.to(dtype, device, batch_shape=batch_shape), *fields)

    def for_cells(self, cells):
        passive = self.passive.for_cells(cells)
        sliced = []
        for value in (self.tension, self.stiffness, self.reference_stretch):
            field = torch.as_tensor(value)
            sliced.append(field[cells] if field.ndim == 2 and field.shape[0] != 1 else field)
        if passive is self.passive and all(
                a is b for a, b in zip(sliced, (self.tension, self.stiffness,
                                                self.reference_stretch))):
            return self
        return self._like(passive, *sliced)

    def at_load(self, factor, time=0.0):
        new = self._like(self.passive.at_load(factor, time), self.tension,
                         self.stiffness, self.reference_stretch)
        new.scale = float(factor) if self.ramp else 1.0
        return new

    def _stretch(self, E):
        """``lambda = sqrt(2 E_11 + 1)``, the fibre stretch in the local frame."""
        return torch.sqrt(torch.clamp(2.0 * E[..., 0, 0] + 1.0, min=1e-12))

    def _active(self, E):
        Ta = torch.as_tensor(self.tension, dtype=E.dtype, device=E.device)
        Ka = torch.as_tensor(self.stiffness, dtype=E.dtype, device=E.device)
        ref = torch.as_tensor(self.reference_stretch, dtype=E.dtype, device=E.device)
        lam = self._stretch(E)
        return self.scale * (Ta + Ka * (lam - ref)), self.scale * Ka, lam

    def stress(self, E):
        S = self.passive.stress(E).clone()
        active, _, _ = self._active(E)
        S[..., 0, 0] = S[..., 0, 0] + active
        return S

    def stress_and_tangent(self, E):
        S, D = self.passive.stress_and_tangent(E)
        active, Ka, lam = self._active(E)
        S = S.clone()
        S[..., 0, 0] = S[..., 0, 0] + active
        D = D.clone()
        # d(Ka (lambda - lambda_ref))/dE_11 = Ka dlambda/dE_11 = Ka / lambda.
        D[..., 0, 0, 0, 0] = D[..., 0, 0, 0, 0] + Ka / lam
        return S, D
