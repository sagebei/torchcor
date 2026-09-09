"""Constitutive behaviour for :mod:`torchcor.mechanics`.

Material laws map a strain measure to stress and the material tangent, and
:class:`MaterialAxes` supplies the local fibre frame they are written in.  Both
are batched over every quadrature point of every element at once; no law here
ever loops in Python.

A law is expressed in its *own* local frame, so adding a transversely isotropic
or orthotropic law -- or, later, an active-contraction term that adds to the
fibre-fibre stress component -- does not touch the assembly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch


__all__ = ["HyperelasticMaterial", "GuccioneMaterial", "NeoHookeanMaterial",
           "ActiveStressMaterial", "IsochoricMaterial", "MaterialAxes",
           "OrientedMaterial", "AugmentedLagrangian"]


class HyperelasticMaterial(ABC):
    """Strain-energy law expressed in the local material (fibre) frame.

    Implementations map the Green-Lagrange strain to the second Piola-Kirchhoff
    stress and the material tangent, both with *minor symmetry* so that the
    assembly can contract with the unsymmetrised discrete gradient operator.
    """

    @abstractmethod
    def stress_and_tangent(self, E: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``E`` ``(..., 3, 3)`` -> ``(S (..., 3, 3), D (..., 3, 3, 3, 3))``."""

    def stress(self, E: torch.Tensor) -> torch.Tensor:
        """Stress only.

        Override where the tangent is expensive: residual-only evaluations
        (line search, load-step probing) never need ``D``, whose ``3**4`` per
        quadrature point dominates the memory traffic of an assembly.
        """
        return self.stress_and_tangent(E)[0]

    def to(self, dtype: torch.dtype, device: torch.device, *, batch_shape=None):
        """Prepare tensor fields once for an optional (cell, quadrature) batch.

        Uniform laws have no fields to move and can return themselves.
        """
        return self

    def for_cells(self, cells: slice):
        """Return this law restricted to a cell batch; uniform laws are shared."""
        return self

    def at_load(self, factor: float):
        """Return this law at a load factor; passive laws are unchanged."""
        return self


class GuccioneMaterial(HyperelasticMaterial):
    r"""Transversely isotropic Fung-type law of Guccione et al.

    .. math::

        W = \frac{C}{2}\left(e^{Q} - 1\right), \qquad
        Q = \sum_{ij} B_{ij} E_{ij}^2

    with, in the fibre frame :math:`(f, s, n)`,

    .. math::

        B = \begin{pmatrix} b_f & b_{fs} & b_{fs} \\
                            b_{fs} & b_t & b_t \\
                            b_{fs} & b_t & b_t \end{pmatrix}.

    This is the constitutive law used by all three problems of the Land et al.
    (2015) cardiac mechanics benchmark.

    Parameters
    ----------
    C, bf, bt, bfs:
        Material constants; ``C`` carries the stress unit (kPa in the benchmark).
    """

    def __init__(self, C: float, bf: float, bt: float, bfs: float) -> None:
        self.C = float(C)
        self.bf = float(bf)
        self.bt = float(bt)
        self.bfs = float(bfs)
        self._B: Optional[torch.Tensor] = None
        self._B_parameters: Optional[Tuple[float, float, float]] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"GuccioneMaterial(C={self.C}, bf={self.bf}, bt={self.bt}, "
                f"bfs={self.bfs})")

    def coefficients(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        parameters = (self.bf, self.bt, self.bfs)
        if (self._B is None or self._B.dtype != dtype or self._B.device != device
                or self._B_parameters != parameters):
            b_f, b_t, b_fs = parameters
            self._B = torch.tensor(
                [[b_f, b_fs, b_fs],
                 [b_fs, b_t, b_t],
                 [b_fs, b_t, b_t]], dtype=dtype, device=device)
            self._B_parameters = parameters
        return self._B

    def stress(self, E: torch.Tensor) -> torch.Tensor:
        B = self.coefficients(E.dtype, E.device)
        BE = B * E
        expQ = torch.exp((BE * E).sum(dim=(-2, -1)))
        return self.C * expQ[..., None, None] * BE

    def stress_and_tangent(self, E: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.coefficients(E.dtype, E.device)

        BE = B * E                                    # (..., 3, 3)
        Q = (BE * E).sum(dim=(-2, -1))                # (...,)
        expQ = torch.exp(Q)

        S = self.C * expQ[..., None, None] * BE

        # D_ijkl = C e^Q [ 2 (B_ij E_ij)(B_kl E_kl)
        #                  + 1/2 B_ij (d_ik d_jl + d_il d_jk) ]
        outer = 2.0 * BE[..., :, :, None, None] * BE[..., None, None, :, :]

        eye = torch.eye(3, dtype=E.dtype, device=E.device)
        iso = 0.5 * B[:, :, None, None] * (
            eye[:, None, :, None] * eye[None, :, None, :]
            + eye[:, None, None, :] * eye[None, :, :, None]
        )
        D = self.C * expQ[..., None, None, None, None] * (outer + iso)
        return S, D


class NeoHookeanMaterial(HyperelasticMaterial):
    r"""Isotropic neo-Hookean law, :math:`W = \tfrac{\mu}{2}(\mathrm{tr}\,C - 3)`.

    Useful as a verification reference and as an isotropic fallback; its
    deviatoric response is linear in ``E`` so the tangent is constant.
    """

    def __init__(self, mu: float) -> None:
        self.mu = float(mu)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"NeoHookeanMaterial(mu={self.mu})"

    def stress(self, E: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, dtype=E.dtype, device=E.device)
        return self.mu * eye.expand_as(E).clone()

    def stress_and_tangent(self, E: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = torch.zeros(E.shape + (3, 3), dtype=E.dtype, device=E.device)
        return self.stress(E), D


class ActiveStressMaterial(HyperelasticMaterial):
    r"""A passive law plus an active tension along the fibre direction.

    In the local material frame the active second Piola-Kirchhoff stress is
    :math:`T_a\, f \otimes f`, so it simply adds ``tension`` to the
    fibre-fibre component.  For a *prescribed* (strain-independent) tension the
    material tangent is unchanged; the stiffness the active stress does
    contribute is geometric, and the assembly already forms that from the total
    stress.

    This is the active response specified by problem 3 of the Land et al.
    benchmark, and is the extension point for contraction models in general:
    anything that produces a fibre tension can be composed with any passive law
    without either of them knowing about the other.

    Parameters
    ----------
    passive:
        The underlying law, e.g. :class:`GuccioneMaterial`.
    tension:
        Active fibre tension, in the stress units of the passive law.  A scalar
        applies everywhere. Use ``(n_cells, 1)`` for one value per cell or
        ``(n_cells, n_quadrature)`` for a quadrature field. A ``(n_quadrature,)``
        or ``(1, n_quadrature)`` field is shared by every cell. Fields are moved
        to the solver's device once and sliced with each assembly batch.
    ramp:
        If true, ``tension`` specifies the full-load value and follows the
        current load factor. Otherwise the prescribed tension stays constant.

    Notes
    -----
    Verified for tangent consistency, but not yet checked against the published
    problem 3 solution. Load scaling is opt-in and leaves the original
    full-load tension unchanged, including after rejected load increments.
    """

    def __init__(self, passive: HyperelasticMaterial, tension, *, ramp: bool = False) -> None:
        self.passive = passive
        self.tension = tension
        self.ramp = bool(ramp)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ActiveStressMaterial({self.passive!r}, tension={self.tension}, "
                f"ramp={self.ramp})")

    def to(self, dtype: torch.dtype, device: torch.device, *, batch_shape=None):
        tension = torch.as_tensor(self.tension, dtype=dtype, device=device)
        if tension.ndim > 2 or not bool(torch.isfinite(tension).all()):
            raise ValueError("active tension must be finite and have at most two dimensions")
        if batch_shape is not None:
            try:
                compatible = torch.broadcast_shapes(tension.shape, batch_shape) == batch_shape
            except RuntimeError:
                compatible = False
            if not compatible:
                raise ValueError(
                    f"active tension shape {tuple(tension.shape)} does not broadcast "
                    f"to the cell/quadrature shape {tuple(batch_shape)}; use "
                    "(n_cells, 1) for per-cell tension")
        passive = self.passive.to(dtype, device, batch_shape=batch_shape)
        return ActiveStressMaterial(passive, tension, ramp=self.ramp)

    def for_cells(self, cells: slice):
        passive = self.passive.for_cells(cells)
        tension = torch.as_tensor(self.tension)
        if tension.ndim == 2 and tension.shape[0] != 1:
            tension = tension[cells]
        elif passive is self.passive:
            return self
        return ActiveStressMaterial(passive, tension, ramp=self.ramp)

    def at_load(self, factor: float):
        passive = self.passive.at_load(factor)
        if self.ramp:
            return ActiveStressMaterial(passive, torch.as_tensor(self.tension)*factor)
        return self if passive is self.passive else ActiveStressMaterial(passive, self.tension)

    def _add_active(self, S: torch.Tensor) -> torch.Tensor:
        Ta = torch.as_tensor(self.tension, dtype=S.dtype, device=S.device)
        S = S.clone()
        S[..., 0, 0] = S[..., 0, 0] + Ta
        return S

    def stress(self, E: torch.Tensor) -> torch.Tensor:
        return self._add_active(self.passive.stress(E))

    def stress_and_tangent(self, E: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        S, D = self.passive.stress_and_tangent(E)
        return self._add_active(S), D


class IsochoricMaterial(HyperelasticMaterial):
    r"""Applies a law to the volume-preserving part of the deformation only.

    Splits the deformation gradient as :math:`F = J^{1/3} \bar{F}` with
    :math:`\det \bar{F} = 1`, and evaluates the passive law at
    :math:`\bar{E} = \tfrac12(J^{-2/3} C - I)`.  The resulting stress

    .. math::

        S = J^{-2/3}\left(\bar{S}
            - \tfrac13 (\bar{S} : \bar{C})\, \bar{C}^{-1}\right)

    satisfies :math:`S : C = 0` exactly, so the passive response exerts *no*
    volumetric driving force.  Without the split the passive law carries a
    volumetric part of its own, which pushes ``J`` away from 1 wherever the
    incompressibility constraint is not directly imposed.

    The tangent follows the analytic chain rule using the wrapped material's
    stress and tangent.  All leading batch dimensions are preserved, and no
    automatic differentiation is needed during a solve.

    Method and references
    ---------------------
    This is the uncoupled energy split used by FEBio and deal.II step-44,
    expressed here as second Piola-Kirchhoff stress and ``D = dS/dE``.  It is
    independent of the element's treatment of the pressure constraint.

    * https://help.febio.org/doxygen/febio4.8/material.html
    * https://www.dealii.org/current/doxygen/deal.II/step_44.html

    Contract
    --------
    The wrapped law follows :class:`HyperelasticMaterial` and is evaluated at
    ``Cbar``.  Apply prescribed active stress *outside* this passive split:
    ``ActiveStressMaterial(IsochoricMaterial(passive), tension)``.  Reversing
    those wrappers projects the active stress too, which defines a different
    constitutive law.
    """

    def __init__(self, passive: HyperelasticMaterial) -> None:
        self.passive = passive

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"IsochoricMaterial({self.passive!r})"

    def to(self, dtype: torch.dtype, device: torch.device, *, batch_shape=None):
        passive = self.passive.to(dtype, device, batch_shape=batch_shape)
        return self if passive is self.passive else IsochoricMaterial(passive)

    def for_cells(self, cells: slice):
        passive = self.passive.for_cells(cells)
        return self if passive is self.passive else IsochoricMaterial(passive)

    def at_load(self, factor: float):
        passive = self.passive.at_load(factor)
        return self if passive is self.passive else IsochoricMaterial(passive)

    @staticmethod
    def _determinant(A: torch.Tensor) -> torch.Tensor:
        """Batched 3x3 determinant using elementwise tensor operations."""
        return (A[..., 0, 0]*(A[..., 1, 1]*A[..., 2, 2] - A[..., 1, 2]*A[..., 2, 1])
                - A[..., 0, 1]*(A[..., 1, 0]*A[..., 2, 2] - A[..., 1, 2]*A[..., 2, 0])
                + A[..., 0, 2]*(A[..., 1, 0]*A[..., 2, 1] - A[..., 1, 1]*A[..., 2, 0]))

    def _kinematics(self, E: torch.Tensor):
        """Volume split without synchronizing CUDA on a singular trial state."""
        eye = torch.eye(3, dtype=E.dtype, device=E.device)
        C = 2.0*E + eye
        scale = self._determinant(C).pow(-1.0/3.0)
        Cbar = scale[..., None, None]*C
        # inv_ex leaves non-finite values for the solver to reject rather than
        # raising on a collapsed line-search trial.
        Cinv = torch.linalg.inv_ex(C)[0]
        return C, Cinv, scale, Cbar, 0.5*(Cbar - eye)

    @staticmethod
    def _project_stress(Sbar, Cbar, Cinv, scale):
        trace = (Sbar*Cbar).sum(dim=(-2, -1))
        S = scale[..., None, None]*Sbar - (trace/3.0)[..., None, None]*Cinv
        return S, trace

    def stress(self, E: torch.Tensor) -> torch.Tensor:
        _, Cinv, scale, Cbar, Ebar = self._kinematics(E)
        return self._project_stress(self.passive.stress(Ebar), Cbar, Cinv, scale)[0]

    def stress_and_tangent(self, E: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        C, Cinv, scale, Cbar, Ebar = self._kinematics(E)
        Sbar, Dbar = self.passive.stress_and_tangent(Ebar)
        S, trace = self._project_stress(Sbar, Cbar, Cinv, scale)

        # P : (a^2 Dbar) : P^T, with a = J^(-2/3) and
        # P = I_sym - (Cinv outer C)/3.  Expanding the two projections avoids
        # allocating and multiplying separate 9x9 projector matrices.
        D = scale.square()[..., None, None, None, None]*Dbar
        DC = torch.einsum("...ijkl,...kl->...ij", D, C)
        CD = torch.einsum("...ij,...ijkl->...kl", C, D)
        CDC = (C*DC).sum(dim=(-2, -1))
        inverse_outer = torch.einsum("...ij,...kl->...ijkl", Cinv, Cinv)
        D = (D - (torch.einsum("...ij,...kl->...ijkl", DC, Cinv)
                  + torch.einsum("...ij,...kl->...ijkl", Cinv, CD))/3.0
             + (CDC/9.0)[..., None, None, None, None]*inverse_outer)

        # Derivatives of the volume scale and stress projector.  The symmetric
        # inverse product includes both minor pairs because D contracts with
        # the assembly's unsymmetrised displacement-gradient operator.
        inverse_symmetric = 0.5*(
            torch.einsum("...ik,...jl->...ijkl", Cinv, Cinv)
            + torch.einsum("...il,...jk->...ijkl", Cinv, Cinv))
        D = (D - (2.0/3.0)*(torch.einsum("...ij,...kl->...ijkl", S, Cinv)
                            + torch.einsum("...ij,...kl->...ijkl", Cinv, S))
             + (2.0*trace/3.0)[..., None, None, None, None]
             * (inverse_symmetric - inverse_outer/3.0))
        return S, D


@dataclass
class MaterialAxes:
    """Local orthonormal material frame ``(f, s, n)`` -- fibre, sheet, normal.

    Each direction may be uniform ``(3,)``, per-cell ``(n_cells, 3)``, or
    per-quadrature ``(n_cells, n_quadrature, 3)``. A callable receives physical
    reference quadrature coordinates and returns directions once during setup.
    ``s`` and ``n`` are completed automatically if not supplied.
    The frame is used to rotate strain into, and stress/tangent out
    of, the material coordinate system in which
    :class:`GuccioneMaterial` is defined.
    """

    f: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]
    s: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None
    n: Optional[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]] = None

    #: Tolerance for the orthonormality checks on a supplied frame.
    atol: float = 1e-8

    def rotation(self, dtype: torch.dtype, device: torch.device, *,
                 reference_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return ``R`` with columns ``(f, s, n)``, preserving batch dimensions.

        The result is always a genuine orthonormal frame.  ``f`` is normalised,
        ``s`` is orthogonalised against it, and the normal is *constructed* as
        ``f x s``.  A supplied ``s`` or ``n`` is validated rather than trusted:
        directions that are zero, non-finite or parallel raise ``ValueError``
        instead of silently producing a rank-deficient "rotation".
        """
        directions = []
        for value, name in zip((self.f, self.s, self.n), ("fibre f", "sheet s", "normal n")):
            if value is None:
                directions.append(None)
                continue
            if callable(value):
                if reference_points is None:
                    raise ValueError("callable material directions need reference_points")
                value = value(reference_points)
            value = torch.as_tensor(value, dtype=dtype, device=device)
            if value.ndim not in (1, 2, 3) or value.shape[-1] != 3 or value.numel() == 0:
                raise ValueError(f"{name} must have shape (3,), (n_cells, 3), "
                                 "or (n_cells, n_quadrature, 3)")
            directions.append(value)
        if any(v is not None and v.ndim == 3 for v in directions):
            directions = [v[:, None, :] if v is not None and v.ndim == 2 else v
                          for v in directions]
        try:
            shape = torch.broadcast_shapes(*(v.shape for v in directions if v is not None))
        except RuntimeError as error:
            raise ValueError("material direction fields have incompatible batch shapes") from error
        f, trial, given = [v.expand(shape) if v is not None else None for v in directions]
        f = self._unit(f, "fibre direction f")

        if trial is None:
            # Any axis not parallel to f; the one f is least aligned with.
            eye = torch.eye(3, dtype=dtype, device=device)
            trial = eye[f.abs().argmin(dim=-1)]
        else:
            self._unit(trial, "sheet direction s")
        s = self._unit(trial - (trial * f).sum(-1, keepdim=True) * f,
                       "sheet direction s (after removing its fibre component; "
                       "is it parallel to f?)")

        n = torch.linalg.cross(f, s, dim=-1)
        if given is not None:
            given = self._unit(given, "normal direction n")
            # A supplied normal may only fix the orientation, not tilt the frame.
            alignment = (given * n).sum(-1)
            if float(alignment.abs().min()) < 1.0 - self.atol:
                raise ValueError(
                    "normal direction n is not orthogonal to both f and s "
                    f"(|n . (f x s)| = {float(alignment.abs().min()):.6f}, expected 1); "
                    "supply an orthogonal frame or leave n unset to have it built")
            n = torch.sign(alignment)[..., None] * n

        return torch.stack([f, s, n], dim=-1)        # columns

    @classmethod
    def _unit(cls, v: torch.Tensor, what: str) -> torch.Tensor:
        """Normalise ``v``, rejecting zero-length or non-finite directions."""
        norm = v.norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(v).all()):
            raise ValueError(f"{what} contains non-finite entries")
        if float(norm.min()) <= cls.atol:
            raise ValueError(f"{what} has zero (or near-zero) length")
        return v / norm

    @staticmethod
    def voigt9_rotation(R: torch.Tensor) -> torch.Tensor:
        """9x9 operator ``Q`` with ``vec(R^T A R) = Q vec(A)`` (row-major vec)."""
        Q = torch.einsum("...ki,...lj->...ijkl", R, R)
        return Q.reshape(R.shape[:-2] + (9, 9))


class OrientedMaterial:
    """A material written in a local frame, evaluated in the global one.

    Which frame a constitutive law is written in is a property of the law, not
    of the assembler, so the rotation lives here: strain is rotated in, stress
    and tangent are rotated back out.  An identity frame is detected once and
    skipped entirely, so an isotropic or axis-aligned problem pays nothing.

    The tangent is returned in 9x9 form, which is what the assembly contracts
    against.
    """

    def __init__(self, material: HyperelasticMaterial,
                 axes: Optional[MaterialAxes],
                 dtype: torch.dtype, device: torch.device, *, batch_shape=None,
                 reference_points: Optional[torch.Tensor] = None) -> None:
        self.material = material.to(dtype, device, batch_shape=batch_shape)
        self.axes = axes
        self.rotation: Optional[torch.Tensor] = None
        self.per_cell = False

        if axes is None:
            return
        R = axes.rotation(dtype, device, reference_points=reference_points)
        if R.dim() == 3:
            R = R[:, None]                         # per-cell, shared quadrature frame
        if batch_shape is not None:
            try:
                compatible = torch.broadcast_shapes(R.shape[:-2], batch_shape) == batch_shape
            except RuntimeError:
                compatible = False
            if not compatible:
                raise ValueError(f"material axes batch shape {tuple(R.shape[:-2])} does not "
                                 f"broadcast to cell/quadrature shape {tuple(batch_shape)}")
        eye = torch.eye(3, dtype=dtype, device=device)
        if R.dim() == 2 and torch.allclose(R, eye, atol=1e-14):
            return                                  # identity frame: no rotation
        self.rotation = MaterialAxes.voigt9_rotation(R)
        self.per_cell = self.rotation.dim() > 2

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        frame = "global axes" if self.rotation is None else (
            "spatial frame" if self.per_cell else "uniform frame")
        return f"OrientedMaterial({self.material!r}, {frame})"

    def response(self, E: torch.Tensor, cells: slice = slice(None),
                 tangent: bool = True, load_factor: float = 1.0
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Stress and (optionally) the 9x9 tangent, both in the global frame."""
        material = self.material.for_cells(cells).at_load(load_factor)
        if self.rotation is None:
            if not tangent:
                return material.stress(E), None
            S, D = material.stress_and_tangent(E)
            return S, D.reshape(D.shape[:-4] + (9, 9))

        Q = (self.rotation[cells] if self.per_cell and self.rotation.shape[0] != 1
             else self.rotation)

        E_local = torch.einsum("...mn,...n->...m", Q, E.flatten(-2)).reshape(E.shape)
        if not tangent:
            S_local = material.stress(E_local)
            S = torch.einsum("...ma,...m->...a", Q, S_local.flatten(-2))
            return S.reshape(E.shape), None

        S_local, D_local = material.stress_and_tangent(E_local)
        S = torch.einsum("...ma,...m->...a", Q, S_local.flatten(-2))
        D = torch.einsum("...ma,...mn,...nb->...ab", Q,
                         D_local.reshape(D_local.shape[:-4] + (9, 9)), Q)
        return S.reshape(E.shape), D


class AugmentedLagrangian:
    r"""Mixed incompressibility with local pressure and dilatation fields.

    The displacement has degree ``p`` and pressure uses complete polynomials
    of degree ``p - 1`` in reference coordinates. For Q2 this is the four-mode
    discontinuous P1 space. All moments use physical reference-volume weights.

    With basis H, mass matrix M = integral(H H^T), and b = integral(H (J-1)),
    the condensed augmented energy is ``lambda.b + kappa/2 b.M^-1.b``.
    The multiplier update is ``lambda += kappa M^-1 b``. Convergence enforces
    the mixed volume moments; it does not enforce pointwise J=1 everywhere.

    This follows the three-field formulation in deal.II step-44 with its
    additional incompressibility multiplier, using a quadratic dilatational
    energy and augmented-Lagrangian updates. See FORMULATION.md.
    The consistent tangent includes a coupling between quadrature points;
    :meth:`stiffness` supplies that term separately from the local stress law.
    """

    def __init__(self, bulk_modulus: float, points: torch.Tensor,
                 weights: torch.Tensor, degree: int) -> None:
        self.kappa = float(bulk_modulus)
        if not 0.0 < self.kappa < float("inf"):
            raise ValueError("bulk_modulus must be finite and positive")
        if degree < 0:
            raise ValueError("pressure degree must be non-negative")
        # Only basis construction loops in Python; every element shares it.
        powers = [(i, j, k) for n in range(degree + 1)
                  for i in range(n + 1) for j in range(n - i + 1)
                  for k in [n - i - j]]
        self.basis = torch.stack([
            points[:, 0]**i * points[:, 1]**j * points[:, 2]**k
            for i, j, k in powers], dim=-1)
        self.weights = weights
        weighted = weights[..., None] * self.basis
        mass = torch.einsum("qm,eqn->emn", self.basis, weighted)
        self.mass_inverse = torch.linalg.inv(mass)
        self.projector = torch.linalg.solve(mass, weighted.mT)
        self.multiplier = weights.new_zeros(weights.shape[0], len(powers))

    def __repr__(self) -> str:
        return (f"AugmentedLagrangian(kappa={self.kappa}, "
                f"pressure_modes={self.basis.shape[1]})")

    def coefficients(self, J: torch.Tensor, cells: slice = slice(None)):
        """Coefficients of the physical-volume projection of J-1."""
        return torch.einsum("emq,eq->em", self.projector[cells], J - 1.0)

    def residual(self, J: torch.Tensor) -> torch.Tensor:
        """Projected volume violation evaluated at the integration points."""
        return self.coefficients(J) @ self.basis.mT

    def response(self, F: torch.Tensor, cells: slice = slice(None),
                 tangent: bool = True):
        """Pressure stress and its local derivative at fixed pressure.

        The derivative of pressure itself is the nonlocal term in stiffness().
        Invalid trial deformations propagate non-finite values for the solver
        to reject, without a device-to-host check at each quadrature point.
        """
        C = F.mT @ F
        J = torch.linalg.det(F)
        Cinv = torch.linalg.inv_ex(C)[0]
        pressure = (self.multiplier[cells]
                    + self.kappa*self.coefficients(J, cells)) @ self.basis.mT
        gJ = pressure * J
        S = gJ[..., None, None] * Cinv
        if not tangent:
            return S, None, J
        cc = Cinv[..., :, :, None, None] * Cinv[..., None, None, :, :]
        cx = (Cinv[..., :, None, :, None] * Cinv[..., None, :, None, :]
              + Cinv[..., :, None, None, :] * Cinv[..., None, :, :, None])
        D = gJ[..., None, None, None, None] * (cc - cx)
        return S, D.flatten(-4, -3).flatten(-2, -1), J

    def stiffness(self, F: torch.Tensor, dNdX: torch.Tensor,
                  cells: slice = slice(None)) -> torch.Tensor:
        """Consistent condensed dilatational stiffness kappa B^T M^-1 B."""
        J = torch.linalg.det(F)
        inverse = torch.linalg.inv_ex(F)[0]
        B = torch.einsum("eq,eqIi,eqaI,qm->emai",
                         self.weights[cells]*J, inverse, dNdX, self.basis)
        B = B.flatten(-2)
        return self.kappa * (B.mT @ self.mass_inverse[cells] @ B)

    def update(self, J: torch.Tensor) -> None:
        self.multiplier.add_(self.kappa*self.coefficients(J))

    def reset(self) -> None:
        self.multiplier.zero_()
