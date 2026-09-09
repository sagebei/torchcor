# Reference formulation for the mechanics takeover

This note specifies the mixed incompressibility implemented in `material.py`
and `assembly.py`. Benchmark accuracy and runtime must be assessed separately
from the consistency of these equations.

The implementation uses the established displacement/pressure/dilatation mixed
formulation with discontinuous linear pressure and dilatation for quadratic
hexahedra. Augmented Lagrangian iterations enforce the dilatation constraint.
The two local mixed fields are eliminated at a fixed augmentation multiplier,
retaining displacement Newton iterations and the sparse GPU solver. This avoids a new global saddle-point
solver while preserving the converged mixed equations. It does not reproduce
deal.II's numerical solver, material model, or benchmark results. The same
discrete equations can be solved with coupled Newton iterations or the
augmented iteration used here; their iteration paths and costs differ.

## Provenance and differences

- [deal.II step-44, version 9.5.0](https://github.com/dealii/dealii/blob/v9.5.0/examples/step-44/doc/intro.dox)
  defines the three-field energy and the `Q_n x DGPM_(n-1) x DGPM_(n-1)`
  interpolation. Its discretization section describes local elimination of
  pressure and dilatation and an additional multiplier constraint on dilatation
  for full incompressibility. The derivation below specializes the volumetric
  energy to a quadratic augmentation and writes that additional constraint
  explicitly. It is not the tutorial's finite-compressibility default.
- [FEBio's three-field solid implementation](https://github.com/febiosoftware/FEBio/blob/dbbcf7ba8e56d0f514b9fca354b2f65ef2595101/FEBioMech/FE3FieldElasticSolidDomain.cpp)
  provides a production example of physical-volume averaging, a separate
  dilatational stiffness term, and multiplier augmentation. Its
  `ElementDilatationalStiffness`, `UpdateElementStress`, and `Augment` methods
  are relevant. The associated header stores one average dilatation, pressure,
  and multiplier per element. **That implementation is the constant-pressure
  case, not an existing four-mode Q2/P1 implementation.** Do not claim otherwise.
- [MFEM example 19](https://mfem.org/examples/#example-19-incompressible-nonlinear-elasticity)
  shows the exact-incompressible displacement/pressure saddle system. It is a
  useful independent equation reference, not the GPU implementation to copy.

The pressure space is a mapped complete polynomial space on the reference cell,
as in [deal.II's `FE_DGPMonomial`](https://github.com/dealii/dealii/blob/v9.5.0/include/deal.II/fe/fe_dgp_monomial.h).
It is not tensor-product Q1, and on non-affine cells its mapped functions are not
necessarily linear in physical coordinates. This distinction must remain in
the documentation.

## Discrete energy and local elimination

For each Q2 reference element, use a vector of pressure basis functions
`H = [1, xi, eta, zeta]`. The code enumerates them as `[1, zeta, eta, xi]`;
this permutation changes coefficient ordering, not the space or equations.
More generally, displacement degree `p` uses complete pressure polynomials of
degree `p-1`. Let `w_q` include the reference-geometry Jacobian and
the Gauss weight. All integrals below are over the reference physical volume.

Define

```text
M_ij = sum_q w_q H_qi H_qj
b_i(u) = sum_q w_q H_qi [J_q(u) - 1]
beta = M^-1 b
```

`M` is a physical-volume mass matrix, generally different on each element.
It is fixed by the undeformed mesh and can be prepared once on the GPU.
An unweighted least-squares projector on the reference Gauss points does not
replace it on curved elements. `AugmentedLagrangian` caches the mass inverse
and the operator `M^-1 H^T W`, and stores multiplier coefficients per element.

Let `z` be coefficients of the independent dilatation minus one, `p` the mixed
pressure coefficients, and `lambda` the augmentation multipliers. With positive
augmentation parameter `kappa`, the element energy at fixed `lambda` is

```text
Phi(u, p, z; lambda)
  = W_iso(u) + kappa/2 z^T M z + p^T [b(u) - M z] + lambda^T M z.
```

Stationarity with respect to `p` and `z` gives

```text
z = M^-1 b = beta
p = lambda + kappa beta.
```

Eliminating **both** local variables gives

```text
Phi_condensed(u; lambda)
  = W_iso(u) + lambda^T b(u) + kappa/2 b(u)^T M^-1 b(u).
```

After a converged displacement solve, update

```text
lambda <- lambda + kappa beta.
```

At convergence `b = 0`, the augmentation term vanishes, leaving mixed
equilibrium equations that do not depend on `kappa`. Solutions on the same
equilibrium branch agree to numerical tolerances; this does not guarantee
convergence or uniqueness of the nonlinear solution for every `kappa`.
The corresponding hydrostatic pressure at a quadrature point is
`g_q = H_q (lambda + kappa beta)`.

This is not pressure-only condensation of an exact-incompressible two-field
system. The latter has a zero pressure/pressure block and cannot be eliminated
that way. Here the pressure/dilatation coupling permits local elimination at a
fixed augmentation multiplier.

## Residual and consistent tangent

Let `B = db/du` (four rows and 81 columns for a Q2 hexahedron). Then

```text
R_vol = B^T (lambda + kappa beta)
K_vol = sum_q w_q g_q d2J_q/du2 + kappa B^T M^-1 B.
```

The second term couples quadrature points through the element's four pressure
modes. The implementation uses this coupling rather than a pointwise penalty
stiffness. The pressure part uses the second Piola stress `S_vol = g J C^-1` with the tangent at
fixed `g`; its material tangent is

```text
D_IJKL = g J [Cinv_IJ Cinv_KL
              - Cinv_IK Cinv_JL - Cinv_IL Cinv_JK].
```

The geometric stiffness completes its displacement linearization.
`AugmentedLagrangian.stiffness` supplies the separate `kappa B^T M^-1 B`
contribution added in assembly. There is no additional local
`kappa J^2 (Cinv outer Cinv)` contribution: that would describe a different
penalty and double-count the intended augmentation.

P1 and P2 explicitly wrap the passive law in `IsochoricMaterial`, using its
analytic stress and tangent. The assembler itself accepts the material supplied
by the caller and does not silently apply that wrapper. For prescribed active
stress, split the passive response first and add the specified active stress
afterward; projecting active stress would change the benchmark's active law.

P3 uses `ActiveStressMaterial(IsochoricMaterial(passive), tension=60, ramp=True)`.
In reference material axes, the prescribed contribution is `S_active = Ta f f^T`.
Its material derivative `dS_active/dE` is zero, while `P_active = F S_active`
contributes geometric stiffness through the total stress. The optional ramp
scales the prescribed full-load tension with the current continuation factor;
it does not mutate the material or accumulate scaling after rejected steps.

Reference-attached material axes can vary per quadrature point. A spatial
callable is evaluated once at `X_q = sum_a N_a(q) X_a`; the resulting rotations
remain GPU tensors and follow the same cell chunks as assembly. The paper's
P3 fibre formula and all ventricular measurements remain in `benchmark/`.
The pinned participant source and the necessary coordinate/active-stress
differences are documented in the [P3 reference notes](benchmark/validation/takeover/reference/problem3_notes.md).

## Quadrature and GPU implementation

For tensor Q2 geometry and displacement, `det(dx/dxi)` has degree at most five
in each reference coordinate. The linear pressure moments have degree at most
six, and the pressure mass integrands at most seven. Thus a four-point Gauss
rule per axis integrates `b` and `M` exactly for polynomial Q2 element maps.
The common three-point rule is not guaranteed to do so. Four points per axis
are the Q2 default for both material and volume contributions; the exponential
passive energy still requires quadrature convergence checks. Surface pressure
uses its own face rule and the consistently differentiated deformed area vector.

Basis values, physical weights, and small mass-matrix inverses are prepared once.
Element operations use CUDA tensors and cell chunks. The state contains four
multiplier coefficients per Q2 element, not one per Gauss point.
The rank-four tangent update uses ordinary small dense batched operations; no
new global pressure CSR blocks or CPU numerical solves are required.

## Independent verification and its scope

[The independent reference tests](tests/test_mixed_reference.py) use explicit
Q2 polynomials and automatic differentiation of independently coded energies.
They check curved-element residuals and tangents, four- versus six-point physical
moments, and the Schur complement of the local uncondensed three-field Hessian.
A small separate coupled displacement/pressure Newton solve includes an
independently integrated nonsymmetric follower pressure and is compared with
the AL result at `kappa=10` and `kappa=100`. These checks test equations and local
elimination; they are not comparisons against an installed deal.II or FEBio solver.

P1 and P2 refinement must additionally compare positions, strain curves, and
dense-sampled local volume errors with unchanged force tolerances. Agreement
of a single position does not establish convergence of every field.

The implemented stopping diagnostic is `max_(element,q) |H_q beta|`: the
projected volume violation evaluated at assembly points. It is not
`max |J-1|`. `volume_report(n_gauss=6)` separately reports the volume-weighted
RMS and sampled maximum of `J-1`, total relative volume change, and the smallest
sampled `J`. Discrete mixed incompressibility is weak incompressibility, not
`J=1` at every material point on a finite mesh.

The usual mixed-element stability expectations assume suitable regular meshes.
The ventricle's collapsed apex does not automatically meet those assumptions.
Positive sampled geometry Jacobians, balanced directional refinement, and
observed convergence provide evidence; the element name alone does not prove
stability for this particular curved, collapsed mesh.
