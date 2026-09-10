"""Reference elements for :mod:`torchcor.mechanics`.

Shape functions and integration rules -- the mathematics *inside* a single
element, independent of any mesh, material or solver.

Two families are provided, and they share one interface, so the assembly and
the mesh never test which one they hold:

* **Tensor-product Lagrange** of arbitrary order ``p`` on :math:`[-1, 1]^d`
  (:class:`LagrangeHex`, :class:`LagrangeQuad`).  Node ordering is
  lexicographic, matching :mod:`torchcor.mechanics.mesh`::

      hex  (l, m, n) -> (l * (p + 1) + m) * (p + 1) + n
      quad (a, b)    -> a * (p + 1) + b

  Separability is what makes the basis cheap: ``N_{lmn}(xi) = L_l L_m L_n``, so
  a 3D basis is three 1D evaluations and an outer product, batched over all
  quadrature points at once.

* **Simplex Lagrange** of order 1 or 2 (:class:`LagrangeTet`,
  :class:`LagrangeTri`), which is what stored unstructured meshes contain.
  Node ordering is the vertices, then the edge midpoints in :attr:`EDGES`
  order -- lexicographic in each edge's two vertices, which is not the order
  Gmsh writes; :mod:`torchcor.mechanics.mesh` gives the permutation.

Every class exposes ``points``/``weights`` (its rule), ``N``/``dN`` (the basis
there), and a static ``basis(order, xi)``/``n_nodes(order)`` for use away from
the quadrature points, e.g. when locating a probe.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


__all__ = ["gauss_legendre_1d", "lagrange_basis_1d", "LagrangeHex",
           "LagrangeQuad", "LagrangeTet", "LagrangeTri"]


def gauss_legendre_1d(n: int, dtype: torch.dtype, device: torch.device
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``n``-point Gauss-Legendre rule on ``[-1, 1]``."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return (torch.as_tensor(x, dtype=dtype, device=device),
            torch.as_tensor(w, dtype=dtype, device=device))


def lagrange_basis_1d(order: int, xi: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lagrange basis of degree ``order`` on equispaced nodes of ``[-1, 1]``.

    Parameters
    ----------
    order:
        Polynomial degree ``p``; the basis has ``p + 1`` functions.
    xi:
        ``(...,)`` evaluation points.

    Returns
    -------
    ``(N, dN)`` each of shape ``(..., p + 1)``.
    """
    p = int(order)
    dtype, device = xi.dtype, xi.device
    nodes = torch.linspace(-1.0, 1.0, p + 1, dtype=dtype, device=device)

    # denom[l] = prod_{m != l} (x_l - x_m)
    sep = nodes[:, None] - nodes[None, :]
    sep = sep + torch.eye(p + 1, dtype=dtype, device=device)
    denom = sep.prod(dim=1)

    diff = xi[..., None] - nodes                              # (..., m)
    eye = torch.eye(p + 1, dtype=torch.bool, device=device)

    # N_l = prod_{m != l} (xi - x_m) / denom_l
    M = diff[..., None, :].expand(*diff.shape[:-1], p + 1, p + 1)
    M = torch.where(eye, torch.ones_like(M), M)
    N = M.prod(dim=-1) / denom

    # dN_l = sum_{k != l} prod_{m != l, k} (xi - x_m) / denom_l
    idx = torch.arange(p + 1, device=device)
    skip = (idx[:, None, None] == idx[None, None, :]) | \
           (idx[None, :, None] == idx[None, None, :])          # [l, k, m]
    A = diff[..., None, None, :].expand(*diff.shape[:-1], p + 1, p + 1, p + 1)
    A = torch.where(skip, torch.ones_like(A), A)
    T = A.prod(dim=-1)                                         # [..., l, k]
    T = torch.where(eye, torch.zeros_like(T), T)               # drop k == l
    dN = T.sum(dim=-1) / denom

    return N, dN


def _supplied(rule, dim: int, dtype: torch.dtype, device: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate a caller-supplied ``(points, weights)`` rule.

    Reproducing another implementation's integration exactly needs its actual
    points and weights, not a matching count: two rules of the same degree are
    generally different rules.
    """
    points, weights = (torch.as_tensor(a, dtype=dtype, device=device) for a in rule)
    if points.ndim != 2 or points.shape[1] != dim or weights.shape != points.shape[:1]:
        raise ValueError(f"a rule for this element needs (n, {dim}) points and "
                         f"(n,) weights, got {tuple(points.shape)} and "
                         f"{tuple(weights.shape)}")
    return points.contiguous(), weights.contiguous()


class _Element:
    """Sizes shared by every reference element."""

    @property
    def n_qp(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_en(self) -> int:
        return int(self.N.shape[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{type(self).__name__}(order={self.order}, "
                f"n_en={self.n_en}, n_qp={self.n_qp})")


class _TensorProduct(_Element):
    """Lagrange basis on the reference cube, with a Gauss-Legendre rule."""

    DIM = 0

    def __init__(self, order: int, n_gauss: Optional[int] = None,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None,
                 rule: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> None:
        self.order = int(order)
        self.n_gauss = int(n_gauss) if n_gauss is not None else self.order + 1
        device = torch.device(device) if device is not None else torch.device("cpu")

        if rule is not None:
            self.points, self.weights = _supplied(rule, self.DIM, dtype, device)
        else:
            g, w = gauss_legendre_1d(self.n_gauss, dtype, device)
            axes = torch.meshgrid(*([g] * self.DIM), indexing="ij")
            wts = torch.meshgrid(*([w] * self.DIM), indexing="ij")
            self.points = torch.stack([a.reshape(-1) for a in axes], dim=1)
            self.weights = torch.stack(wts).prod(dim=0).reshape(-1)
        self.N, self.dN = self.basis(self.order, self.points)

    @classmethod
    def n_nodes(cls, order: int) -> int:
        return (int(order) + 1) ** cls.DIM

    @classmethod
    def volume_quadrature(cls, order: int) -> int:
        """Gauss points per axis for the volume terms of an order-``p`` cell.

        An ``n``-point Gauss rule is exact to degree ``2n - 1``.  This is the
        curved isoparametric requirement -- it includes the pressure mass
        matrix, so Q2 needs four points per axis rather than the three an
        affine cell would.
        """
        return max(int(order) + 1, (5*int(order) - 1)//2)

    @classmethod
    def surface_quadrature(cls, order: int) -> int:
        """Gauss points per axis for a surface term of an order-``p`` face.

        A follower pressure integrand is cubic in the surface shape functions
        -- the deformed position against two surface tangents -- so it reaches
        degree ``3p - 1``, not the ``p + 1`` an affine term would.
        """
        return max(int(order) + 1, (3*int(order) + 1)//2)

    @classmethod
    def mass_quadrature(cls, order: int) -> int:
        """Gauss points per axis for the consistent mass matrix.

        Two shape functions against the reference-map determinant reaches
        degree ``5p - 1`` per direction on a curved Qp cell, which the volume
        rule -- chosen for the nonlinear terms -- does not cover.  The mass
        matrix is constant and built once, so it takes its own rule rather
        than raising the cost of every Newton iteration.
        """
        return max(cls.volume_quadrature(order), -(-5*int(order)//2))

    @staticmethod
    def basis(order: int, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Basis and reference gradients at ``xi`` of shape ``(n, dim)``."""
        L = [lagrange_basis_1d(order, xi[:, d]) for d in range(xi.shape[-1])]

        def outer(derivative: int) -> torch.Tensor:
            out = L[0][derivative == 0]
            for d in range(1, len(L)):
                out = (out[:, :, None] * L[d][derivative == d][:, None, :]
                       ).reshape(xi.shape[0], -1)
            return out

        N = outer(-1)                                    # no derivative
        dN = torch.stack([outer(d) for d in range(len(L))], dim=-1)
        return N, dN


class LagrangeHex(_TensorProduct):
    """Tensor-product Lagrange hexahedron.

    Node ordering matches :mod:`torchcor.mechanics.mesh`:
    ``(l, m, n) -> (l * (p + 1) + m) * (p + 1) + n``.

    Attributes
    ----------
    points:  ``(n_qp, 3)`` reference quadrature points.
    weights: ``(n_qp,)`` quadrature weights.
    N:       ``(n_qp, n_en)`` shape functions.
    dN:      ``(n_qp, n_en, 3)`` reference gradients.
    """

    DIM = 3


class LagrangeQuad(_TensorProduct):
    """Tensor-product Lagrange quadrilateral, used for surface terms.

    Node ordering ``(a, b) -> a * (p + 1) + b`` with ``a`` along ``xi_1``.
    """

    DIM = 2


def _collapsed_rule(n: int, dim: int, dtype: torch.dtype, device: torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gauss rule on the reference simplex, by the Duffy transformation.

    Mapping the cube onto the simplex collapses one face to a point, and the
    Jacobian of that collapse is a polynomial, so a tensor-product Gauss rule
    on the cube integrates polynomials on the simplex exactly.  It costs more
    points than a bespoke symmetric rule but is generated for any order and is
    checkable against exact monomial integrals, which is what the tests do.
    """
    x, w = gauss_legendre_1d(n, dtype, device)
    x, w = 0.5*(x + 1.0), 0.5*w                                # onto [0, 1]
    if dim == 2:
        u, v = torch.meshgrid(x, x, indexing="ij")
        wu, wv = torch.meshgrid(w, w, indexing="ij")
        points = torch.stack([u, v*(1.0 - u)], dim=-1).reshape(-1, 2)
        return points, (wu*wv*(1.0 - u)).reshape(-1)
    u, v, t = torch.meshgrid(x, x, x, indexing="ij")
    wu, wv, wt = torch.meshgrid(w, w, w, indexing="ij")
    points = torch.stack([u, v*(1.0 - u), t*(1.0 - u)*(1.0 - v)], dim=-1).reshape(-1, 3)
    return points, (wu*wv*wt*(1.0 - u)**2*(1.0 - v)).reshape(-1)


class _Simplex(_Element):
    """Lagrange basis of degree one or two on a reference simplex.

    Nodes are the vertices, then the edge midpoints in :attr:`EDGES` order,
    which is lexicographic in each edge's two vertices.
    The basis is written in the barycentric coordinates
    ``lambda = (1 - sum(xi), xi)``, whose gradients are constant, so the
    quadratic basis costs one product per node.
    """

    EDGES: Tuple[Tuple[int, int], ...] = ()
    DIM = 0

    def __init__(self, order: int, n_gauss: Optional[int] = None,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None,
                 rule: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> None:
        self.order = int(order)
        if self.order not in (1, 2):
            raise ValueError(f"simplex order must be 1 or 2, got {order}")
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.n_gauss = int(n_gauss) if n_gauss is not None else self.order + 1
        self.points, self.weights = (
            _supplied(rule, self.DIM, dtype, device) if rule is not None
            else _collapsed_rule(self.n_gauss, self.DIM, dtype, device))
        self.N, self.dN = self.basis(self.order, self.points)

    @classmethod
    def n_nodes(cls, order: int) -> int:
        return cls.DIM + 1 + (len(cls.EDGES) if int(order) == 2 else 0)

    @classmethod
    def volume_quadrature(cls, order: int) -> int:
        """Gauss points per axis for the volume terms of an order-``p`` cell.

        The collapsed rule loses two orders to the Duffy Jacobian, so it is
        exact to degree ``2n - 3`` rather than the ``2n - 1`` of the cube --
        which is why the tensor-product count cannot simply be reused: at
        ``p = 1`` it gives degree one against a degree-two mass integrand.
        The target here is ``2p + 1``: the mass matrix of an affine cell is
        degree ``2p``, with one order over for a linear weight.  Genuinely
        curved simplices need more, and take it through ``quadrature_order``.
        """
        return (2*int(order) + 5)//2

    @classmethod
    def surface_quadrature(cls, order: int) -> int:
        """Gauss points per axis for a surface term of an order-``p`` face.

        Degree ``3p - 1``, as for the cube, but reached three orders later:
        ``2n - 3 >= 3p - 1``.  On the flat facets a stored linear mesh actually
        has, the integrand is only degree ``p`` and any of these is exact; the
        margin is for a genuinely curved quadratic face.
        """
        return max(int(order) + 1, (3*int(order) + 3)//2)

    @classmethod
    def mass_quadrature(cls, order: int) -> int:
        """Gauss points per axis for the consistent mass matrix.

        Total degree ``5p - 3`` on a curved Pp cell, against the ``2n - 3`` the
        collapsed rule reaches.  Built once, so it takes its own rule; on the
        affine cells a stored mesh has, this agrees with the volume rule.
        """
        return max(cls.volume_quadrature(order), -(-5*int(order)//2))

    @classmethod
    def basis(cls, order: int, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Basis and reference gradients at ``xi`` of shape ``(n, DIM)``."""
        dim = cls.DIM
        lam = torch.cat([1.0 - xi.sum(dim=-1, keepdim=True), xi], dim=-1)
        grad = torch.zeros(dim + 1, dim, dtype=xi.dtype, device=xi.device)
        grad[0] = -1.0
        grad[1:] = torch.eye(dim, dtype=xi.dtype, device=xi.device)

        if int(order) == 1:
            return lam, grad.expand(xi.shape[0], -1, -1).clone()

        n_nodes = cls.n_nodes(2)
        N = torch.empty(xi.shape[0], n_nodes, dtype=xi.dtype, device=xi.device)
        dNdlam = torch.zeros(xi.shape[0], n_nodes, dim + 1,
                             dtype=xi.dtype, device=xi.device)
        for i in range(dim + 1):
            N[:, i] = lam[:, i]*(2.0*lam[:, i] - 1.0)
            dNdlam[:, i, i] = 4.0*lam[:, i] - 1.0
        for k, (a, b) in enumerate(cls.EDGES):
            j = dim + 1 + k
            N[:, j] = 4.0*lam[:, a]*lam[:, b]
            dNdlam[:, j, a] = 4.0*lam[:, b]
            dNdlam[:, j, b] = 4.0*lam[:, a]
        return N, dNdlam @ grad


class LagrangeTet(_Simplex):
    """Four- or ten-node tetrahedron."""

    EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    DIM = 3


class LagrangeTri(_Simplex):
    """Three- or six-node triangle, used for surface terms."""

    EDGES = ((0, 1), (0, 2), (1, 2))
    DIM = 2
