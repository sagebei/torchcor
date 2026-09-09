"""Reference elements for :mod:`torchcor.mechanics`.

Shape functions and integration rules -- the mathematics *inside* a single
element, independent of any mesh, material or solver.

Everything is a tensor-product Lagrange basis of arbitrary order ``p`` on
:math:`[-1, 1]^d`, evaluated once per element family and reused for every
element in the mesh.  Node ordering matches
:mod:`torchcor.mechanics.mesh`::

    hex  (l, m, n) -> (l * (p + 1) + m) * (p + 1) + n
    quad (a, b)    -> a * (p + 1) + b

Separability is what makes the basis cheap: ``N_{lmn}(xi) = L_l L_m L_n``, so a
3D basis is three 1D evaluations and an outer product, batched over all
quadrature points at once.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


__all__ = ["gauss_legendre_1d", "lagrange_basis_1d", "LagrangeHex", "LagrangeQuad"]


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


class LagrangeHex:
    """Tensor-product Lagrange hexahedron with a Gauss-Legendre rule.

    Node ordering matches :mod:`torchcor.mechanics.mesh`:
    ``(l, m, n) -> (l * (p + 1) + m) * (p + 1) + n``.

    Attributes
    ----------
    points:  ``(n_qp, 3)`` reference quadrature points.
    weights: ``(n_qp,)`` quadrature weights.
    N:       ``(n_qp, n_en)`` shape functions.
    dN:      ``(n_qp, n_en, 3)`` reference gradients.
    """

    def __init__(self, order: int, n_gauss: Optional[int] = None,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None) -> None:
        self.order = int(order)
        self.n_gauss = int(n_gauss) if n_gauss is not None else self.order + 1
        device = torch.device(device) if device is not None else torch.device("cpu")

        g, w = gauss_legendre_1d(self.n_gauss, dtype, device)
        gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
        wx, wy, wz = torch.meshgrid(w, w, w, indexing="ij")
        self.points = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
        self.weights = (wx * wy * wz).reshape(-1)

        L = [lagrange_basis_1d(self.order, self.points[:, d]) for d in range(3)]
        n0, d0 = L[0]
        n1, d1 = L[1]
        n2, d2 = L[2]

        def outer(a, b, c):
            return (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :]
                    ).reshape(self.points.shape[0], -1)

        self.N = outer(n0, n1, n2)
        self.dN = torch.stack([outer(d0, n1, n2), outer(n0, d1, n2), outer(n0, n1, d2)],
                              dim=-1)

    @property
    def n_qp(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_en(self) -> int:
        return int(self.N.shape[1])


class LagrangeQuad:
    """Tensor-product Lagrange quadrilateral, used for surface (pressure) terms.

    Node ordering ``(a, b) -> a * (p + 1) + b`` with ``a`` along ``xi_1``.
    """

    def __init__(self, order: int, n_gauss: Optional[int] = None,
                 dtype: torch.dtype = torch.float64,
                 device: Optional[torch.device] = None) -> None:
        self.order = int(order)
        self.n_gauss = int(n_gauss) if n_gauss is not None else self.order + 1
        device = torch.device(device) if device is not None else torch.device("cpu")

        g, w = gauss_legendre_1d(self.n_gauss, dtype, device)
        gx, gy = torch.meshgrid(g, g, indexing="ij")
        wx, wy = torch.meshgrid(w, w, indexing="ij")
        self.points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        self.weights = (wx * wy).reshape(-1)

        n0, d0 = lagrange_basis_1d(self.order, self.points[:, 0])
        n1, d1 = lagrange_basis_1d(self.order, self.points[:, 1])

        self.N = (n0[:, :, None] * n1[:, None, :]).reshape(self.points.shape[0], -1)
        self.dN = torch.stack(
            [(d0[:, :, None] * n1[:, None, :]).reshape(self.points.shape[0], -1),
             (n0[:, :, None] * d1[:, None, :]).reshape(self.points.shape[0], -1)],
            dim=-1,
        )

    @property
    def n_qp(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_en(self) -> int:
        return int(self.N.shape[1])
