"""Structured return type for domain builder functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperct import Complex


@dataclass
class DomainResult:
    """Result of a domain builder function.

    Attributes
    ----------
    HC : Complex
        The simplicial complex with the filled domain mesh.
    bV : set
        Set of all boundary vertex objects.
    boundary_groups : dict[str, set]
        Named boundary regions.  Standard keys (when applicable):
        ``'walls'``, ``'inlet'``, ``'outlet'``.
        Custom keys depend on the domain type.
    dim : int
        Spatial dimension (2 or 3).
    metadata : dict[str, Any]
        Domain-specific metadata (radius, length, volume, etc.)
        for downstream use.
    """

    HC: Complex
    bV: set
    boundary_groups: dict[str, set] = field(default_factory=dict)
    dim: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)

    def tag_boundaries(self) -> None:
        """Set ``v.boundary = True/False`` on every vertex based on *bV*.

        Must be called before ``compute_vd()``.  Called automatically
        by every domain builder.
        """
        for v in self.HC.V:
            v.boundary = v in self.bV

    def summary(self) -> str:
        """One-line human-readable summary."""
        n_verts = self.HC.V.size()
        groups_str = ", ".join(
            f"{k}: {len(v)}" for k, v in self.boundary_groups.items()
        )
        return (
            f"DomainResult: {n_verts} vertices, {len(self.bV)} boundary, "
            f"dim={self.dim}, groups=[{groups_str}]"
        )
