"""
Analytical integration over dual cell geometries.

Provides exact and high-accuracy integration of scalar and vector fields
over dual cell domains extracted from hyperct simplicial complexes.
Used for validating numerical DDG operators against analytically
integrated solutions.

Two integration paths are available:

- **Divergence theorem** (``_divergence_theorem``): Numeric integration
  via Gauss-Legendre quadrature on the cell boundary.  Exact for
  polynomial fields when sufficient quadrature points are used.

- **Sympy** (``_sympy_integration``): Exact symbolic integration via
  sympy.  Optional dependency; used for machine-precision verification.

Both paths use the divergence theorem identity::

    ∫_V ∇f dV = ∮_{∂V} f · n dA

to reduce volume integrals to boundary integrals.
"""

from ._divergence_theorem import (
    integrated_gradient_1d,
    integrated_gradient_2d,
    integrated_gradient_3d,
)

from ._integrated_comparison import (
    integrated_pressure_error,
    integrated_l2_norm,
    compare_stress_force,
    volume_averaged_scalar,
)

__all__ = [
    "integrated_gradient_1d",
    "integrated_gradient_2d",
    "integrated_gradient_3d",
    "integrated_pressure_error",
    "integrated_l2_norm",
    "compare_stress_force",
    "volume_averaged_scalar",
]

try:
    from ._sympy_integration import (
        integrated_gradient_sympy_1d,
        integrated_gradient_sympy_2d,
        integrated_gradient_sympy_3d,
    )
    __all__ += [
        "integrated_gradient_sympy_1d",
        "integrated_gradient_sympy_2d",
        "integrated_gradient_sympy_3d",
    ]
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
