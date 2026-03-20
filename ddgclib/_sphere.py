import numpy as np
import copy
from ._curvatures import *
import matplotlib.pyplot as plt

from ddgclib.geometry._parametric_surfaces import parametric_surface, _second_axis_boundary


def sphere_N(R, Phi, Theta, refinement=2, cdist=1e-10, equilibrium=True):
    """Spherical surface mesh.  Delegates to ``parametric_surface()``."""
    theta_range = (Theta[0], Theta[-1])
    phi_range = (Phi[0], Phi[-1])

    def f(theta, phi):
        return (
            R * np.cos(theta) * np.sin(phi),
            R * np.sin(theta) * np.sin(phi),
            R * np.cos(phi),
        )

    return parametric_surface(
        f, [theta_range, phi_range], refinement, cdist=1e-8,
        boundary_fn=_second_axis_boundary,
    )
