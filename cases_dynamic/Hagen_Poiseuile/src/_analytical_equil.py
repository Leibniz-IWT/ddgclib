import numpy as np
from scipy.spatial import Delaunay  # if needed for mesh checks

def poiseuille_analytical_2d(x, G=1.0, mu=1.0, h=1.0, y_lb=0.0, y_ub=1.0):
    """Analytical velocity u_x(y) for planar Poiseuille (channel along x)."""
    y = x[1]
    # Standard: u(y) = (G/(2*mu)) * y * (h - y) with plates at 0 and h
    return (G / (2 * mu)) * (y - y_lb) * (y_ub - (y - y_lb))  # shift if y_lb != 0

def poiseuille_analytical_3d(x, U_max=1.0, R=1.0):
    """Analytical axial velocity u_z(r) for Hagen-Poiseuille tube (axis z)."""
    r = np.linalg.norm(x[:2])  # radial distance, assume centered at (0,0)
    if r > R:
        return 0.0  # outside tube (wall)
    return U_max * (1.0 - (r / R)**2)

def P_gradient_analytical(x, G=1.0, axis=0):  # G = -dp/dz or -dp/dx
    """Linear pressure gradient along flow axis (returns scalar)."""
    return -G * x[axis]  # P(x) = P0 - G * x_axis; gradient potential

def set_equilibrium_IC_2d(HC, G=1.0, mu=1.0, h=1.0, y_lb=0.0, y_ub=1.0, u_in=None):
    """Set fully developed Poiseuille IC on 2D HC (velocity + pressure gradient)."""
    for v in HC.V:
        if u_in is None:
            ux = poiseuille_analytical_2d(v.x_a, G=G, mu=mu, h=h, y_lb=y_lb, y_ub=y_ub)
        else:
            ux = u_in  # constant plug for testing
        v.u = np.array([ux, 0.0])
        P_i = P_gradient_analytical(v.x_a, G=G, axis=0)
        v.P = np.array([P_i, P_i])  # diagonal per your convention
    return HC

def set_equilibrium_IC_3d(HC, U_max=1.0, R=1.0, G=1.0, axis=2):
    """Set fully developed Hagen-Poiseuille IC on 3D tube HC."""
    for v in HC.V:
        uz = poiseuille_analytical_3d(v.x_a, U_max=U_max, R=R)
        v.u = np.array([0.0, 0.0, uz])
        P_i = P_gradient_analytical(v.x_a, G=G, axis=axis)
        v.P = np.array([P_i, P_i, P_i])  # extend to 3 components if needed; adjust du/dP accordingly
    return HC

def test_analytical_equilibrium_2d(HC, tol=1e-10, G=1.0, mu=1.0, h=1.0):
    """Verify dudt ≈ 0 and velocity error small."""
    errors_u = []
    dudt_max = 0.0
    for v in HC.V:
        u_anal = poiseuille_analytical_2d(v.x_a, G=G, mu=mu, h=h)
        errors_u.append(np.abs(v.u[0] - u_anal))
        dudt_val = dudt(v, dim=2, mu=mu)  # your dudt
        dudt_max = max(dudt_max, np.linalg.norm(dudt_val))
    max_err_u = max(errors_u)
    print(f"2D test: max velocity error = {max_err_u:.2e}, max |dudt| = {dudt_max:.2e}")
    assert max_err_u < tol and dudt_max < tol, "Equilibrium test failed"
    return max_err_u, dudt_max

def test_analytical_equilibrium_3d(HC, tol=1e-10, U_max=1.0, R=1.0, G=1.0, mu=1.0):
    """Verify dudt ≈ 0 and velocity error small (3D)."""
    errors_u = []
    dudt_max = 0.0
    for v in HC.V:
        uz_anal = poiseuille_analytical_3d(v.x_a, U_max=U_max, R=R)
        errors_u.append(np.abs(v.u[2] - uz_anal))
        dudt_val = dudt(v, dim=3, mu=mu)
        dudt_max = max(dudt_max, np.linalg.norm(dudt_val))
    max_err_u = max(errors_u)
    print(f"3D test: max velocity error = {max_err_u:.2e}, max |dudt| = {dudt_max:.2e}")
    assert max_err_u < tol and dudt_max < tol, "Equilibrium test failed"
    return max_err_u, dudt_max