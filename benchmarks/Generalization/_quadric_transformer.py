from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple
from _plot import plot_quadric_surface
import math

EPS = 1e-12
@dataclass
class QuadricResult:
    type: str
    axis: np.ndarray | None
    centre: np.ndarray
    new_coeffs: Tuple[float, ...]
    scale_factors1: Tuple[float, float, float]
    scale_factors2: Tuple[float, float, float]
    jacobian: float
    R: np.ndarray

    def asdict(self):
        d = asdict(self)
        for k in ("axis", "centre"):
            if d[k] is not None:
                d[k] = d[k].tolist()
        for k in ("scaling_matrix", "transform", "canonical_matrix"):
            d[k] = d[k].tolist()
        d["new_coeffs"] = list(d["new_coeffs"])
        d["postscale_factors"] = list(d["postscale_factors"])
        return d
    
def detect_surface_of_revolution(coeffs, tol=1e-8):
    """
    Detects axis and type of surface of revolution from quadric coefficients.
    Returns (axis_vector or None, geometry_name or None).
    """
    A, B, C, D, E, F, G, H, I, J = coeffs
    Q = np.array([
        [A, D/2, E/2],
        [D/2, B, F/2],
        [E/2, F/2, C]
    ], dtype=float)
    b = np.array([G, H, I], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(Q)
    idxs = np.argsort(eigvals)
    eigvals = eigvals[idxs]
    eigvecs = eigvecs[:, idxs]
    zero_mask = np.abs(eigvals) <= tol
    # Identify axis if two eigenvalues are equal (surface of revolution)
    axis = None 
    if np.isclose(eigvals[0], eigvals[1], atol=tol):
        axis = eigvecs[:, 2]
    elif np.isclose(eigvals[1], eigvals[2], atol=tol):
        axis = eigvecs[:, 0]
    elif np.isclose(eigvals[0], eigvals[2], atol=tol):
        axis = eigvecs[:, 1]
    elif np.count_nonzero(zero_mask) == 1:
        axis = eigvecs[:, np.where(zero_mask)[0][0]]    
    # Otherwise, axis remains None

    positive = eigvals > tol
    negative = eigvals < -tol
    zero = np.abs(eigvals) <= tol
    n_pos = np.count_nonzero(positive)
    n_neg = np.count_nonzero(negative)
    n_zero = np.count_nonzero(zero)
    linear_proj = 0.0 if axis is None else np.abs(np.dot(axis, b))

    # Classify quadric type
    name = None
    # --- helpers --------------------------------------------------------
    is_cone = (n_pos == 2 and n_neg == 1) or (n_pos == 1 and n_neg == 2)
    J_is_zero = abs(J) <= tol           # constant term

    # --- classification -------------------------------------------------
    if is_cone and J_is_zero and linear_proj <= tol:
        name = "cone of revolution"
    elif n_pos == 3:
        name = "ellipsoid of revolution (spheroid)"
    elif n_neg == 3:
        name = "imaginary ellipsoid of revolution"
    elif n_pos == 2 and n_neg == 1:
        name = "hyperboloid of revolution (one sheet)"
    elif n_pos == 1 and n_neg == 2:
        name = "hyperboloid of revolution (two sheets)"
    elif (n_pos == 2 and n_neg == 1) or (n_pos == 1 and n_neg == 2):
        name = "cone of revolution"
    elif n_pos == 2 and n_zero == 1:
        name = "cylinder of revolution"
    elif n_pos == 1 and n_neg == 1 and n_zero == 1 and linear_proj <= tol:
        name = "hyperbolic cylinder"
    elif n_pos == 1 and n_neg == 1 and n_zero == 1 and linear_proj > tol:
        name = "hyperbolic paraboloid"
    elif n_pos == 1 and n_zero == 2:
        name = "parabolic cylinder"
    elif n_pos == 2 and n_zero == 1 and linear_proj > tol:
        name = "paraboloid of revolution"
    elif n_pos == 2 and n_zero == 1 and np.abs(linear_proj) <= tol:
        name = "elliptic cylinder"
    elif n_neg == 2 and n_zero == 1 and np.abs(linear_proj) <= tol:
        name = "imaginary elliptic cylinder"
    # Optionally, you can add more special cases
    # else: name = None

    if name is not None:
        return axis, name
    else:
        return None, None

def ellipsoid_volume_from_coeffs(coeffs):
    """
    Given 10 quadric coefficients (A, B, C, D, E, F, G, H, I, J) for a surface:
      A x^2 + B y^2 + C z^2 + D x y + E x z + F y z + G x + H y + I z + J = 0,
    where G=H=I=0 and the surface is centered at the origin,
    returns the ellipsoid volume.
    """
    A, B, C, D, E, F, G, H, I, J = coeffs
    if any(abs(t) > 1e-10 for t in (G, H, I)):
        raise ValueError("This function assumes the ellipsoid is centered at the origin (G, H, I = 0).")
    if abs(J + 1) > 1e-8 and abs(J - 1) > 1e-8:
        print("Warning: The constant term is not ±1. The code will normalize to -1 if needed.")
    
    # Build the 3x3 symmetric matrix
    Q = np.array([
        [A, D/2, E/2],
        [D/2, B, F/2],
        [E/2, F/2, C]
    ], dtype=float)
    # Eigenvalues (principal axes coefficients)
    eigvals = np.linalg.eigvalsh(Q)
    if np.any(eigvals <= 0):
        raise ValueError("Not an ellipsoid (quadratic form not positive definite)!")
    axes = 1/np.sqrt(eigvals)
    volume = (4/3) * np.pi * np.prod(axes)
    return volume, axes

def quadric_major_z_minor_x(coeffs, tol=1e-8):
    """
    Rotate a quadric so that
      • if the quadratic form Q is singular, z'-terms are removed completely
        (C' = E' = F' = I' = 0);
      • otherwise it behaves like the original:   major → z',  minor → x'.

    Returns (new_coeffs, R) and never raises for a full-rank Q.

    coeffs = (A, B, C, D, E, F, G, H, I, J)
    """
    A, B, C, D, E, F, G, H, I_lin, J = coeffs
    Q = np.array([[A, D/2, E/2],
                  [D/2, B, F/2],
                  [E/2, F/2, C]], dtype=float)
    b = np.array([G, H, I_lin], dtype=float)

    # ------------------------------------------------------------------ 1 —
    eigvals, eigvecs = np.linalg.eigh(Q)
    abs_eig = np.abs(eigvals)
    null_mask = np.isclose(eigvals, 0.0,
                           atol=tol*np.max(abs_eig) + tol)

    # ======================================================================
    #  CASE 1  –  Singular Q  →  try to annihilate every z'-term
    # ======================================================================
    if np.any(null_mask):
        null_basis = eigvecs[:, null_mask]      # (3,k),  k = 1 or 2

        # ----- choose new z'-axis  c3  -------------------------------
        c3 = None
        if np.linalg.norm(b) < tol:                 # no linear term
            c3 = null_basis[:, 0]
        elif null_basis.shape[1] == 1:              # 1-D null-space
            v = null_basis[:, 0]
            if abs(np.dot(v, b)) < tol:
                c3 = v                              # b ⟂ v  ⇒ good
        else:                                       # 2-D null-space
            n1, n2 = null_basis.T[:2]
            β1, β2 = np.dot(n1, b), np.dot(n2, b)
            if abs(β1) < tol and abs(β2) < tol:
                c3 = n1
            else:                                   # rotate inside null-plane
                θ = math.atan2(-β1, β2)
                c3 = math.cos(θ)*n1 + math.sin(θ)*n2

        # If we found such a c3, finish the “xy-only” rotation
        if c3 is not None:
            c3 /= np.linalg.norm(c3)

            # x' = eigen-direction of largest |λ|, made ⟂ c3
            c1 = eigvecs[:, np.argmax(abs_eig)]
            c1 -= np.dot(c1, c3)*c3
            c1 /= np.linalg.norm(c1)

            c2 = np.cross(c3, c1)
            c2 /= np.linalg.norm(c2)

            R = np.column_stack((c1, c2, c3))
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1

            Qp = R.T @ Q @ R
            bp = R.T @ b

            A_p, B_p, C_p = np.diag(Qp)
            D_p = 2*Qp[0, 1]
            E_p = 2*Qp[0, 2]
            F_p = 2*Qp[1, 2]
            G_p, H_p, I_p = bp

            # succeeded in killing all z'-terms?
            if (abs(C_p) < tol and abs(E_p) < tol
                    and abs(F_p) < tol and abs(I_p) < tol):
                return (A_p, B_p, C_p, D_p, E_p, F_p,
                        G_p, H_p, I_p, J), R
    # ======================================================================
    #  CASE 2  –  full-rank Q   (or “xy-only” not possible)
    # ======================================================================
    # Original behaviour:   z' = eigvec with smallest |λ|,
    #                       x' = largest |λ|,   y' = remaining.
    zero_idx = np.argmin(abs_eig)
    other = [i for i in range(3) if i != zero_idx]
    other = sorted(other, key=lambda i: -abs_eig[i])

    v_z = eigvecs[:, zero_idx]
    v_x = eigvecs[:, other[0]]
    v_y = eigvecs[:, other[1]]

    R = np.column_stack((v_x, v_y, v_z))
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    Qp = R.T @ Q @ R
    bp = R.T @ b

    A_p, B_p, C_p = np.diag(Qp)
    D_p = 2*Qp[0, 1]
    E_p = 2*Qp[0, 2]
    F_p = 2*Qp[1, 2]
    G_p, H_p, I_p = bp

    return (A_p, B_p, C_p, D_p, E_p, F_p,
            G_p, H_p, I_p, J), R

def _quadric_matrix(_A,_B,_C,_D,_E,_F):
    return np.array([[_A,_D/2,_E/2],[_D/2,_B,_F/2],[_E/2,_F/2,_C]],float)

def evaluate_quadric(coeffs:Tuple[float,...], p:np.ndarray) -> float:
    _A, _B, _C, _D, _E, _F, _G, _H, _I, _J = coeffs
    #print("evaluate_quadric coeffs_inside", coeffs,"test point", p)
    x, y, z = p
    return (
        _A*x**2 +
        _B*y**2 +
        _C*z**2 +
        _D*x*y +
        _E*x*z +
        _F*y*z +
        _G*x +
        _H*y +
        _I*z +
        _J
    )

def postscale_coeffs(coeffs):
    a1, a2, a3, a4, a5, a6, b1, b2, b3, c = coeffs
    def scale_factor(a):
        return 1.0 / np.sqrt(abs(a)) if abs(a) > 1e-10 else 1.0
    sx = scale_factor(a1)
    sy = scale_factor(a2)
    sz = scale_factor(a3)
    a1_ = a1 * sx * sx
    a2_ = a2 * sy * sy
    a3_ = a3 * sz * sz
    a4_ = a4 * sx * sy
    a5_ = a5 * sx * sz
    a6_ = a6 * sy * sz
    b1_ = b1 * sx
    b2_ = b2 * sy
    b3_ = b3 * sz
    c_ = c
    detJ = abs(sx * sy * sz)
    return (a1_, a2_, a3_, a4_, a5_, a6_, b1_, b2_, b3_, c_), (sx, sy, sz)

def analyse_quadric(*coeffs:float, eps:float=EPS, prescale:bool=True, postscale:bool=True) -> QuadricResult:
    coeffs_org = tuple(map(float, coeffs))
    coeffs, Rotation = quadric_major_z_minor_x(coeffs)


    #print("After rotation coeffs:", tuple(float(x) for x in coeffs))
    #print("Rotation matrix:\n", Rotation)

    scale_factors1 = (1.0, 1.0, 1.0)

    if abs(coeffs[-1] + 1) > 1e-12 and coeffs[-1] !=0:  # if _J is not already -1
        scale = -1.0 / coeffs[-1]
        coeffs = tuple(c * scale for c in coeffs)
    #print("after j = -1 coeffs***************", coeffs)
    #print("after j = -1 coeffs***************", tuple(float(x) for x in coeffs))

    if prescale:
        coeffs, scale_factors1 = postscale_coeffs(coeffs)
    _A,_B,_C,_D,_E,_F,_G,_H,_I,_J = map(float, coeffs)

    #print("after scaling_quadric coeffs", tuple(float(x) for x in coeffs), "scale_factors1",tuple(float(x) for x in scale_factors1))

    A = _quadric_matrix(_A,_B,_C,_D,_E,_F)
    #print("analyse_quadric matrix A:\n", A)

    b_vec = np.array([_G,_H,_I])
    #print("b_vec", b_vec)


    centre, *_ = np.linalg.lstsq(A, -b_vec, rcond=None)

    #print("centre", centre)
    lam, V = np.linalg.eigh(A)
    zeros = np.sum(np.abs(lam) < eps)
    #print("eigenvalues", lam, "V", V, "zeros", zeros)
    sym_type = "none"
    axis_vec = None

    #print("eigenvalues", lam)

    if zeros == 0 and np.allclose(lam, lam[0], atol=eps):
        sym_type = "sphere"
        axis_vec = None
        print("SPHERE",_A, _B, _C, _D, _E, _F, _G, _H, _I, _J)
        a = _A
        g, f, h, c = _G/2, _H/2, _I/2, _J
        centre = (-g/a, -f/a, -h/a)
        r = np.sqrt(abs(g**2 + f**2 + h**2 - a*c) / a**2)
        #print("g**2",g**2," f**2",f**2, "h**2", h**2, "a*c", a*c, "a**2", a**2)
        #print(f"Detected sphere: center = {centre}, radius r = {r}")
        #print("Volume", 4/3 * np.pi * r**3*scale_factors1[0]* scale_factors1[1] * scale_factors1[2])
       # print("Theory Volume", ellipsoid_volume_from_coeffs(coeffs_org))
        R = np.eye(3)
         
        #transfer
    elif zeros == 2:                        # rank-1 quadratic form
        sym_type = "translation1"
        print("Translation symmetry detected")

        # ------------------------------------------------------------
        # 1.   axis of the cylinder/paraboloid
        # ------------------------------------------------------------
        axis_vec = V[:, np.argmin(np.abs(lam))]

        # ------------------------------------------------------------
        # 2.   shift that kills linear terms  (complete the square)
        # ------------------------------------------------------------
        a, b, c = _A, _B, _C
        g, f, h = _G, _H, _I        # <-- NO division by 2 here

        centre = np.zeros(3)
        if abs(a) > eps: centre[0] = -g / (2.0 * a)
        if abs(b) > eps: centre[1] = -f / (2.0 * b)
        if abs(c) > eps: centre[2] = -h / (2.0 * c)
        print("centre (translation applied):", centre)

        # ------------------------------------------------------------
        # 3.   update all coefficients after the shift
        #      x -> x̂ + centre[0],  y -> ŷ + centre[1],  z -> ẑ + centre[2]
        # ------------------------------------------------------------
        dx, dy, dz = centre
        # new linear part
        _G = g + 2*a*dx + _D*dy + _E*dz
        _H = f + 2*b*dy + _D*dx + _F*dz
        _I = h + 2*c*dz + _E*dx + _F*dy
        # force the ones we aimed to kill to exact zero
        if abs(a) > eps: _G = 0.0
        if abs(b) > eps: _H = 0.0
        if abs(c) > eps: _I = 0.0
        # new constant term
        _J = (_J
            + g*dx + f*dy + h*dz
            + a*dx**2 + b*dy**2 + c*dz**2
            + _D*dx*dy + _E*dx*dz + _F*dy*dz)

        # ------------------------------------------------------------
        # 4.   keep the same rotation matrix
        # ------------------------------------------------------------
        R = V

        # finally pack everything back into coeffs if your code
        # expects that further down:
        coeffs = (_A, _B, _C, _D, _E, _F, _G, _H, _I, _J)

    elif zeros == 1:                       # rank-2 quadratic form
        sym_type = "translation"
        print("Translation symmetry detected")
        # ------------------------------------------------------------------ 1 —
        # axis of the cylinder / paraboloid  (null-eigenvector)
        axis_vec = V[:, np.argmin(np.abs(lam))]

        # ------------------------------------------------------------------ 2 —
        # shift that annihilates linear terms in the range(Q)
        Q = np.array([[_A, _D/2, _E/2],
                    [_D/2, _B, _F/2],
                    [_E/2, _F/2, _C]], dtype=float)
        b = np.array([_G, _H, _I], dtype=float)

        centre = -0.5 * np.linalg.pinv(Q) @ b         # minimal-norm translation
        dx, dy, dz = centre
        print("centre (translation applied):", centre)

        # ------------------------------------------------------------------ 3 —
        # update all coefficients after the shift
        #     x -> x̂ + dx,  y -> ŷ + dy,  z -> ẑ + dz
        _G = _G + 2*_A*dx + _D*dy + _E*dz
        _H = _H + 2*_B*dy + _D*dx + _F*dz
        _I = _I + 2*_C*dz + _E*dx + _F*dy
        # force linear terms whose quadratic coeff. is non-zero to zero
        if abs(_A) > eps: _G = 0.0
        if abs(_B) > eps: _H = 0.0
        if abs(_C) > eps: _I = 0.0

        _J = (_J
            + b @ centre                # g*dx + f*dy + h*dz
            + centre @ Q @ centre)      # quadratic contribution

        # ------------------------------------------------------------------ 4 —
        # rotation matrix that aligns e3 with the symmetry axis
        if axis_vec is not None:
            e3 = axis_vec / np.linalg.norm(axis_vec)
            tmp = np.array([1, 0, 0]) if abs(e3[0]) < 0.9 else np.array([0, 1, 0])
            e1 = np.cross(e3, tmp);  e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)
            R = np.column_stack((e1, e2, e3))
        else:
            R = V                                     # fall-back

        # (optional) repack if later code expects the tuple
        coeffs = (_A, _B, _C, _D, _E, _F, _G, _H, _I, _J)

    elif zeros == 0:                    # full-rank quadratic form
        # ------------------------------------------------------------------ 1 —
        # (a) build quadratic matrix Q and linear vector b
        Q = np.array([[_A, _D/2, _E/2],
                    [_D/2, _B, _F/2],
                    [_E/2, _F/2, _C]], dtype=float)
        b = np.array([_G, _H, _I], dtype=float)

        # (b) centre that kills the linear terms:  Q·c = −b/2
        centre = -0.5 * np.linalg.solve(Q, b)   # exact for full rank
        dx, dy, dz = centre

        # ------------------------------------------------------------------ 2 —
        # update linear and constant terms after the shift
        _G = _G + 2*_A*dx + _D*dy + _E*dz      # → 0
        _H = _H + 2*_B*dy + _D*dx + _F*dz      # → 0
        _I = _I + 2*_C*dz + _E*dx + _F*dy      # → 0
        _J = (_J
            + b @ centre                     # linear contribution
            + centre @ Q @ centre)           # quadratic contribution

        # the quadratic part (A…F) is unchanged
        coeffs = (_A, _B, _C, _D, _E, _F, _G, _H, _I, _J)

        # ------------------------------------------------------------------ 3 —
        # detect axis of rotational symmetry, if any
        pairs = [(0, 1), (0, 2), (1, 2)]
        equal_pair = next((p for p in pairs if abs(lam[p[0]] - lam[p[1]]) < eps),
                        None)
        if equal_pair is not None:
            sym_type = "rotation"
            axis_index = ({0, 1, 2} - set(equal_pair)).pop()
            axis_vec = V[:, axis_index]

        # ------------------------------------------------------------------ 4 —
        # build rotation matrix whose e₃-axis is the symmetry (if found)
        if axis_vec is not None:
            e3 = axis_vec / np.linalg.norm(axis_vec)
            tmp = np.array([1, 0, 0]) if abs(e3[0]) < 0.9 else np.array([0, 1, 0])
            e1 = np.cross(e3, tmp); e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)
            R = np.column_stack((e1, e2, e3))
        else:
            R = V
    else:
        R = V

    new_coeffs = coeffs

    #print("+++++++++before analyse_quadric coeffs", new_coeffs)

    postscale_factors = (1.0, 1.0, 1.0)
    #print("pre new_coeffs", new_coeffs)

    if postscale:
        new_coeffs, scale_factors2 = postscale_coeffs(new_coeffs)
    


    postscale_factors = (
        scale_factors2[0] * scale_factors1[0],
        scale_factors2[1] * scale_factors1[1],
        scale_factors2[2] * scale_factors1[2]
    )
    jacobian = postscale_factors[0] * postscale_factors[1] * postscale_factors[2]
    if sym_type == "sphere":
        new_coeffs = (1,1,1,0,0,0,0,0,0,-r**2)  

    return QuadricResult(sym_type, axis_vec, centre, new_coeffs, scale_factors1, scale_factors2, jacobian,Rotation)




