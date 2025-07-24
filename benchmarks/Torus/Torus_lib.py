import numpy as np
import pandas as pd
from scipy.optimize import root
import multiprocessing
import time
import meshio
from tqdm import tqdm
import numpy as np
import numpy as np
from scipy.integrate import quad
# ====== Torus Parameters ======
R_MAJOR = 1.0
r_MINOR = 0.3
n_major = 12
n_minor = 6
tol = 1e-4
max_depth = 6
# =============================

def torus_F(x, y, z, R=R_MAJOR, r=r_MINOR):
    rho = np.sqrt(x**2 + y**2)
    return (rho - R)**2 + z**2 - r**2

def torus_grad_F(x, y, z, R=R_MAJOR, r=r_MINOR):
    rho = np.sqrt(x**2 + y**2) + 1e-12
    dFdx = 2 * (rho - R) * (x / rho)
    dFdy = 2 * (rho - R) * (y / rho)
    dFdz = 2 * z
    return np.array([dFdx, dFdy, dFdz])

def normal_projection_to_torus(P0, R=R_MAJOR, r=r_MINOR, t0=0.0):
    def f_t(t):
        grad = torus_grad_F(*P0, R, r)
        n = grad / np.linalg.norm(grad)
        P = P0 + t * n
        return torus_F(P[0], P[1], P[2], R, r)
    sol = root(f_t, t0)
    t_star = sol.x[0]
    grad = torus_grad_F(*P0, R, r)
    n = grad / np.linalg.norm(grad)
    P_proj = P0 + t_star * n
    return P_proj

def tet_volume(pts):
    p0, p1, p2, p3 = pts
    return abs(np.dot(p1-p0, np.cross(p2-p0, p3-p0))) / 6

def patch_area_and_volume(V0, V1, V2, R=R_MAJOR, r=r_MINOR):
    area = np.linalg.norm(np.cross(V1-V0, V2-V0)) / 2
    P0 = normal_projection_to_torus(V0, R, r)
    P1 = normal_projection_to_torus(V1, R, r)
    P2 = normal_projection_to_torus(V2, R, r)
    tets = [
        [V0, V1, V2, P1],
        [P0, P1, P2, V0],
        [V0, V2, P1, P2]
    ]
    volume = sum(tet_volume(np.array(tet)) for tet in tets)
    return area, volume

def adaptive_patch_integrate(V0, V1, V2, R=R_MAJOR, r=r_MINOR, tol=1e-5, depth=0, max_depth=8):
    area, volume = patch_area_and_volume(V0, V1, V2, R, r)
    if area < 1e-14:
        return [(area, volume)]
    mid01 = (V0 + V1) / 2
    mid12 = (V1 + V2) / 2
    mid20 = (V2 + V0) / 2
    Pm01 = normal_projection_to_torus(mid01, R, r)
    Pm12 = normal_projection_to_torus(mid12, R, r)
    Pm20 = normal_projection_to_torus(mid20, R, r)
    flat_mids = [mid01, mid12, mid20]
    curved_mids = [Pm01, Pm12, Pm20]
    err = max(np.linalg.norm(cm - fm) for cm, fm in zip(curved_mids, flat_mids))
    if (area > tol or err > tol*5) and depth < max_depth:
        return (
            adaptive_patch_integrate(V0, mid01, mid20, R, r, tol, depth+1, max_depth) +
            adaptive_patch_integrate(mid01, V1, mid12, R, r, tol, depth+1, max_depth) +
            adaptive_patch_integrate(mid20, mid12, V2, R, r, tol, depth+1, max_depth) +
            adaptive_patch_integrate(mid01, mid12, mid20, R, r, tol, depth+1, max_depth)
        )
    else:
        return [(area, volume)]


def cylinder_angle(x, y, R):
    angle = np.arctan2(y, x)
    if angle < -1e-12:
        #print("###########angle################",angle)
        angle = angle + (2 * np.pi)
        #print("after###########angle################",angle)
    # Still handle edge case where angle == 2π due to floating point
    if np.isclose(angle, 2 * np.pi):
        angle = 0.0
    return angle


def extend_arc_point_to_z_torus(P1,r_1, P2,r_2, z_target, R):
    theta1 = cylinder_angle(P1[0], P1[1], r_1)
    theta2 = cylinder_angle(P2[0], P2[1],r_2)

    if abs(theta1 - 2*np.pi) < 1e-6 and theta2 < (np.pi / 2):
         theta1 = 0.0

    if abs(theta2 - 2*np.pi) < 1e-6 and theta1 < (np.pi / 2):
         theta2 = 0.0

    dz = P2[2] - P1[2]
    if abs(dz) < 1e-12:
        return np.array([R * np.cos(theta1), R * np.sin(theta1), z_target])
    t = (z_target - P1[2]) / dz
    theta = theta1 + t * (theta2 - theta1)
    x = R * np.cos(theta)
    y = R * np.sin(theta)

    return np.array([x, y, z_target])


def frustum_volume(rA, rB, h):
    """
    Computes the volume of a frustum (truncated cone) with
    bottom radius rA, top radius rB, and height h.
    All units in meters; returns volume in m^3.
    """
    return (np.pi * h / 3) * (rA**2 + rA * rB + rB**2)

 
 
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero vector!")
    return v / norm

def swept_segment_volume_AB(R, r, A, B):
    """
    Compute swept volumes for the major and minor segments formed by A and B
    on a circle of radius r, whose center is a distance R from the z-axis.
    Returns: (V_major, V_minor)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # Compute tube center on xy-plane, keep as 3D [x, y, 0]
    xy = np.array([A[0], A[1]])
    norm_xy = np.linalg.norm(xy)
    if norm_xy < 1e-12:
        raise ValueError("A is on the z axis, cannot determine circle center.")
    center = R * xy / norm_xy
    center = np.array([center[0], center[1], 0.0])  # Make 3D

    # Normal of the cross-section plane
    normal = np.cross(center, np.array([0,0,1]))
    if np.linalg.norm(normal) < 1e-12:
        normal = np.array([0,1,0])
    normal = normalize(normal)

    # In-plane axes
    ex = normalize(A - center)
    ey = np.cross(normal, ex)
    def angle_on_circle(P):
        v = (P - center) / r
        x, y = np.dot(v, ex), np.dot(v, ey)
        return np.arctan2(y, x)
    phiA = angle_on_circle(A)
    phiB = angle_on_circle(B)
    def centroid_and_area(phiA, phiB, use_major=False):
        dphi = np.abs(phiB - phiA)
        if dphi > np.pi:
            dphi = 2*np.pi - dphi
        if use_major:
            dphi = 2*np.pi - dphi
        if np.isclose(dphi, 0) or np.isclose(dphi, 2*np.pi):
            y_c = 0.0
            phi0 = (phiA + phiB) / 2
        else:
            y_c = 4*r*(np.sin(dphi/2))**3 / (3*(dphi - np.sin(dphi)))
            phi0 = (phiA + phiB) / 2
            if use_major:
                phi0 += np.pi
        local = y_c * np.cos(phi0) * ex + y_c * np.sin(phi0) * ey
        centroid = center + local
        area = 0.5 * r**2 * (dphi - np.sin(dphi))
        return centroid, area
    # Minor and major
    centroid_minor, area_minor = centroid_and_area(phiA, phiB, use_major=False)
    centroid_major, area_major = centroid_and_area(phiA, phiB, use_major=True)
    # Distance from z axis to centroid
    R_minor = np.linalg.norm([centroid_minor[0], centroid_minor[1]])
    R_major = np.linalg.norm([centroid_major[0], centroid_major[1]])
    V_minor = 2 * np.pi * R_minor * area_minor
    V_major = 2 * np.pi * R_major * area_major
    return V_minor


def find_rotated_point(B, r_B, theta_CA):
    """
    B: original point, array-like, shape (2,) or (3,)
    r_B: radius of the circle B is on
    theta_CA: angle to rotate B by (in radians)
    """
    theta_B = np.arctan2(B[1], B[0])   # get angle of B
    theta_D = theta_B + theta_CA       # rotate by theta_CA
    D_2d = np.array([r_B * np.cos(theta_D), r_B * np.sin(theta_D)])
    if len(B) == 3:
        D = np.array([D_2d[0], D_2d[1], B[2]])  # keep original z
    else:
        D = D_2d
    return D

def triangle_area(A, B, C):
    """
    Returns the area of triangle ABC given 3D coordinates.
    """
    AB = B - A
    AC = C - A
    cross = np.cross(AB, AC)
    area = 0.5 * np.linalg.norm(cross)
    return area


def torus_nearest_direction_orientation(A, B, C, R, r):
    """
    Returns:
      orientation: 'inward' or 'outward'
      G: centroid of triangle
      P_surf: nearest point on torus surface to centroid
      v_to_surface: vector from centroid to nearest surface (not normalized)
      normal_to_surface: unit vector from centroid to nearest surface
    """
    G = (A + B + C) / 3
    x, y, z = G
    theta = np.arctan2(y, x)
    cx, cy = R * np.cos(theta), R * np.sin(theta)
    dx, dy, dz = x - cx, y - cy, z
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    if d < 1e-12:
        px, py, pz = cx + r, cy, 0
    else:
        px = cx + r * dx / d
        py = cy + r * dy / d
        pz =      r * dz / d
    P_surf = np.array([px, py, pz])
    v_to_surface = P_surf - G
    norm = np.linalg.norm(v_to_surface)
    normal_to_surface = v_to_surface / (norm + 1e-14)  # unit vector
    v_to_axis = -G
    #orientation = "inward" if np.dot(v_to_surface, v_to_axis) > 0 else "outward"
    orientation = "inward" if np.dot(G[:2], normal_to_surface[:2]) < 0 else "outward"
    return orientation, G, P_surf, v_to_surface, normal_to_surface


def r_of_h(h, r_a, r_b, z):
    return r_a + (r_b - r_a) * h / z

def theta_of_h(h, theta_a, z):
    return theta_a * (1 - h / z)

def segment_volume_integrand(h, r_a, r_b, theta_a, z):
    r = r_a + (r_b - r_a) * h / z
    theta = theta_a * (1 - h / z)
    return 0.5 * r**2 * (theta - np.sin(theta))

def segment_volume_cone(theta, r_a, r_b, z):
    """
    Computes the volume generated by stacking circle segments from h=0 to h=z,
    where radius changes linearly from r_a to r_b, and segment angle is theta (radians).
    """
    def r_h(h):
        return r_a + (r_b - r_a) * (h / z)
    
    def segment_area(h):
        return 0.5 * r_h(h)**2 * (theta - np.sin(theta))
    
    volume, _ = quad(segment_area, 0, z)
    return volume


def curved_tet_volume(A, B, R, C, D, N=1000, verbose=False):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    theta_A = np.arctan2(A[1], A[0])
    theta_B = np.arctan2(B[1], B[0])
    dtheta = theta_B - theta_A 

    theta_vals = theta_A + np.linspace(0, dtheta, N)
    z_vals = np.linspace(A[2], B[2], N)
    arc_pts = np.stack([R * np.cos(theta_vals), R * np.sin(theta_vals), z_vals], axis=1)

    # Sum tiny curved tetrahedra (P_i, P_{i+1}, C, D)
    total = 0.0
    for i in range(N-1):
        P0 = arc_pts[i]
        P1 = arc_pts[i+1]
        v1 = P1 - P0
        v2 = C - P0
        v3 = D - P0
        tiny_vol = np.abs(np.dot(np.cross(v2, v3), v1)) / 6.0
        total += tiny_vol

    if verbose:
        print(f"Curved tet volume (arc AB): {total:.8f}")

    return total

def rotate_point_z_simple(B, r_B, theta_CE):
    """
    Rotate B (2D or 3D point) by theta_CE (radians) about the origin, keeping distance r_B.
    If B is 3D, keeps the same z as B.
    """
    B = np.array(B)
    alpha = np.arctan2(B[1], B[0])
    alpha_new = alpha + theta_CE
    if len(B) == 3:
        B_new = np.array([r_B * np.cos(alpha_new), r_B * np.sin(alpha_new), B[2]])
    else:
        B_new = np.array([r_B * np.cos(alpha_new), r_B * np.sin(alpha_new)])
    return B_new


def pyramid_volume(A, B, C, D, E):
    # Area of ABC
    area_ABC = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    # Area of CDA
    area_CDA = 0.5 * np.linalg.norm(np.cross(D - C, A - C))
    S_ABCD = area_ABC + area_CDA

    # Normal of the base
    n = np.cross(B - A, C - A)
    n_norm = np.linalg.norm(n)
    a, b, c = n
    # Height from E to plane of base
    h = np.abs(np.dot(n, E - A)) / n_norm

    V = (1/3) * S_ABCD * h
    return V
 

def curved_arc_tet_volume_ABCE(A, B, C, E, r_E, theta_CE, n_steps=1000):
    """
    Computes the volume of a 'curved wedge' ABCE, where:
      - Only edge CE is an arc of a circle (centered at (0,0,E[2]), radius r_E, angle theta_CE)
      - Other edges (AB, AE, BC, BE, AC) are straight
    A, B, C, E: 3D points
    r_E: radius of circle for arc CE (should match |C-center|, |E-center|)
    theta_CE: angle (in radians) from E to C around the circle (direction matters)
    n_steps: integration steps (higher = more accurate)
    Returns: float, the curved-wedge volume
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    E = np.array(E)
    z0 = E[2]
    center = np.array([0, 0, z0])

    # Get angle for E (start) and arc direction
    theta_E = np.arctan2(E[1] - center[1], E[0] - center[0])

    def arc_point(t):
        theta = theta_E + t * theta_CE
        x = center[0] + r_E * np.cos(theta)
        y = center[1] + r_E * np.sin(theta)
        z = z0
        return np.array([x, y, z])

    def tet_vol(P0, P1, Q, R_):
        v1 = P1 - P0
        v2 = Q - P0
        v3 = R_ - P0
        return abs(np.dot(v1, np.cross(v2, v3))) / 6

    vol = 0.0
    for i in range(n_steps):
        t0 = i / n_steps
        t1 = (i + 1) / n_steps
        Q0 = arc_point(t0)
        Q1 = arc_point(t1)
        # Tetrahedron: (A, B, Q0, Q1)
        v = tet_vol(A, B, Q0, Q1)
        vol += v
    return vol


def cone_volume_patch_inside(A, B, C,R,r,k, triangle_id=None):
    EPS = 1e-10

    r_A = np.linalg.norm(A[:2])
    r_B = np.linalg.norm(B[:2])
    r_C = np.linalg.norm(C[:2])

    O = np.array([0.0, 0.0, (B[2] + A[2]) / 2])
    h_full = abs(B[2] - A[2])
    frustum_volume_ABC = frustum_volume(r_A, r_B, h_full)
    V_t = frustum_volume_ABC 

    theta_A = cylinder_angle(A[0], A[1], r_A)
    theta_B = cylinder_angle(B[0], B[1], r_B)
    theta_C = cylinder_angle(C[0], C[1], r_C)
    original_theta_A = np.arctan2(A[1], A[0])
    original_theta_B = np.arctan2(B[1], B[0])
    original_theta_C = np.arctan2(C[1], C[0])
    original_theta_CB = (original_theta_C - original_theta_B)    
    original_theta_CA = (original_theta_C - original_theta_A)

    if abs(C[2] - A[2]) < EPS: 
        theta_CA = (theta_C - theta_A) % (2 * np.pi)
        D = find_rotated_point(B, r_B, -original_theta_CA)# -theta_CA
        if theta_CA < 1e-8:
            return 0.0
        if theta_CA >= np.pi:
            theta_CA = 2 * np.pi - theta_CA        
        triangle_area_COA = 0.5 * r_A ** 2 * np.sin(theta_CA)
        triangle_area_BOD = 0.5 * r_B ** 2 * np.sin(theta_CA)
        splittimes = 2 * np.pi / (theta_CA + 1e-12)
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6

        Vpatch = (swept_segment_volume_AB(R, r, C, B) - (V_t# + swept_segment_volume_AB(R, r, C, B) 
                   - 
                    V_OABC * splittimes - 
                    V_OABD * splittimes -
                    triangle_area_BOD* splittimes * h_full / 3 * 1/2 -
                    triangle_area_COA* splittimes * h_full / 3 * 1/2
                    ) )/ splittimes *(r_A**k/(r_A**k + r_B**k)) 

    if abs(C[2] - B[2]) < EPS: 
        theta_CB = (theta_C - theta_B) % (2 * np.pi)
        if theta_CB < 1e-8:
            return 0.0
        if theta_CB >= np.pi:
            theta_CB = 2 * np.pi - theta_CB
        D = find_rotated_point(A, r_A,-original_theta_CB ) #  -theta_CB
        
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6


        if theta_CB < 1e-12:
            return 0.0
        if theta_CB >= np.pi:
            theta_CB = 2 * np.pi - theta_CB
        triangle_area_COB = 0.5 * r_B**2 * np.sin(theta_CB)
        triangle_area_DOA = 0.5 * r_A**2 * np.sin(theta_CB)
        splittimes = 2 * np.pi / (theta_CB + 1e-12)
        # Combine for patch
        Vpatch = (swept_segment_volume_AB(R, r, A, C) - (
            V_t #+ swept_segment_volume_AB(R, r, A, B)
            - V_OABC * splittimes
            - V_OABD * splittimes
            - triangle_area_COB * splittimes * h_full / 3 * 0.5
            - triangle_area_DOA * splittimes * h_full / 3 * 0.5
        )) / splittimes *(r_B**k / (r_B**k + r_A**k)) 

    
    return Vpatch



def cone_volume_patch(A, B, C,R,r,k, triangle_id=None):
    #swept_segment_volume_ABC= swept_segment_volume_AB(R, r, C, B)
    EPS = 1e-10

    r_A = np.linalg.norm(A[:2])
    r_B = np.linalg.norm(B[:2])
    r_C = np.linalg.norm(C[:2])


    O = np.array([0.0, 0.0, (B[2] + A[2]) / 2])
    h_full = abs(B[2] - A[2])
    frustum_volume_ABC = frustum_volume(r_A, r_B, h_full)
    V_t = frustum_volume_ABC 
    theta_A = cylinder_angle(A[0], A[1], r_A)
    theta_B = cylinder_angle(B[0], B[1], r_B)
    theta_C = cylinder_angle(C[0], C[1], r_C)
    original_theta_A = np.arctan2(A[1], A[0])
    original_theta_B = np.arctan2(B[1], B[0])
    original_theta_C = np.arctan2(C[1], C[0])
    original_theta_CB = (original_theta_C - original_theta_B)    
    original_theta_CA = (original_theta_C - original_theta_A)

    if abs(C[2] - A[2]) < EPS: 
        theta_CA = (theta_C - theta_A) % (2 * np.pi)
        D = find_rotated_point(B, r_B, -original_theta_CA) # -theta_CA
        if theta_CA < 1e-8:
            return 0.0
        if theta_CA >= np.pi:
            theta_CA = 2 * np.pi - theta_CA        
        triangle_area_COA = 0.5 * r_A ** 2 * np.sin(theta_CA)
        triangle_area_BOD = 0.5 * r_B ** 2 * np.sin(theta_CA)
        splittimes = 2 * np.pi / (theta_CA + 1e-12)
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6


        Vpatch = (V_t + swept_segment_volume_AB(R, r, C, B)  - 
                    V_OABC * splittimes - 
                    V_OABD * splittimes -
                    triangle_area_BOD* splittimes * h_full / 3 * 1/2 -
                    triangle_area_COA* splittimes * h_full / 3 * 1/2
                    ) / splittimes * (r_A**k/(r_A**k + r_B**k)) 

    if abs(C[2] - B[2]) < EPS: 
        theta_CB = (theta_C - theta_B) % (2 * np.pi)
        if theta_CB < 1e-8:
            return 0.0
        if theta_CB >= np.pi:
            theta_CB = 2 * np.pi - theta_CB
        D = find_rotated_point(A, r_A, -original_theta_CB) #-theta_CB
        
        V_OABC = abs(np.linalg.det(np.vstack([A - O, B - O, C - O]))) / 6
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6


        if theta_CB < 1e-12:
            return 0.0
        if theta_CB >= np.pi:
            theta_CB = 2 * np.pi - theta_CB
        triangle_area_COB = 0.5 * r_B**2 * np.sin(theta_CB)
        triangle_area_DOA = 0.5 * r_A**2 * np.sin(theta_CB)
        splittimes = 2 * np.pi / (theta_CB + 1e-12)
        # Combine for patch
        Vpatch = (
            V_t + swept_segment_volume_AB(R, r, A, B)
            - V_OABC * splittimes
            - V_OABD * splittimes
            - triangle_area_COB * splittimes * h_full / 3 * 0.5
            - triangle_area_DOA * splittimes * h_full / 3 * 0.5
        ) / splittimes  *  (r_B**k / (r_B**k + r_A**k)) 


    #print("triangle*",triangle_id,"A", A,"B", B,"C",C,"D", D)
    
    return Vpatch

def classify_triangle_side(A, B, C, R):
    rA = np.sqrt(A[0]**2 + A[1]**2)
    rB = np.sqrt(B[0]**2 + B[1]**2)
    rC = np.sqrt(C[0]**2 + C[1]**2)
    if rA > R and rB > R and rC > R:
        return "outside"
    elif rA < R and rB < R and rC < R:
        return "inside"
    else:
        return "between"


def wedge_volume_geodesic_ABCD(A, B, C, R,r,triangle_id=None):
    k=1
    zvals = [A[2], B[2], C[2]]
    idxs = np.argsort(zvals)
    top = [A, B, C][idxs[2]]
    mid = [A, B, C][idxs[1]]
    bot = [A, B, C][idxs[0]]
    A, B, C = top, bot, mid
    r_A = np.linalg.norm(A[:2])
    r_B = np.linalg.norm(B[:2])
    r_C = np.linalg.norm(C[:2])   
    r_E = r_C    
    orientation, G, P_surf, v_to_surface, normal_to_surface = torus_nearest_direction_orientation(A, B, C, R_MAJOR, r_MINOR)

    EPS = 1e-10
    side = classify_triangle_side(A, B, C, R_MAJOR)

    # Special case: C at same z as A -> use solid angle patch on mapped sphere!
    if abs(C[2] - A[2]) < EPS and side != "between":
        
        r_A = np.linalg.norm(A[:2])  # sqrt(x^2 + y^2)
        r_B = np.linalg.norm(B[:2])       
        #print(f"r_A: {r_A}, r_B: {r_B}")
        theta_A = cylinder_angle(A[0], A[1],r_A)
        theta_B = cylinder_angle(B[0], B[1],r_B)
        theta_C = cylinder_angle(C[0], C[1],r_A)


        #print("**theta_CA",(theta_C - theta_A))
        #theta_CA = (theta_C - theta_A)
        
        #print(f"theta_A: {theta_A}, theta_C: {theta_C}")
        theta_CA = (theta_C - theta_A) % (2 * np.pi)

        original_theta_A = np.arctan2(A[1], A[0])
        original_theta_C = np.arctan2(C[1], C[0])
        original_theta_B = np.arctan2(B[1], B[0])
        original_theta_CA = (original_theta_A - original_theta_C)
        if original_theta_CA >= np.pi:
            original_theta_CA = 2 * np.pi - original_theta_CA 

        #print("original_theta_CA",original_theta_CA)

        D = find_rotated_point(B, r_B,-(original_theta_C - original_theta_A)) #-theta_CA
        theta_D = cylinder_angle(D[0], D[1],r_A)
        original_theta_D = np.arctan2(D[1], D[0])
        
        original_theta_AD = (original_theta_D - original_theta_A) % (2 * np.pi)
        if original_theta_AD >= np.pi:
            original_theta_AD = 2 * np.pi - original_theta_AD 
       #print("original_theta_A",original_theta_A,"original_theta_B",original_theta_B,"original_theta_C",original_theta_C,"original_theta_D",original_theta_D)

        if original_theta_AD > 1.5*original_theta_CA: 
            #print("triangle_id****", triangle_id)
            #D = find_rotated_point(B, r_B, original_theta_CA)
            theta_D = cylinder_angle(D[0], D[1],r_A)
            original_theta_D = np.arctan2(D[1], D[0])



        if theta_CA < 1e-8:
            return 0.0
        if theta_CA >= np.pi:
            theta_CA = 2 * np.pi - theta_CA
        triangle_area_COA = 0.5 * r_A ** 2 * np.sin(theta_CA)
        triangle_area_BOD = 0.5 * r_B ** 2 * np.sin(theta_CA)
        splittimes = 2 * np.pi / (theta_CA + 1e-12)
        h_full = abs(C[2] - B[2])
        frustum_volume_ABC = frustum_volume(r_A, r_B, h_full)
        swept_segment_volume_ABC= swept_segment_volume_AB(R, r, C, B)

        ratio = r_A / r_B

        V_t = frustum_volume_ABC + swept_segment_volume_ABC # h_full * np.pi * R ** 2
        O = np.array([0.0, 0.0, (B[2] + A[2]) / 2])
        mat = np.vstack([A - O, B - O, C - O])
        V_OABC= abs(np.linalg.det(mat)) / 6
        #k=1
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6
        ratio_ABC = V_OABC/(V_OABC + V_OABD)
        ratio_ABD = V_OABD/(V_OABC + V_OABD)

        Vpatch = cone_volume_patch(A, B, C,R,r,k, triangle_id) 


        return 1, Vpatch

    # Special case: C at same z as B (torus version)
    if abs(C[2] - B[2]) < EPS and side != "between":
        r_B = np.linalg.norm(B[:2])
        r_A = np.linalg.norm(A[:2])
        theta_B = cylinder_angle(B[0], B[1], r_B)
        
        theta_C = cylinder_angle(C[0], C[1], r_B)
      
        theta_CB = (theta_C - theta_B) % (2 * np.pi)
        if theta_CB < 1e-12:
            return 0.0
        if theta_CB >= np.pi:
            theta_CB = 2 * np.pi - theta_CB
        #if theta_C -theta_B < 1e-12:


        triangle_area_COB = 0.5 * r_B**2 * np.sin(theta_CB)
        triangle_area_DOA = 0.5 * r_A**2 * np.sin(theta_CB)
        splittimes = 2 * np.pi / (theta_CB + 1e-12)
        h_full = abs(A[2] - B[2])
        
        # Core volumes:
        frustum_volume_ABC = frustum_volume(r_B, r_A, h_full)
        swept_segment_volume_ABC = swept_segment_volume_AB(R, r, A, B)
        V_t = frustum_volume_ABC + swept_segment_volume_ABC

        O = np.array([0.0, 0.0, (B[2] + A[2]) / 2])
        mat = np.vstack([A - O, B - O, C - O])
        V_OABC = abs(np.linalg.det(mat)) / 6

        original_theta_B = np.arctan2(B[1], B[0])
        original_theta_C = np.arctan2(C[1], C[0])
        original_theta_CB = (original_theta_C - original_theta_B)
        #print("original_theta_CB",original_theta_CB)

        D = find_rotated_point(A, r_A, -original_theta_CB) #-theta_CB
        
        theta_D = cylinder_angle(D[0], D[1], r_A)
        theta_A = cylinder_angle(A[0], A[1],r_A)
        theta_AD = (theta_D - theta_A) % (2 * np.pi)
        if theta_AD < 1e-12:
            return 0.0
        if theta_AD >= np.pi:
            theta_AD = 2 * np.pi - theta_AD

        original_theta_A = np.arctan2(A[1], A[0])
        original_theta_D = np.arctan2(D[1], D[0])
        original_theta_AD = (original_theta_D - original_theta_A)

        
        V_OABD = abs(np.linalg.det(np.vstack([B - O, D - O, A - O]))) / 6
        ratio_ABC = V_OABC/(V_OABC + V_OABD)
        ratio_ABD = V_OABD/(V_OABC + V_OABD)

        #k=1
        # Combine for patch
        Vpatch = (
            V_t
            - V_OABC * splittimes
            - V_OABD * splittimes
            - triangle_area_COB * splittimes * h_full / 3 * 0.5
            - triangle_area_DOA * splittimes * h_full / 3 * 0.5
        ) / splittimes * ratio_ABC# (r_B**k / (r_B**k + r_A**k))  # (you can adjust weighting if needed)


        Vpatch = cone_volume_patch(A, B, C,R,r,k, triangle_id) 

        return 2,Vpatch


    # GENERAL CASE: C is at neither same z as A nor B
    E = extend_arc_point_to_z_torus(A, r_A, B, r_B, C[2], r_C)
    
    if side == "between":
        pts = [A, B, C]
        dists = [np.linalg.norm(p[:2]) for p in pts]
        idx_furthest = np.argmax(dists)
        idx_nearest = np.argmin(dists)
        # C is the index not in furthest or nearest
        idx_remaining = ({0,1,2} - {idx_furthest, idx_nearest}).pop()
        A, B, C = pts[idx_furthest], pts[idx_nearest], pts[idx_remaining]
        r_C = np.linalg.norm(C[:2])
        r_B = np.linalg.norm(B[:2])
        r_A = np.linalg.norm(A[:2])        
        E = extend_arc_point_to_z_torus(A, r_A, B, r_B, C[2], r_C) # AEC outside and ECB inside
        V_ABCE = abs(np.linalg.det(np.vstack([C - E, A - E, B - E]))) / 6
        #print("triangle_id", triangle_id, "side", side, "A", A, "B", B, "C", C,"E", E)
 
        theta_A = cylinder_angle(A[0], A[1], r_A)
        theta_B = cylinder_angle(B[0], B[1], r_B)
        theta_C = cylinder_angle(C[0], C[1], r_C)
        theta_E = cylinder_angle(E[0], E[1], r_E)
        #print("theta_C", theta_C, "theta_E", theta_E)
        theta_CE = (theta_E - theta_C) % (2 * np.pi)
        if theta_CE >= np.pi:
            theta_CE = 2 * np.pi - theta_CE

        splittimes_CE = 2 * np.pi / theta_CE

        #return 4,  Vpatch_in + Vpatch_out + V_ABCE
        return 4, (swept_segment_volume_AB(R, r, A, E) * (r_C**k )/(r_A**k + r_C**k ) +
                    swept_segment_volume_AB(R, r, B, E)* (r_C**k )/(r_B**k + r_C**k ) )/splittimes_CE + V_ABCE 
        #
        #side = classify_triangle_side(A, C, E, R_MAJOR)

    theta_C = cylinder_angle(C[0], C[1], r_C)
    theta_E = cylinder_angle(E[0], E[1], r_E)
    #print("theta_C", theta_C, "theta_E", theta_E)
    theta_CE = (theta_E - theta_C) % (2 * np.pi)
    if theta_CE >= np.pi:
        theta_CE = 2 * np.pi - theta_CE

    splittimes1 = 2 * np.pi / theta_CE
    h_full1 = abs(C[2] - A[2])
    h_full2 = abs(C[2] - B[2])
    h_full = abs(B[2] - A[2])
    if orientation == "inward": 
        swept_segment_volume_AC = swept_segment_volume_AB(R, r, A, C)/splittimes1#* (r_C**2/(r_A**2 + r_C**2)) 
        V_ABCE = abs(np.linalg.det(np.vstack([C - E, A - E, B - E]))) / 6

        Vpatch = ( 
                   cone_volume_patch_inside(A, E, C,R,r,k, triangle_id) +
                       cone_volume_patch_inside(E, B, C,R,r,k, triangle_id)) + min(curved_tet_volume(E, C, r_C, A, B), V_ABCE) 
        
        return 5, Vpatch

        #print(triangle_id, "Inward orientation detected for C at same z as A")


    # Core volumes:
    frustum_volume_AEC = frustum_volume(r_E, r_A, h_full1)
    swept_segment_volume_AC = swept_segment_volume_AB(R, r, A, C)
    V_t1 = swept_segment_volume_AC + frustum_volume_AEC
    #if orientation == "inward": V_t1 =frustum_volume_AEC
    
    O1 = np.array([0.0, 0.0, (C[2] + A[2]) / 2])
    mat1 = np.vstack([A - O1, E - O1, C - O1])
    V_O1AEC = abs(np.linalg.det(mat1)) / 6

    D = find_rotated_point(A, r_A, -theta_CE)
    #D = find_rotated_point(B, r_B, -theta_CA)
    #print("E",E,"A",A,"B",B,"C",C,"D",D,"O1",O1)
    V_O1ACD = abs(np.linalg.det(np.vstack([C - O1, D - O1, A - O1]))) / 6
    
    triangle_area_COE = 0.5 * r_C**2 * np.sin(theta_CE)
    triangle_area_DOA = 0.5 * r_A**2 * np.sin(theta_CE)
    #k=1

    Vpart1 =abs(V_t1
                - V_O1AEC * splittimes1
                - V_O1ACD * splittimes1 
                - triangle_area_COE * splittimes1 * h_full1 * 0.5 / 3
                - triangle_area_DOA * splittimes1 * h_full1 * 0.5 / 3           
                )/ splittimes1 * (r_C**k / (r_C**k + r_A**k))
    
    # print("##Vpart1",Vpart1,"V_t1",V_t1,"V_O1AEC",V_O1AEC,"V_O1ACD",V_O1ACD,
    #       "triangle_area_COE", triangle_area_COE, "triangle_area_DOA", triangle_area_DOA,
    #       "splittimes1", splittimes1, "h_full1", h_full1,(r_C / (r_C + r_A)))

    #triangle_area_COE = 0.5 * r_C**2 * np.sin(theta_CE)
    splittimes2 = 2 * np.pi / theta_CE
    h_full2 = abs(C[2] - B[2])

    # Core volumes:
    frustum_volume_BEC = frustum_volume(r_C, r_B, h_full2)
    swept_segment_volume_BC = swept_segment_volume_AB(R, r, B, C)
    V_t2 = frustum_volume_BEC + swept_segment_volume_BC

    ratio_AEC = frustum_volume_AEC/(frustum_volume_BEC+frustum_volume_AEC)
    ratio_BEC = frustum_volume_BEC/(frustum_volume_BEC+frustum_volume_AEC)
    #print("ratio_AEC", ratio_AEC, "ratio_BEC", ratio_BEC)
    #if orientation == "inward": V_t2 = frustum_volume_BEC


    O2 = np.array([0.0, 0.0, (B[2] + C[2]) / 2])
    mat2 = np.vstack([E - O2, B - O2, C - O2])
    V_O2CBE = abs(np.linalg.det(mat2)) / 6

    D2 = find_rotated_point(B, r_B, -theta_CE)
    V_O2BCD2 = abs(np.linalg.det(np.vstack([C - O2, D2 - O2, B - O2]))) / 6

    triangle_area_D2OB = 0.5 * r_B**2 * np.sin(theta_CE)

    Vpart2 =abs(V_t2  
                - V_O2CBE * splittimes2 
                - V_O2BCD2 * splittimes2 
                - triangle_area_COE * splittimes2 * h_full2 / 3 * 0.5
                - triangle_area_D2OB * splittimes2 * h_full2 / 3 * 0.5               
                )/ splittimes2 * (r_C**k / (r_C**k + r_B**k))
    
    
    mat3 = np.vstack([A - E, B - E, C- E])
    Vpart3 = abs(np.linalg.det(mat3)) / 6      
    Vpatch =  cone_volume_patch(A, E, C,R,r,k, triangle_id) + cone_volume_patch(E, B,C ,R,r,k, triangle_id) + Vpart3

    return 3, Vpatch

def adaptive_triangle_worker(args):
    idx, tri, points, tol, max_depth = args
    A, B, C = points[tri]
    result = adaptive_patch_integrate(A, B, C, tol=tol, max_depth=max_depth)
    total_area = sum(x[0] for x in result)
    total_volume = sum(x[1] for x in result)
    V_flat = abs(np.dot(A, np.cross(B, C))) / 6
    flat_area = np.linalg.norm(np.cross(B - A, C - A)) / 2
    Vcorrection = wedge_volume_geodesic_ABCD(A, B, C, R_MAJOR, r_MINOR, idx)
    side = classify_triangle_side(A, B, C, R_MAJOR)


    # Compute pointdirection (inward/outward) using centroid
    pointdirection, _, _, _,_ = torus_nearest_direction_orientation(A, B, C, R_MAJOR, r_MINOR)
    row = {
        'triangle_id': idx,
        'curved_patch_volume': total_volume,
        'Vcorrection': Vcorrection,
        'flat_volume': V_flat,
        'correction': total_volume - V_flat,
        'patch_area': total_area,
        'flat_area': flat_area,
        'surface_triangle': True,
        'pointdirection': pointdirection,  # <-- new column
        'side': side   
    }
    return row, total_volume

def torus_theoretical_volume(R, r):
    return 2 * np.pi**2 * R * r**2

def extract_surface_triangles(tets):
    from collections import defaultdict
    face_count = defaultdict(int)
    faces = []
    for tet in tets:
        faces_in_tet = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]
        for face in faces_in_tet:
            face_count[face] += 1
    surface_faces = [list(face) for face, count in face_count.items() if count == 1]
    return np.array(surface_faces, dtype=int)

if __name__ == "__main__":
    mesh_filename = "torus_tet_coarse.msh"
    mesh = meshio.read(mesh_filename)
    points = mesh.points
    cells = mesh.cells_dict
    faces = cells["triangle"] if "triangle" in cells else []
    tet_cells = cells["tetra"] if "tetra" in cells else []

    print(f"Loaded mesh: {mesh_filename}")
    print(f"Number of surface triangles: {len(faces)}")
    print(f"Number of tetrahedra: {len(tet_cells)}")

    # --- Piecewise linear (PL) volume over all tetrahedra ---
    V_piecewise = 0.0
    for tet in tet_cells:
        pts4 = points[tet]
        V_piecewise += tet_volume(pts4)

    # --- Patch/curved volume ---
    V_theory = torus_theoretical_volume(R_MAJOR, r_MINOR)
    print(f"Torus theoretical volume: {V_theory:.8f}")


    args_list = [(idx, tri, points, tol, max_depth) for idx, tri in enumerate(faces)]
    print(f"Using {multiprocessing.cpu_count()} CPU cores...")

    t0 = time.time()
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(adaptive_triangle_worker, args_list), total=len(args_list)))
    csv_rows = []
    V_patch_sum = 0.0
    for row, volume in results:
        csv_rows.append(row)
        V_patch_sum += volume
    df = pd.DataFrame(csv_rows)
    df.to_csv("adaptive_triangle_patch_correction.csv", index=False)
    print("\nWrote triangle patch corrections to adaptive_triangle_patch_correction.csv")
    t1 = time.time()

    rel_error_patchsum = (V_patch_sum + V_piecewise - V_theory) / V_theory * 100
    rel_error_piecewise = (V_piecewise - V_theory) / V_theory * 100
    V_patchsum_solidangle = sum(
        row['Vcorrection'][1] if isinstance(row['Vcorrection'], tuple) else row['Vcorrection']
        for row, _ in results
    )
    rel_error_patchsum_solidangle = (V_patchsum_solidangle + V_piecewise - V_theory) / V_theory * 100

    print(f"\n==== VOLUME SUMMARY ====")
    print(f"Theoretical value (torus):            {V_theory:.8f}")
    print(f"Piecewise sum of all tets:            {V_piecewise:.8f}")    
    print(f"Patch-based total (curved patch sum): {V_patch_sum:.8f}")

    print(f"Number of torus surface triangles:    {len(faces)}")
    print(f"Relative error (curved patch sum):    {rel_error_patchsum:.6f}%")
    print(f"Relative error (piecewise tets):      {rel_error_piecewise:.6f}%")
    print(f"Elapsed time (patch sum):             {t1-t0:.2f} s")

    print(f"\n==== VOLUME SUMMARY Solid Angle ====")
    print(f"Theoretical value (torus):            {V_theory:.8f}")
    print(f"Piecewise sum of all tets:            {V_piecewise:.8f}")
    print(f"Patch-based total (solid angle sum):  {V_patchsum_solidangle:.8f}")
    print(f"Number of torus surface triangles:    {len(faces)}")
    print(f"Relative error (piecewise tets):      {rel_error_piecewise:.6f}%")
    print(f"Relative error (solid angle sum):     {rel_error_patchsum_solidangle:.6f}%")
    print(f"Elapsed time (patch sum):             {t1-t0:.2f} s")
