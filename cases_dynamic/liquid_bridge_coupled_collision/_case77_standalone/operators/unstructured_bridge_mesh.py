"""Unstructured tetra meshes for a sphere contacting a thin liquid film.

The liquid-air bridge is represented by an arc-length meridian.  It may turn
inward through a neck before joining the outer film, so it is not restricted to
the graph ``z=h(r)``.  The meridian is revolved and tetrahedralized with Gmsh;
all boundary sets are then recovered from tetra connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from scipy.special import i0


@dataclass(frozen=True)
class YoungLaplaceMeridian:
    r_m: np.ndarray
    z_m: np.ndarray
    tangent_angle_rad: np.ndarray
    pressure_jump_pa: float
    arc_length_m: float
    contact_radius_m: float
    rim_radius_m: float
    neck_radius_m: float


@dataclass(frozen=True)
class UnstructuredBridgeMesh:
    points_m: np.ndarray
    tets: np.ndarray
    boundary_faces: np.ndarray
    free_surface_faces: np.ndarray
    sphere_faces: np.ndarray
    substrate_faces: np.ndarray
    outer_wall_faces: np.ndarray
    contact_line_vertices: np.ndarray
    sphere_vertices: np.ndarray
    substrate_vertices: np.ndarray
    outer_wall_vertices: np.ndarray
    outer_film_vertices: np.ndarray
    bridge_surface_vertices: np.ndarray
    rim_vertices: np.ndarray
    contact_radius_m: float
    rim_radius_m: float
    pressure_jump_pa: float
    seed_neck_resolution_m: float
    # ``None`` denotes the historical full 2*pi revolution.  A finite angle
    # denotes an opt-in computational wedge whose two meridian copies are
    # tied by the exact-axisymmetric Stokes reduction.  The scale converts
    # extensive wedge geometry (for example tetra volume) to its equivalent
    # full-circle value; it must not be applied to the assembled wedge
    # equations because every term in those equations already has the same
    # angular measure.
    axisymmetric_wedge_angle_rad: float | None = None
    axisymmetric_wedge_midpoint_angle_rad: float = 0.0
    axisymmetric_full_circle_scale: float = 1.0
    axisymmetric_wedge_side_faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=int)
    )


def sphere_lower_z(
    radius_m: np.ndarray | float,
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
) -> np.ndarray | float:
    radius = np.asarray(radius_m, dtype=float)
    sphere_radius = float(sphere_radius_m)
    clipped = np.minimum(radius, sphere_radius * (1.0 - 1.0e-12))
    height = float(sphere_tip_z_m) + sphere_radius - np.sqrt(
        np.maximum(sphere_radius * sphere_radius - clipped * clipped, 0.0)
    )
    if np.isscalar(radius_m):
        return float(height)
    return height


def gravity_film_height(
    radius_m: np.ndarray,
    *,
    center_height_m: float,
    substrate_radius_m: float,
    capillary_length_m: float,
    edge_height_m: float = 0.0,
) -> np.ndarray:
    """Siekman gravity-shaped film, with an optional meshing edge floor."""

    radius = np.asarray(radius_m, dtype=float)
    edge_i0 = float(i0(float(substrate_radius_m) / float(capillary_length_m)))
    normalized = (
        edge_i0 - i0(radius / float(capillary_length_m))
    ) / max(edge_i0 - 1.0, 1.0e-30)
    height = float(edge_height_m) + (
        float(center_height_m) - float(edge_height_m)
    ) * normalized
    return np.maximum(height, float(edge_height_m))


def solve_zero_angle_sphere_film_meridian(
    *,
    contact_radius_m: float,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    film_junction_z_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    samples: int = 48,
    tolerance: float = 2.0e-5,
) -> YoungLaplaceMeridian:
    """Solve a local axisymmetric Young-Laplace bridge by arc length.

    The solid and outer film are completely wetting.  Starting at the sphere,
    the interface is tangent to the lower spherical surface with the opposite
    orientation; at the film junction it is horizontal.  The pressure jump and
    total arc length are solved parameters, not prescribed profile constants.
    """

    contact_radius = float(contact_radius_m)
    sphere_radius = float(sphere_radius_m)
    if not 0.0 < contact_radius < sphere_radius:
        raise ValueError("contact_radius_m must lie inside the sphere radius")
    contact_z = float(
        sphere_lower_z(
            contact_radius,
            sphere_radius_m=sphere_radius,
            sphere_tip_z_m=float(sphere_tip_z_m),
        )
    )
    axial = math.sqrt(max(sphere_radius * sphere_radius - contact_radius * contact_radius, 1.0e-30))
    solid_angle = math.atan2(contact_radius, axial)
    contact_tangent = solid_angle - math.pi

    xi = np.linspace(0.0, 1.0, max(120, int(samples) * 4))
    radial_guess = contact_radius + max(0.02 * contact_radius, 2.0e-6) * xi
    height_guess = contact_z + (float(film_junction_z_m) - contact_z) * xi
    tangent_guess = contact_tangent * (1.0 - xi)
    guess = np.vstack((radial_guess, height_guess, tangent_guess))
    gamma = float(surface_tension_n_m)
    rho_g = float(density_kg_m3) * float(gravity_m_s2)

    def ode(_xi: np.ndarray, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        pressure_jump = float(params[0])
        arc_length = math.exp(float(np.clip(params[1], -40.0, 10.0)))
        radius = np.maximum(state[0], 1.0e-12)
        height = state[1]
        tangent = state[2]
        curvature = (
            pressure_jump + rho_g * (height - float(film_junction_z_m))
        ) / max(gamma, 1.0e-30)
        return np.vstack(
            (
                arc_length * np.cos(tangent),
                arc_length * np.sin(tangent),
                arc_length * (curvature - np.sin(tangent) / radius),
            )
        )

    def boundary(left: np.ndarray, right: np.ndarray, _params: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                left[0] - contact_radius,
                left[1] - contact_z,
                left[2] - contact_tangent,
                right[1] - float(film_junction_z_m),
                right[2],
            ),
            dtype=float,
        )

    pressure_guess = gamma / max(contact_radius, 1.0e-12)
    arc_guess = max(0.15 * contact_radius, abs(contact_z - float(film_junction_z_m)))
    result = solve_bvp(
        ode,
        boundary,
        xi,
        guess,
        p=np.asarray((pressure_guess, math.log(max(arc_guess, 1.0e-18))), dtype=float),
        tol=float(tolerance),
        max_nodes=30000,
    )
    if not bool(result.success):
        raise RuntimeError(f"Young-Laplace bridge solve failed: {result.message}")
    query = np.linspace(0.0, 1.0, max(12, int(samples)))
    solved = np.asarray(result.sol(query), dtype=float)
    return YoungLaplaceMeridian(
        r_m=solved[0],
        z_m=solved[1],
        tangent_angle_rad=solved[2],
        pressure_jump_pa=float(result.p[0]),
        arc_length_m=math.exp(float(result.p[1])),
        contact_radius_m=contact_radius,
        rim_radius_m=float(solved[0, -1]),
        neck_radius_m=float(np.min(solved[0])),
    )


def solve_sphere_film_meridian_for_contact_angle_and_tangent(
    *,
    contact_radius_m: float,
    contact_angle_rad: float,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    film_junction_z_m: float,
    film_junction_tangent_rad: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    samples: int = 64,
    tolerance: float = 2.0e-5,
    previous: YoungLaplaceMeridian | None = None,
) -> YoungLaplaceMeridian:
    """Solve the local Young-Laplace bridge from its two wetting angles.

    The sphere contact radius and the apparent dynamic contact angle are
    supplied by the contact-line kinematics.  The transported outer film
    supplies the tangent at ``z=h0``.  Pressure jump, arc length, bridge
    volume, and footprint radius are outputs.  This is the natural endpoint
    problem for coupling a PR37 contact-line step to the tetrahedral bridge
    volume flux: it does not prescribe an experimental radius or volume.
    """

    contact = float(contact_radius_m)
    sphere = float(sphere_radius_m)
    angle = float(contact_angle_rad)
    rim_tangent = float(film_junction_tangent_rad)
    gamma = float(surface_tension_n_m)
    rho_g = float(density_kg_m3) * float(gravity_m_s2)
    if not 0.0 < contact < sphere:
        raise ValueError("contact_radius_m must lie inside the sphere radius")
    if not np.isfinite(angle) or not 0.0 <= angle < math.pi:
        raise ValueError("contact_angle_rad must lie in [0, pi)")
    if not np.isfinite(rim_tangent) or abs(rim_tangent) >= 0.5 * math.pi:
        raise ValueError(
            "film_junction_tangent_rad must lie in (-pi/2, pi/2)"
        )
    if not np.isfinite(gamma) or gamma <= 0.0 or rho_g <= 0.0:
        raise ValueError("surface tension and density*gravity must be positive")

    capillary_length = math.sqrt(gamma / rho_g)
    contact_z = float(
        sphere_lower_z(
            contact,
            sphere_radius_m=sphere,
            sphere_tip_z_m=float(sphere_tip_z_m),
        )
    )
    solid_angle = math.asin(float(np.clip(contact / sphere, -0.999999, 0.999999)))
    contact_tangent = solid_angle - math.pi + angle
    film_z = float(film_junction_z_m)
    xi = np.linspace(0.0, 1.0, max(180, int(samples) * 4))

    if previous is None or len(previous.r_m) < 4:
        seed = solve_zero_angle_sphere_film_meridian(
            contact_radius_m=contact,
            sphere_radius_m=sphere,
            sphere_tip_z_m=float(sphere_tip_z_m),
            film_junction_z_m=film_z,
            density_kg_m3=float(density_kg_m3),
            gravity_m_s2=float(gravity_m_s2),
            surface_tension_n_m=gamma,
            samples=max(24, int(samples)),
            tolerance=max(float(tolerance), 1.0e-4),
        )
    else:
        seed = previous
    source = np.linspace(0.0, 1.0, len(seed.r_m))
    radius_guess = np.interp(xi, source, np.asarray(seed.r_m, dtype=float))
    height_guess = np.interp(xi, source, np.asarray(seed.z_m, dtype=float))
    tangent_guess = np.interp(
        xi, source, np.asarray(seed.tangent_angle_rad, dtype=float)
    )
    radius_guess += contact - float(radius_guess[0])
    height_guess += contact_z - float(height_guess[0])
    # A linear endpoint correction is only a Newton initial guess.  Both
    # endpoint angles are imposed exactly by the boundary residual below.
    tangent_guess += (
        contact_tangent - float(tangent_guess[0])
    ) * (1.0 - xi) + (
        rim_tangent - float(tangent_guess[-1])
    ) * xi
    guess = np.vstack(
        (
            radius_guess / capillary_length,
            height_guess / capillary_length,
            tangent_guess,
        )
    )
    pressure_guess = float(seed.pressure_jump_pa) * capillary_length / gamma
    arc_guess = max(float(seed.arc_length_m) / capillary_length, 1.0e-14)
    contact_nd = contact / capillary_length
    contact_z_nd = contact_z / capillary_length
    film_z_nd = film_z / capillary_length

    def ode(_xi: np.ndarray, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        pressure = float(params[0])
        arc_length = math.exp(float(np.clip(params[1], -40.0, 10.0)))
        radius = np.maximum(state[0], 1.0e-12)
        height = state[1]
        tangent = state[2]
        curvature = pressure + height - film_z_nd
        return np.vstack(
            (
                arc_length * np.cos(tangent),
                arc_length * np.sin(tangent),
                arc_length * (curvature - np.sin(tangent) / radius),
            )
        )

    def boundary(
        left: np.ndarray, right: np.ndarray, _params: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            (
                left[0] - contact_nd,
                left[1] - contact_z_nd,
                left[2] - contact_tangent,
                right[1] - film_z_nd,
                right[2] - rim_tangent,
            ),
            dtype=float,
        )

    result = solve_bvp(
        ode,
        boundary,
        xi,
        guess,
        p=np.asarray((pressure_guess, math.log(arc_guess)), dtype=float),
        tol=float(tolerance),
        max_nodes=50000,
    )
    if not bool(result.success):
        raise RuntimeError(
            "Contact-angle Young-Laplace solve failed: " + str(result.message)
        )
    query = np.linspace(0.0, 1.0, max(12, int(samples)))
    solved = np.asarray(result.sol(query), dtype=float)
    radius_m = capillary_length * solved[0]
    height_m = capillary_length * solved[1]
    profile = YoungLaplaceMeridian(
        r_m=radius_m,
        z_m=height_m,
        tangent_angle_rad=solved[2],
        pressure_jump_pa=float(result.p[0]) * gamma / capillary_length,
        arc_length_m=capillary_length * math.exp(float(result.p[1])),
        contact_radius_m=contact,
        rim_radius_m=float(radius_m[-1]),
        neck_radius_m=float(np.min(radius_m)),
    )
    if not np.all(np.isfinite(radius_m)) or not np.all(np.isfinite(height_m)):
        raise RuntimeError("Contact-angle Young-Laplace solution is not finite")
    return profile


def sphere_meridian_volume_integral(
    contact_radius_m: float,
    *,
    sphere_radius_m: float,
) -> float:
    """Return ``pi integral(r^2 dz)`` along the lower sphere from 0 to CL."""

    contact = float(contact_radius_m)
    sphere = float(sphere_radius_m)
    radial = np.linspace(0.0, contact, 600)
    axial = sphere - np.sqrt(np.maximum(sphere * sphere - radial * radial, 0.0))
    return float(np.trapezoid(math.pi * radial * radial, axial))


def bridge_volume_from_meridian(
    profile: YoungLaplaceMeridian,
    *,
    sphere_radius_m: float,
) -> float:
    """Liquid volume above the undisturbed film enclosed by a bridge profile."""

    free_integral = float(
        np.trapezoid(math.pi * np.asarray(profile.r_m) ** 2, np.asarray(profile.z_m))
    )
    solid_integral = sphere_meridian_volume_integral(
        float(profile.contact_radius_m),
        sphere_radius_m=float(sphere_radius_m),
    )
    return float(-(free_integral + solid_integral))


def solve_zero_angle_sphere_film_meridian_for_volume_and_tangent(
    *,
    bridge_volume_m3: float,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    film_junction_z_m: float,
    film_junction_tangent_rad: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    samples: int = 160,
    tolerance: float = 3.0e-5,
    previous: YoungLaplaceMeridian | None = None,
) -> YoungLaplaceMeridian:
    """Solve a zero-angle fixed-sphere bridge at fixed volume and rim slope.

    Unlike :func:`solve_sphere_film_meridian_for_contact_and_volume`, the
    sphere contact radius is *not* prescribed.  The Young--Laplace pressure,
    meridian arc length, and sphere polar contact angle are the three BVP
    parameters.  The four dimensionless states are radius, height, tangent
    angle, and ``pi integral(r**2 dz)``.  Consequently both the requested
    bridge volume and the measured/transported outer-film tangent can be
    imposed without adding a geometric closure.

    ``previous`` is an optional continuation seed.  It changes only the
    nonlinear initial iterate; it is never used as an additional constraint.
    In particular, a zero film-junction tangent recovers the same equilibrium
    family as :func:`solve_zero_angle_sphere_film_meridian`.
    """

    target_volume = float(bridge_volume_m3)
    sphere = float(sphere_radius_m)
    tip = float(sphere_tip_z_m)
    film_z = float(film_junction_z_m)
    rim_tangent = float(film_junction_tangent_rad)
    gamma = float(surface_tension_n_m)
    rho_g = float(density_kg_m3) * float(gravity_m_s2)
    if not np.isfinite(target_volume) or target_volume <= 0.0:
        raise ValueError("bridge_volume_m3 must be finite and positive")
    if not np.isfinite(sphere) or sphere <= 0.0:
        raise ValueError("sphere_radius_m must be finite and positive")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("surface_tension_n_m must be finite and positive")
    if not np.isfinite(rho_g) or rho_g <= 0.0:
        raise ValueError("density_kg_m3 * gravity_m_s2 must be positive")
    if not np.isfinite(rim_tangent) or abs(rim_tangent) >= 0.5 * math.pi:
        raise ValueError(
            "film_junction_tangent_rad must be finite and lie in (-pi/2, pi/2)"
        )

    capillary_length = math.sqrt(gamma / rho_g)
    sphere_nd = sphere / capillary_length
    tip_nd = tip / capillary_length
    film_z_nd = film_z / capillary_length
    target_volume_nd = target_volume / capillary_length**3
    xi = np.linspace(0.0, 1.0, max(220, int(samples) * 3))

    def sphere_integral_from_angle_nd(polar_angle: float) -> float:
        # pi integral(r**2 dz) on the lower spherical cap.  The cap-height
        # form avoids cancellation of 2/3-cos(a)+cos(a)**3/3 for small a.
        cap_height = 2.0 * sphere_nd * math.sin(0.5 * polar_angle) ** 2
        return math.pi * (
            sphere_nd * cap_height * cap_height - cap_height**3 / 3.0
        )

    def profile_guess(
        profile: YoungLaplaceMeridian,
    ) -> tuple[np.ndarray, np.ndarray]:
        old_x = np.linspace(0.0, 1.0, len(profile.r_m))
        radius = np.interp(xi, old_x, np.asarray(profile.r_m, dtype=float))
        height = np.interp(xi, old_x, np.asarray(profile.z_m, dtype=float))
        tangent = np.interp(
            xi, old_x, np.asarray(profile.tangent_angle_rad, dtype=float)
        )
        integral = np.zeros_like(xi)
        integral[1:] = np.cumsum(
            0.5
            * math.pi
            * (radius[1:] ** 2 + radius[:-1] ** 2)
            * np.diff(height)
        )
        contact = float(profile.contact_radius_m)
        polar_angle = math.asin(float(np.clip(contact / sphere, 1.0e-10, 0.999999)))
        state = np.vstack(
            (
                radius / capillary_length,
                height / capillary_length,
                tangent,
                integral / capillary_length**3,
            )
        )
        params = np.asarray(
            (
                float(profile.pressure_jump_pa) * capillary_length / gamma,
                math.log(max(float(profile.arc_length_m) / capillary_length, 1.0e-14)),
                polar_angle,
            ),
            dtype=float,
        )
        return state, params

    def zero_tangent_seed() -> YoungLaplaceMeridian:
        # The prescribed-contact solver supplies a branch-safe seed only.  A
        # fresh three-parameter BVP below still solves contact radius from the
        # volume and boundary tangent.  The zero-tangent volume is strictly
        # increasing on the physical branch used here.
        cache: dict[float, tuple[float, YoungLaplaceMeridian]] = {}

        def residual(contact: float) -> float:
            key = float(contact)
            if key not in cache:
                candidate = solve_zero_angle_sphere_film_meridian(
                    contact_radius_m=key,
                    sphere_radius_m=sphere,
                    sphere_tip_z_m=tip,
                    film_junction_z_m=film_z,
                    density_kg_m3=float(density_kg_m3),
                    gravity_m_s2=float(gravity_m_s2),
                    surface_tension_n_m=gamma,
                    samples=max(120, int(samples)),
                    tolerance=max(float(tolerance), 2.0e-5),
                )
                cache[key] = (
                    bridge_volume_from_meridian(candidate, sphere_radius_m=sphere),
                    candidate,
                )
            return cache[key][0] - target_volume

        # Contacts below about 0.1% of the sphere radius are both far below
        # the mesh-resolved regime and unnecessarily stiff for the auxiliary
        # prescribed-contact seed solve.  At R=5 mm this lower endpoint has a
        # bridge volume O(1e-9 uL), five orders below the requested test range.
        lower = max(1.0e-3 * sphere, 1.0e-6)
        upper = min(0.70 * sphere, sphere - 1.0e-8)
        lower_value = residual(lower)
        upper_value = residual(upper)
        if lower_value > 0.0 or upper_value < 0.0:
            raise RuntimeError(
                "Requested bridge volume is outside the zero-angle fixed-sphere "
                "Young-Laplace seed branch"
            )
        contact = float(
            brentq(
                residual,
                lower,
                upper,
                xtol=max(2.0e-11, 2.0e-8 * sphere),
                rtol=2.0e-8,
            )
        )
        # Re-evaluate at the converged root rather than relying on brentq's
        # last internal evaluation, which is not guaranteed to equal it.
        residual(contact)
        return cache[contact][1]

    def ode(_xi: np.ndarray, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        pressure_nd = float(params[0])
        arc_length_nd = math.exp(float(np.clip(params[1], -40.0, 10.0)))
        radius = np.maximum(state[0], 1.0e-11)
        height = state[1]
        tangent = state[2]
        curvature_nd = pressure_nd + height - film_z_nd
        return np.vstack(
            (
                arc_length_nd * np.cos(tangent),
                arc_length_nd * np.sin(tangent),
                arc_length_nd * (curvature_nd - np.sin(tangent) / radius),
                arc_length_nd * math.pi * radius * radius * np.sin(tangent),
            )
        )

    def solve_once(
        guess: np.ndarray,
        parameters: np.ndarray,
        *,
        volume_nd: float,
        tangent_rad: float,
    ):
        def boundary(
            left: np.ndarray, right: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            polar_angle = float(params[2])
            contact_nd = sphere_nd * math.sin(polar_angle)
            contact_z_nd = tip_nd + sphere_nd * (1.0 - math.cos(polar_angle))
            target_free_integral_nd = -float(volume_nd) - (
                sphere_integral_from_angle_nd(polar_angle)
            )
            return np.asarray(
                (
                    left[0] - contact_nd,
                    left[1] - contact_z_nd,
                    left[2] - (polar_angle - math.pi),
                    left[3],
                    right[1] - film_z_nd,
                    right[2] - float(tangent_rad),
                    right[3] - target_free_integral_nd,
                ),
                dtype=float,
            )

        result = solve_bvp(
            ode,
            boundary,
            xi,
            guess,
            p=np.asarray(parameters, dtype=float),
            tol=float(tolerance),
            max_nodes=100000,
        )
        if not bool(result.success):
            raise RuntimeError(
                "Volume-and-tangent Young-Laplace solve failed: "
                f"{result.message}"
            )
        polar_angle = float(result.p[2])
        if not 0.0 < polar_angle < 0.5 * math.pi:
            raise RuntimeError(
                "Volume-and-tangent Young-Laplace solve left the lower-sphere branch"
            )
        return result

    use_previous = previous is not None and len(previous.r_m) >= 4
    if use_previous:
        previous_contact = float(previous.contact_radius_m)
        previous_angle = math.asin(
            float(np.clip(previous_contact / sphere, 1.0e-10, 0.999999))
        )
        contact_tangent_error = abs(
            float(previous.tangent_angle_rad[0]) - (previous_angle - math.pi)
        )
        use_previous = contact_tangent_error <= 2.0e-3

    if use_previous:
        guess, parameters = profile_guess(previous)
        previous_volume = bridge_volume_from_meridian(previous, sphere_radius_m=sphere)
        previous_volume_nd = max(previous_volume / capillary_length**3, 1.0e-16)
        previous_tangent = float(previous.tangent_angle_rad[-1])
        volume_steps = int(
            math.ceil(
                abs(math.log(max(target_volume_nd, 1.0e-30) / previous_volume_nd))
                / math.log(1.8)
            )
        )
        tangent_steps = int(math.ceil(abs(rim_tangent - previous_tangent) / 0.08))
        continuation_steps = max(1, volume_steps, tangent_steps)
        start_log_volume = math.log(previous_volume_nd)
        target_log_volume = math.log(target_volume_nd)
        result = None
        for step in range(1, continuation_steps + 1):
            fraction = step / continuation_steps
            next_volume_nd = math.exp(
                start_log_volume + fraction * (target_log_volume - start_log_volume)
            )
            next_tangent = previous_tangent + fraction * (
                rim_tangent - previous_tangent
            )
            old_endpoint_tangent = float(guess[2, -1])
            guess[2] += (next_tangent - old_endpoint_tangent) * xi
            result = solve_once(
                guess,
                parameters,
                volume_nd=next_volume_nd,
                tangent_rad=next_tangent,
            )
            guess = np.asarray(result.sol(xi), dtype=float)
            parameters = np.asarray(result.p, dtype=float)
    else:
        seed = zero_tangent_seed()
        guess, parameters = profile_guess(seed)
        result = solve_once(
            guess,
            parameters,
            volume_nd=target_volume_nd,
            tangent_rad=0.0,
        )
        guess = np.asarray(result.sol(xi), dtype=float)
        parameters = np.asarray(result.p, dtype=float)
        tangent_steps = max(1, int(math.ceil(abs(rim_tangent) / 0.08)))
        for step in range(1, tangent_steps + 1):
            next_tangent = rim_tangent * step / tangent_steps
            guess[2] += (next_tangent - float(guess[2, -1])) * xi
            result = solve_once(
                guess,
                parameters,
                volume_nd=target_volume_nd,
                tangent_rad=next_tangent,
            )
            guess = np.asarray(result.sol(xi), dtype=float)
            parameters = np.asarray(result.p, dtype=float)

    query = np.linspace(0.0, 1.0, max(12, int(samples)))

    def sampled_result_volume(current_result) -> float:
        sampled = np.asarray(current_result.sol(query), dtype=float)
        sampled_profile = YoungLaplaceMeridian(
            r_m=capillary_length * sampled[0],
            z_m=capillary_length * sampled[1],
            tangent_angle_rad=sampled[2],
            pressure_jump_pa=float(current_result.p[0]) * gamma / capillary_length,
            arc_length_m=(
                capillary_length * math.exp(float(current_result.p[1]))
            ),
            contact_radius_m=sphere * math.sin(float(current_result.p[2])),
            rim_radius_m=capillary_length * float(sampled[0, -1]),
            neck_radius_m=capillary_length * float(np.min(sampled[0])),
        )
        return bridge_volume_from_meridian(
            sampled_profile,
            sphere_radius_m=sphere,
        )

    # The tetra boundary is the returned piecewise-linear meridian, not the
    # dense collocation curve.  Correct only the O(ds**2) representation
    # error so that a deliberately coarse, mesh-resolved profile encloses the
    # requested volume.  The correction tends to zero under meridian
    # refinement and changes no boundary angle or material parameter.
    represented_target_nd = target_volume_nd
    for _ in range(4):
        represented_volume = sampled_result_volume(result)
        representation_error = represented_volume - target_volume
        if abs(representation_error) <= max(2.0e-7 * target_volume, 1.0e-23):
            break
        if not math.isfinite(represented_volume) or represented_volume <= 0.0:
            raise RuntimeError(
                "Volume-and-tangent Young-Laplace sampling produced invalid volume"
            )
        represented_target_nd *= target_volume / represented_volume
        result = solve_once(
            np.asarray(result.sol(xi), dtype=float),
            np.asarray(result.p, dtype=float),
            volume_nd=represented_target_nd,
            tangent_rad=rim_tangent,
        )
    else:
        raise RuntimeError(
            "Volume-and-tangent Young-Laplace polyline did not conserve volume"
        )

    solved = np.asarray(result.sol(query), dtype=float)
    radius_m = capillary_length * solved[0]
    height_m = capillary_length * solved[1]
    polar_angle = float(result.p[2])
    contact_radius = sphere * math.sin(polar_angle)
    contact_z = tip + sphere * (1.0 - math.cos(polar_angle))

    # The sphere occupies z >= z_lower(r).  Apart from the common contact
    # point, a valid bridge meridian therefore has z <= z_lower for r < R.
    inside_radial = radius_m < sphere * (1.0 - 1.0e-10)
    solid_lower = np.asarray(
        sphere_lower_z(
            radius_m[inside_radial],
            sphere_radius_m=sphere,
            sphere_tip_z_m=tip,
        ),
        dtype=float,
    )
    penetration = height_m[inside_radial] - solid_lower
    geometry_tolerance = max(2.0e-9, 5.0e-6 * sphere)
    if penetration.size and float(np.max(penetration)) > geometry_tolerance:
        raise RuntimeError(
            "Volume-and-tangent Young-Laplace meridian penetrates the solid sphere"
        )
    if abs(float(radius_m[0]) - contact_radius) > geometry_tolerance or abs(
        float(height_m[0]) - contact_z
    ) > geometry_tolerance:
        raise RuntimeError("Young-Laplace contact point failed its sphere constraint")

    return YoungLaplaceMeridian(
        r_m=radius_m,
        z_m=height_m,
        tangent_angle_rad=solved[2],
        pressure_jump_pa=float(result.p[0]) * gamma / capillary_length,
        arc_length_m=capillary_length * math.exp(float(result.p[1])),
        contact_radius_m=contact_radius,
        rim_radius_m=float(radius_m[-1]),
        neck_radius_m=float(np.min(radius_m)),
    )


def solve_sphere_film_meridian_for_contact_and_volume(
    *,
    contact_radius_m: float,
    bridge_volume_m3: float,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    film_junction_z_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    contact_angle_rad: float = 0.0,
    film_junction_tangent_rad: float = 0.0,
    enforce_zero_contact_angle: bool = False,
    enforce_contact_angle: bool = False,
    samples: int = 52,
    previous: YoungLaplaceMeridian | None = None,
) -> YoungLaplaceMeridian:
    """Young-Laplace bridge at fixed mesh CL and conserved bridge volume.

    Pressure, arc length, contact angle, and film-junction radius are solved
    by default.  With ``enforce_contact_angle=True`` the supplied dynamic
    angle is imposed and the junction tangent becomes an output;
    ``enforce_zero_contact_angle=True`` is the backward-compatible shorthand
    for imposing zero.  ``contact_angle_rad`` is otherwise only the nonlinear
    initial guess.  No experimental bridge radius or angle enters the solve.
    """

    contact = float(contact_radius_m)
    sphere = float(sphere_radius_m)
    target_volume = max(float(bridge_volume_m3), 1.0e-18)
    solid_angle = math.asin(float(np.clip(contact / sphere, -0.999999, 0.999999)))
    fixed_contact_angle = bool(
        enforce_zero_contact_angle or enforce_contact_angle
    )
    imposed_contact_angle = (
        0.0 if bool(enforce_zero_contact_angle) else float(contact_angle_rad)
    )
    contact_tangent = solid_angle - math.pi + imposed_contact_angle
    # ``previous`` is a nonlinear initial guess, not proof of equilibrium.
    # A transported Lagrangian contour can have the requested contact point,
    # volume, and endpoint angle while its interior curvature is far from the
    # Young--Laplace pressure balance.  The former endpoint-only early return
    # therefore made local pressure relaxation a no-op precisely on accepted
    # ALE states.  Always evaluate the BVP; solve_bvp already converges in one
    # Newton update when ``previous`` is an actual solution.
    contact_z = float(
        sphere_lower_z(
            contact,
            sphere_radius_m=sphere,
            sphere_tip_z_m=float(sphere_tip_z_m),
        )
    )
    solid_volume_integral = sphere_meridian_volume_integral(
        contact,
        sphere_radius_m=sphere,
    )
    target_free_integral = -target_volume - solid_volume_integral
    xi = np.linspace(0.0, 1.0, max(180, int(samples) * 4))

    if previous is not None and len(previous.r_m) >= 4:
        old_x = np.linspace(0.0, 1.0, len(previous.r_m))
        radius_guess = np.interp(xi, old_x, np.asarray(previous.r_m, dtype=float))
        radius_guess += contact - float(radius_guess[0])
        height_guess = np.interp(xi, old_x, np.asarray(previous.z_m, dtype=float))
        height_guess += contact_z - float(height_guess[0])
        tangent_guess = np.interp(
            xi, old_x, np.asarray(previous.tangent_angle_rad, dtype=float)
        )
        pressure_guess = float(previous.pressure_jump_pa)
        arc_guess = float(previous.arc_length_m)
        angle_guess = (
            imposed_contact_angle
            if fixed_contact_angle
            else float(previous.tangent_angle_rad[0] - (solid_angle - math.pi))
        )
        tangent_guess += (
            solid_angle - math.pi + angle_guess - float(tangent_guess[0])
        )
    else:
        seed = solve_zero_angle_sphere_film_meridian(
            contact_radius_m=contact,
            sphere_radius_m=sphere,
            sphere_tip_z_m=float(sphere_tip_z_m),
            film_junction_z_m=float(film_junction_z_m),
            density_kg_m3=float(density_kg_m3),
            gravity_m_s2=float(gravity_m_s2),
            surface_tension_n_m=float(surface_tension_n_m),
            samples=max(24, int(samples)),
            tolerance=1.0e-4,
        )
        old_x = np.linspace(0.0, 1.0, len(seed.r_m))
        radius_guess = np.interp(xi, old_x, seed.r_m)
        height_guess = np.interp(xi, old_x, seed.z_m)
        tangent_guess = np.interp(xi, old_x, seed.tangent_angle_rad)
        tangent_guess += contact_tangent - float(tangent_guess[0])
        pressure_guess = float(seed.pressure_jump_pa)
        arc_guess = float(seed.arc_length_m)
        angle_guess = float(contact_angle_rad)

    free_guess = np.zeros_like(xi)
    free_guess[1:] = np.cumsum(
        0.5
        * math.pi
        * (radius_guess[1:] ** 2 + radius_guess[:-1] ** 2)
        * np.diff(height_guess)
    )
    if abs(float(free_guess[-1])) > 1.0e-30:
        free_guess *= target_free_integral / float(free_guess[-1])
    guess = np.vstack((radius_guess, height_guess, tangent_guess, free_guess))
    gamma = float(surface_tension_n_m)
    rho_g = float(density_kg_m3) * float(gravity_m_s2)
    # Solve the BVP in capillary-length units.  In SI units the boundary
    # residual combines millimetre coordinates with an O(1e-10 m3) volume
    # integral, which makes continuation between 0.1 s timesteps needlessly
    # ill-conditioned.  With l_c=sqrt(gamma/rho g), gravity has unit
    # coefficient and every boundary residual is O(1).
    capillary_length = math.sqrt(gamma / max(rho_g, 1.0e-30))
    contact_nd = contact / capillary_length
    contact_z_nd = contact_z / capillary_length
    film_z_nd = float(film_junction_z_m) / capillary_length
    target_free_integral_nd = target_free_integral / capillary_length**3
    radius_guess_nd = radius_guess / capillary_length
    height_guess_nd = height_guess / capillary_length
    free_guess_nd = free_guess / capillary_length**3
    pressure_guess_nd = pressure_guess * capillary_length / gamma
    arc_guess_nd = arc_guess / capillary_length
    guess_nd = np.vstack(
        (radius_guess_nd, height_guess_nd, tangent_guess, free_guess_nd)
    )

    def ode(_xi: np.ndarray, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        pressure_jump = float(params[0])
        arc_length = math.exp(float(np.clip(params[1], -40.0, 10.0)))
        radius = np.maximum(state[0], 1.0e-12)
        height = state[1]
        tangent = state[2]
        curvature = pressure_jump + height - film_z_nd
        return np.vstack(
            (
                arc_length * np.cos(tangent),
                arc_length * np.sin(tangent),
                arc_length * (curvature - np.sin(tangent) / radius),
                arc_length * math.pi * radius * radius * np.sin(tangent),
            )
        )

    def boundary(left: np.ndarray, right: np.ndarray, _params: np.ndarray) -> np.ndarray:
        if fixed_contact_angle:
            return np.asarray(
                (
                    left[0] - contact_nd,
                    left[1] - contact_z_nd,
                    left[2] - contact_tangent,
                    left[3],
                    right[1] - film_z_nd,
                    right[3] - target_free_integral_nd,
                ),
                dtype=float,
            )
        solved_contact_tangent = solid_angle - math.pi + float(_params[2])
        return np.asarray(
            (
                left[0] - contact_nd,
                left[1] - contact_z_nd,
                left[2] - solved_contact_tangent,
                left[3],
                right[1] - film_z_nd,
                right[2] - float(film_junction_tangent_rad),
                right[3] - target_free_integral_nd,
            ),
            dtype=float,
        )

    solver_parameters = (
        np.asarray(
            (
                pressure_guess_nd,
                math.log(max(arc_guess_nd, 1.0e-18)),
            ),
            dtype=float,
        )
        if fixed_contact_angle
        else np.asarray(
            (
                pressure_guess_nd,
                math.log(max(arc_guess_nd, 1.0e-18)),
                angle_guess,
            ),
            dtype=float,
        )
    )

    result = solve_bvp(
        ode,
        boundary,
        xi,
        guess_nd,
        p=solver_parameters,
        tol=8.0e-5,
        max_nodes=100000,
    )
    if not bool(result.success):
        raise RuntimeError(f"Volume-constrained Young-Laplace solve failed: {result.message}")
    query = np.linspace(0.0, 1.0, max(12, int(samples)))
    solved = np.asarray(result.sol(query), dtype=float)
    return YoungLaplaceMeridian(
        r_m=capillary_length * solved[0],
        z_m=capillary_length * solved[1],
        tangent_angle_rad=solved[2],
        pressure_jump_pa=float(result.p[0]) * gamma / capillary_length,
        arc_length_m=capillary_length * math.exp(float(result.p[1])),
        contact_radius_m=contact,
        rim_radius_m=capillary_length * float(solved[0, -1]),
        neck_radius_m=capillary_length * float(np.min(solved[0])),
    )


def resolved_contact_radius(
    *,
    target_rim_offset_m: float,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    film_junction_z_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
) -> tuple[float, YoungLaplaceMeridian]:
    """Choose only the numerical seed radius from a resolvable rim offset."""

    target = max(float(target_rim_offset_m), 1.0e-9)

    def residual(contact_radius: float) -> float:
        profile = solve_zero_angle_sphere_film_meridian(
            contact_radius_m=float(contact_radius),
            sphere_radius_m=float(sphere_radius_m),
            sphere_tip_z_m=float(sphere_tip_z_m),
            film_junction_z_m=float(film_junction_z_m),
            density_kg_m3=float(density_kg_m3),
            gravity_m_s2=float(gravity_m_s2),
            surface_tension_n_m=float(surface_tension_n_m),
            samples=24,
            tolerance=1.0e-4,
        )
        return float(profile.rim_radius_m - contact_radius - target)

    lower = max(20.0e-6, 0.004 * float(sphere_radius_m))
    upper = min(0.65 * float(sphere_radius_m), 2.5e-3)
    f_lower = residual(lower)
    f_upper = residual(upper)
    if f_lower * f_upper > 0.0:
        raise RuntimeError(
            "Requested neck resolution is outside the Young-Laplace seed bracket"
        )
    contact = float(brentq(residual, lower, upper, xtol=1.0e-8, rtol=1.0e-6))
    profile = solve_zero_angle_sphere_film_meridian(
        contact_radius_m=contact,
        sphere_radius_m=float(sphere_radius_m),
        sphere_tip_z_m=float(sphere_tip_z_m),
        film_junction_z_m=float(film_junction_z_m),
        density_kg_m3=float(density_kg_m3),
        gravity_m_s2=float(gravity_m_s2),
        surface_tension_n_m=float(surface_tension_n_m),
        samples=52,
        tolerance=2.0e-5,
    )
    return contact, profile


def _tet_subdivision(element_name: str, row: list[int]) -> list[list[int]]:
    if element_name.startswith("Tetrahedron") and len(row) >= 4:
        return [row[:4]]
    if element_name.startswith("Pyramid") and len(row) >= 5:
        a, b, c, d, apex = row[:5]
        return [[a, b, c, apex], [a, c, d, apex]]
    if element_name.startswith("Prism") and len(row) >= 6:
        a, b, c, d, e, f = row[:6]
        return [[a, b, c, d], [b, c, d, e], [c, d, e, f]]
    if element_name.startswith("Hexahedron") and len(row) >= 8:
        a, b, c, d, e, f, g, h = row[:8]
        return [
            [a, b, d, e],
            [b, c, d, g],
            [b, d, e, g],
            [b, e, f, g],
            [d, e, g, h],
        ]
    return []


def _extract_tets(gmsh_module) -> tuple[np.ndarray, np.ndarray]:
    node_tags, coordinates, _ = gmsh_module.model.mesh.getNodes()
    tags = np.asarray(node_tags, dtype=np.int64)
    points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
    index = {int(tag): i for i, tag in enumerate(tags)}
    tets: list[list[int]] = []
    element_types, _, element_nodes = gmsh_module.model.mesh.getElements(3)
    for element_type, flat_nodes in zip(element_types, element_nodes):
        name, _dim, _order, node_count, _local, _primary = (
            gmsh_module.model.mesh.getElementProperties(element_type)
        )
        rows = np.asarray(flat_nodes, dtype=np.int64).reshape((-1, int(node_count)))
        for raw in rows:
            row = [index[int(tag)] for tag in raw]
            tets.extend(_tet_subdivision(str(name), row))
    return points, np.asarray(tets, dtype=int)


def _extract_planar_triangles(
    gmsh_module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract primary-node triangles from the current x-z plane mesh."""

    node_tags, coordinates, _ = gmsh_module.model.mesh.getNodes()
    tags = np.asarray(node_tags, dtype=np.int64)
    points = np.asarray(coordinates, dtype=float).reshape((-1, 3))
    index = {int(tag): i for i, tag in enumerate(tags)}
    triangles: list[list[int]] = []
    element_types, _, element_nodes = gmsh_module.model.mesh.getElements(2)
    for element_type, flat_nodes in zip(element_types, element_nodes):
        name, _dim, _order, node_count, _local, _primary = (
            gmsh_module.model.mesh.getElementProperties(element_type)
        )
        if not str(name).startswith("Triangle"):
            continue
        rows = np.asarray(flat_nodes, dtype=np.int64).reshape((-1, int(node_count)))
        for raw in rows:
            triangles.append([index[int(tag)] for tag in raw[:3]])
    return points, np.asarray(triangles, dtype=int), tags


def _revolve_planar_triangles(
    planar_points: np.ndarray,
    triangles: np.ndarray,
    azimuthal_sectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Revolve an x-z triangulation into an anisotropic full-360 tetra mesh."""

    base = np.asarray(planar_points, dtype=float)
    tri = np.asarray(triangles, dtype=int).reshape((-1, 3))
    sectors = max(int(azimuthal_sectors), 8)
    radius = np.abs(base[:, 0])
    axis = radius <= max(1.0e-12, 1.0e-9 * float(np.max(radius)))
    index = np.full((len(base), sectors), -1, dtype=int)
    points: list[tuple[float, float, float]] = []
    base_vertex: list[int] = []
    for vertex in range(len(base)):
        if axis[vertex]:
            index[vertex, :] = len(points)
            points.append((0.0, 0.0, float(base[vertex, 2])))
            base_vertex.append(vertex)
            continue
        for sector in range(sectors):
            angle = 2.0 * math.pi * sector / sectors
            index[vertex, sector] = len(points)
            points.append(
                (
                    float(radius[vertex] * math.cos(angle)),
                    float(radius[vertex] * math.sin(angle)),
                    float(base[vertex, 2]),
                )
            )
            base_vertex.append(vertex)

    tetrahedra: list[list[int]] = []
    for raw in tri:
        ordered = np.sort(raw)
        axis_count = int(np.count_nonzero(axis[ordered]))
        for sector in range(sectors):
            following = (sector + 1) % sectors
            lower = [int(index[v, sector]) for v in ordered]
            upper = [int(index[v, following]) for v in ordered]
            if axis_count == 0:
                tetrahedra.extend(
                    (
                        [lower[0], lower[1], lower[2], upper[2]],
                        [lower[0], lower[1], upper[1], upper[2]],
                        [lower[0], upper[0], upper[1], upper[2]],
                    )
                )
            elif axis_count == 1:
                axis_position = int(np.flatnonzero(axis[ordered])[0])
                solid_axis = lower[axis_position]
                other = [position for position in range(3) if position != axis_position]
                first, second = other
                tetrahedra.extend(
                    (
                        [solid_axis, lower[first], lower[second], upper[second]],
                        [solid_axis, lower[first], upper[second], upper[first]],
                    )
                )
            elif axis_count == 2:
                radial_position = int(np.flatnonzero(~axis[ordered])[0])
                axis_positions = [position for position in range(3) if position != radial_position]
                tetrahedra.append(
                    [
                        lower[axis_positions[0]],
                        lower[axis_positions[1]],
                        lower[radial_position],
                        upper[radial_position],
                    ]
                )

    points_array = np.asarray(points, dtype=float)
    tets = np.asarray(tetrahedra, dtype=int)
    if tets.size == 0:
        finite = np.all(np.isfinite(base), axis=1) if base.ndim == 2 else np.zeros(0, dtype=bool)
        raise RuntimeError(
            "Planar-to-axisymmetric revolution produced no tetrahedra "
            f"(planar_points={len(base)}, triangles={len(tri)}, "
            f"finite_points={int(np.count_nonzero(finite))}, "
            f"axis_points={int(np.count_nonzero(axis))})"
        )
    tets = tets.reshape((-1, 4))
    tet_points = points_array[tets]
    signed = np.einsum(
        "ij,ij->i",
        tet_points[:, 1] - tet_points[:, 0],
        np.cross(tet_points[:, 2] - tet_points[:, 0], tet_points[:, 3] - tet_points[:, 0]),
    )
    negative = signed < 0.0
    if np.any(negative):
        swapped = tets[negative, 1].copy()
        tets[negative, 1] = tets[negative, 2]
        tets[negative, 2] = swapped
    keep = np.abs(signed) > 6.0e-26
    return points_array, tets[keep], np.asarray(base_vertex, dtype=int)


def _extrude_planar_triangles_to_axisymmetric_wedge(
    planar_points: np.ndarray,
    triangles: np.ndarray,
    wedge_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrude an x-z triangulation into one narrow, genuine 3-D wedge.

    The two boundary copies lie at ``+/- wedge_angle_rad/2``.  An internal
    copy at the angular midpoint permits a mirrored tetrahedral split of the
    two half-prisms.  Besides preserving reflection symmetry, this ensures
    that every off-midplane P2 edge midpoint has a partner with the same
    ``(r,z)`` coordinate for an exact-axisymmetric Taylor--Hood reduction.
    Off-axis triangular half-prisms are split into three tetrahedra, while
    half-prisms touching the rotation axis reduce to two or one tetrahedra.
    Axis vertices are shared by all three planes, which avoids zero-volume
    duplicate-axis elements.

    The returned Boolean array has one row per 3-D vertex and two columns.  It
    records membership of the lower and upper artificial angular planes; an
    axis vertex belongs to both.  Callers use it to remove those two planes
    from the physical liquid-air/solid boundary patches.
    """

    base = np.asarray(planar_points, dtype=float)
    tri = np.asarray(triangles, dtype=int).reshape((-1, 3))
    angle = float(wedge_angle_rad)
    if not math.isfinite(angle) or not 0.0 < angle < math.pi:
        raise ValueError(
            "axisymmetric_wedge_angle_rad must lie strictly between 0 and pi"
        )
    radius = np.abs(base[:, 0])
    axis = radius <= max(1.0e-12, 1.0e-9 * float(np.max(radius)))
    index = np.full((len(base), 3), -1, dtype=int)
    points: list[tuple[float, float, float]] = []
    base_vertex: list[int] = []
    side_membership: list[tuple[bool, bool]] = []
    half_angle = 0.5 * angle
    side_angles = (-half_angle, 0.0, half_angle)
    for vertex in range(len(base)):
        if axis[vertex]:
            index[vertex, :] = len(points)
            points.append((0.0, 0.0, float(base[vertex, 2])))
            base_vertex.append(vertex)
            side_membership.append((True, True))
            continue
        for side, side_angle in enumerate(side_angles):
            index[vertex, side] = len(points)
            points.append(
                (
                    float(radius[vertex] * math.cos(side_angle)),
                    float(radius[vertex] * math.sin(side_angle)),
                    float(base[vertex, 2]),
                )
            )
            base_vertex.append(vertex)
            side_membership.append((side == 0, side == 2))

    tetrahedra: list[list[int]] = []
    for raw in tri:
        # Use the same global planar-vertex ordering for every prism.  Adjacent
        # prisms then choose identical diagonals on their common quadrilateral
        # face, so the tetra complex remains conforming.
        ordered = np.sort(raw)
        axis_count = int(np.count_nonzero(axis[ordered]))
        lower = [int(index[v, 0]) for v in ordered]
        middle = [int(index[v, 1]) for v in ordered]
        upper = [int(index[v, 2]) for v in ordered]
        if axis_count == 0:
            tetrahedra.extend(
                (
                    [lower[0], lower[1], lower[2], middle[2]],
                    [lower[0], lower[1], middle[1], middle[2]],
                    [lower[0], middle[0], middle[1], middle[2]],
                    [upper[0], upper[1], upper[2], middle[2]],
                    [upper[0], upper[1], middle[1], middle[2]],
                    [upper[0], middle[0], middle[1], middle[2]],
                )
            )
        elif axis_count == 1:
            axis_position = int(np.flatnonzero(axis[ordered])[0])
            solid_axis = lower[axis_position]
            first, second = [
                position for position in range(3) if position != axis_position
            ]
            tetrahedra.extend(
                (
                    [solid_axis, lower[first], lower[second], middle[second]],
                    [solid_axis, lower[first], middle[second], middle[first]],
                    [solid_axis, upper[first], upper[second], middle[second]],
                    [solid_axis, upper[first], middle[second], middle[first]],
                )
            )
        elif axis_count == 2:
            radial_position = int(np.flatnonzero(~axis[ordered])[0])
            axis_positions = [
                position for position in range(3) if position != radial_position
            ]
            tetrahedra.append(
                [
                    lower[axis_positions[0]],
                    lower[axis_positions[1]],
                    lower[radial_position],
                    middle[radial_position],
                ]
            )
            tetrahedra.append(
                [
                    upper[axis_positions[0]],
                    upper[axis_positions[1]],
                    upper[radial_position],
                    middle[radial_position],
                ]
            )

    points_array = np.asarray(points, dtype=float)
    tets = np.asarray(tetrahedra, dtype=int).reshape((-1, 4))
    if tets.size == 0:
        raise RuntimeError("Planar-to-axisymmetric wedge extrusion produced no tetrahedra")
    tet_points = points_array[tets]
    signed = np.einsum(
        "ij,ij->i",
        tet_points[:, 1] - tet_points[:, 0],
        np.cross(
            tet_points[:, 2] - tet_points[:, 0],
            tet_points[:, 3] - tet_points[:, 0],
        ),
    )
    negative = signed < 0.0
    if np.any(negative):
        swapped = tets[negative, 1].copy()
        tets[negative, 1] = tets[negative, 2]
        tets[negative, 2] = swapped
    keep = np.abs(signed) > 6.0e-26
    return (
        points_array,
        tets[keep],
        np.asarray(base_vertex, dtype=int),
        np.asarray(side_membership, dtype=bool),
    )


def boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    tet = np.asarray(tets, dtype=int).reshape((-1, 4))
    faces = np.vstack(
        (tet[:, (0, 1, 2)], tet[:, (0, 1, 3)], tet[:, (0, 2, 3)], tet[:, (1, 2, 3)])
    )
    canonical = np.sort(faces, axis=1)
    unique, first, count = np.unique(canonical, axis=0, return_index=True, return_counts=True)
    del unique
    return faces[first[count == 1]]


def _ordered_loop(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    ids = np.unique(np.asarray(vertices, dtype=int))
    angle = np.arctan2(points[ids, 1], points[ids, 0])
    return ids[np.argsort(np.mod(angle, 2.0 * math.pi))]


def _face_edges(faces: np.ndarray) -> np.ndarray:
    """Return unique canonical edges from a triangular face set."""

    face = np.asarray(faces, dtype=int).reshape((-1, 3))
    if face.size == 0:
        return np.empty((0, 2), dtype=int)
    edges = np.vstack((face[:, (0, 1)], face[:, (1, 2)], face[:, (2, 0)]))
    return np.unique(np.sort(edges, axis=1), axis=0)


def _shared_face_boundary_vertices(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vertices on the edge loop shared by two boundary-face patches."""

    first_edges = _face_edges(first)
    second_edges = _face_edges(second)
    if first_edges.size == 0 or second_edges.size == 0:
        return np.empty(0, dtype=int)
    edge_dtype = np.dtype((np.void, first_edges.dtype.itemsize * first_edges.shape[1]))
    first_view = np.ascontiguousarray(first_edges).view(edge_dtype).ravel()
    second_view = np.ascontiguousarray(second_edges).view(edge_dtype).ravel()
    shared = first_edges[np.isin(first_view, second_view)]
    return np.unique(shared) if shared.size else np.empty(0, dtype=int)


def _distance_to_meridian_polyline(
    query_rz: np.ndarray,
    polyline_r: np.ndarray,
    polyline_z: np.ndarray,
) -> np.ndarray:
    """Minimum 2D distance from meridian points to a polyline."""

    query = np.asarray(query_rz, dtype=float).reshape((-1, 2))
    polyline = np.column_stack(
        (np.asarray(polyline_r, dtype=float), np.asarray(polyline_z, dtype=float))
    )
    start = polyline[:-1]
    delta = polyline[1:] - start
    length2 = np.sum(delta * delta, axis=1)
    result = np.full(len(query), np.inf, dtype=float)
    for first in range(0, len(query), 2048):
        block = query[first : first + 2048]
        relative = block[:, None, :] - start[None, :, :]
        projection = np.sum(relative * delta[None, :, :], axis=2) / np.maximum(
            length2[None, :], 1.0e-30
        )
        projection = np.clip(projection, 0.0, 1.0)
        closest = start[None, :, :] + projection[:, :, None] * delta[None, :, :]
        distance2 = np.sum((block[:, None, :] - closest) ** 2, axis=2)
        result[first : first + len(block)] = np.sqrt(np.min(distance2, axis=1))
    return result


def _prune_collinear_meridian_vertices(
    radius_m: np.ndarray,
    height_m: np.ndarray,
    *,
    tolerance_m: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove redundant vertices without changing a meridian polyline.

    A Lagrangian complete-wetting contour can carry several material samples
    on the same nearly straight rim segment.  Passing sub-resolution samples
    to a constrained mesher adds no resolved geometry, but makes edge recovery
    arbitrarily ill conditioned.  A point is removed only when its distance
    from the segment joining its retained neighbours is below the caller's
    declared geometric tolerance; the endpoints, contact and rim are kept.
    """

    radius = np.asarray(radius_m, dtype=float).reshape(-1)
    height = np.asarray(height_m, dtype=float).reshape(-1)
    if radius.shape != height.shape:
        raise ValueError("radius_m and height_m must have equal shapes")
    if len(radius) <= 2:
        return radius.copy(), height.copy()
    point = np.column_stack((radius, height))
    kept = [0]
    for index in range(1, len(point) - 1):
        start = point[kept[-1]]
        end = point[index + 1]
        chord = end - start
        length2 = float(np.dot(chord, chord))
        if length2 <= 1.0e-30:
            kept.append(index)
            continue
        relative = point[index] - start
        fraction = float(np.dot(relative, chord) / length2)
        closest = start + np.clip(fraction, 0.0, 1.0) * chord
        distance = float(np.linalg.norm(point[index] - closest))
        if not (-1.0e-12 <= fraction <= 1.0 + 1.0e-12):
            kept.append(index)
        elif distance > float(tolerance_m):
            kept.append(index)
    kept.append(len(point) - 1)
    kept_array = np.asarray(kept, dtype=int)
    return radius[kept_array], height[kept_array]


def _densify_meridian_segments(
    radius_m: np.ndarray,
    height_m: np.ndarray,
    *,
    maximum_segment_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a boundary polyline so every physical edge is mesh-resolved."""

    radius = np.asarray(radius_m, dtype=float).reshape(-1)
    height = np.asarray(height_m, dtype=float).reshape(-1)
    if radius.shape != height.shape or len(radius) < 2:
        raise ValueError("A meridian needs at least two matched coordinates")
    maximum = float(maximum_segment_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_segment_m must be finite and positive")
    dense_radius = [float(radius[0])]
    dense_height = [float(height[0])]
    for index in range(len(radius) - 1):
        length = float(
            math.hypot(
                radius[index + 1] - radius[index],
                height[index + 1] - height[index],
            )
        )
        pieces = max(1, int(math.ceil(length / maximum)))
        for piece in range(1, pieces + 1):
            fraction = float(piece) / float(pieces)
            dense_radius.append(
                float(radius[index] + fraction * (radius[index + 1] - radius[index]))
            )
            dense_height.append(
                float(height[index] + fraction * (height[index + 1] - height[index]))
            )
    return np.asarray(dense_radius), np.asarray(dense_height)


def _densify_meridian_segments_near_contact_and_neck(
    radius_m: np.ndarray,
    height_m: np.ndarray,
    *,
    fine_segment_m: float,
    coarse_segment_m: float,
    transition_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Densify only near the moving contact line and bridge neck.

    The contact line is the first meridian point and the neck is the minimum
    radius.  Segment size increases smoothly with arc-length distance from
    those two features, so the refined band follows them while old interior
    regions coarsen as the bridge grows.  Added points remain exactly on the
    supplied polyline; no normal displacement or profile fitting occurs.
    """

    radius = np.asarray(radius_m, dtype=float).reshape(-1)
    height = np.asarray(height_m, dtype=float).reshape(-1)
    if radius.shape != height.shape or len(radius) < 2:
        raise ValueError("A meridian needs at least two matched coordinates")
    fine = float(fine_segment_m)
    coarse = max(float(coarse_segment_m), fine)
    transition = float(transition_length_m)
    if not all(math.isfinite(value) and value > 0.0 for value in (fine, coarse, transition)):
        raise ValueError("Adaptive meridian sizes must be finite and positive")
    segment = np.hypot(np.diff(radius), np.diff(height))
    arc = np.concatenate(([0.0], np.cumsum(segment)))
    neck_arc = float(arc[int(np.argmin(radius))])
    dense_radius = [float(radius[0])]
    dense_height = [float(height[0])]
    def local_size(arc_position: float) -> float:
        feature_distance = min(
            float(arc_position), abs(float(arc_position) - neck_arc)
        )
        ratio = feature_distance / transition
        blend = ratio * ratio / (1.0 + ratio * ratio)
        return fine + (coarse - fine) * blend

    for index, length in enumerate(segment):
        # Bisect against the smallest local target at the interval endpoints
        # and midpoint.  Unlike a single midpoint test, this guarantees that
        # no coarse interval can terminate at the contact line or neck.
        accepted_fractions: list[float] = []
        pending = [(0.0, 1.0)]
        while pending:
            start, end = pending.pop()
            midpoint = 0.5 * (start + end)
            interval_length = float(length) * (end - start)
            start_arc = float(arc[index]) + start * float(length)
            midpoint_arc = float(arc[index]) + midpoint * float(length)
            end_arc = float(arc[index]) + end * float(length)
            maximum = min(
                local_size(start_arc),
                local_size(midpoint_arc),
                local_size(end_arc),
            )
            if interval_length <= maximum:
                accepted_fractions.append(end)
            else:
                pending.append((midpoint, end))
                pending.append((start, midpoint))
        for fraction in sorted(set(accepted_fractions)):
            dense_radius.append(
                float(radius[index] + fraction * (radius[index + 1] - radius[index]))
            )
            dense_height.append(
                float(height[index] + fraction * (height[index + 1] - height[index]))
            )
    return np.asarray(dense_radius), np.asarray(dense_height)


def build_sphere_film_bridge_mesh(
    *,
    sphere_radius_m: float,
    sphere_tip_z_m: float,
    substrate_radius_m: float,
    film_thickness_m: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    surface_tension_n_m: float,
    resolved_neck_width_m: float,
    azimuthal_sectors: int = 16,
    outer_film_samples: int = 44,
    sphere_arc_samples: int = 20,
    sphere_surface_mesh_size_m: float | None = None,
    film_vertical_layers: int = 0,
    film_layer_radial_cutoff_m: float | None = None,
    edge_height_m: float = 1.0e-6,
    bulk_mesh_size_m: float = 4.5e-4,
    anisotropic_revolution: bool = True,
    axisymmetric_wedge_angle_rad: float | None = None,
    write_msh_path: str | None = None,
    bridge_profile: YoungLaplaceMeridian | None = None,
    outer_profile_r_m: np.ndarray | None = None,
    outer_profile_z_m: np.ndarray | None = None,
    continuous_profile_r_m: np.ndarray | None = None,
    continuous_profile_z_m: np.ndarray | None = None,
    adaptive_contact_neck_refinement: bool = False,
    adaptive_refinement_transition_m: float = 4.0e-5,
    adaptive_bridge_max_segment_m: float | None = None,
    preserve_bridge_profile_vertices: bool = False,
) -> UnstructuredBridgeMesh:
    """Build a tetra mesh with a resolved bridge overhang.

    ``axisymmetric_wedge_angle_rad=None`` preserves the historical full
    360-degree revolution.  A finite angle is an opt-in narrow 3-D wedge for
    an exact-axisymmetric solve; it does not silently change existing cases.
    """

    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("The unstructured bridge mesh requires the gmsh package") from exc

    capillary_length = math.sqrt(
        float(surface_tension_n_m)
        / max(float(density_kg_m3) * float(gravity_m_s2), 1.0e-30)
    )
    if bridge_profile is None:
        contact_radius, profile = resolved_contact_radius(
            target_rim_offset_m=float(resolved_neck_width_m),
            sphere_radius_m=float(sphere_radius_m),
            sphere_tip_z_m=float(sphere_tip_z_m),
            film_junction_z_m=float(film_thickness_m),
            density_kg_m3=float(density_kg_m3),
            gravity_m_s2=float(gravity_m_s2),
            surface_tension_n_m=float(surface_tension_n_m),
        )
    else:
        profile = bridge_profile
        contact_radius = float(profile.contact_radius_m)
    rim_radius = float(profile.rim_radius_m)
    continuous_radius: np.ndarray | None = None
    continuous_height: np.ndarray | None = None
    if continuous_profile_r_m is not None or continuous_profile_z_m is not None:
        if continuous_profile_r_m is None or continuous_profile_z_m is None:
            raise ValueError(
                "Both continuous_profile_r_m and continuous_profile_z_m are required"
            )
        continuous_radius = np.asarray(continuous_profile_r_m, dtype=float)
        continuous_height = np.asarray(continuous_profile_z_m, dtype=float)
        if (
            continuous_radius.shape != continuous_height.shape
            or continuous_radius.ndim != 1
            or len(continuous_radius) < 4
            or not np.all(np.isfinite(continuous_radius))
            or not np.all(np.isfinite(continuous_height))
        ):
            raise ValueError("The continuous free-surface profile is invalid")
        # Do not hand Gmsh boundary edges far below the declared physical
        # neck resolution.  Such interpolation duplicates create a
        # near-zero-area cusp and can make the valid inward-turning meridian
        # unmeshable.  This changes only boundary sampling: the transported
        # state and its diagnostic volume remain untouched.
        if not bool(preserve_bridge_profile_vertices):
            minimum_boundary_segment = max(
                1.0e-12, 0.1 * float(resolved_neck_width_m)
            )
            retained = [0]
            for index in range(1, len(continuous_radius) - 1):
                separation = math.hypot(
                    float(
                        continuous_radius[index]
                        - continuous_radius[retained[-1]]
                    ),
                    float(
                        continuous_height[index]
                        - continuous_height[retained[-1]]
                    ),
                )
                if separation >= minimum_boundary_segment:
                    retained.append(index)
            retained.append(len(continuous_radius) - 1)
            retained_index = np.asarray(retained, dtype=int)
            continuous_radius = continuous_radius[retained_index]
            continuous_height = continuous_height[retained_index]
        if abs(float(continuous_radius[0]) - contact_radius) > max(
            0.25 * float(resolved_neck_width_m), 1.0e-10
        ):
            raise ValueError("Continuous profile does not start at the contact line")
        if abs(float(continuous_radius[-1]) - float(substrate_radius_m)) > max(
            0.25 * float(resolved_neck_width_m), 1.0e-10
        ):
            raise ValueError("Continuous profile does not end at the substrate edge")
    if outer_profile_r_m is not None and outer_profile_z_m is not None:
        source_r = np.asarray(outer_profile_r_m, dtype=float)
        source_z = np.asarray(outer_profile_z_m, dtype=float)
        order = np.argsort(source_r)
        source_r = source_r[order]
        source_z = source_z[order]
        keep = np.concatenate(([True], np.diff(source_r) > 1.0e-14))
        source_r = source_r[keep]
        source_z = source_z[keep]
        inside = (source_r >= rim_radius - 1.0e-12) & (
            source_r <= float(substrate_radius_m) + 1.0e-12
        )
        outer_radius = source_r[inside].copy()
        outer_height = source_z[inside].copy()
        if len(outer_radius) < 3:
            raise ValueError("The supplied outer profile must contain at least three radii")
        outer_radius[0] = rim_radius
        outer_radius[-1] = float(substrate_radius_m)
        outer_height[0] = float(film_thickness_m)
        outer_height[-1] = max(float(edge_height_m), float(outer_height[-1]))
    else:
        outer_s = np.linspace(0.0, 1.0, max(10, int(outer_film_samples)))
        outer_radius = rim_radius + (
            float(substrate_radius_m) - rim_radius
        ) * outer_s**1.55
        outer_height = gravity_film_height(
            outer_radius,
            center_height_m=float(film_thickness_m),
            substrate_radius_m=float(substrate_radius_m),
            capillary_length_m=capillary_length,
            edge_height_m=float(edge_height_m),
        )

    polygon: list[tuple[float, float]] = []
    sizes: list[float] = []

    def add(radius: float, height: float, size: float) -> None:
        point = (float(radius), float(height))
        if polygon and np.linalg.norm(np.asarray(polygon[-1]) - np.asarray(point)) <= 1.0e-14:
            return
        polygon.append(point)
        sizes.append(float(size))

    add(0.0, 0.0, min(float(bulk_mesh_size_m), 0.4 * float(film_thickness_m)))
    add(float(substrate_radius_m), 0.0, float(bulk_mesh_size_m))
    add(float(substrate_radius_m), float(outer_height[-1]), float(bulk_mesh_size_m))
    if continuous_radius is not None and continuous_height is not None:
        continuous_segment = np.hypot(
            np.diff(continuous_radius), np.diff(continuous_height)
        )
        if np.any(continuous_segment <= 1.0e-14):
            raise ValueError("Continuous free-surface profile has a collapsed edge")
        continuous_spacing = np.empty_like(continuous_radius)
        continuous_spacing[0] = continuous_segment[0]
        continuous_spacing[-1] = continuous_segment[-1]
        if len(continuous_spacing) > 2:
            continuous_spacing[1:-1] = np.maximum(
                continuous_segment[:-1], continuous_segment[1:]
            )
        for radius, height, spacing in zip(
            continuous_radius[::-1][1:],
            continuous_height[::-1][1:],
            continuous_spacing[::-1][1:],
        ):
            local_size = max(
                float(resolved_neck_width_m),
                min(1.05 * float(spacing), float(bulk_mesh_size_m)),
            )
            add(float(radius), float(height), local_size)
    outer_spacing = np.empty_like(outer_radius)
    outer_spacing[0] = outer_radius[1] - outer_radius[0]
    outer_spacing[-1] = outer_radius[-1] - outer_radius[-2]
    if len(outer_spacing) > 2:
        outer_spacing[1:-1] = np.maximum(
            outer_radius[1:-1] - outer_radius[:-2],
            outer_radius[2:] - outer_radius[1:-1],
        )
    outer_boundary_iterator = (
        zip(
            outer_radius[::-1][1:],
            outer_height[::-1][1:],
            outer_spacing[::-1][1:],
        )
        if continuous_radius is None
        else ()
    )
    for radius, height, spacing in outer_boundary_iterator:
        distance = max(float(radius) - rim_radius, float(resolved_neck_width_m))
        # Preserve the computed ALE profile as the actual free-surface
        # polyline.  A characteristic length smaller than one profile segment
        # makes Gmsh insert collinear midpoint rings; the cotangent curvature
        # then aliases onto alternating original rings.  The local profile
        # spacing already supplies the requested neck refinement.
        local_size = max(
            float(np.clip(
                0.12 * distance,
                2.0 * resolved_neck_width_m,
                bulk_mesh_size_m,
            )),
            min(1.05 * float(spacing), float(bulk_mesh_size_m)),
        )
        add(float(radius), float(height), float(local_size))
    raw_profile_radius = np.asarray(profile.r_m, dtype=float)
    raw_profile_height = np.asarray(profile.z_m, dtype=float)
    resolved_geometry_tolerance = max(
        1.0e-12, 0.001 * float(resolved_neck_width_m)
    )
    # A deliberately vanishing preconnected topology has no resolved vertical
    # span; its tiny hairpin defines connectivity and must not be simplified
    # away.  Once a physical bridge spans more than the tolerance, the same
    # tolerance removes only redundant sub-resolution material sampling.
    if bool(preserve_bridge_profile_vertices):
        # A material-profile initializer promises that every supplied
        # meridian vertex becomes a boundary degree of freedom.  Even a
        # picometre collinearity tolerance can remove an almost-flat Case29
        # startup ring, changing both its eight-ring topology and the
        # discrete Heron force before the first physical solve.
        profile_radius = raw_profile_radius.copy()
        profile_height = raw_profile_height.copy()
    else:
        pruning_tolerance = (
            resolved_geometry_tolerance
            if float(np.ptp(raw_profile_height))
            >= 2.0 * resolved_geometry_tolerance
            else 1.0e-12
        )
        profile_radius, profile_height = _prune_collinear_meridian_vertices(
            raw_profile_radius,
            raw_profile_height,
            # The free-boundary finite-element resolution is declared
            # explicitly by resolved_neck_width_m. Curvature is a second
            # derivative: pruning at one percent of a neck cell left only
            # about ten meridian rings and biased the discrete FHeron pressure
            # even though the position error looked small. Keep points whose
            # chord defect reaches one part per thousand of the declared neck
            # resolution. This is a curvature-convergence setting, not an
            # experimental scale.
            tolerance_m=pruning_tolerance,
        )
    if float(np.ptp(raw_profile_height)) >= 2.0 * resolved_geometry_tolerance:
        # A constrained curve keeps its endpoints, but Gmsh is free to use a
        # long straight segment as one boundary edge.  That left a 124 um gap
        # immediately after a nominal 1.25 um neck edge, eliminating the DDG
        # curvature and ALE degrees of freedom needed by FHeron.  Explicitly
        # split every resolved bridge segment at the declared neck resolution.
        # The vanishing t=0 topology is excluded so its flat preallocated
        # branch is not globally over-refined before physical contact.
        if bool(adaptive_contact_neck_refinement):
            adaptive_maximum = (
                float(bulk_mesh_size_m)
                if adaptive_bridge_max_segment_m is None
                else min(
                    float(adaptive_bridge_max_segment_m),
                    float(bulk_mesh_size_m),
                )
            )
            profile_radius, profile_height = (
                _densify_meridian_segments_near_contact_and_neck(
                    profile_radius,
                    profile_height,
                    fine_segment_m=float(resolved_neck_width_m),
                    coarse_segment_m=adaptive_maximum,
                    transition_length_m=float(adaptive_refinement_transition_m),
                )
            )
        else:
            if bool(preserve_bridge_profile_vertices) and len(profile_radius) > 2:
                # The positive seed explicitly declares its first
                # CL-adjacent material edge.  Resolve curvature downstream,
                # but do not split that edge into sub-micron vertices which
                # the advancing CL would immediately overtake.
                dense_radius, dense_height = _densify_meridian_segments(
                    profile_radius[1:],
                    profile_height[1:],
                    maximum_segment_m=float(resolved_neck_width_m),
                )
                profile_radius = np.concatenate(
                    (profile_radius[:1], dense_radius)
                )
                profile_height = np.concatenate(
                    (profile_height[:1], dense_height)
                )
            else:
                profile_radius, profile_height = _densify_meridian_segments(
                    profile_radius,
                    profile_height,
                    maximum_segment_m=float(resolved_neck_width_m),
                )
    profile_segment = np.hypot(
        np.diff(profile_radius), np.diff(profile_height)
    )
    profile_spacing = np.empty_like(profile_radius)
    profile_spacing[0] = profile_segment[0]
    profile_spacing[-1] = profile_segment[-1]
    if len(profile_spacing) > 2:
        profile_spacing[1:-1] = np.maximum(
            profile_segment[:-1], profile_segment[1:]
        )
    if (
        bool(preserve_bridge_profile_vertices)
        and polygon
        and abs(float(polygon[-1][0]) - float(profile_radius[-1]))
        <= 1.0e-14
        and abs(float(polygon[-1][1]) - float(profile_height[-1]))
        <= resolved_geometry_tolerance
    ):
        # The rim was inserted by the outer-film iterator. Give that shared
        # endpoint the bridge-side spacing as well, otherwise Gmsh may split
        # the last declared material edge solely because the outer-film
        # characteristic length is smaller.
        sizes[-1] = max(
            float(sizes[-1]),
            1.05 * float(profile_spacing[-1]),
        )
    bridge_boundary_iterator = (
        zip(
            profile_radius[::-1][1:],
            profile_height[::-1][1:],
            profile_spacing[::-1][1:],
        )
        if continuous_radius is None
        else ()
    )
    for radius, height, spacing in bridge_boundary_iterator:
        if bool(preserve_bridge_profile_vertices):
            local_size = max(
                float(resolved_neck_width_m),
                1.05 * float(spacing),
            )
        else:
            local_size = max(
                float(resolved_neck_width_m),
                min(1.05 * float(spacing), float(bulk_mesh_size_m)),
            )
        add(float(radius), float(height), local_size)
    contact_polygon_index = len(polygon) - 1
    # Cluster the exact circular boundary near the three-phase contact line.
    # A uniform polygonal sphere has a long first chord which lies inside the
    # convex solid and can cross a micron-scale tangent bridge.  That creates a
    # self-intersecting liquid polygon even though both analytic surfaces are
    # valid.  Quadratic arc spacing removes the chord error without changing
    # either physical surface or globally over-refining the sphere.
    sphere_sample_count = max(12, int(sphere_arc_samples))
    sphere_parameter = np.linspace(0.0, 1.0, sphere_sample_count)
    sphere_radius_samples = contact_radius * (1.0 - sphere_parameter**2)
    sphere_height_samples = sphere_lower_z(
        sphere_radius_samples,
        sphere_radius_m=float(sphere_radius_m),
        sphere_tip_z_m=float(sphere_tip_z_m),
    )
    sphere_segment = np.hypot(
        np.diff(sphere_radius_samples), np.diff(sphere_height_samples)
    )
    sphere_spacing = np.empty_like(sphere_radius_samples)
    sphere_spacing[0] = sphere_segment[0]
    sphere_spacing[-1] = sphere_segment[-1]
    if len(sphere_spacing) > 2:
        sphere_spacing[1:-1] = np.maximum(
            sphere_segment[:-1], sphere_segment[1:]
        )
    # Neck-scale elements are needed at the triple line, but applying that
    # size to the entire wetted sphere wastes nearly all tetrahedra on a
    # fixed, analytic solid boundary.  Case-specific callers may cap the
    # sphere size while retaining the quadratically clustered exact arc.
    # ``None`` preserves the historical globally fine sphere mesh.
    if sphere_surface_mesh_size_m is None:
        sphere_sizes = np.full_like(
            sphere_radius_samples, 2.0 * float(resolved_neck_width_m)
        )
    else:
        sphere_sizes = np.clip(
            1.05 * sphere_spacing,
            float(resolved_neck_width_m),
            float(sphere_surface_mesh_size_m),
        )
    for radius, height, local_size in zip(
        sphere_radius_samples[1:],
        sphere_height_samples[1:],
        sphere_sizes[1:],
    ):
        add(float(radius), float(height), float(local_size))

    # Radial neck resolution and azimuthal resolution are independent.  The
    # short meridian segments resolve the neck; a larger characteristic size
    # around the revolved loop avoids thousands of unnecessary CL vertices.
    boundary_size_floor = max(0.5 * float(resolved_neck_width_m), 0.25e-6)
    sizes = [max(float(size), boundary_size_floor) for size in sizes]
    gmsh.initialize()
    revolved_base_vertex: np.ndarray | None = None
    revolved_side_membership: np.ndarray | None = None
    planar_boundary_sets: dict[str, np.ndarray] | None = None
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        # The resolved coalescence neck can contain adjacent meridian points
        # below OCC's default 1e-7 model-unit tolerance.  Treating those
        # distinct points as coincident prevents creation of the physical
        # inward-turning neck.  The model is expressed in metres, so use a
        # tolerance safely below the smallest supported free-surface edge.
        gmsh.option.setNumber("Geometry.Tolerance", 1.0e-12)
        gmsh.option.setNumber("Geometry.MatchMeshTolerance", 1.0e-11)
        gmsh.model.add("ddgclib_sphere_film_bridge")
        # The axisymmetric path only needs a constrained x-z triangulation.
        # OCC's generic inverse surface mapper can lose the macroscopic film
        # domain when the exact zero-angle liquid/solid cusp becomes very
        # narrow, leaving just the first two triangles at the contact line.
        # Gmsh's built-in planar kernel represents the same polygon directly
        # in its natural coordinates and remains well posed at that cusp.
        # Retain OCC for the optional true 3-D revolve below.
        geometry = gmsh.model.geo if bool(anisotropic_revolution) else gmsh.model.occ
        point_tags = [
            geometry.addPoint(radius, 0.0, height, size)
            for (radius, height), size in zip(polygon, sizes)
        ]
        line_tags = [
            geometry.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
            for i in range(len(point_tags))
        ]
        if bool(anisotropic_revolution):
            wire = geometry.addCurveLoop(line_tags)
        else:
            wire = geometry.addWire(line_tags)
        surface = geometry.addPlaneSurface([wire])
        embedded_film_points: list[int] = []
        layer_count = max(0, int(film_vertical_layers))
        if layer_count:
            cutoff = (
                float(substrate_radius_m)
                if film_layer_radial_cutoff_m is None
                else min(
                    float(substrate_radius_m),
                    float(film_layer_radial_cutoff_m),
                )
            )
            for radius, height, spacing_local in zip(
                outer_radius[1:], outer_height[1:], outer_spacing[1:]
            ):
                # A through-thickness support point at the substrate's outer
                # radius lies on the outer-wall boundary, not in the planar
                # liquid interior.  Embedding it as a surface-interior point
                # makes Gmsh return an empty triangulation when the requested
                # layer cutoff reaches the complete Case69 film radius.
                if (
                    float(radius) > cutoff
                    or float(radius)
                    >= float(substrate_radius_m) - 1.0e-12
                ):
                    break
                for layer in range(1, layer_count + 1):
                    fraction = float(layer) / float(layer_count + 1)
                    local_size = max(
                        float(resolved_neck_width_m),
                        min(
                            1.05 * float(spacing_local),
                            float(bulk_mesh_size_m),
                        ),
                    )
                    embedded_film_points.append(
                        geometry.addPoint(
                            float(radius),
                            0.0,
                            fraction * float(height),
                            local_size,
                        )
                    )
        geometry.synchronize()
        if embedded_film_points:
            gmsh.model.mesh.embed(0, embedded_film_points, 2, surface)
        gmsh.option.setNumber("Mesh.MeshSizeMin", boundary_size_floor)
        gmsh.option.setNumber(
            "Mesh.MeshSizeMax",
            max(float(bulk_mesh_size_m), max(float(size) for size in sizes)),
        )
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        if bool(anisotropic_revolution):
            # The just-resolved complete-wetting wedge can be valid yet too
            # anisotropic for one particular 2-D meshing algorithm.  Try the
            # standard Delaunay and frontal variants on the identical
            # boundary before declaring the physical polygon unmeshable.
            # This changes only volume-mesh connectivity; the computed free
            # surface and every boundary point remain unchanged.
            planar_points = np.empty((0, 3), dtype=float)
            triangles = np.empty((0, 3), dtype=int)
            planar_tags = np.empty(0, dtype=int)
            # Frontal-Delaunay (6) recovers an acute constrained wetting cusp
            # without the unbounded edge-recovery pass that the plain
            # Delaunay algorithm (5) can enter as the point-contact homotopy
            # opens.  Keep the other standard algorithms as connectivity-only
            # fallbacks for less anisotropic historical meshes.
            # Frontal-Delaunay is robust for the unresolved startup cusp, but
            # can spend minutes recovering edges once the bridge contains a
            # long, inward-turning neck.  DelQuad's frontal kernel handles
            # that developed constrained polygon directly.  Switch only at
            # twelve declared neck cells, the same resolution gate used by
            # the local bridge geometry; this is connectivity control, not a
            # physical or experimental parameter.
            developed_bridge = (
                float(np.ptp(raw_profile_height))
                >= 12.0 * float(resolved_neck_width_m)
                and float(np.ptp(raw_profile_radius))
                >= 12.0 * float(resolved_neck_width_m)
            )
            planar_algorithms = (
                (8, 6, 5) if developed_bridge else (6, 8, 5)
            )
            for algorithm in planar_algorithms:
                gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm", float(algorithm))
                gmsh.model.mesh.generate(2)
                planar_points, triangles, planar_tags = _extract_planar_triangles(
                    gmsh
                )
                if len(triangles):
                    break
            tag_to_planar = {
                int(tag): index for index, tag in enumerate(planar_tags)
            }

            def planar_nodes_on_lines(tags: list[int]) -> np.ndarray:
                node_tags: set[int] = set()
                for line_tag in tags:
                    entity_tags, _coordinates, _parametric = gmsh.model.mesh.getNodes(
                        1, int(line_tag), includeBoundary=True
                    )
                    node_tags.update(int(tag) for tag in entity_tags)
                return np.asarray(
                    [tag_to_planar[tag] for tag in node_tags if tag in tag_to_planar],
                    dtype=int,
                )

            planar_boundary_sets = {
                "substrate": planar_nodes_on_lines([line_tags[0]]),
                "outer": planar_nodes_on_lines([line_tags[1]]),
                "free": planar_nodes_on_lines(
                    line_tags[2:contact_polygon_index]
                ),
                "sphere": planar_nodes_on_lines(
                    line_tags[contact_polygon_index : len(line_tags) - 1]
                ),
            }
            if axisymmetric_wedge_angle_rad is None:
                points, tets, revolved_base_vertex = _revolve_planar_triangles(
                    planar_points, triangles, int(azimuthal_sectors)
                )
            else:
                (
                    points,
                    tets,
                    revolved_base_vertex,
                    revolved_side_membership,
                ) = _extrude_planar_triangles_to_axisymmetric_wedge(
                    planar_points,
                    triangles,
                    float(axisymmetric_wedge_angle_rad),
                )
        else:
            occ = gmsh.model.occ
            revolved = occ.revolve(
                [(2, surface)],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0 * math.pi,
            )
            occ.removeAllDuplicates()
            occ.synchronize()
            if not any(dim == 3 for dim, _tag in revolved):
                raise RuntimeError("Gmsh did not create a revolved liquid volume")
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.model.mesh.generate(3)
            if write_msh_path is not None:
                gmsh.write(str(write_msh_path))
            points, tets = _extract_tets(gmsh)
    finally:
        # Repeated ALE remeshing can otherwise leave OCC/Geo model storage in
        # Gmsh's native allocator until process exit, even though the Python
        # model objects are gone.  Clearing the model is numerically neutral
        # and keeps long fixed-timestep runs at bounded memory.
        try:
            gmsh.clear()
        finally:
            gmsh.finalize()

    # The anisotropic revolve can receive two planar node tags at an exact
    # curve junction (most often the bridge/outer-film rim).  Revolving both
    # tags creates coincident 3-D vertices and zero-volume tetrahedra even
    # though the meridian polygon itself is valid.  Merge exact coordinate
    # duplicates before classifying or assembling the volume mesh.  Exact
    # matching is intentionally used here: nearby physical neck nodes must
    # remain distinct.
    unique_points, first_occurrence, inverse_point = np.unique(
        np.asarray(points, dtype=float),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    if len(unique_points) != len(points):
        points = unique_points
        tets = inverse_point[np.asarray(tets, dtype=int)]
        if revolved_base_vertex is not None:
            revolved_base_vertex = np.asarray(revolved_base_vertex, dtype=int)[
                first_occurrence
            ]
        if revolved_side_membership is not None:
            revolved_side_membership = np.asarray(
                revolved_side_membership, dtype=bool
            )[first_occurrence]

    tet_points = points[tets]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        tet_points[:, 1] - tet_points[:, 0],
        np.cross(tet_points[:, 2] - tet_points[:, 0], tet_points[:, 3] - tet_points[:, 0]),
    )
    tets = tets[np.abs(signed_six_volume) > 6.0e-26]
    used_vertices = np.unique(tets)
    old_to_new = np.full(len(points), -1, dtype=int)
    old_to_new[used_vertices] = np.arange(len(used_vertices), dtype=int)
    points = points[used_vertices]
    if revolved_base_vertex is not None:
        revolved_base_vertex = revolved_base_vertex[used_vertices]
    if revolved_side_membership is not None:
        revolved_side_membership = revolved_side_membership[used_vertices]
    tets = old_to_new[tets]
    boundary = boundary_faces_from_tets(tets)
    radius = np.hypot(points[:, 0], points[:, 1])
    if revolved_side_membership is None:
        wedge_side_face = np.zeros(len(boundary), dtype=bool)
    else:
        wedge_side_face = np.all(
            revolved_side_membership[boundary, 0], axis=1
        ) | np.all(revolved_side_membership[boundary, 1], axis=1)
    if revolved_base_vertex is not None and planar_boundary_sets is not None:
        substrate_vertex = np.isin(
            revolved_base_vertex, planar_boundary_sets["substrate"]
        )
        outer_vertex = np.isin(revolved_base_vertex, planar_boundary_sets["outer"])
        sphere_vertex_from_entity = np.isin(
            revolved_base_vertex, planar_boundary_sets["sphere"]
        )
        free_vertex_from_entity = np.isin(
            revolved_base_vertex, planar_boundary_sets["free"]
        )
        substrate_face = np.all(substrate_vertex[boundary], axis=1)
        outer_face = np.all(outer_vertex[boundary], axis=1)
        sphere_face = np.all(sphere_vertex_from_entity[boundary], axis=1)
        free_face = np.all(free_vertex_from_entity[boundary], axis=1)
        unresolved = ~(sphere_face | free_face | substrate_face | outer_face)
    else:
        substrate_vertex = points[:, 2] <= max(
            1.0e-10, 0.03 * resolved_neck_width_m
        )
        outer_vertex = radius >= float(substrate_radius_m) - max(
            2.0 * resolved_neck_width_m, 1.0e-7
        )
        substrate_face = np.all(substrate_vertex[boundary], axis=1)
        outer_face = np.all(outer_vertex[boundary], axis=1)
        sphere_face = np.zeros(len(boundary), dtype=bool)
        free_face = np.zeros(len(boundary), dtype=bool)
        unresolved = ~(substrate_face | outer_face)
    substrate_face &= ~wedge_side_face
    outer_face &= ~wedge_side_face
    centroid = np.mean(points[boundary], axis=1)
    centroid_rz = np.column_stack(
        (np.hypot(centroid[:, 0], centroid[:, 1]), centroid[:, 2])
    )
    boundary_vertex_rz = np.stack(
        (radius[boundary], points[boundary, 2]), axis=2
    )
    sphere_distance = np.mean(
        _distance_to_meridian_polyline(
            boundary_vertex_rz.reshape((-1, 2)),
            sphere_radius_samples,
            sphere_height_samples,
        ).reshape((-1, 3)),
        axis=1,
    )
    free_reference_radius = (
        np.asarray(continuous_radius, dtype=float)
        if continuous_radius is not None
        else np.concatenate((np.asarray(profile.r_m), outer_radius[1:]))
    )
    free_reference_height = (
        np.asarray(continuous_height, dtype=float)
        if continuous_height is not None
        else np.concatenate((np.asarray(profile.z_m), outer_height[1:]))
    )
    free_distance = np.mean(
        _distance_to_meridian_polyline(
            boundary_vertex_rz.reshape((-1, 2)),
            free_reference_radius,
            free_reference_height,
        ).reshape((-1, 3)),
        axis=1,
    )
    # Boundary-entity tags become ambiguous where a tangent micron-scale
    # bridge meets a polygonal sphere: Gmsh can assign the first one or two
    # liquid-air segments to the neighboring sphere curve.  Classify every
    # non-wall boundary face against the two exact meridian polylines instead
    # of trusting those inherited tags near the junction.  This makes their
    # shared edge the actual analytic sphere/liquid/air triple line.
    # The two angular wedge planes are computational symmetry planes, not
    # liquid-air or liquid-solid interfaces.  They remain in boundary_faces
    # for topology/ALE bookkeeping but never enter a physical surface force.
    liquid_boundary = ~(substrate_face | outer_face | wedge_side_face)
    sphere_face = liquid_boundary & (sphere_distance < free_distance)
    free_face = liquid_boundary & ~sphere_face
    sphere_faces = boundary[sphere_face]
    free_faces = boundary[free_face]
    sphere_vertices = np.unique(sphere_faces)
    free_vertices = np.unique(free_faces)
    contact_vertices = _shared_face_boundary_vertices(sphere_faces, free_faces)
    minimum_contact_vertices = (
        2 if axisymmetric_wedge_angle_rad is not None else 8
    )
    if contact_vertices.size < minimum_contact_vertices:
        contact_height = float(
            sphere_lower_z(
                contact_radius,
                sphere_radius_m=float(sphere_radius_m),
                sphere_tip_z_m=float(sphere_tip_z_m),
            )
        )
        boundary_vertices = np.unique(boundary)
        contact_distance = np.hypot(
            radius[boundary_vertices] - float(contact_radius),
            points[boundary_vertices, 2] - contact_height,
        )
        nearest_contact_distance = float(np.min(contact_distance))
        contact_vertices = boundary_vertices[
            contact_distance
            <= nearest_contact_distance
            + max(0.05 * float(resolved_neck_width_m), 1.0e-10)
        ]
    contact_loop = _ordered_loop(points, contact_vertices)
    if contact_loop.size < minimum_contact_vertices:
        raise RuntimeError("Failed to recover the sphere-liquid-air contact loop")

    # Split the one continuous liquid-air surface into bridge and outer-film
    # patches by their generating meridian polylines, then recover the rim as
    # the shared topological edge.  A radial tolerance band is invalid for a
    # small overhanging bridge: at early times its entire surface can lie
    # within 0.02*r_CL of r_rim and was consequently mislabeled as the rim.
    free_vertex_rz = np.stack(
        (radius[free_faces], points[free_faces, 2]), axis=2
    )
    bridge_patch_distance = np.mean(
        _distance_to_meridian_polyline(
            free_vertex_rz.reshape((-1, 2)),
            np.asarray(profile.r_m, dtype=float),
            np.asarray(profile.z_m, dtype=float),
        ).reshape((-1, 3)),
        axis=1,
    )
    outer_patch_distance = np.mean(
        _distance_to_meridian_polyline(
            free_vertex_rz.reshape((-1, 2)),
            np.asarray(outer_radius, dtype=float),
            np.asarray(outer_height, dtype=float),
        ).reshape((-1, 3)),
        axis=1,
    )
    bridge_patch_face = bridge_patch_distance <= outer_patch_distance
    bridge_free_faces = free_faces[bridge_patch_face]
    outer_free_faces = free_faces[~bridge_patch_face]
    bridge_vertices = np.unique(bridge_free_faces)
    outer_film_vertices = np.unique(outer_free_faces)
    rim_vertices = _shared_face_boundary_vertices(
        bridge_free_faces, outer_free_faces
    )
    if rim_vertices.size < minimum_contact_vertices:
        boundary_free_vertices = np.unique(free_faces)
        rim_distance = np.hypot(
            radius[boundary_free_vertices] - float(rim_radius),
            points[boundary_free_vertices, 2] - float(film_thickness_m),
        )
        nearest_rim_distance = float(np.min(rim_distance))
        rim_vertices = boundary_free_vertices[
            rim_distance
            <= nearest_rim_distance
            + max(0.05 * float(resolved_neck_width_m), 1.0e-10)
        ]
    # Never allow a mesher failure to masquerade as a valid, nearly empty
    # bridge solve.  An OCC cusp failure previously retained a handful of
    # contact-line tetrahedra while silently dropping the whole substrate and
    # outer-film domain.  These are topology/inventory guards, not physical
    # calibration parameters.
    substrate_vertices = np.unique(boundary[substrate_face])
    outer_wall_vertices = np.unique(boundary[outer_face])
    minimum_loop_vertices = (
        2
        if axisymmetric_wedge_angle_rad is not None
        else max(8, int(azimuthal_sectors))
    )
    if len(substrate_vertices) < minimum_loop_vertices:
        raise RuntimeError("Tetra mesh lost the substrate boundary")
    if len(outer_wall_vertices) < minimum_loop_vertices:
        raise RuntimeError("Tetra mesh lost the outer-wall boundary")
    if len(outer_film_vertices) < minimum_loop_vertices:
        raise RuntimeError("Tetra mesh lost the outer free-film boundary")
    if float(np.max(radius)) < 0.99 * float(substrate_radius_m):
        raise RuntimeError("Tetra mesh no longer spans the physical substrate")
    tetrahedron = points[tets]
    six_volume = np.abs(
        np.einsum(
            "ij,ij->i",
            tetrahedron[:, 1] - tetrahedron[:, 0],
            np.cross(
                tetrahedron[:, 2] - tetrahedron[:, 0],
                tetrahedron[:, 3] - tetrahedron[:, 0],
            ),
        )
    )
    mesh_volume = float(np.sum(six_volume) / 6.0)
    nominal_film_volume = (
        math.pi * float(substrate_radius_m) ** 2 * float(film_thickness_m)
    )
    if axisymmetric_wedge_angle_rad is not None:
        # The straight-sided tetra wedge subtends a chordal area fraction
        # sin(alpha)/(2*pi), rather than the ideal sector alpha/(2*pi).
        nominal_film_volume *= math.sin(float(axisymmetric_wedge_angle_rad)) / (
            2.0 * math.pi
        )
    if mesh_volume < 0.10 * nominal_film_volume:
        raise RuntimeError(
            "Tetra mesh liquid inventory collapsed during remeshing: "
            f"{mesh_volume:.6e} m^3"
        )
    return UnstructuredBridgeMesh(
        points_m=points,
        tets=tets,
        boundary_faces=boundary,
        free_surface_faces=free_faces,
        sphere_faces=sphere_faces,
        substrate_faces=boundary[substrate_face],
        outer_wall_faces=boundary[outer_face],
        contact_line_vertices=contact_loop,
        sphere_vertices=sphere_vertices,
        substrate_vertices=substrate_vertices,
        outer_wall_vertices=outer_wall_vertices,
        outer_film_vertices=outer_film_vertices,
        bridge_surface_vertices=bridge_vertices,
        rim_vertices=np.unique(rim_vertices),
        contact_radius_m=contact_radius,
        rim_radius_m=rim_radius,
        pressure_jump_pa=float(profile.pressure_jump_pa),
        seed_neck_resolution_m=float(resolved_neck_width_m),
        axisymmetric_wedge_angle_rad=(
            None
            if axisymmetric_wedge_angle_rad is None
            else float(axisymmetric_wedge_angle_rad)
        ),
        # Sample inside one mirrored half-wedge, away from both the angular
        # boundary and the internal reflection plane where a tetra MINI
        # bubble vanishes.  The positive quarter-angle is equivalent by
        # reflection; storing one ray makes all flux evaluators agree.
        axisymmetric_wedge_midpoint_angle_rad=(
            0.0
            if axisymmetric_wedge_angle_rad is None
            else -0.25 * float(axisymmetric_wedge_angle_rad)
        ),
        axisymmetric_full_circle_scale=(
            1.0
            if axisymmetric_wedge_angle_rad is None
            else 2.0
            * math.pi
            / max(math.sin(float(axisymmetric_wedge_angle_rad)), 1.0e-30)
        ),
        axisymmetric_wedge_side_faces=boundary[wedge_side_face],
    )
