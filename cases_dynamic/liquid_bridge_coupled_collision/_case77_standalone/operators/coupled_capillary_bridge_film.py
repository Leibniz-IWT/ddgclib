"""Mixed-dimensional capillary bridge--film coupling operators.

The tetrahedral bridge and the axisymmetric lubrication film share one
junction.  Positive junction rate denotes liquid transferred from the outer
film into the bridge:

    q_r = -M(h) dp_f/dr,
    Q_J = -2*pi*r_J*q_r,
    dV_bridge/dt = Q_J,
    dV_film/dt = -Q_J.

The functions in this module contain no experimental trajectory, rate cap, or
profile reconstruction.  They provide the reusable physical closure and the
strictly conservative PR33 target-volume exchange used by Case57.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class JunctionFlux:
    """Axisymmetric lubrication flux at the bridge--film junction."""

    bridge_rate_m3_s: float
    radial_flux_m2_s: float
    mobility_m3_pa_s: float
    pressure_drop_pa: float
    pressure_gradient_pa_m: float
    film_thickness_m: float
    junction_radius_m: float
    pressure_length_m: float


@dataclass(frozen=True)
class SweptJunctionFlux:
    """Finite-contact film capture by a contact line on a sphere."""

    bridge_rate_m3_s: float
    contact_radius_m: float
    contact_speed_m_s: float
    film_height_m: float
    sphere_swept_height_m: float
    effective_swept_height_m: float


@dataclass(frozen=True)
class ConservativeTransfer:
    """Conservative receiver/donor target-volume update."""

    target_volumes_m3: np.ndarray
    actual_bridge_rate_m3_s: float
    bridge_volume_change_m3: float
    film_volume_change_m3: float
    net_volume_change_m3: float
    positivity_limited: bool


@dataclass(frozen=True)
class PrescribedFluxFilmDiagnostics:
    """Diagnostics for one conservative axisymmetric film update."""

    junction_rate_m3_s: float
    film_volume_change_m3: float
    conservation_residual_m3: float
    minimum_height_m: float
    minimum_radius_m: float
    junction_pressure_pa: float
    maximum_pressure_pa: float
    minimum_pressure_pa: float
    linear_solve_count: int


def axisymmetric_sphere_swept_junction_flux(
    *,
    contact_radius_m: float,
    film_height_m: float,
    sphere_radius_m: float,
    contact_speed_m_s: float,
) -> SweptJunctionFlux:
    """Return the signed liquid rate swept into a sphere--film bridge.

    The axisymmetric finite-contact closure is

    ``Q_cap = 2*pi*r_CL*(h_J + r_CL**2/(2*sqrt(R**2-r_CL**2)))*U_CL``.

    Positive contact speed advances the contact line and transfers liquid
    from the film into the bridge.  Negative speed reverses that transfer.
    No trajectory data, fitted rate, or numerical cap enters this operator.
    """

    radius = float(contact_radius_m)
    height = float(film_height_m)
    sphere = float(sphere_radius_m)
    speed = float(contact_speed_m_s)
    if not math.isfinite(sphere) or sphere <= 0.0:
        raise ValueError("sphere_radius_m must be finite and positive")
    if not math.isfinite(radius) or radius < 0.0 or radius >= sphere:
        raise ValueError(
            "contact_radius_m must be finite and lie in [0, sphere_radius_m)"
        )
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("film_height_m must be finite and nonnegative")
    if not math.isfinite(speed):
        raise ValueError("contact_speed_m_s must be finite")
    axial_radius = math.sqrt(sphere * sphere - radius * radius)
    sphere_height = radius * radius / (2.0 * axial_radius)
    effective_height = height + sphere_height
    rate = (
        2.0
        * math.pi
        * radius
        * effective_height
        * speed
    )
    return SweptJunctionFlux(
        bridge_rate_m3_s=float(rate),
        contact_radius_m=radius,
        contact_speed_m_s=speed,
        film_height_m=height,
        sphere_swept_height_m=float(sphere_height),
        effective_swept_height_m=float(effective_height),
    )


def lubrication_mobility(
    film_thickness_m: float,
    viscosity_pa_s: float,
    *,
    slip_length_m: float = 0.0,
) -> float:
    """Return the pressure-driven film mobility.

    The no-slip term is ``h**3/(3*mu)``.  A nonnegative Navier slip length
    adds ``b*h**2/mu``; the default is the no-slip model used in the paper.
    """

    height = float(film_thickness_m)
    viscosity = float(viscosity_pa_s)
    slip = float(slip_length_m)
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("film_thickness_m must be finite and nonnegative")
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity_pa_s must be finite and positive")
    if not math.isfinite(slip) or slip < 0.0:
        raise ValueError("slip_length_m must be finite and nonnegative")
    return height**3 / (3.0 * viscosity) + slip * height**2 / viscosity


def advance_axisymmetric_film_prescribed_junction_flux(
    radius_m: np.ndarray,
    height_m: np.ndarray,
    *,
    junction_rate_m3_s: float,
    dt_s: float,
    surface_tension_n_m: float,
    viscosity_pa_s: float,
    density_kg_m3: float,
    gravity_m_s2: float,
    slip_length_m: float = 0.0,
    mobility_average: str = "harmonic",
    picard_iterations: int = 2,
    picard_relaxation: float = 0.65,
    picard_tolerance_m: float = 2.0e-10,
    minimum_height_m: float = 1.0e-9,
    junction_sink_fraction: np.ndarray | None = None,
    junction_pressure_pa: float | None = None,
) -> tuple[np.ndarray, PrescribedFluxFilmDiagnostics]:
    """Advance a radial film with a prescribed conservative junction flux.

    The lagged-mobility backward-Euler system discretizes

    ``h_t + (1/r) d(r*q)/dr = 0``,
    ``q = -M(h) dp/dr``,
    ``p = -gamma*(1/r)d(r*h_r)/dr + rho*g*h``.

    ``junction_rate_m3_s`` is positive when liquid leaves the film and enters
    the bridge.  By default the inner through-flow is therefore
    ``2*pi*r*q = -junction_rate_m3_s``.  When
    ``junction_sink_fraction`` is provided, the same rate is removed as a
    normalized conservative cell sink instead.  The outer edge is
    impermeable.
    Radial slope is zero at both pressure-curvature boundaries by default.  If
    ``junction_pressure_pa`` is supplied, pressure continuity replaces the
    zero-slope condition at the inner boundary:
    ``p_f(radius_m[0]) = junction_pressure_pa``.  The outer slope remains zero.
    No fitted dimple width, height floor, rate cap, or experimental profile
    enters this operator.  A step that crosses ``minimum_height_m`` is
    rejected instead of clipped.
    """

    radius = np.asarray(radius_m, dtype=float)
    old_height = np.asarray(height_m, dtype=float)
    if radius.ndim != 1 or old_height.shape != radius.shape or radius.size < 5:
        raise ValueError("radius_m and height_m must be matching 1-D arrays with at least five entries")
    if np.any(~np.isfinite(radius)) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius_m must be finite and strictly increasing")
    if float(radius[0]) <= 0.0:
        raise ValueError("the axisymmetric film annulus must start at positive radius")
    if np.any(~np.isfinite(old_height)) or np.any(old_height <= 0.0):
        raise ValueError("height_m must be finite and strictly positive")
    dt = float(dt_s)
    gamma = float(surface_tension_n_m)
    viscosity = float(viscosity_pa_s)
    density = float(density_kg_m3)
    gravity = float(gravity_m_s2)
    junction_rate = float(junction_rate_m3_s)
    junction_pressure = (
        None if junction_pressure_pa is None else float(junction_pressure_pa)
    )
    slip = float(slip_length_m)
    minimum_height = float(minimum_height_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("surface_tension_n_m must be finite and positive")
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity_pa_s must be finite and positive")
    if not math.isfinite(density) or density < 0.0:
        raise ValueError("density_kg_m3 must be finite and nonnegative")
    if not math.isfinite(gravity) or gravity < 0.0:
        raise ValueError("gravity_m_s2 must be finite and nonnegative")
    if not math.isfinite(junction_rate):
        raise ValueError("junction_rate_m3_s must be finite")
    if junction_pressure is not None and not math.isfinite(junction_pressure):
        raise ValueError("junction_pressure_pa must be finite when supplied")
    if not math.isfinite(slip) or slip < 0.0:
        raise ValueError("slip_length_m must be finite and nonnegative")
    if not math.isfinite(minimum_height) or minimum_height <= 0.0:
        raise ValueError("minimum_height_m must be finite and positive")

    count = int(radius.size)
    faces = np.empty(count + 1, dtype=float)
    faces[0] = float(radius[0])
    faces[1:-1] = 0.5 * (radius[:-1] + radius[1:])
    faces[-1] = float(radius[-1])
    cell_area = math.pi * (faces[1:] ** 2 - faces[:-1] ** 2)
    if np.any(cell_area <= 0.0):
        raise ValueError("radius_m does not define positive annular control volumes")
    sink_fraction: np.ndarray | None
    if junction_sink_fraction is None:
        sink_fraction = None
    else:
        sink_fraction = np.asarray(junction_sink_fraction, dtype=float)
        if sink_fraction.shape != radius.shape:
            raise ValueError("junction_sink_fraction must match radius_m")
        if np.any(~np.isfinite(sink_fraction)) or np.any(sink_fraction < 0.0):
            raise ValueError("junction_sink_fraction must be finite and nonnegative")
        fraction_sum = float(np.sum(sink_fraction))
        if fraction_sum <= 0.0:
            raise ValueError("junction_sink_fraction must have positive sum")
        sink_fraction = sink_fraction / fraction_sum

    # Integrated radial-gradient operator S = 2*pi*r*h_r at faces, with
    # zero-slope curvature boundaries.  Dividing its face difference by each
    # annular area gives the axisymmetric Laplacian.
    slope_flux = sparse.lil_matrix((count + 1, count), dtype=float)
    for face in range(1, count):
        spacing = float(radius[face] - radius[face - 1])
        coefficient = 2.0 * math.pi * float(faces[face]) / spacing
        slope_flux[face, face - 1] = -coefficient
        slope_flux[face, face] = coefficient
    face_difference = sparse.lil_matrix((count, count + 1), dtype=float)
    for cell in range(count):
        face_difference[cell, cell] = -1.0
        face_difference[cell, cell + 1] = 1.0
    face_difference = face_difference.tocsr()
    laplacian = (
        sparse.diags(1.0 / cell_area, format="csr")
        @ face_difference
        @ slope_flux.tocsr()
    )
    pressure_operator = (
        density * gravity * sparse.identity(count, format="csr")
        - gamma * laplacian
    )
    pressure_offset = np.zeros(count, dtype=float)
    if junction_pressure is not None:
        # The inner prescribed through-flow remains the first junction
        # condition.  Pressure continuity replaces the former zero-slope
        # curvature condition and supplies the second one.
        pressure_operator = pressure_operator.tolil()
        pressure_operator[0, :] = 0.0
        pressure_operator = pressure_operator.tocsr()
        pressure_offset[0] = junction_pressure

    mode = str(mobility_average).lower()
    if mode not in {"harmonic", "arithmetic"}:
        raise ValueError("mobility_average must be 'harmonic' or 'arithmetic'")
    coefficient_height = old_height.copy()
    candidate = old_height.copy()
    linear_solve_count = 0
    volume_matrix = sparse.diags(cell_area / dt, format="csr")
    boundary_through_flow = np.zeros(count + 1, dtype=float)
    distributed_sink_rate = np.zeros(count, dtype=float)
    if sink_fraction is None:
        boundary_through_flow[0] = -junction_rate
    else:
        distributed_sink_rate = junction_rate * sink_fraction

    for iteration in range(max(1, int(picard_iterations))):
        linear_solve_count = iteration + 1
        nodal_mobility = np.asarray(
            [
                lubrication_mobility(value, viscosity, slip_length_m=slip)
                for value in coefficient_height
            ],
            dtype=float,
        )
        through_pressure_gradient = sparse.lil_matrix(
            (count + 1, count), dtype=float
        )
        for face in range(1, count):
            left = float(nodal_mobility[face - 1])
            right = float(nodal_mobility[face])
            if mode == "harmonic":
                mobility = 2.0 * left * right / max(left + right, 1.0e-300)
            else:
                mobility = 0.5 * (left + right)
            spacing = float(radius[face] - radius[face - 1])
            coefficient = (
                2.0 * math.pi * float(faces[face]) * mobility / spacing
            )
            # Through-flow is -2*pi*r*M*dp/dr.
            through_pressure_gradient[face, face - 1] = coefficient
            through_pressure_gradient[face, face] = -coefficient
        pressure_gradient = through_pressure_gradient.tocsr()
        through_height = pressure_gradient @ pressure_operator
        through_pressure_offset = np.asarray(
            pressure_gradient @ pressure_offset,
            dtype=float,
        )
        transport = face_difference @ through_height
        lhs = volume_matrix + transport
        rhs = (
            cell_area * old_height / dt
            - face_difference @ boundary_through_flow
            - distributed_sink_rate
            - face_difference @ through_pressure_offset
        )
        candidate = np.asarray(spsolve(lhs.tocsc(), rhs), dtype=float)
        if np.any(~np.isfinite(candidate)):
            raise RuntimeError("axisymmetric film solve returned a non-finite height")
        if float(np.min(candidate)) < minimum_height:
            raise RuntimeError(
                "axisymmetric film step crossed the minimum admissible "
                f"height ({float(np.min(candidate)):.6e} < {minimum_height:.6e} m)"
            )
        residual = float(np.max(np.abs(candidate - coefficient_height)))
        if residual <= float(picard_tolerance_m):
            break
        relaxation = float(np.clip(picard_relaxation, 1.0e-3, 1.0))
        coefficient_height = (
            (1.0 - relaxation) * coefficient_height
            + relaxation * candidate
        )

    pressure = np.asarray(
        pressure_operator @ candidate + pressure_offset,
        dtype=float,
    )
    film_volume_change = float(np.dot(cell_area, candidate - old_height))
    conservation_residual = film_volume_change + dt * junction_rate
    minimum_index = int(np.argmin(candidate))
    diagnostics = PrescribedFluxFilmDiagnostics(
        junction_rate_m3_s=junction_rate,
        film_volume_change_m3=film_volume_change,
        conservation_residual_m3=float(conservation_residual),
        minimum_height_m=float(candidate[minimum_index]),
        minimum_radius_m=float(radius[minimum_index]),
        junction_pressure_pa=float(pressure[0]),
        maximum_pressure_pa=float(np.max(pressure)),
        minimum_pressure_pa=float(np.min(pressure)),
        linear_solve_count=int(linear_solve_count),
    )
    return candidate, diagnostics


def axisymmetric_junction_flux(
    *,
    donor_pressure_pa: float,
    bridge_pressure_pa: float,
    film_thickness_m: float,
    viscosity_pa_s: float,
    junction_radius_m: float,
    pressure_length_m: float,
    slip_length_m: float = 0.0,
) -> JunctionFlux:
    """Evaluate the signed local finite-volume form of the film PDE.

    ``donor_pressure_pa`` is sampled just outside the junction and
    ``bridge_pressure_pa`` on its bridge side.  A positive pressure drop
    therefore drives an inward radial flux and a positive bridge-volume rate.
    """

    radius = float(junction_radius_m)
    length = float(pressure_length_m)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("junction_radius_m must be finite and nonnegative")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("pressure_length_m must be finite and positive")
    pressure_drop = float(donor_pressure_pa) - float(bridge_pressure_pa)
    if not math.isfinite(pressure_drop):
        raise ValueError("junction pressures must be finite")
    mobility = lubrication_mobility(
        film_thickness_m,
        viscosity_pa_s,
        slip_length_m=slip_length_m,
    )
    pressure_gradient = pressure_drop / length
    radial_flux = -mobility * pressure_gradient
    bridge_rate = -2.0 * math.pi * radius * radial_flux
    return JunctionFlux(
        bridge_rate_m3_s=float(bridge_rate),
        radial_flux_m2_s=float(radial_flux),
        mobility_m3_pa_s=float(mobility),
        pressure_drop_pa=float(pressure_drop),
        pressure_gradient_pa_m=float(pressure_gradient),
        film_thickness_m=float(film_thickness_m),
        junction_radius_m=radius,
        pressure_length_m=length,
    )


def conservative_transfer_target_volumes(
    base_volumes_m3: np.ndarray,
    *,
    bridge_mask: np.ndarray,
    film_mask: np.ndarray,
    requested_bridge_rate_m3_s: float,
    dt_s: float,
) -> ConservativeTransfer:
    """Apply a signed, exactly conservative bridge--film transfer.

    Positive rate drains the film and supplies the bridge.  Negative rate
    performs the reverse transfer.  The only limiter is donor positivity:
    no donor target volume is permitted to become negative.
    """

    base = np.asarray(base_volumes_m3, dtype=float)
    bridge = np.asarray(bridge_mask, dtype=bool)
    film = np.asarray(film_mask, dtype=bool)
    if base.ndim != 1 or bridge.shape != base.shape or film.shape != base.shape:
        raise ValueError("volume and mask arrays must have matching 1-D shapes")
    if np.any(~np.isfinite(base)) or np.any(base < 0.0):
        raise ValueError("base_volumes_m3 must be finite and nonnegative")
    if np.any(bridge & film):
        raise ValueError("bridge_mask and film_mask must be disjoint")
    if not np.any(bridge) or not np.any(film):
        raise ValueError("both bridge and film transfer regions are required")
    dt = float(dt_s)
    requested_rate = float(requested_bridge_rate_m3_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(requested_rate):
        raise ValueError("requested_bridge_rate_m3_s must be finite")

    bridge_weights = base[bridge].copy()
    film_weights = base[film].copy()
    bridge_sum = float(np.sum(bridge_weights))
    film_sum = float(np.sum(film_weights))
    if bridge_sum <= 0.0:
        bridge_weights = np.ones_like(bridge_weights)
        bridge_sum = float(np.sum(bridge_weights))
    if film_sum <= 0.0:
        film_weights = np.ones_like(film_weights)
        film_sum = float(np.sum(film_weights))
    bridge_weights /= bridge_sum
    film_weights /= film_sum

    requested_change = dt * requested_rate
    if requested_change >= 0.0:
        actual_change = min(requested_change, float(np.sum(base[film])))
    else:
        actual_change = -min(-requested_change, float(np.sum(base[bridge])))

    target = base.copy()
    target[bridge] += actual_change * bridge_weights
    target[film] -= actual_change * film_weights
    bridge_change = float(np.sum(target[bridge] - base[bridge]))
    film_change = float(np.sum(target[film] - base[film]))
    net_change = float(np.sum(target - base))
    positivity_limited = not math.isclose(
        actual_change,
        requested_change,
        rel_tol=1.0e-13,
        abs_tol=1.0e-30,
    )
    return ConservativeTransfer(
        target_volumes_m3=target,
        actual_bridge_rate_m3_s=float(actual_change / dt),
        bridge_volume_change_m3=bridge_change,
        film_volume_change_m3=film_change,
        net_volume_change_m3=net_change,
        positivity_limited=positivity_limited,
    )
