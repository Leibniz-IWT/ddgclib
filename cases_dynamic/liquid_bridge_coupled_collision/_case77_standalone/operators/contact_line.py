"""Contact-line operators for ddgclib free-surface calculations.

The functions here are deliberately geometry-light: case drivers provide the
contact ring, local velocities, and force direction, while this module provides
the Cox-Voinov dynamic-angle relation and the corresponding discrete
unbalanced Young line force.
"""

from __future__ import annotations

import math

import numpy as np


def cox_log_factor(macro_length_m: float, slip_length_m: float) -> float:
    """Return ln(L/lambda) with a conservative lower bound."""

    macro = max(float(macro_length_m), 1.0e-30)
    slip = max(float(slip_length_m), 1.0e-30)
    return float(math.log(max(macro / slip, 1.0)))


def cox_voinov_dynamic_angle(
    *,
    theta_eq_rad: float,
    slide_speed_m_s: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    macro_length_m: float,
    slip_length_m: float,
    min_angle_rad: float = 0.0,
    max_angle_rad: float = math.radians(25.0),
) -> float:
    """Cox-Voinov dynamic contact angle for a moving contact line.

    This matches the PR37 form
    ``theta_dyn**3 = theta_eq**3 + 9 Ca ln(L/lambda)``.
    """

    theta_eq = float(theta_eq_rad)
    capillary_number = (
        float(viscosity_pa_s)
        * float(slide_speed_m_s)
        / max(float(surface_tension_n_m), 1.0e-30)
    )
    theta_dyn = float(
        np.cbrt(
            max(
                theta_eq**3
                + 9.0 * capillary_number * cox_log_factor(macro_length_m, slip_length_m),
                float(min_angle_rad) ** 3,
            )
        )
    )
    return float(np.clip(theta_dyn, float(min_angle_rad), float(max_angle_rad)))


def cox_molecular_dynamic_angle(
    *,
    theta_eq_rad: float,
    slide_speed_m_s: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    macro_length_m: float,
    slip_length_m: float,
    contact_line_friction_pa_s: float,
    min_angle_rad: float = 0.0,
    max_angle_rad: float = math.radians(170.0),
) -> float:
    """Return the apparent angle for Cox plus molecular line friction."""

    theta_cox = cox_voinov_dynamic_angle(
        theta_eq_rad=theta_eq_rad,
        slide_speed_m_s=slide_speed_m_s,
        viscosity_pa_s=viscosity_pa_s,
        surface_tension_n_m=surface_tension_n_m,
        macro_length_m=macro_length_m,
        slip_length_m=slip_length_m,
        min_angle_rad=min_angle_rad,
        max_angle_rad=max_angle_rad,
    )
    cosine = (
        math.cos(theta_cox)
        - max(float(contact_line_friction_pa_s), 0.0)
        * abs(float(slide_speed_m_s))
        / max(float(surface_tension_n_m), 1.0e-30)
    )
    return float(
        np.clip(
            math.acos(float(np.clip(cosine, -1.0, 1.0))),
            float(min_angle_rad),
            float(max_angle_rad),
        )
    )


def cox_inverse_contact_line_speed(
    *,
    theta_geo_rad: float,
    theta_eq_rad: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    macro_length_m: float,
    slip_length_m: float,
) -> float:
    """Return the Cox contact-line speed implied by a geometric angle."""

    return float(
        float(surface_tension_n_m)
        * (float(theta_geo_rad) ** 3 - float(theta_eq_rad) ** 3)
        / (
            9.0
            * max(float(viscosity_pa_s), 1.0e-30)
            * max(cox_log_factor(macro_length_m, slip_length_m), 1.0e-30)
        )
    )


def cox_molecular_inverse_contact_line_speed(
    *,
    theta_geo_rad: float,
    theta_eq_rad: float,
    viscosity_pa_s: float,
    surface_tension_n_m: float,
    macro_length_m: float,
    slip_length_m: float,
    contact_line_friction_pa_s: float,
) -> float:
    """Invert the PR37 Cox law with molecular contact-line friction.

    The advancing force balance is

    ``cos(theta_geo) = cos(theta_cox(U)) - zeta_cl*U/gamma``.

    Cox hydrodynamic wedge dissipation and molecular kinetic friction are
    therefore retained as two physical resistance mechanisms.  The solve is
    monotone and introduces no mobility or experimental speed parameter.
    """

    theta_geo = float(theta_geo_rad)
    theta_eq = float(theta_eq_rad)
    if theta_geo <= theta_eq:
        return 0.0
    zeta = max(float(contact_line_friction_pa_s), 0.0)
    if zeta <= 0.0:
        return cox_inverse_contact_line_speed(
            theta_geo_rad=theta_geo,
            theta_eq_rad=theta_eq,
            viscosity_pa_s=viscosity_pa_s,
            surface_tension_n_m=surface_tension_n_m,
            macro_length_m=macro_length_m,
            slip_length_m=slip_length_m,
        )
    gamma = max(float(surface_tension_n_m), 1.0e-30)
    target_cosine = math.cos(theta_geo)
    upper = gamma * max(math.cos(theta_eq) - target_cosine, 0.0) / zeta
    if upper <= 0.0:
        return 0.0

    def residual(speed: float) -> float:
        theta_cox = cox_voinov_dynamic_angle(
            theta_eq_rad=theta_eq,
            slide_speed_m_s=float(speed),
            viscosity_pa_s=viscosity_pa_s,
            surface_tension_n_m=gamma,
            macro_length_m=macro_length_m,
            slip_length_m=slip_length_m,
            min_angle_rad=theta_eq,
            max_angle_rad=max(theta_geo, theta_eq),
        )
        return math.cos(theta_cox) - zeta * float(speed) / gamma - target_cosine

    lower = 0.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return float(0.5 * (lower + upper))


def relax_radius_toward_target(
    *,
    current_radius_m: float,
    target_radius_m: float,
    dt_s: float,
    relaxation_fraction: float,
    max_speed_m_s: float,
) -> float:
    """Return one bounded relaxation step for a circular contact radius."""

    current = float(current_radius_m)
    target = float(target_radius_m)
    fraction = float(np.clip(relaxation_fraction, 0.0, 1.0))
    max_step = max(float(max_speed_m_s), 0.0) * max(float(dt_s), 0.0)
    desired = fraction * (target - current)
    if max_step > 0.0:
        desired = float(np.clip(desired, -max_step, max_step))
    return current + desired


def ring_segment_lengths(points_m: np.ndarray) -> np.ndarray:
    """Return the per-vertex arc length represented by a closed contact ring."""

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return np.zeros(points.shape[0] if points.ndim == 2 else 0, dtype=float)
    previous_len = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    next_len = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return 0.5 * (previous_len + next_len)


def cox_contact_line_force_ring(
    *,
    points_m: np.ndarray,
    velocities_m_s: np.ndarray,
    line_direction: np.ndarray,
    surface_tension_n_m: float,
    theta_eq_rad: float,
    viscosity_pa_s: float,
    macro_length_m: float,
    slip_length_m: float,
    solid_velocity_m_s: np.ndarray | None = None,
    slide_direction: np.ndarray | None = None,
    dynamic: bool = True,
    min_angle_rad: float = 0.0,
    max_angle_rad: float = math.radians(25.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete Cox contact-line force for one closed ring.

    Returns ``(forces, theta_dyn, slide_speed)`` with one row per ring vertex.
    The caller maps these ring-local forces back to the global mesh vertices.
    The returned force is the unbalanced Young/Cox contribution, so it is zero
    when the dynamic and equilibrium contact angles are equal.
    """

    points = np.asarray(points_m, dtype=float)
    velocities = np.asarray(velocities_m_s, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (n, 3).")
    if velocities.shape != points.shape:
        raise ValueError("velocities_m_s must have the same shape as points_m.")

    direction = np.asarray(line_direction, dtype=float)
    if direction.ndim == 1:
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-30:
            raise ValueError("line_direction must be nonzero.")
        direction = np.broadcast_to(direction / direction_norm, points.shape).copy()
    elif direction.shape == points.shape:
        direction_norm = np.linalg.norm(direction, axis=1)
        valid_direction = direction_norm > 1.0e-30
        if not np.all(valid_direction):
            raise ValueError("every row of line_direction must be nonzero.")
        direction = direction / direction_norm[:, None]
    else:
        raise ValueError("line_direction must have shape (3,) or (n, 3).")

    if solid_velocity_m_s is None:
        solid_velocity = np.zeros(3, dtype=float)
    else:
        solid_velocity = np.asarray(solid_velocity_m_s, dtype=float)

    if slide_direction is None:
        radius = np.hypot(points[:, 0], points[:, 1])
        slide = np.zeros_like(points)
        valid = radius > 1.0e-30
        slide[valid, 0] = points[valid, 0] / radius[valid]
        slide[valid, 1] = points[valid, 1] / radius[valid]
    else:
        slide = np.asarray(slide_direction, dtype=float)
        if slide.ndim == 1:
            slide = np.broadcast_to(slide, points.shape).copy()
        if slide.shape != points.shape:
            raise ValueError("slide_direction must have shape (3,) or (n, 3).")
        norm = np.linalg.norm(slide, axis=1)
        valid = norm > 1.0e-30
        slide = np.divide(slide, norm[:, None], out=np.zeros_like(slide), where=valid[:, None])

    relative_velocity = velocities - solid_velocity[None, :]
    slide_speed = np.sum(relative_velocity * slide, axis=1)
    theta = np.full(points.shape[0], float(theta_eq_rad), dtype=float)
    if bool(dynamic):
        theta = np.asarray(
            [
                cox_voinov_dynamic_angle(
                    theta_eq_rad=float(theta_eq_rad),
                    slide_speed_m_s=float(speed),
                    viscosity_pa_s=float(viscosity_pa_s),
                    surface_tension_n_m=float(surface_tension_n_m),
                    macro_length_m=float(macro_length_m),
                    slip_length_m=float(slip_length_m),
                    min_angle_rad=float(min_angle_rad),
                    max_angle_rad=float(max_angle_rad),
                )
                for speed in slide_speed
            ],
            dtype=float,
        )

    line_lengths = ring_segment_lengths(points)
    # PR37 ``_cox_uy_contact_line_liquid_force_map`` uses a slide direction
    # opposite to the positive advancing tangent used by this public operator:
    #
    #   s_PR37 = -d_adv,
    #   F_i = gamma*(cos(theta_dyn) - cos(theta_eq))*ell_i*s_PR37
    #       = gamma*(cos(theta_eq) - cos(theta_dyn))*ell_i*d_adv.
    #
    # ``slide`` and the signed speed below use ``d_adv``.  Keeping the
    # coefficient in the same convention prevents reversing the liquid force.
    force_mag = float(surface_tension_n_m) * line_lengths * (
        math.cos(float(theta_eq_rad)) - np.cos(theta)
    )
    return force_mag[:, None] * slide, theta, slide_speed
