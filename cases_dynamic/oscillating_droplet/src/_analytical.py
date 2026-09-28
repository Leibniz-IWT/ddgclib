"""Analytical solutions for oscillating droplet (Lamb/Rayleigh).

Provides the linearised analytical solution for small-amplitude
oscillations of an inviscid or viscous droplet about its spherical
equilibrium shape.

The solution is parameterised by mode number ``l`` (l=2 is the
ellipsoidal fundamental mode).

References
----------
- Rayleigh (1879): inviscid frequency.
- Lamb (1932): viscous damping rate.
- Prosperetti (1980): complete initial-value solution.
"""
from __future__ import annotations

import numpy as np


def rayleigh_frequency(
    l: int, gamma: float, rho: float, R0: float, dim: int = 3,
    rho_outer: float | None = None,
) -> float:
    """Rayleigh angular frequency for mode *l*.

    Parameters
    ----------
    l : int
        Mode number (l >= 2).
    gamma : float
        Surface tension [N/m].
    rho : float
        Droplet (inner) density [kg/m³].
    R0 : float
        Equilibrium radius [m].
    dim : int
        2 or 3.
    rho_outer : float or None
        Outer-phase density.  When given, applies the Miller–Scriven
        two-phase inertia correction; the single-phase Rayleigh formula
        is recovered in the limit ``rho_outer → 0``.

    Returns
    -------
    float
        Angular frequency omega [rad/s].

    References
    ----------
    Rayleigh (1879); Miller & Scriven, J. Fluid Mech. 32:417-435 (1968).
    """
    if dim == 3:
        if rho_outer is None:
            # Single-phase Rayleigh (drop in vacuum)
            omega_sq = l * (l - 1) * (l + 2) * gamma / (rho * R0 ** 3)
        else:
            # Miller-Scriven two-phase: effective inertia
            # [(l+1)*rho_in + l*rho_out]
            inertia = (l + 1) * rho + l * rho_outer
            omega_sq = l * (l - 1) * (l + 1) * (l + 2) * gamma \
                / (inertia * R0 ** 3)
    elif dim == 2:
        if rho_outer is None:
            omega_sq = (l ** 3 - l) * gamma / (rho * R0 ** 3)
        else:
            # 2D two-phase: symmetric inner/outer potential decay,
            # effective inertia (rho + rho_outer)
            omega_sq = (l ** 3 - l) * gamma \
                / ((rho + rho_outer) * R0 ** 3)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    return np.sqrt(max(omega_sq, 0.0))


def lamb_damping_rate(
    l: int, mu: float, rho: float, R0: float, dim: int = 3,
) -> float:
    """Lamb viscous damping rate for mode *l*.

    Parameters
    ----------
    l : int
        Mode number.
    mu : float
        Dynamic viscosity [Pa·s].
    rho : float
        Droplet density [kg/m³].
    R0 : float
        Equilibrium radius [m].
    dim : int
        2 or 3.

    Returns
    -------
    float
        Damping rate beta [1/s].

    Notes
    -----
    Both prefactors follow from Lamb's weak-viscosity energy-dissipation
    method: an inviscid irrotational mode ``phi ~ r^l Y_l(theta)`` is
    substituted into the viscous dissipation integral
    ``D = 2 mu int e_ij e_ij dV``, and ``beta = <D>/(2 E_total)`` with
    ``E_total = 2 <KE>`` for SHM.

    - 3D: ``(l-1)(2l+1) mu/(rho R^2)`` — Lamb, *Hydrodynamics* 6th ed.
      (1932) §355; Chandrasekhar (1961) §92; Prosperetti (1980).
    - 2D: ``2 l (l-1) mu/(rho R^2)`` — derived in Aalilija, Gandin &
      Hachem (2020), *Comput. Fluids* 197:104362, Eq. (34a), by the
      same Lamb method applied to the 2D disk (cross-section of an
      infinite cylinder, interior flow, inviscid exterior). The paper
      notes that no prior analytical 2D formula had been published.
      For ``l=2`` the coefficient is 4.
    """
    if dim == 3:
        return (l - 1) * (2 * l + 1) * mu / (rho * R0 ** 2)
    elif dim == 2:
        return 2 * l * (l - 1) * mu / (rho * R0 ** 2)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def damped_frequency(omega: float, beta: float) -> float:
    """Damped oscillation frequency.

    Returns ``sqrt(omega^2 - beta^2)`` if underdamped, else 0.
    """
    discriminant = omega**2 - beta**2
    if discriminant > 0:
        return np.sqrt(discriminant)
    return 0.0


def radius_perturbation(
    t: float | np.ndarray,
    theta: float | np.ndarray,
    R0: float,
    epsilon: float,
    l: int,
    omega: float,
    beta: float,
) -> float | np.ndarray:
    """Analytical interface radius R(theta, t).

    For an initially deformed droplet with:
        R(theta, 0) = R0 * (1 + epsilon * cos(l * theta))
        dR/dt(theta, 0) = 0  (started from rest)

    The linearised solution is:
        R(theta, t) = R0 + epsilon * R0 * exp(-beta*t)
                      * cos(omega_d * t) * cos(l * theta)

    where omega_d = damped_frequency(omega, beta).

    For the overdamped case (beta > omega), the cosine term becomes
    a decaying exponential (no oscillation).
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    omega_d = damped_frequency(omega, beta)
    envelope = np.exp(-beta * t)

    if omega_d > 0:
        # Underdamped: oscillatory decay
        temporal = envelope * np.cos(omega_d * t)
    else:
        # Overdamped: pure exponential decay (dominant root)
        # Two exponential modes: exp(-(beta ± sqrt(beta²-omega²))t)
        # Dominant (slower) mode:
        delta = np.sqrt(beta**2 - omega**2)
        # Solution satisfying R'(0)=0:
        # R(t) = A*exp(-(beta-delta)*t) + B*exp(-(beta+delta)*t)
        # with A+B=1, -(beta-delta)*A - (beta+delta)*B = 0
        if delta > 1e-30:
            A = (beta + delta) / (2 * delta)
            B = 1.0 - A
            temporal = A * np.exp(-(beta - delta) * t) + B * np.exp(-(beta + delta) * t)
        else:
            # Critically damped
            temporal = (1.0 + beta * t) * np.exp(-beta * t)

    return R0 + epsilon * R0 * temporal * np.cos(l * theta)


def max_radius_envelope(
    t: float | np.ndarray,
    R0: float,
    epsilon: float,
    omega: float,
    beta: float,
    l: int = 2,
) -> float | np.ndarray:
    """Maximum interface radius over all angles at time t.

    This is R(theta=0, t) since cos(l*0) = 1 for any ``l``.
    """
    return radius_perturbation(t, 0.0, R0, epsilon, l, omega, beta)


def pressure_jump_analytical(
    gamma: float, R0: float, dim: int = 3,
) -> float:
    """Young-Laplace equilibrium pressure jump.

    Returns
    -------
    float
        ΔP = γ/R (2D) or ΔP = 2γ/R (3D).
    """
    if dim == 3:
        return 2 * gamma / R0
    return gamma / R0


# =====================================================================
# Two-fluid reference (lane C, 2026-07-29) — NEW functions only.
# The single-fluid functions above are pinned by regression tests and
# must not change; everything below augments them with a reference that
# accounts for the outer bath (rho_o, mu_o).  See
# docs_temp/debug_session/laneC-two-fluid-reference.md.
# =====================================================================


def lamb_damping_rate_two_fluid(
    l: int, mu: float, rho: float, R0: float,
    mu_outer: float, rho_outer: float, dim: int = 2,
) -> float:
    """Closed-form two-fluid damping rate by Lamb's dissipation method (2D).

    Derivation (energy method, both phases irrotational):
    take the inviscid two-fluid mode ``phi_i ~ (r/R)^l cos(l theta)``
    (inner) and ``phi_o ~ (R/r)^l cos(l theta)`` (outer), matched in
    ``u_r`` at ``r = R``.  With velocity amplitude ``a`` the peak
    kinetic energies are equal in structure,
    ``KE = (pi/2) rho l a^2 R^{2l}`` per phase (hence the
    ``rho + rho_outer`` inertia in :func:`rayleigh_frequency`), and the
    bulk dissipation ``D = 2 mu int e_ij e_ij dA = mu oint
    d(|u|^2)/dn ds`` evaluates to ``4 pi l^2 (l-1) mu a^2 R^{2l-2}``
    (inner) and ``4 pi l^2 (l+1) mu_o a^2 R^{2l-2}`` (outer).  With
    ``beta = <D>/(2 E) = D_peak/(4 KE_peak)``:

        beta = 2 l [ (l-1) mu + (l+1) mu_outer ]
               / [ (rho + rho_outer) R0^2 ]

    Limits: ``mu_outer = rho_outer = 0`` recovers the single-fluid
    :func:`lamb_damping_rate` (2D) exactly.

    .. warning::
       This is the *bulk-dissipation* (potential-flow) estimate.  For a
       genuine two-fluid interface the tangential-velocity mismatch of
       the two potential flows spawns an interfacial vortical layer
       whose dissipation scales as ``sqrt(mu)`` and *dominates* the
       ``O(mu)`` bulk terms at small viscosity (Miller & Scriven,
       J. Fluid Mech. 32:417 (1968), for the 3D analogue) — verified
       here numerically against :func:`two_fluid_dispersion_roots_2d`.
       Use the dispersion-relation root for quantitative work; this
       closed form is the documented "at minimum" estimate.
    """
    if dim != 2:
        raise NotImplementedError(
            "lamb_damping_rate_two_fluid: only dim=2 is implemented; "
            "for 3D see Miller & Scriven (1968)."
        )
    return 2 * l * ((l - 1) * mu + (l + 1) * mu_outer) \
        / ((rho + rho_outer) * R0 ** 2)


def mode_temporal_ivp(
    t: float | np.ndarray, omega: float, beta: float,
) -> float | np.ndarray:
    """Exact started-from-rest temporal factor of a damped mode.

    Solves ``x'' + 2 beta x' + omega^2 x = 0`` with ``x(0) = 1``,
    ``x'(0) = 0`` in all regimes:

    - underdamped (beta < omega):
      ``exp(-beta t) [cos(w_d t) + (beta/w_d) sin(w_d t)]``,
      ``w_d = sqrt(omega^2 - beta^2)``
    - overdamped (beta > omega): the biexponential
      ``A exp(-(beta-delta) t) + B exp(-(beta+delta) t)`` with
      ``delta = sqrt(beta^2 - omega^2)``, ``A = (beta+delta)/(2 delta)``,
      ``B = 1 - A``
    - critically damped: ``(1 + beta t) exp(-beta t)``

    Note: the pinned :func:`radius_perturbation` omits the
    ``(beta/w_d) sin`` term in its underdamped branch (weak-damping
    approximation, kept for regression continuity).  The two-fluid
    reference has ``beta/w_d ~ 1.25``, so the exact term matters and
    this function is used instead.
    """
    t = np.asarray(t, dtype=float)
    disc = omega ** 2 - beta ** 2
    if disc > 1e-30:
        w_d = np.sqrt(disc)
        return np.exp(-beta * t) * (
            np.cos(w_d * t) + (beta / w_d) * np.sin(w_d * t)
        )
    delta = np.sqrt(max(beta ** 2 - omega ** 2, 0.0))
    if delta > 1e-30:
        A = (beta + delta) / (2 * delta)
        B = 1.0 - A
        return A * np.exp(-(beta - delta) * t) \
            + B * np.exp(-(beta + delta) * t)
    return (1.0 + beta * t) * np.exp(-beta * t)


def radius_perturbation_two_fluid(
    t: float | np.ndarray,
    theta: float | np.ndarray,
    R0: float,
    epsilon: float,
    l: int,
    omega: float,
    beta: float,
) -> float | np.ndarray:
    """Two-fluid-reference interface radius R(theta, t).

    Same initial conditions as :func:`radius_perturbation`
    (``R(theta,0) = R0 (1 + epsilon cos(l theta))``, started from
    rest) but with the *exact* started-from-rest temporal factor
    :func:`mode_temporal_ivp` and ``(omega, beta)`` taken from the
    two-fluid dispersion relation (:func:`two_fluid_omega_beta_2d`).

    Approximation (labelled per lane-C ground rules): the deformation
    is projected onto the *least-damped normal mode alone* with
    second-order-ODE started-from-rest weights.  The full viscous
    initial-value problem (Prosperetti, J. Fluid Mech. 100:333 (1980),
    3D analogue) adds a short vorticity-diffusion transient before the
    least-damped mode dominates; that transient is neglected here.
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    temporal = mode_temporal_ivp(t, omega, beta)
    return R0 + epsilon * R0 * temporal * np.cos(l * theta)


def _bessel_ratio_iv(l: int, z: complex) -> complex:
    """I_l'(z)/I_l(z) via exponentially scaled Bessels (overflow-safe)."""
    from scipy import special
    num = 0.5 * (special.ive(l - 1, z) + special.ive(l + 1, z))
    return num / special.ive(l, z)


def _bessel_ratio_kv(l: int, z: complex) -> complex:
    """K_l'(z)/K_l(z) via exponentially scaled Bessels (overflow-safe)."""
    from scipy import special
    num = -0.5 * (special.kve(l - 1, z) + special.kve(l + 1, z))
    return num / special.kve(l, z)


def two_fluid_dispersion_det_2d(
    s: complex, l: int, gamma: float, mu: float, rho: float, R0: float,
    mu_outer: float, rho_outer: float,
) -> complex:
    """Determinant of the 2D two-fluid viscous normal-mode system.

    Formulation (linearised incompressible Navier–Stokes about rest,
    perturbations ``~ exp(s t + i l theta)``, streamfunction
    ``u = curl(psi z)``, so ``(nabla^2 - q^2) nabla^2 psi = 0`` with
    ``q^2 = s rho / mu`` per phase).  Radial parts, normalised at
    ``r = R0``:

    - inner (regular at 0):  ``A (r/R)^l + B I_l(q_i r)/I_l(q_i R)``
    - outer (decaying):      ``C (R/r)^l + D K_l(q_o r)/K_l(q_o R)``

    The harmonic (potential) parts carry the pressure
    (``p_i = -i rho s A (r/R)^l``, ``p_o = +i rho_o s C (R/r)^l``);
    the ``I_l/K_l`` (vortical) parts carry none.  Four interface
    conditions at ``r = R0`` (sharp interface, no gravity, unbounded
    outer phase — the case's no-slip wall at 5 R0 is NOT modelled):

    1. ``u_r`` continuous:      ``f_i = f_o``
    2. ``u_theta`` continuous:  ``f_i' = f_o'``
    3. tangential stress continuous:
       ``mu [ -f'' + f'/R - l^2 f / R^2 ]`` matches
    4. normal stress jump balances the perturbation curvature:
       ``(-p_o + 2 mu_o du_r/dr) - (-p_i + 2 mu_i du_r/dr)
       = gamma (l^2 - 1) eta / R^2`` with kinematic
       ``s eta = u_r(R)``

    Roots of ``det = 0`` are the normal-mode growth rates ``s``
    (``Re s < 0``: decay).  Verified limits (see lane-C log):
    inviscid → ``s = ± i omega`` with the ``rho + rho_outer`` inertia
    of :func:`rayleigh_frequency`; vanishing outer phase + weak ``mu``
    → ``Re s → -lamb_damping_rate(2D)``; weak-viscosity two-fluid
    damping scales as ``sqrt(mu)`` (interfacial vortical layer,
    Miller–Scriven mechanism).  For roots off the negative real axis
    ``Re(q) > 0`` in both phases, so the eigenfunctions genuinely
    decay in r — these are discrete normal modes, no
    analytic-continuation caveat.
    """
    s = complex(s)
    if s == 0:
        return complex(np.nan)
    q_i = np.sqrt(s * rho / mu)
    q_o = np.sqrt(s * rho_outer / mu_outer)
    x = q_i * R0
    y = q_o * R0
    p2 = q_i * _bessel_ratio_iv(l, x)
    p4 = q_o * _bessel_ratio_kv(l, y)
    # T[f] = -f'' + f'/R - l^2 f/R^2 (tangential stress operator) on the
    # four normalised basis functions, evaluated at r=R0:
    T1 = -2 * l * (l - 1) / R0 ** 2
    T2 = -(x ** 2 + 2 * l ** 2) / R0 ** 2 + 2 * p2 / R0
    T3 = -2 * l * (l + 1) / R0 ** 2
    T4 = -(y ** 2 + 2 * l ** 2) / R0 ** 2 + 2 * p4 / R0
    g = gamma * (l ** 2 - 1) / (s * R0 ** 3)
    M = np.zeros((4, 4), dtype=complex)
    M[0] = [1, 1, -1, -1]                       # u_r continuity
    M[1] = [l / R0, p2, l / R0, -p4]            # u_theta continuity
    M[2] = [mu * T1, mu * T2, -mu_outer * T3, -mu_outer * T4]
    M[3, 0] = -rho * s / l - 2 * mu * (l - 1) / R0 ** 2 - g
    M[3, 1] = -2 * mu * (p2 / R0 - 1 / R0 ** 2) - g
    M[3, 2] = -rho_outer * s / l - 2 * mu_outer * (l + 1) / R0 ** 2
    M[3, 3] = 2 * mu_outer * (p4 / R0 - 1 / R0 ** 2)
    # Row scalings (roots unaffected; conditioning only)
    M[1] *= R0
    M[2] *= R0 ** 2 / max(mu, mu_outer)
    M[3] *= R0 ** 2 * l / (rho * max(abs(s), 1.0))
    return complex(np.linalg.det(M))


def _newton_complex(f, s0: complex, tol: float = 1e-13,
                    maxit: int = 100) -> complex | None:
    """Complex Newton iteration with central-difference derivative."""
    s = complex(s0)
    for _ in range(maxit):
        h = 1e-7 * max(abs(s), 1e-3)
        fs = f(s)
        d = (f(s + h) - f(s - h)) / (2 * h)
        if not np.isfinite(d) or d == 0:
            return None
        s_new = s - fs / d
        if not np.isfinite(s_new):
            return None
        if abs(s_new - s) < tol * max(abs(s), 1.0):
            return s_new
        s = s_new
    return None


def two_fluid_dispersion_roots_2d(
    l: int, gamma: float, mu: float, rho: float, R0: float,
    mu_outer: float, rho_outer: float,
) -> list[complex]:
    """Decaying roots of the 2D two-fluid dispersion relation.

    Multi-seed complex Newton on
    :func:`two_fluid_dispersion_det_2d` over a physical seed grid
    (scales: inviscid ``omega`` and the two viscous rates
    ``mu/(rho R0^2)``), deduplicated, sorted least-damped first
    (largest ``Re s``).  Roots are folded to ``Im s >= 0`` (they come
    in conjugate pairs).  Spurious ``|s| ~ 0`` Newton fixed points
    (the ``1/s`` curvature term) are filtered.

    All four fluid parameters must be positive — the ``K_l`` basis
    degenerates for a vanishing outer phase; use tiny-but-finite
    outer values to probe the single-fluid limit (done in the tests).
    """
    if min(mu, rho, mu_outer, rho_outer) <= 0:
        raise ValueError(
            "two_fluid_dispersion_roots_2d requires mu, rho, mu_outer, "
            "rho_outer > 0 (use lamb_damping_rate/rayleigh_frequency "
            "for the single-fluid problem)."
        )
    omega = np.sqrt((l ** 3 - l) * gamma / ((rho + rho_outer) * R0 ** 3))
    S = max(omega, mu / (rho * R0 ** 2), mu_outer / (rho_outer * R0 ** 2))

    def f(s):
        return two_fluid_dispersion_det_2d(
            s, l, gamma, mu, rho, R0, mu_outer, rho_outer)

    roots: list[complex] = []
    for re in np.linspace(-4 * S, -0.02 * S, 8):
        for im in np.linspace(0.0, 2.5 * S, 6):
            r = _newton_complex(f, re + 1j * im)
            if r is None or r.real > 0 or abs(r) < 1e-4 * S:
                continue
            r = complex(r.real, abs(r.imag))
            if all(abs(r - r0) > 1e-6 * S for r0 in roots):
                roots.append(r)
    roots.sort(key=lambda z: -z.real)
    return roots


def two_fluid_omega_beta_2d(
    l: int, gamma: float, mu: float, rho: float, R0: float,
    mu_outer: float, rho_outer: float,
) -> tuple[float, float]:
    """Effective ``(omega, beta)`` of the least-damped two-fluid mode.

    Maps the dominant root(s) of
    :func:`two_fluid_dispersion_roots_2d` onto the damped-oscillator
    parameters consumed by :func:`radius_perturbation_two_fluid` /
    :func:`mode_temporal_ivp`:

    - complex pair ``s = -beta ± i w_d``:
      ``beta = -Re s``, ``omega = |s|`` (then
      ``sqrt(omega^2 - beta^2) = w_d`` exactly);
    - two real roots ``-lambda_1, -lambda_2`` (aperiodic regime):
      ``beta = (lambda_1 + lambda_2)/2``,
      ``omega = sqrt(lambda_1 lambda_2)`` (the biexponential of
      :func:`mode_temporal_ivp` then has exactly those rates).

    For the case parameters (rho 800/1000, mu 0.5/0.1, gamma 0.05,
    R0 0.01, l=2) this gives ``s = -6.8326 ± 5.4707i`` — the true
    two-fluid mode is weakly *oscillatory* (beta 6.83, omega 8.75),
    not overdamped like the single-fluid Lamb mapping (beta 25,
    omega 12.91, slow rate 3.59 1/s).
    """
    roots = two_fluid_dispersion_roots_2d(
        l, gamma, mu, rho, R0, mu_outer, rho_outer)
    if not roots:
        raise RuntimeError("no dispersion roots found")
    s1 = roots[0]
    if abs(s1.imag) > 1e-8 * abs(s1):
        return float(abs(s1)), float(-s1.real)
    reals = [r for r in roots if abs(r.imag) <= 1e-8 * abs(r)]
    if len(reals) < 2:
        raise RuntimeError(
            f"least-damped root {s1} is real but no second real root "
            "was found to form the aperiodic (omega, beta) pair"
        )
    lam1, lam2 = -reals[0].real, -reals[1].real
    return float(np.sqrt(lam1 * lam2)), float(0.5 * (lam1 + lam2))


def kinetic_energy_envelope(
    t: float | np.ndarray,
    R0: float,
    epsilon: float,
    rho: float,
    omega: float,
    beta: float,
    dim: int = 3,
) -> float | np.ndarray:
    """Approximate kinetic energy decay envelope.

    For small perturbations the KE decays as exp(-2*beta*t)
    times the initial KE.
    """
    t = np.asarray(t, dtype=float)
    # Initial KE scales as rho * R0^dim * (epsilon * R0 * omega)^2
    if dim == 3:
        KE0 = 0.5 * rho * (4 / 3) * np.pi * R0**3 * (epsilon * R0 * omega)**2
    else:
        KE0 = 0.5 * rho * np.pi * R0**2 * (epsilon * R0 * omega)**2
    return KE0 * np.exp(-2 * beta * t)
