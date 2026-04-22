# Electrolysis Hydrogen Bubble (dynamic, multiphase)

Proof-of-concept dynamic Lagrangian simulation of a hydrogen gas
bubble immersed in a column of electrolyte, with gravity acting
downward and a placeholder linear gas-generation rate standing in
for the full electrochemical reaction.  The bottom wall of the
domain is the "electrode" where hydrogen is produced.

This case is a stripped-down precursor to a full electrolysis
simulation:

- **No electrostatics.**  Surface tension `gamma` is constant, no
  double-layer, no charge transport.
- **No species diffusion / Butler-Volmer.**  Gas mass is injected at
  a constant rate `dm/dt` spread across the gas-phase dual volume
  (see `src/_reaction.py`).  This is a hook for the eventual
  reaction-diffusion coupling.
- **Real physics that *is* there:** surface tension with Young-Laplace
  pre-load, viscous + pressure stress from the two-phase Cauchy
  momentum equation, compressible Tait-Murnaghan EOS on both phases
  so hydrostatic pressure emerges from the solver, and gravity as a
  body force on every dual cell.

## Files

    electrolysis_bubble_2D.py   -- 2D run (square box, bubble offset
                                   near the electrode via an
                                   explicit off-centre builder)
    electrolysis_bubble_3D.py   -- 3D run (cubic box, bubble at
                                   origin, electrode wall L_domain
                                   below the bubble)
    view_polyscope.py           -- interactive snapshot viewer
    src/
        _params.py              -- physical / numerical parameters
        _setup.py               -- mesh builder, custom boundary
                                   conditions, gas/liquid IC
        _reaction.py            -- linear gas-mass injection (placeholder)
        _analytical.py          -- capillary length, Fritz detachment R
        _plot_helpers.py        -- diagnostics + static plots
    fig/                        -- output plots and animations
    results/snapshots/          -- StateHistory JSON dumps for polyscope

## Run

    # 2D
    python cases_dynamic/electrolysis_bubble/electrolysis_bubble_2D.py

    # 3D
    python cases_dynamic/electrolysis_bubble/electrolysis_bubble_3D.py

    # Replay the snapshots interactively
    python cases_dynamic/electrolysis_bubble/view_polyscope.py --dim 2

## Geometry

**2D**: outer box fixed at `[-L, L]²`, bubble offset below the
origin at `y = -nucleation_frac · (L - 2R₀)`.  The electrode is the
bottom wall at `y = -L`, the bubble sits `L - 2R₀ · nucleation_frac`
above it at `t=0`.  Built via a custom off-centre combined-mesh
builder (`_build_offcenter_bubble_box_2d`) because the canonical
`droplet_in_box_2d` co-shifts the outer box with the bubble.

**3D**: outer box centred at origin, bubble centred at origin.  The
electrode wall at `z = -L` is always `L` below the bubble; the
bubble rises toward the top electrode / free surface under
buoyancy as gas mass accumulates.  Uses the canonical
`droplet_in_box_3d` (the 3D off-centre builder proved numerically
fragile in the barycentric dual pipeline; 3D off-centre placement
is future work).

## Physics summary

Reference scales (water-like electrolyte, softened EOS for CFL):

| Parameter       | Value        | Notes                                    |
|-----------------|--------------|------------------------------------------|
| `gamma`         | 0.072 N/m    | Water-air surface tension                |
| `rho_liq`       | 1000 kg/m³   | Water                                    |
| `rho_gas`       | 10 kg/m³     | Softened H2 (real H2 ~0.09)              |
| `mu_liq`        | 0.1 Pa·s     | Raised 100 x above real water for stability |
| `mu_gas`        | 0.01 Pa·s    | Raised above real H2 but still 10 x < mu_liq |
| `g`             | 9.81 m/s²    |                                          |
| `R0`            | 1 mm         | Initial bubble radius                    |
| `L_domain`      | 4 mm         | Half-side of outer box (8 mm³)           |
| `K_liq = K_gas` | 1e5 Pa       | Softened, *matched* bulk moduli          |

**Matched bulk moduli** is the single biggest stability decision:
allowing `K_gas < K_liq` (the physical ordering) lets the gas phase
drop below its EOS reference density under any transient, which
collapses `P = K·(ρ/ρ₀ − 1)` into negative territory and drives
runaway inflation.  With `K_gas = K_liq` the gas is effectively a
"lighter liquid" with the correct density contrast; mass injection
still drives growth, gravity still produces buoyancy.

Derived:

    capillary length:    λ = sqrt(gamma/(rho_diff * g)) ~ 2.7 mm
    Fritz detach radius: R_det = (3 * R0 * λ² / 2)^(1/3) ~ 2.23 mm

## Initial conditions

1. **Hydrostatic pressure** is pre-imposed in the liquid by solving
   `p(y) = P0 + rho_liq · g · (wall_top - y)` for the per-vertex
   density via the EOS, then setting `m_phase[0] = rho · V_liq`.
   This eliminates the startup pressure wave that would otherwise
   dwarf every other dynamical signal for the first ~100 steps.
2. **Young-Laplace + gas hydrostatic** pre-load in the gas phase:
   `P_gas(y) = P_liq(y_top_of_bubble) + gamma/R +
   rho_gas * g * (y_top - y)`.  The small `rho_gas · g · h`
   correction is important — without it the interior gas has no
   pressure support against gravity and accelerates wildly.
3. **NaN sanitisation** after the refresh.  In 3D the barycentric
   dual construction produces NaN dual volumes at domain corners;
   those vertices are clipped to zero mass in both the IC and the
   per-step `dudt_fn`.  Without the sanitisation, a single corner
   NaN poisons every subsequent Delaunay retopology.

## Numerics

- **CFL**: `dt = cfl_safety · dx_min / c_s` with `cfl_safety = 0.05`
  (2D) and `0.025` (3D — the 3D Heron curvature and DEC dual
  polygon are more sensitive to mesh skew under advection).
- **Retopology**: `_retopologize_multiphase` (full Delaunay +
  multiphase relabeling).  Dual-only retopology (no Delaunay flips)
  was tried but was markedly less stable on this problem — the
  opposite of the observation in `static_droplet_2D`, because with
  gravity the mesh accumulates drift that the Delaunay pass needs
  to relax.
- **`WallClampBC`** (defined in `src/_setup.py`) is a custom BC
  that clips interior vertices crossing a wall plane back to
  `level + min_gap` and zeroes the wall-normal velocity.  Used on
  both the bottom (electrode) and top walls.

## Expected output

**2D**: bubble starts at R ≈ 1 mm at the nucleation site; as gas
mass is injected it grows, deforms, and its interface migrates
toward the electrode.  `R_eq` climbs to ~1.4-1.5 mm over the
simulation window; `gas_com.y` and `min_iface_z` track the bubble
centroid and its lowest interface vertex.

**3D**: bubble grows from R ≈ 1 mm to ~1.9 mm while the interface
starts to elongate.  The Delaunay retopology loses some interface
vertices as the bubble deforms past this point (a known limitation
on coarse 3D meshes with aggressive growth rates).

## Known limitations

- Delaunay retopology on a Lagrangian multiphase mesh does not
  preserve a thin detaching interface indefinitely; bubble mass is
  slowly lost once the interface starts to neck.  A full
  detachment-and-rise event would need either adaptive interface-
  preserving remesh (not yet available in 3D) or a level-set /
  volume-of-fluid add-on to the Lagrangian pipeline.
- The 3D setup uses the centred-bubble geometry because a robust
  3D off-centre builder is not yet available.  The bubble
  therefore starts in the middle of the box; physically this is
  equivalent to a freshly-detached bubble that has just left the
  electrode.
- Contact angle is not enforced.  A dedicated contact-angle BC
  (planned in `FEATURES.md`) would replace the hard clamp and give
  a proper receding / advancing angle.
- Gas / liquid bulk moduli are `matched` to keep the EOS in the
  positive-pressure regime; real compressibility contrast would
  require a better gas EOS (e.g. `IdealGas`) combined with the same
  safeguards, which we leave to a later iteration.
