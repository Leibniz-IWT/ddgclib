# Two-Particle Liquid Bridge Case Study

Capillary liquid bridge formation and agglomeration between two wetted
silica glass spheres.

## What This Demonstrates

- **Capillary bridge formation**: Two wetted particles approach each other.
  When the surface-to-surface gap drops below a critical distance, a
  liquid bridge forms between them.
- **Capillary force**: The toroidal bridge approximation (Willett et al. 2000)
  produces an attractive capillary force that pulls the particles together.
- **Hertz contact**: When particles overlap (or nearly touch), Hertz
  nonlinear elastic repulsion (F ∝ δ^{3/2}) balances the capillary
  attraction.
- **Agglomeration**: The combined capillary attraction and contact damping
  bring the particles to rest at a small equilibrium gap — a wet
  agglomerate.

## Physics

| Model               | Description                             |
|---------------------|-----------------------------------------|
| Capillary bridge    | Toroidal approximation, F₀ = 2πγR*cos θ |
| Rupture criterion   | Lian et al. 1993: s_rupt = (1+0.5θ)V^{1/3} |
| Contact repulsion   | Hertz: F_n = 4/3 E* √R* δ^{3/2}        |
| Damping             | Linear viscous: F_d = −γ_n v_n          |
| Integration         | Velocity Verlet (symplectic, 2nd order) |

## Running the Simulation

```bash
cd cases_dynamic/liquid_bridge
python liquid_bridge_case.py
```

This produces:
- `results/final_state.json` — final particle state (loadable with `load_particles`)
- `results/history.json` — time-series data for post-processing
- `fig/separation.png` — surface separation vs time
- `fig/velocity.png` — particle velocities vs time
- `fig/capillary_force.png` — capillary force magnitude vs time
- `fig/summary.png` — combined multi-panel figure

## 3D Visualization (Polyscope)

Requires `polyscope` (`pip install polyscope`).

```bash
python visualize_bridge.py               # interactive viewer
python visualize_bridge.py --screenshot   # save frames to fig/frames/
```

Controls in the polyscope GUI:
- **Frame slider** — scrub through the simulation
- **Play checkbox** — auto-advance frames
- Info panel shows time, separation, bridge count, and capillary force

## Expected Output

With default parameters (1 mm silica glass spheres, water):

- **Bridge forms** at ~30 ms when surface gap ≤ 0.5 mm
- **Final separation** oscillates around ~300 μm (underdamped agglomerate)
- **Capillary force** peaks at ~0.39 μN (2πγR*cos θ)
- **Centre-of-mass drift** remains < 1e-10 m (momentum conservation)
- **Bridge survives** throughout (rupture distance ≈ 1.26 mm > max oscillation)

## Parameter Customization

Edit `src/_params.py` to change:

```python
R = 1e-3           # particle radius [m]
gamma = 0.072      # surface tension [N/m]
theta = radians(30) # contact angle [rad]
v_approach = 0.01  # approach speed per particle [m/s]
```

Or override programmatically:

```python
from cases_dynamic.liquid_bridge_dem.src._setup import setup_liquid_bridge
ps, detector, model, bridge_mgr = setup_liquid_bridge(R=5e-4, gamma=0.05)
```

## References

- Lian, G., Thornton, C., & Adams, M. J. (1993). A theoretical study of the
  liquid bridge forces between two rigid spherical bodies. *J. Colloid
  Interface Sci.*, 161(1), 138–147.
- Willett, C. D., Adams, M. J., Johnson, S. A., & Seville, J. P. K. (2000).
  Capillary bridges between two spherical bodies. *Langmuir*, 16(24),
  9396–9405.
