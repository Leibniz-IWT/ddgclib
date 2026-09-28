# Oscillating Droplet Case Study

Multiphase oscillating droplet validation case.  An ellipsoidally
perturbed droplet relaxes toward its equilibrium shape under surface
tension, with analytical comparison to the Lamb/Rayleigh solution.

## Running the simulations

```bash
# 2D (overdamped, ~2 min)
python cases_dynamic/oscillating_droplet/oscillating_droplet_2D.py

# 3D (overdamped, ~5 min)
python cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py

# 2D mesh convergence study
python cases_dynamic/oscillating_droplet/mesh_convergence_2D.py
```

## Outputs

All outputs are saved **within this case directory**:

```
cases_dynamic/oscillating_droplet/
    fig/                                    # Plots and animations
        oscillating_droplet_2D_fluid.png    # Final pressure + velocity snapshot
        oscillating_droplet_2D_phases.png   # Final phase field
        oscillating_droplet_2D_radius.png   # R_max(t) vs analytical
        oscillating_droplet_2D_energy.png   # KE(t)
        oscillating_droplet_2D.mp4          # 2D animation
        oscillating_droplet_3D.mp4          # 3D animation
    results/
        snapshots/                          # JSON state snapshots for replay
            state_000000_t0.000062.json
            state_000001_t0.000684.json
            ...
```

## Viewing results in polyscope

After running a simulation, use the interactive polyscope viewer:

```bash
# Case-specific viewer (auto-detects snapshot path):
python cases_dynamic/oscillating_droplet/view_polyscope.py

# Or use the generic library viewer:
python -m ddgclib.scripts.view_polyscope \
    --snapshots cases_dynamic/oscillating_droplet/results/snapshots/ \
    --scalars p --vectors u
```

## Generating animations

Animations are generated automatically when the simulation scripts run.
They use the standard `dynamic_plot_fluid` from `ddgclib.visualization`
with multiphase overlays (`phase_field`, `interface_field`, `reference_R`).

To regenerate manually from saved snapshots:

```python
from ddgclib.data import StateHistory
from ddgclib.visualization import dynamic_plot_fluid
from ddgclib.scripts.view_polyscope import load_history_from_dir

history, HC = load_history_from_dir('cases_dynamic/oscillating_droplet/results/snapshots/',
                                     fields=('u', 'p', 'phase', 'is_interface'))

dynamic_plot_fluid(history, HC,
                   save_path='output.mp4',
                   phase_field='phase',
                   interface_field='is_interface',
                   reference_R=0.01)
```

## Parameters

See `src/_params.py` for physical parameters.  Two parameter sets:

- **Overdamped** (default): high-viscosity oil, monotonic decay
- **Underdamped**: water-like, oscillatory decay

## Physics

See `INTERFACE_STRESS.md` for the stress computation design and
`FEATURES.md` (project root) for planned improvements.
