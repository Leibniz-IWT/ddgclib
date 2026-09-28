# Liquid Bridge Equilibrium Benchmarks

This folder contains five benchmark scripts and their canonical outputs.

## Cases

1. `Case_1_equilibrium_particle_particle_bridge_benchmark.py`
   - exact catenoid baseline
   - mesh displacement field is zero by construction
   - dynamic path uses damped `symplectic_euler(...)` for relaxation on the exact catenoid

2. `Case_2_perturbed_mesh_equilibrium_particle_particle_bridge_benchmark.py`
   - deterministic perturbed-mesh case
   - mesh PNGs use displacement from the exact catenoid

3. `Case_3_off_equilibrium_liquid_bridge_particle_particle_bridge_benchmark.py`
   - off-equilibrium liquid-bridge case
   - mesh PNGs use displacement from the exact catenoid

4. `Case_4_off_equilibrium_relaxation_particle_particle_bridge_benchmark.py`
   - off-equilibrium relaxation case
   - mesh PNGs use displacement from the exact catenoid

5. `Case_5_volumetric_stress_equilibrium_particle_particle_bridge_benchmark.py`
   - volumetric catenoid-like bridge benchmark for `stress.py`
   - zero-pressure / zero-velocity equilibrium run through `symplectic_euler(...)`
   - exercises the volumetric `stress.py` operator path honestly on a 3D bridge volume
   - also benchmarks the extracted boundary surface against the Endres catenoid reference

## Canonical outputs

Each case writes into its own folder under:

- `out/Case_1`
- `out/Case_2`
- `out/Case_3`
- `out/Case_4`
- `out/Case_5`

Cases 1 to 4 produce:

- one reproduced static/dynamic/reference figure
- one hold-time figure
- four mesh PNGs (`ref 2` to `ref 5`)

Case 5 currently produces:

- one reproduced static/dynamic/reference-style stress figure
- one hold-time stress-residual figure
- one DDG/FD capillary companion figure
- four mesh PNGs (`ref 0` to `ref 3`)

## Current hold-time setup

Cases 1 to 4 are currently aligned on:

- `dt = 2e-6`
- `max hold time = 2e-4`
- sampled times:
  - `0.000002`
  - `0.000010`
  - `0.000020`
  - `0.000040`
  - `0.000100`
- `0.000200`

Case 5 currently uses the same:

- `dt = 2e-6`
- `max hold time = 2e-4`
- sampled times:
  - `0.000002`
  - `0.000010`
  - `0.000020`
  - `0.000040`
  - `0.000100`
  - `0.000200`

## Current plotting conventions

### Reproduced static/dynamic/reference figure

- Cases 1 to 4:
  - right axis: DDG capillary force error (%)
  - left axis: DDG integration error (%)
  - Endres (2024) reference curves remain separate reference series
- Case 5:
  - right axis: stress capillary force error (%) on the extracted boundary surface
  - left axis: DDG integration error (%)
  - Endres (2024) reference curves remain separate reference series

### Hold-time figure

- DDG and FD capillary force error are compared directly
- orange solid: DDG
- blue dashed: FD

### Case 5 figures

- reproduced 6-series figure now uses **Case 5 stress capillary error** against the Endres reference on the Case 5 boundary surface
- hold-time figure remains the **volumetric Cauchy stress residual** plot
- DDG/FD capillary comparison remains a separate companion figure
- dynamic evaluations use the integrator's built-in thread workers (`workers=8` by default on this machine, overridable with `DDGCLIB_CASE5_WORKERS`)
- for the hold-time residual plot, the target value is **zero** and the current residual is numerically zero to machine precision / machine-scale

## Case 1 operator coverage

Case 1 now reports the axial capillary-force error from the three operator families Stefan asked for on the same equilibrium bridge mesh:

- built-in `stress.py` combined with a **benchmark-local** interface surface-tension term
- built-in `multiphase_stress.py`
- curvature / DDG

The benchmark also still keeps the direct `surface_tension.py` path as a cross-check. The library file `ddgclib/operators/stress.py` itself remains a plain pressure/viscous Cauchy-stress operator; the extra capillary combination lives in the benchmark case code.

## Case 5 operator coverage

Case 5 adds the missing volumetric `stress.py` path on a thickened catenoid-like bridge volume:

- dynamic integrator: `symplectic_euler(...)`
- operator: `stress.py`
- geometry: volumetric bridge mesh built from a structured catenoid-like volume with boundary counts `8, 16, 32, 64`

Case 5 now also reports boundary-surface capillary-force error on its own meshes using:

- built-in `stress.py` combined with a **benchmark-local** interface surface-tension term
- built-in `multiphase_stress.py`
- curvature / DDG

and compares the stress-path capillary curve against the Endres reference in the reproduced 6-series figure. The separate hold-time plot still reports volumetric Cauchy stress residuals.

In other words, the Case 5 hold-time figure is **not** an analytical catenoid-force error plot. It is a residual-to-zero check: at equilibrium, the volumetric Cauchy stress residual should be zero, and the current benchmark values are numerically zero to machine precision / machine-scale.

## Mesh coloring

- Cases 2 to 4: red displacement map relative to the exact catenoid
- Case 1: same displacement-map renderer, but the field is zero everywhere
- Case 5: same displacement-map renderer style, with a zero field against the exact volumetric catenoid-like bridge

## Console printout

When each script runs, it now prints:

- the saved reproduced figure path
- the saved hold-time figure path
- DDG and FD capillary force errors (%) at the script's max hold time
- Case 1 also prints the built-in `surface_tension`, built-in `multiphase_stress`, and curvature/DDG capillary force errors (%)
- Case 5 prints the max-hold-time volumetric `stress.py` axial and max-local stress residuals (%) together with the worker count used for the dynamic run
- Case 5 also prints the built-in stress-path, built-in `multiphase_stress`, and curvature/DDG capillary force errors (%) on its own boundary-surface benchmark path

## Current max-hold-time DDG/FD capillary results

The current benchmark printouts at `t = 0.000200 s` and `dt = 2e-06 s` are:

### Case 1

| ref | n_boundary | n_total | DDG_capillary_% | FD_capillary_% |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8 | 36 | `4.163336342344e-15` | `5.388281834606e-10` |
| 3 | 16 | 136 | `1.387778780781e-15` | `4.377976713793e-10` |
| 4 | 32 | 528 | `3.469446951954e-16` | `1.557550222333e-10` |
| 5 | 64 | 2080 | `2.385244779468e-15` | `3.157284591829e-12` |

### Case 2

| ref | n_boundary | n_total | DDG_capillary_% | FD_capillary_% |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8 | 36 | `2.775557561563e-15` | `5.782411587263e-17` |
| 3 | 16 | 136 | `1.734723475977e-15` | `1.683858492455e-11` |
| 4 | 32 | 528 | `2.862293735362e-15` | `2.525755745684e-10` |
| 5 | 64 | 2080 | `1.279358563533e-15` | `9.366347001470e-11` |

### Case 3

| ref | n_boundary | n_total | DDG_capillary_% | FD_capillary_% |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8 | 36 | `3.997913111675e-10` | `8.082422173809e-10` |
| 3 | 16 | 136 | `1.476260086397e-10` | `5.725051075571e-10` |
| 4 | 32 | 528 | `1.829803669695e-11` | `3.788629566333e-11` |
| 5 | 64 | 2080 | `1.108522995619e-11` | `1.126067921092e-10` |

### Case 4

| ref | n_boundary | n_total | DDG_capillary_% | FD_capillary_% |
| --- | ---: | ---: | ---: | ---: |
| 2 | 8 | 36 | `1.136206793410e-03` | `3.787348272044e-04` |
| 3 | 16 | 136 | `2.741100105733e-04` | `9.137008150082e-05` |
| 4 | 32 | 528 | `1.215860940661e-06` | `4.050346631707e-07` |
| 5 | 64 | 2080 | `5.163323169629e-07` | `1.722640202278e-07` |

### Case 5

| ref | n_boundary | n_total | DDG_capillary_% | FD_capillary_% |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8 | 81 | `5.551115123126e-15` | `1.616482989131e-09` |
| 1 | 16 | 289 | `5.551115123126e-15` | `8.924359442992e-10` |
| 2 | 32 | 1617 | `1.110223024625e-14` | `9.976744907644e-10` |
| 3 | 64 | 8385 | `9.367506770275e-15` | `3.346628374000e-10` |

## Practical note

The scripts are intended to keep only canonical outputs in each `out/Case_*` folder. Older experimental PNG variants may be removed during cleanup.
