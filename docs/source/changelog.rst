Changelog
=========

v0.4.3 (current)
-----------------

- **Multiphase framework**: ``MultiphaseSystem``, ``PhaseProperties``,
  ``MultiphaseEOS``, per-phase dual volume splitting, sharp interface tracking.
- **Multiphase operators**: ``multiphase_stress_force``,
  ``multiphase_dudt_i``, integrated surface tension (2D + 3D).
- **Multiphase ICs**: ``PhaseAssignment``, ``MultiphaseMass``,
  ``MultiphasePressure`` with Young-Laplace jump.
- **Multiphase visualization**: ``record_multiphase_frame``,
  ``dynamic_plot_multiphase`` with phase/interface overlays.
- **Multiphase domain builders**: ``droplet_in_box_2d``,
  ``droplet_in_box_3d``.
- **Mass redistribution**: pressure-preserving ``redistribute_mass_multiphase``
  for retriangulation.
- **Multiphase retopologization**: ``_retopologize_multiphase`` preserving
  phase labels and interface identification.
- **Equation of State module**: ``TaitMurnaghan``, ``IdealGas``,
  ``MultiphaseEOS`` dispatcher.
- **Surface tension operators**: Heron-formula curvature (3D),
  integrated tangent-vector curvature (2D).
- **Case studies**: oscillating droplet (2D/3D), dam break (2D/3D).
- **2D curvature operators**: ``integrated_curvature_normal_2d``,
  ``surface_tension_force_2d``.
- **Mass redistribution operators**: single-phase and multiphase
  pressure-preserving redistribution after retriangulation.

v0.4.2
------

- Adaptive remeshing (2D): ``hyperct.remesh`` with edge split/collapse/flip.
- Domain builders: ``rectangle``, ``disk``, ``cylinder_volume``, ``ball``,
  ``l_shape``, ``annulus``, ``box``, ``pipe``.
- DEM submodule: particles, contacts, Hertz/linear-spring force models,
  sintered bonds, capillary bridges, fluid coupling, I/O.
- Integrated Cauchy stress tensor operators.
- Dynamic integrators: Euler, symplectic Euler, RK45, adaptive.
- Boundary condition framework: no-slip, Dirichlet, outlet, periodic inlet.
- Initial condition framework with composable ICs.
- GPU acceleration via PyTorch backend.
