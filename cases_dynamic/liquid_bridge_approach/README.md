# Liquid Bridge Approach Cases

This folder contains the Pitois et al. (2000) Fig. 6 approaching-sphere
validation cases.  The cases were developed from the PR #35 / Case 12
separation workflow, but use an approach-specific initial mesh and motion:

- two ruby spheres with radius `R = 4.0 mm`,
- liquid 2 with `V = 1.10 uL`,
- initial gap `D0/R = 0.200`,
- lower sphere approaching a fixed upper sphere at `v = 10 um/s`,
- canonical t0 contact-line radius `r_CL = 1.071 mm`.

The active scripts intentionally keep the requested spelling `finial` in the
file and result-folder names.

## Contents

- `case_1_finial.py` to `case_5_finial.py`: five final approach case entry
  points.
- `run_fig6_approach_case12.py`: shared Fig. 6 approach runner.
- `_ddgclib_case_core.py`: local ddgclib/PR35 Case 12 solver core used by the
  approach scripts.
- `fig6_t0_finial.py`: rebuilds the final t0 mesh from the committed
  tetrahedral source mesh by adding boundary surface triangles.
- `fig6_t0_finial/mesh/fig6_approach_t0.msh`: final initial mesh used by all
  five cases.
- `fig6_t0_finial/mesh/fig6_approach_t0_source_gmsh_iter0012.msh`: source
  tetrahedral Gmsh mesh before boundary triangles are added.
- `py_for_png_gif/`: scripts used to render the PPT PNG/GIF assets.
- `20260617_liquid_bridge_approach_comparison_5cases_v2.pptx`: comparison deck
  for the five approach cases.

The case result folders, generated GIFs, generated PNGs, sparse histories, and
per-step histories are intentionally not committed.  The comparison deck embeds
the final result plots and animations.  Rerunning a case creates a result folder
with the same stem as the case script, for example `case_5_finial/`.

## Equations Used in the Raw Force Report

The reported scalar force is the top-sphere axial reaction, plotted positive
for attraction:

`F(t) = -(F_top,z)`.

The total top-sphere force assembled in the approach cases is

`F_top = F_Heron + F_Cauchy + F_PR33 + F_lub + F_hydro + F_Cox`.

Only terms enabled by a given case contribute.  Case 1 does not include the
lubrication pressure term.  Cases 2 to 5 include it.

The Newtonian wall traction uses the Cauchy stress tensor

`sigma = -p I + mu (grad u + grad u^T)`,

and the wall force form

`F_wall = integral_A sigma n dA`.

The lubrication wall-pressure contribution is

`F_lub = integral_Awet sigma_lub n dA`, with `sigma_lub = -p_lub I`.

The pressure profile used in Cases 2 to 5 is

`p_lub(r) = C_lub * 3 mu |dh/dt| (a_lub^2 - r^2) / h^3`, for
`0 <= r <= a_lub`.

For large gaps this term decays rapidly because of the `h^-3` dependence.  It
is not hard-switched to zero unless viscosity, approach speed, liquid volume,
or the solver lubrication flag is zero.

The Pitois-style gorge force used by the baseline capillary report is

`F_gorge = Delta p_g pi r_g^2 + 2 pi gamma r_g`.

## Case Summary

| Case | Force model change | What worked | What did not work | Why |
| --- | --- | --- | --- | --- |
| 1 | Baseline PR35/Case 12 approach recipe with gorge pressure, Heron surface tension, wall-viscous Cauchy traction, hydrostatic force, and Cox contact-line force. | Captures the attractive capillary branch up to about 60 s. | Misses the near-contact repulsive branch after about 60 s. | The raw reported force does not include the squeeze-film pressure that develops between approaching spheres at small gap. |
| 2 | Adds gap-dependent lubrication wall pressure with `C_lub = 1.55`. | Produces the required repulsive branch near contact. | Late-time response can become too sharp and can show post-60 s mismatch. | The pressure term is physically needed, but the constant strength plus unconstrained contact-line motion is numerically too abrupt near contact. |
| 3 | Keeps `C_lub = 1.55` and caps contact-line motion to `0.5 um/step`. | Reduces late force spikes compared with Case 2. | Still not smooth enough in the final branch. | The contact-line cap improves geometry continuity, but the lubrication strength is still high near the smallest gaps. |
| 4 | Uses a tighter `0.2 um/step` contact-line cap and a late-gap lubrication scale table. | Improves smoothness and the post-60 s trend. | Uses a gap-dependent scale table, so it is less attractive as a predictive model. | The scale table compensates the near-contact branch but is closer to a calibration curve than a single model parameter. |
| 5 | Uses one global lubrication coefficient `C_lub = 1.39` and motion-scaled contact-line smoothing. | Best current balance: smoother force curve and better post-60 s match without a D/R lookup table. | `C_lub = 1.39` is still calibrated on Fig. 6 and needs independent validation. | Keeping one coefficient avoids the most obvious overfitting while retaining the physically required lubrication wall pressure. |

Case 5 is the recommended current candidate for a ddgclib patch, but it should
not yet be called predictive.  The next validation step is to repeat the same
coefficient on independent approach data and perform mesh/time-step sensitivity
tests.

## Reproducing the Stored Cases

From this folder, regenerate the final t0 mesh:

```bash
python3 fig6_t0_finial.py
```

Run a case:

```bash
python3 case_5_finial.py
```

By default, the final scripts record sparse checkpoints every 1000 steps and
write the smooth force curve from every simulation step when
`history_every_step=True`.  Increase `--mesh-snapshot-every` when new GIF frames
are needed.

Generate PPT-style GIFs:

```bash
python3 py_for_png_gif/1window_gif.py --case-dir case_5_finial
python3 py_for_png_gif/2windows_gif.py --case-dir case_5_finial
python3 py_for_png_gif/ppt_cover.py --case-dir case_5_finial
```

## Literature References

1. Pitois, O., Moucheront, P., and Chateau, X. "Liquid bridge between two
   moving spheres: an experimental study of viscosity effects." Journal of
   Colloid and Interface Science 231, 26-31 (2000),
   DOI: 10.1006/jcis.2000.7096.
2. Reynolds, O. "On the Theory of Lubrication and its Application to Mr.
   Beauchamp Tower's Experiments." Philosophical Transactions of the Royal
   Society of London 177, 157-234 (1886), pp. 188-190, Eqs. (17), (21), (24),
   and (27), DOI: 10.1098/rstl.1886.0005.
3. Batchelor, G. K. "An Introduction to Fluid Dynamics." Cambridge University
   Press (1967), DOI: 10.1017/CBO9780511800955.
4. Meyer, M., Desbrun, M., Schroder, P., and Barr, A. H. "Discrete
   Differential-Geometry Operators for Triangulated 2-Manifolds." Visualization
   and Mathematics III (2003).
5. Leibniz-IWT/ddgclib PR #35: Case 12 separation workflow used as the
   numerical base for these approach cases.
