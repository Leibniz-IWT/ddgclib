# Prompt for the next session: continue the ddgclib dynamics campaign as one major workflow

Written 2026-10-06 at the hand-off of the local session. Paste the section
"PROMPT" into a fresh Claude Code session (cloud or local) after both
repositories are cloned side by side and pushed to the commits named below.
Everything the prompt refers to is in the repository; nothing lives only in a
local scratch directory.

## State at hand-off

| repo | branch | HEAD at hand-off | remote |
|---|---|---|---|
| ddgclib | master | the commit that adds this file (lane F unreviewed, see below) | git@github.com:Leibniz-IWT/ddgclib.git |
| hyperct | master | `48d163a` (lane Q) | git@github.com:Stefan-Endres/hyperct.git |

ddgclib depends on hyperct at or after `48d163a` (lanes S, L, T, Q added
`rebuild_simplex_cache_3d`, `HC.V.move_all`, deterministic iteration order and
`simplex_dual_face_areas`). Both must be pushed.

Campaign lanes finished and committed on master (one commit per lane, each
after an independent review with green test suites): R (single-phase
conservative remap), S (exact setup volumes), P (hydrostatic column on
presets, `delaunay_material`), L (`frozen_set` axis, vertex-move collisions),
H (Hagen-Poiseuille 2D/3D pinned, simplex-gradient fluxes), T (deterministic
runs), O (`area_orientation` axis), B (full droplet outer mesh, baselines
re-pinned), Q (`edge_area_source` axis, exact 3D faces), M (`curvature_path`
shrunk to two values), W (every setup builds its force and retopology from
`SolverMethods`), F (dam-break phase ledger and face closure; implemented,
suites green, independent review cut off by the session end, see below).

Test suites at hand-off (from the ddgclib root, hyperct on the path):

| suite | command | result |
|---|---|---|
| ddgclib fast | `pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider` | 1240 passed, 12 skipped, 2 xfailed |
| ddgclib slow | `pytest ddgclib/tests -q -m slow -p no:cacheprovider` | 34 passed, 1 xfailed |
| hyperct | `cd ../hyperct && pytest hyperct/tests -k "not benchmark" -q -p no:cacheprovider` | 340 passed, 38 skipped, 6 xfailed |

Pinned main benchmark (cases_dynamic/oscillating_droplet/baselines, tracked
in git with their methods block): 2D preset `oscillating_droplet_2D` l2
0.17439096487276182 / tail 0.9998871416222597; 3D preset
`oscillating_droplet_3D` l2 0.24811443136179492 / tail 0.0841737962816189.

## Environment for a cloud session

```bash
git clone git@github.com:Leibniz-IWT/ddgclib.git
git clone git@github.com:Stefan-Endres/hyperct.git
cd ddgclib
ln -s ../hyperct/hyperct hyperct          # the untracked symlink every runner expects
conda env create -f environment.yml -n ddg || pip install -e . numpy scipy matplotlib pytest
pip install -e ../hyperct                 # live hyperct, never a wheel (a stale 0.3.5 wheel shadowed the tree once)
python -c "import hyperct, ddgclib; print(hyperct.__file__)"   # must point into ../hyperct
pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider      # expect 1240 passed
```

`CLAUDE.md` and `DEVELOPMENT.md` are git-ignored in this repository by the
owner's choice, so a fresh clone has neither. The rules below are
self-contained; if the owner wants the project CLAUDE.md in the cloud
session, `git add -f CLAUDE.md DEVELOPMENT.md` before pushing.

## PROMPT

You are continuing the ddgclib dynamics debugging campaign as one major
workflow. ddgclib is a Lagrangian discrete-differential-geometry fluid solver
(this repository); its mesh backend hyperct is the sibling repository
symlinked as `./hyperct`. The owner's standing instructions: get every
relevant dynamic test case working with the oscillating droplet as the main
benchmark; always ship a fix into the library as long as the method is
separated behind the method wrappers (`ddgclib.methods`: `SolverMethods`,
`AXES`, `PRESETS`) and no code is duplicated; commit verified work on master
(no branches until the owner organises branching); never touch the owner's
unrelated work.

Read first, in this order: `debugging_plan.md` (the status entries from
2026-10-01 onward and the section "Reproducibility protocol for method
lanes", which is binding), `METHODS.md` (every method axis with status and
evidence, the presets, the case matrix in section 4), the lane logs in
`docs_temp/debug_session/` named by each lane brief, and
`docs_temp/11_dynamics_audit_2026-09-25.md` for the original findings.

### How the campaign runs

One lane at a time on the shared working tree, as a Workflow script. The
script of the last workflow is `docs_temp/workflows/ddgclib-lanes-3.js`
(the two earlier ones are beside it). Reuse its structure as is: for each
lane an implementer agent, then an independent reviewer that re-runs the
suites and re-measures the lane's headline numbers, up to two fix rounds,
then a commit agent that stages the lane's files by explicit path and commits
on master (hyperct first when it changed), then the next lane. Before
launching, replace its two constants: `PY` (the interpreter of the ddg
environment) and `SCRATCH` (a scratch directory outside the repositories),
and replace every `bash ${SCRATCH}/lane_diff.sh` with
`bash docs_temp/workflows/lane_diff.sh` (the repo-resident helper prints the
lane's diff against HEAD in both repositories and excludes the owner's
unrelated uncommitted work). Replace the `LANES` array with the lanes below
and the `meta.phases` list to match. Keep the rule texts of the script; they
encode every lesson of the campaign (never execute scripts under
`cases_mean_flow/`, `benchmarks/`, `tutorials/` or `test_cases/`, not even
from a copy: some chdir to the absolute repo path and once overwrote the
owner's manuscript figures; deletion may be denied, so never create files you
would need to delete; probe outputs go to scratch; anything a lane log needs
for reproduction lives in the repository; off limits for edits:
`cases_dynamic/capillary_rise_energy_grad/` and the dynCA runners and
`src/_setup_dynca.py` under `cases_dynamic/capillary_rise/`, which belong to
another agent).

The commit step: the git index must be empty before staging (`git diff
--cached --stat`; a stale index is cleared with `git reset -q`, index only);
stage with explicit paths; never stage `.gitignore`, anything under
`cases_mean_flow/`, `benchmarks/`, `tutorials/`, `test_cases/`,
`cases_dynamic/capillary_rise/`, `cases_dynamic/capillary_rise_energy_grad/`,
`ddgclib/tests/test_integrated_validation.py`, generated outputs or logs;
message with the repo's prefix convention `ENH:` / `BUG:` / `MAINT:`, the
lane id in the first line, a body with the headline numbers and every pin
that moved as old -> new, no em dashes, and the last line
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; no push.

Test commands and baselines are in the table above. Every lane ends with the
three suites green, a lane log `docs_temp/debug_session/lane<ID>-<slug>.md`
(what changed, measurements with exact numbers and the `SolverMethods`
configuration behind each, measured DO-NOTs, known limits, how to reproduce),
a new entry at the top of the status log in `debugging_plan.md`, evidence and
status in `ddgclib/methods/_axes.py`, `python -m ddgclib.methods --update
METHODS.md`, and the case matrix row in section 4 of `METHODS.md` edited by
hand. Pins stay bit-identical unless the lane deliberately changes the
method behind them; then A/B through presets (`preset` versus
`preset.replace(...)`), re-pin with the methods block, old and new on record,
the old behaviour reachable as a registered option with status `broken` when
that costs one flag. Never loosen a tolerance or delete an assertion to make
a test pass. Validation by integrated comparisons (`ddgclib.analytical`),
never point-wise pressure comparisons. Stay in the Lagrangian formalism.

### Lanes, in this order

**Lane F-review (first, short).** Lane F (dam-break phase ledger and face
closure) is implemented and committed on master without a second-party
review: the reviewer was cut off by the session end. Run the review step of
the script on it as a lane of its own: `git show --stat HEAD~N..` for the
lane F commit, read `docs_temp/debug_session/laneF-ledger-and-face-closure.md`,
re-run the suites, re-measure its headline claims (alpha_art 0.2 and 0.1
complete the 1585-step 2D horizon with no vertex outside; the 3D preset
completes 793 steps on `edge_area_source='p_ij_simplex'`; the shipped
alpha 0.3 run is bit-identical, digest `952d4544676ca366`; the four new pins
in `test_case_dam_break.py::TestDamBreakPins`; the two new axes
`phase_ledger` and `face_closure` with their defaults flipped and the old
values kept as `broken`; `split_method='simplex'` opt-in), check the rules,
and fix what is wrong at the cause, then commit the fix as `lane F-review`.

**Lane G: electrolysis 3D and shearing plate.** Evidence: METHODS.md case
matrix rows; `docs_temp/audit_2026-09-25/cases_caprise_dam_bubble_shear.md`;
lane B log (the shearing-plate rescale (1, 2/3) maps outer vertices at
(0, +-0.0075) onto the droplet poles and the case's on_collision='evict' loop
deletes both interface poles; the 3D shearing setup crashes on the same
collision); lane L log (the periodic path implements neither
`frozen_set='membership'` nor the lane Q axis; `SolverMethods` raises); lane
F log (electrolysis_bubble_3D carried 26 stranded (vertex, phase) pairs from
its setup, released at the first rebuild under the new `phase_ledger`
default; not re-measured). Goal: both cases run through presets on library
paths with every new axis available on the periodic path, each with a pinned
smoke test and a physical check. Tasks: reproduce the 3D gas-phase loss
through the preset and find whether it is the gas injection, the one-vertex
phase in the redistribution, the wall clamp or the ledger (lane F's method
may already apply); fix in the library behind the registry; physical check:
a static bubble of the injected volume holds its Laplace pressure to the
discretisation error and the injected gas mass is conserved to round-off.
Shearing plate: move the rescale into the builder with `move_all` (scale
only vertices outside the droplet ring, or push a rescaled vertex that lands
inside R0 + h radially out); make `retopologize_multiphase_periodic`
(`ddgclib/methods/_retopo.py`) and `ddgclib/geometry/periodic.py` forward
`frozen_set`, `edge_area_source`, `area_orientation`, `phase_ledger`,
`face_closure` and `remap` like the non-periodic path, or raise with a clear
message; fix the seam-edge asymmetry of the 2D periodic area vectors lane O
counted (62 of 886 directed edges) if it is on the path; physical check: a
droplet under weak shear keeps its volume and its interface for the short
run and the periodic seam carries no spurious force (sum of forces on a
quiescent droplet at round-off). Pins: electrolysis 2D and 3D smoke
(per-phase mass drift, bubble volume), shearing plate 2D short run (droplet
volume, interface vertex count, deformation parameter), 3D setup plus a few
steps. Make `setup_shearing_plate_droplet` re-entrant or document why not.

**Lane I: static capillary rise and hand-built 3D setups.** Evidence:
METHODS.md rows `capillary_rise/capillary_rise_2D.py` and `_3D.py`
(hand-rolled scaffolds); `docs_temp/audit_2026-09-25/cases_caprise_dam_bubble_shear.md`;
lane P log (hydrostatic column on presets, `delaunay_material`, the
free-surface flutter and the saddle of the discrete equilibrium, artificial
viscosity); lane R log section 3 (free-surface limits of the remap); lane S
log section 7 (hand-built 3D complexes still read fan-walk volumes at setup:
cube_flow, cube2droplet, `Hagen_Poiseuile/src/_geometry.py`). You may add
files in `cases_dynamic/capillary_rise/` and edit `capillary_rise_2D.py`,
`capillary_rise_3D.py` and the non-dynCA src helpers; the dynCA files and
`capillary_rise_energy_grad/` stay untouched. Goal: both static runners go
through presets on library integrators, reach the static Jurin height for
their contact angle to the discretisation error, and are pinned; every
hand-built 3D setup gets exact simplex volumes through one library call
(`rebuild_simplex_cache_2d` / `_3d` from lane S, applied by a helper or a
builder argument, with a tiling test). Connectivity by measurement as lane
P did (`dual_only` versus `delaunay_material` + `remap='conservative'`),
walls frozen by membership, the wall and contact-angle condition as a
registered BC in the library if none exists, gravity through `body_force`,
EOS through `pressure_model`. Validation: meniscus height against Jurin's
law and the meniscus shape against the Young-Laplace solution for the tube
width, integrated comparisons, two refinement levels. Pins: meniscus height
and max|u| at the end, 2D fast and 3D slow; presets
`capillary_rise_static_2D/3D`; README and methods.json. Report what the
static case says about the free-surface items lane P left open and what the
dynCA work (read-only) would need from the library next.

**Lane X: every runner under cases_dynamic/ runs or is retired.** Evidence:
METHODS.md case matrix (cube2droplet: broken import `Cube2droplet`;
liquid_bridge_approach: broken import; cube_flow stalls after step 0;
bc_demo_v2: ImportError `_rebuild_nn_from_delaunay`; liquid_bridge_equilibrium
Case 1 and 5: AttributeError 'vd'; liquid_bridge_cfd_dem: AttributeError
'u'); lane W log (not converted: `cube2droplet/*`, `liquid_bridge_cfd_dem/*`,
`oscillating_droplet_p_ref/scripts/*`, the stale
`oscillating_droplet/diagnose_split_methods.py`). Run each runner's shortest
mode from a scratch copy; fix import errors by pointing at the library
function that replaced the old one, never by re-adding dead code; convert
demos with hand-rolled loops to presets where the library can do it
(cube_flow, bc_demo); the frozen-surface liquid-bridge cases through
`connectivity='frozen'` presets; retire what duplicates a shipped case or
depends on removed features by saying so in the case matrix and in a header
docstring (do not delete files; list what the owner should delete). One fast
headless smoke test per revived runner. The case matrix gets one honest row
per runner with its preset name.

**Lane V: impenetrability and the interface vote (the two blockers every
free-surface case shares).** Evidence: lane L known limit 1 (a fluid vertex
can pass between two wall vertices; HP2D had one 5.1e-3 outside the top wall
at t = 30; dam-break refinement 4 and its `split_method='simplex'` arm end
with a vertex through the floor; the case-local `WallClampBC` of the
electrolysis case and the dynCA clamp are the two existing clamps); lane F
follow-ups (an interface-aware simplex vote so a one-cell liquid tongue
survives the collapse, since a tie now goes to the lower phase ID and erases
a lone liquid vertex; a mass-conserving per-phase sliver merge or a registered
mass floor for the refinement 4 small-cell spike; the 3D dam break on wall
half cells). Implement a library wall clamp (planar and, if cheap, general
wall: put the vertex back on the wall and zero its normal velocity, behind a
registered BC and a method axis for the policy), replacing the two case-local
clamps; an interface-aware vote as a registered `split_method` or vote
option; the sliver merge extended to `m_phase` behind an axis. Measure each
on the dam break (alpha_art 0.1 and refinement 4 through the horizon), on
HP2D and on the electrolysis case; adopt by the flip rule; re-pin what
moves.

**Lane D (last): the oscillating droplet with every new method.** The owner
asked for this explicitly: "when you finish all lanes please also try your
newest methods/fixes on the oscillating droplet too". Build the arm list
from the registry: every explicit axis value that applies to the multiphase
droplet and that was added or changed since 2026-09-25, each as
`preset.replace(...)` on the 2D and the 3D preset: `frozen_set='membership'`;
`edge_area_source='p_ij_simplex'` (3D); `area_orientation` (default now,
legacy as control); `remap='conservative'` in 3D now that lanes O, Q and B
changed what it reads; `projection_every` in {2, 5} with and without the 3D
remap; `redistribute_mass=False`; `phase_ledger` and `face_closure` arms
(the droplet runs at P0 = 0 where lane F measured them inert: confirm);
`split_method='simplex'`; `curvature_path='csf_dual'` as the measured-worse
control; lane V's clamp and vote if registered; then the best combinations
(at most eight). Score every arm with the case's own score (l2 against the
two-fluid reference, tail growth, linf, mass drift, R_max peak, KE max, the
quarter-mean error channels of lane G, July) and record methods.json next
to each; run 3D arms in parallel processes (deterministic since lane T; a
full 3D run is about 8 min). Adopt by the flip rule (better l2 AND tail
against the pin), re-pin with the methods block, keep the previous preset
as a named preset. Write the result as a short section at the top of the
`debugging_plan.md` status log titled "Where the oscillating droplet stands
(2026-10)": the pinned numbers before and after the campaign (2D 0.17479 in
September), the physics gap that remains against the two-fluid reference
with its best current attribution (lane Q showed the 3D pin is a
cancellation between an outward bump and an over-decay; lane H of July
attributed the 2D over-decay to the every-step projection), and the one or
two next experiments the evidence supports.

### Known open items to carry in the lane logs (not lanes of their own)

- `frozen_set='membership'` is implemented for `connectivity='delaunay'`
  only (adaptive, periodic, delaunay_material, dual_only_bare raise); a
  constrained-vertex set in `hyperct.remesh` would open it for adaptive.
- The 2D periodic branch returns different dual segments from the two ends
  of 62 of 886 seam edges (lane O); lane G touches it.
- Hydrostatic column: `hydrostatic_2D_periodic` runs free-slip side walls,
  not periodic connectivity; the column needs artificial viscosity (alpha
  0.05 suffices) because a free-surface flutter and a saddle of the
  discrete equilibrium are damped, not fixed; the water-viscosity blow-up
  at 64 acoustic times is unexplained (lane P).
- `delaunay_material` is exact in 2D only while every old boundary edge
  survives (an L-shape at rest grows by 1e-2 at the re-entrant corner); in
  3D it leaves slivers of order 1e-6 of the volume per call (no facet
  recovery) (lane P).
- Hagen-Poiseuille has no pressure solve, so continuity is not enforced;
  the 2D error falls by about 3 per refinement instead of 4 (lane H).
- The bare `pytest -q` at the hyperct root fails at collection (duplicate
  test module names under `archives/` and `hyperc_rl_quick_figs_delete/`);
  use `pytest hyperct/tests -k "not benchmark"`.
- `CLAUDE.md` still says 3D projected domains need `_retopologize()` before
  `compute_vd()`; since lane S the builders carry the simplex cache.
- Stray probe outputs the owner may delete: `osc3d.log` in the repo root,
  `cases_dynamic/Hagen_Poiseuile/results/laneL_smoke/`, ten
  `*laneLprobe_smoke*` files under `cases_dynamic/capillary_rise/fig` and
  `results`.

### Deliverable of the session

Every lane committed on master in both repositories with green suites, the
registry and `METHODS.md` current, `debugging_plan.md` with one entry per
lane and the droplet summary on top, and a closing message to the owner
that states, for each case in the METHODS.md case matrix, whether it runs
through a preset, what its pin is, and what remains.
