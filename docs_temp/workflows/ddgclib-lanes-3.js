export const meta = {
  name: 'ddgclib-lanes-3',
  description: 'ddgclib campaign lanes W, F, G, I, X, D in sequence: method plumbing for every multiphase setup, dam-break slivers, electrolysis 3D and shearing plate, static capillary rise, broken demos, final oscillating-droplet evaluation',
  phases: [
    { title: 'W methods reach every setup', detail: 'all multiphase setups build their force from SolverMethods; recorded axes are applied axes' },
    { title: 'F dam break slivers', detail: 'air sliver-cell ejection blocker, 2D and 3D, through presets' },
    { title: 'G electrolysis 3D and shearing plate', detail: 'gas phase lost in 3D; shearing-plate rescale deletes poles; periodic path axes' },
    { title: 'I static capillary rise', detail: 'capillary_rise 2D/3D scaffolds on presets; hand-built 3D setups' },
    { title: 'X broken demos', detail: 'cube2droplet, liquid bridges, bc_demo, cube_flow: run or retire, case matrix honest' },
    { title: 'D oscillating droplet with every new method', detail: 'A/B of every new axis on the 2D and 3D droplet, adoption by the flip rule, re-pin' },
  ],
}

const SCRATCH = '/tmp/claude-1000/-home-endres-projects-ddgclib/aeffc932-b55d-49f0-9349-f03613d0bde4/scratchpad'
const PY = '/home/endres/anaconda3/envs/ddg/bin/python'

const COMMON = `
You are continuing the ddgclib dynamics debugging campaign. ddgclib is a Lagrangian discrete-differential-geometry fluid solver (repo root /home/endres/projects/ddgclib). Its mesh backend hyperct lives at /home/endres/projects/hyperct and is symlinked into the repo as ./hyperct. The user's standing instructions: get all relevant dynamic test cases working with the oscillating droplet as the main benchmark; always ship a fix into the library as long as the method is separated behind the method wrappers (ddgclib.methods: SolverMethods, AXES, PRESETS) and no code is duplicated.

READ FIRST (skim what is not relevant to your lane):
- debugging_plan.md: the status entries from 2026-10-01 onward (lanes R, S, P, L, H, T, O, B, Q, M and anything newer) and the section "Reproducibility protocol for method lanes" (binding rules, including rule 8 on process dependence).
- METHODS.md: every method axis with status and evidence, the presets, and the case matrix (section 4).
- docs_temp/11_dynamics_audit_2026-09-25.md and the reports in docs_temp/audit_2026-09-25/.
- The lane logs in docs_temp/debug_session/ that your brief names.

RULES
1. Python is ${PY}, run from the repo root /home/endres/projects/ddgclib. The shell prints conda start-up noise on every command; ignore it.
2. Git: never commit, stash, checkout, reset, restore or otherwise use git to change either working tree or history (the only exception is the dedicated commit step that runs after a lane passes review). Read-only git (log, diff, status, show) is fine. Both repos are on master and every earlier lane is committed, so HEAD is the pre-lane state: the lane's changes are exactly what bash ${SCRATCH}/lane_diff.sh prints (add --stat for the file list); it excludes unrelated uncommitted work of the user and of another agent.
3. Off limits, never edit AND never execute anything from them, not even from a scratch copy (several scripts chdir to an absolute repo path and overwrite the user's manuscript files; this happened once): cases_mean_flow/, benchmarks/, tutorials/, test_cases/, manuscript directories. Also off limits for edits: cases_dynamic/capillary_rise_energy_grad/ (another agent owns it); in cases_dynamic/capillary_rise/ do not change the behaviour of the dynCA runners or src/_setup_dynca.py. If a library change of yours can move the numbers of those runners, say so in your lane log.
4. Library over case code: a method a case needs goes into ddgclib (or hyperct) behind a registered axis in ddgclib/methods/_axes.py with a SolverMethods field or builder in ddgclib/methods/_config.py. Cases consume a preset from ddgclib/methods/_presets.py and write methods.json with record_methods. No duplicate hand-rolled loops or closures in case files. Keep changes minimal and surgical, match the existing style, add nothing speculative.
5. Pins: every pinned number in ddgclib/tests must stay bit-identical unless your lane deliberately changes the method behind it. If it does, A/B through presets (preset versus preset.replace(...)), re-pin together with the methods block (baselines under cases_dynamic/oscillating_droplet/baselines carry one), and record old and new values in the lane log. Never loosen a tolerance or delete an assertion to make a test pass. When a lane replaces a default because the old behaviour is a defect, keep the old behaviour reachable as a registered option with status 'broken' when that costs a flag, so earlier results stay reproducible; say so if it is not feasible.
6. Scratch and outputs: deletion (rm) is denied in this environment, so do not create files you would need to delete. ${SCRATCH} lives under /tmp and is wiped by a machine restart (this happened once mid-lane), so anything the lane log needs for reproduction (detector and diagnose scripts) must live in the repository, not in scratch. Probe runs write only under ${SCRATCH}/lane<ID>/ (run case runners from scratch copies or pass an explicit output directory). Only results that are meant to be kept go into the case directory (fig/, results/). A diagnose script worth keeping goes next to its case as diagnose_*.py.
7. Tests. Fast suite: ${PY} -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider (baseline at the start of this workflow: 1227 passed, 12 skipped, 2 xfailed, about 130 s). Slow pinned battery: ${PY} -m pytest ddgclib/tests -q -m slow -p no:cacheprovider (baseline: 32 passed, 1 xfailed, about 240 s). hyperct, when you touch it: cd /home/endres/projects/hyperct && ${PY} -m pytest hyperct/tests -k "not benchmark" -q -p no:cacheprovider (baseline 340 passed; the bare pytest -q at the hyperct root fails at collection for unrelated reasons). All must be green when you finish; earlier lanes may have raised the counts.
8. Use validation by integrated comparisons (ddgclib.analytical), never point-wise pressure comparisons. Stay in the Lagrangian formalism.
9. Documentation when done: a lane log docs_temp/debug_session/lane<ID>-<slug>.md (what changed and where, measurements with exact numbers and the SolverMethods configuration behind each, measured DO-NOTs, known limits, how to reproduce); a new entry at the TOP of the status log in debugging_plan.md; the DEVELOPMENT.md checklist; evidence and status in ddgclib/methods/_axes.py; then ${PY} -m ddgclib.methods --update METHODS.md and edit the case matrix rows in section 4 of METHODS.md by hand. Never use an em dash in prose.
10. If part of the brief turns out wrong or infeasible, do everything that is achievable and state exactly what is left and why. Report honestly: failing tests with their output, unverified claims marked as unverified. State numbers from process-dependent runs as ranges over several fresh processes.
`

const LANES = [
  {
    id: 'W', phase: 'W methods reach every setup',
    brief: `
LANE W: every multiphase setup builds its force from SolverMethods, so a recorded axis is an applied axis.

Evidence: lane M log (setup_oscillating_droplet(methods=) and run_a5b(methods=) now build dudt_fn through methods.dudt_fn; the dam-break, electrolysis and shearing-plate setups still build their own partial(multiphase_dudt_i, ...)), lane O log known limit 1 (area_orientation on those presets is recorded but not applied), METHODS.md section 5. Code: cases_dynamic/dam_break/src/, cases_dynamic/electrolysis_bubble/src/_setup.py, cases_dynamic/shearing_plate_droplet/src/_setup.py, cases_dynamic/oscillating_droplet/src/_setup.py (the pattern to copy), ddgclib/methods/_config.py.

Goal. No case file builds a force partial, a retopology closure or integrator kwargs by hand. Each setup takes methods= and derives split_method, redistribute_mass, curvature_path, area_orientation, edge_area_source, frozen_set and every other force or retopology choice from it; the runners pass the preset; methods.json is the truth for every case. Bit-identity at the defaults is the acceptance test.

Tasks.
1. Inventory every call site in cases_dynamic/ (not capillary_rise*) that constructs partial(multiphase_dudt_i or dudt_i), that passes split_method / redistribute_mass / curvature_path explicitly, or that builds retopology kwargs by hand. Include diagnose_*.py drivers that run the solver.
2. Convert them to methods= the way setup_oscillating_droplet does, keeping the same partial at the defaults (prove it with the keyword-equality tests of test_methods.py, extended per setup) and the same numbers: every pinned test, the two droplet baselines, the dam-break, electrolysis and shearing-plate smoke digests of lanes L and B.
3. Where a setup needs a geometry or physics object the builder does not take (gravity closure, NaN guard, gas injection, wall clamp), keep it as a wrapper around methods.dudt_fn(..., body_force=...) in the library builder if it is generic, or as the thinnest case-local wrapper if it is case physics; say which.
4. Remove the redundant explicit kwargs the runners still pass alongside methods= (lane M review item), and the private _retopologize call in Hagen_Poiseuile_2D.py post-processing if a public refresh on SolverMethods is the right home for it (add it as a small method if so).
5. Update METHODS.md section 5 and the case matrix so 'recorded but not applied' disappears everywhere it is no longer true.
`,
  },
  {
    id: 'F', phase: 'F dam break slivers',
    brief: `
LANE F: the dam break through the whole horizon in 2D and a working 3D run.

Evidence: METHODS.md case matrix rows for dam_break; docs_temp/debug_session/laneF-dam-break-unstick.md (July: conservative remap makes the collapse run; blocker = air sliver-cell F/m ejection); laneL log (with walls held by membership the run is still lost 8 to 23 steps after the hull arm); laneB log (dam_break_3D setup phases are now voted on builder tetrahedra; re-check); lane S follow-up (remove the auto-Delaunay footgun in MultiphaseSystem.assign_simplex_phases*). Presets dam_break_2D (delaunay + conservative remap, frozen_set membership) and dam_break_3D (dual_only).

Problem. Near the moving free surface and the wall corner, thin air cells (slivers) get a tiny dual volume and mass; the stress force divided by that mass ejects the vertex at a speed far above the wave speed and the run is lost. The remap keeps the pressure field invariant but does not bound F/m on a degenerate cell.

Goal. dam_break_2D runs the full 0.2 s horizon at the shipped alpha_art and at least one lower value without an ejection, with a pinned regression test (fast variant plus a slow full-horizon variant); dam_break_3D runs its smoke horizon through a preset and is pinned; the fix is a registered library method, not a case hack.

Tasks.
1. Reproduce the ejection on the current library through the preset: step, vertex, cell volume, mass, |F|, |a|, phase and neighbourhood; show whether it is a vertex created by inlet/merge, a corner vertex, or a bulk vertex whose cell collapsed in a reconnection, and whether the mass ledger (per-phase masses after the remap) or the geometry is at fault.
2. Candidate library methods, each behind a registered axis or a registered option on an existing axis, measured one at a time through preset.replace(...): mass-conserving merge of a sliver vertex into its neighbour (ddgclib.multiphase.mass_conserving_merge exists; a merge threshold axis), a minimum dual-volume floor in the acceleration (a registered limiter, reported honestly as a regularisation), the material Delaunay connectivity of lane P (keeps the fluid domain, no convex fill of the free surface), exact 3D edge areas of lane Q for the 3D case, the frozen-set policy. Adopt by the protocol: the arm that survives the horizon with the smallest change to the resolved part of the flow, and say what it costs.
3. Compare the surviving run against the reference the case already carries (the laneF KE_liq peak 1.0369e-3 J at 0.0506 s was measured on the lossy mesh; re-measure) and against the classical dam-break front position law for the resolution used; report the numbers.
4. Pins: KE_liq peak and time, front position at two times, mass drift, for 2D (fast variant at reduced refinement, slow full) and 3D (smoke).
`,
  },
  {
    id: 'G', phase: 'G electrolysis 3D and shearing plate',
    brief: `
LANE G: the 3D electrolysis bubble keeps its gas phase; the shearing-plate droplet survives its setup and its shear.

Evidence: METHODS.md case matrix rows for electrolysis_bubble and shearing_plate_droplet; docs_temp/audit_2026-09-25/cases_caprise_dam_bubble_shear.md; laneB log (shearing plate: the anisotropic rescale (1, 2/3) maps outer vertices at (0, +-0.0075) onto the droplet poles and the case's on_collision='evict' loop deletes both interface poles; the 3D setup crashes on the same collision; electrolysis 2D on the full mesh ends with KE 4.26e-02 J and no reference ranks it); laneL log (periodic path does not implement frozen_set='membership'; the lane Q axis is not forwarded on the periodic path either); lane P (delaunay_material); lane K/R (single-phase remap) do not apply (multiphase).

Goal. Both cases run through presets on library paths, with every new axis available on the periodic path, and each has a pinned smoke test with a physical check.

Tasks.
1. Electrolysis 3D: reproduce the loss of the gas phase (by t about 1.1e-4 s per the matrix) on the current library through the preset; find whether it is the gas injection in the callback, the per-phase redistribution of a one-vertex phase, the wall clamp, or a sliver (lane F's method may apply). Fix in the library behind the registry. Physical check: a static bubble of the injected volume holds its Laplace pressure to the discretisation error, and the injected gas mass is conserved to round-off.
2. Shearing plate: move the rescale into the builder (scale only the vertices outside the droplet ring, or push a rescaled vertex that lands inside R0 + h radially out) with move_all, so no interface vertex is deleted; make the periodic multiphase retopology (ddgclib/methods/_retopo.py:retopologize_multiphase_periodic and ddgclib/geometry/periodic.py) forward frozen_set, edge_area_source, area_orientation and remap like the non-periodic path, or raise with a clear message for what it cannot do; fix the seam-edge asymmetry of the 2D periodic area vectors that lane O counted (62 of 886 directed edges) if it is on the path. Physical check: a droplet under weak shear keeps its volume and its interface for the full short run, and the periodic seam carries no spurious force (sum of forces on a quiescent droplet at round-off).
3. Pins: electrolysis 2D and 3D smoke (mass drift per phase, bubble volume), shearing plate 2D short run (droplet volume, interface vertex count, deformation parameter) and 3D setup plus a few steps.
4. Setups that are not re-entrant (setup_shearing_plate_droplet second call in one process crashes): fix or document at the function.
`,
  },
  {
    id: 'I', phase: 'I static capillary rise',
    brief: `
LANE I: the static capillary-rise scaffolds on presets, and hand-built 3D setups on exact volumes.

Evidence: METHODS.md case matrix rows capillary_rise/capillary_rise_2D.py and _3D.py (hand-rolled, _recompute_duals, static-angle Washburn body force, 'scaffold'); docs_temp/audit_2026-09-25/cases_caprise_dam_bubble_shear.md; laneP log (hydrostatic column on presets, delaunay_material, free-surface flutter and the saddle of the discrete equilibrium, artificial viscosity); laneR log section 3 (free-surface limits of the remap); laneS log section 7 and reviewer note (hand-built 3D complexes that do not come from a builder still read the fan-walk volumes at setup: setup_hydrostatic(dim=3) if still so, cube_flow, cube2droplet, Hagen_Poiseuile/src/_geometry.py). OFF LIMITS: cases_dynamic/capillary_rise_energy_grad/ and the dynCA runners and src/_setup_dynca.py in cases_dynamic/capillary_rise/ (another agent); you may add new files in cases_dynamic/capillary_rise/ and edit capillary_rise_2D.py, capillary_rise_3D.py and the non-dynCA src helpers.

Goal. capillary_rise_2D.py and _3D.py run through presets on library integrators (no hand-rolled loop), reach the static Jurin height for their contact angle to the discretisation error, and are pinned; every 3D setup in cases_dynamic/ that builds a Complex by hand gets exact simplex volumes through one library call.

Tasks.
1. Library: one helper (or an argument of the existing builders) that gives a hand-built Complex its simplex cache (rebuild_simplex_cache_2d/_3d from lane S) and apply it to every such setup; test that cache_dual_volumes tiles the domain for each.
2. Capillary rise static: read the two scaffolds; replace the loop by SolverMethods (connectivity: dual_only or delaunay_material + remap, chosen by measurement as lane P did; frozen walls by membership; the wall and contact-angle condition as a registered BC in the library if the case needs one that does not exist); gravity through body_force; EOS through pressure_model.
3. Validation: equilibrium meniscus height against Jurin's law and the meniscus shape against the Young-Laplace solution for the tube width used, with integrated comparisons; convergence with refinement (two levels at least).
4. Pins: meniscus height and max|u| at the end for 2D (fast) and 3D (slow if needed); presets capillary_rise_static_2D/3D; README and methods.json per the case convention.
5. Report what the static case says about the free-surface items lane P left open (flutter, saddle, artificial viscosity), and what the dynCA work (read-only) would need from the library next.
`,
  },
  {
    id: 'X', phase: 'X broken demos',
    brief: `
LANE X: every runner in cases_dynamic/ either runs or is retired, and the case matrix says which.

Evidence: METHODS.md case matrix (cube2droplet: broken import Cube2droplet; liquid_bridge_approach: broken import; cube_flow stalls after step 0; bc_demo); laneL log pre-existing failures seen in its runner scan (liquid_bridge_equilibrium Case 1 and Case 5 AttributeError 'vd'; liquid_bridge_cfd_dem AttributeError 'u'; bc_demo_v2.py ImportError _rebuild_nn_from_delaunay). Rule 3 applies: run only scripts under cases_dynamic/ (never cases_mean_flow, benchmarks, tutorials, test_cases), from scratch copies or with explicit output directories, never leaving outputs in the case directories except where the case convention keeps them.

Goal. Every runner under cases_dynamic/ (except capillary_rise* and the ones other lanes own) starts, runs its short or smoke mode headlessly, and writes methods.json; or it is retired with one line in the case matrix saying why. No duplicate solver code remains in a demo.

Tasks.
1. Run each runner's shortest mode from a scratch copy with the live hyperct on the path; list every failure with the traceback head.
2. Fix import errors by pointing at the library function that replaced the old one (not by re-adding dead code); convert demos with hand-rolled loops to SolverMethods presets where the library can do it (cube_flow, bc_demo); retire files that duplicate a shipped case or depend on removed features, by stating so in the case matrix and in a short header docstring (do not delete files: deletion is denied here; say which files the user should delete).
3. liquid_bridge_*: the frozen-surface-mesh cases (retopologize_fn=False) and the DEM-coupled case: make them run through connectivity='frozen' presets and record methods.json; the semi-implicit liquid_bridge_approach loop is case physics: run it or document precisely why not.
4. Add a fast smoke test per revived runner (a few steps, headless) so they cannot break silently again.
5. Case matrix of METHODS.md section 4: one row per runner, status from this lane's run, with the preset name.
`,
  },
  {
    id: 'D', phase: 'D oscillating droplet with every new method',
    brief: `
LANE D: try every new method on the oscillating droplet, 2D and 3D, and adopt what wins.

The user asked for this explicitly: 'when you finish all lanes please also try your newest methods/fixes on the oscillating droplet too'. The droplet is the campaign's main benchmark. Evidence: cases_dynamic/oscillating_droplet/ (runners, src, baselines with their methods blocks, diagnose_* drivers of lanes J, M, Q and the box_shift driver of lane B), debugging_plan.md status entries of July (lanes C, D, E, G, H: the exact two-fluid reference, the conservative remap, the 2D over-decay attributed to the every-step projection, the 3D bump / over-decay cancellation, the redistribution lever) and of October (lanes R, S, T, O, B, Q, M and this workflow), METHODS.md presets oscillating_droplet_2D / _2D_dual_only / _2D_bare_delaunay / _2D_projection2 / _3D / _3D_delaunay and the flip rule of the protocol.

Current pins (lane B, full outer mesh): 2D preset oscillating_droplet_2D l2 0.17439096487276182 / tail 0.9998871416222597; projection2 arm l2 0.0388 / tail 1.394; 3D preset oscillating_droplet_3D (dual_only) l2 0.24811443136179492 / tail 0.0841737962816189; 3D delaunay 1.529.

Tasks.
1. Build the arm list from the registry: every explicit axis value that applies to the multiphase droplet and that was added or changed since 2026-09-25, each as preset.replace(...) on the 2D and the 3D preset: frozen_set='membership'; edge_area_source='p_ij_simplex' (3D); area_orientation (default now, legacy as the control); remap='conservative' in 3D now that lanes O, Q and B changed what it reads (the July rejection was measured on the lossy mesh with the fan cache); projection_every in {2, 5} with and without the 3D remap; redistribute_mass=False (the noredist end member); curvature_path='csf_dual' as the measured-worse control; connectivity='delaunay_material' if it applies to a closed box (say so if not); the lane F sliver method if it is a registered axis; the lane W plumbing makes all of them reach the force. Then the best combinations (at most eight), chosen from the single-axis results.
2. Score every arm with the case's own score (l2 against the two-fluid reference, tail growth, linf, mass drift, R_max peak, KE max, the quarter-mean error channels of lane G) and record methods.json next to each score. Run the 3D arms in parallel processes (they are deterministic since lane T); budget the wall time (a full 3D run is about 8 min).
3. Adopt by the protocol's flip rule (better l2 AND tail against the pin), re-pin the baselines with the methods block when a preset changes, and keep the previous preset as a named preset so the old number stays reproducible. Where a candidate improves one channel and worsens another, say so with the numbers and do not flip.
4. Write the result as a short section at the top of the debugging_plan.md status log titled 'Where the oscillating droplet stands (2026-10)': the pinned numbers before and after this campaign (2D 0.17479 in September), the physics gap that remains against the two-fluid reference with its best current attribution, and the one or two next experiments that the evidence supports.
`,
  },
]

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'What was done and what was measured, with exact numbers' },
    lane_log: { type: 'string', description: 'Path of the lane log written' },
    files_changed: { type: 'array', items: { type: 'string' } },
    pins_changed: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, old: { type: 'string' }, new: { type: 'string' }, reason: { type: 'string' } }, required: ['name', 'old', 'new', 'reason'] } },
    tests: { type: 'object', properties: { fast: { type: 'string' }, slow: { type: 'string' }, hyperct: { type: 'string' } }, required: ['fast', 'slow', 'hyperct'] },
    case_status: { type: 'array', items: { type: 'object', properties: { case: { type: 'string' }, status: { type: 'string' }, evidence: { type: 'string' } }, required: ['case', 'status', 'evidence'] } },
    not_done: { type: 'array', items: { type: 'string' }, description: 'Parts of the brief left open, with the reason' },
    followups: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'lane_log', 'files_changed', 'pins_changed', 'tests', 'case_status', 'not_done', 'followups'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fail'] },
    tree_green: { type: 'boolean', description: 'fast, slow and (if touched) hyperct suites all pass and METHODS.md is current' },
    goals_met: { type: 'string', enum: ['full', 'partial', 'none'] },
    tests: { type: 'object', properties: { fast: { type: 'string' }, slow: { type: 'string' }, hyperct: { type: 'string' } }, required: ['fast', 'slow', 'hyperct'] },
    remeasured: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, result: { type: 'string' }, confirmed: { type: 'boolean' } }, required: ['claim', 'result', 'confirmed'] } },
    blocking_issues: { type: 'array', items: { type: 'string' } },
    nonblocking_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdict', 'tree_green', 'goals_met', 'tests', 'remeasured', 'blocking_issues', 'nonblocking_issues'],
}

function implPrompt(lane) {
  return `${COMMON}
${lane.brief}
Finish only when the test suites are green and the documentation of rule 9 is written. Your structured result is read by a reviewer who will re-run the tests and re-measure your headline numbers.`
}

function verifyPrompt(lane, impl, round) {
  return `${COMMON}
YOU ARE THE INDEPENDENT REVIEWER of lane ${lane.id} (review round ${round + 1}). You did not write it. Do not edit any source, test or documentation file; you may write scratch files under ${SCRATCH}/review_${lane.id}/. Your job is to find what is wrong or unproven.

The lane brief was:
${lane.brief}
The implementer reported:
${JSON.stringify(impl, null, 1)}

Steps.
1. Read the complete lane diff: bash ${SCRATCH}/lane_diff.sh (add --stat for the file list); re-pinned baselines under cases_dynamic/oscillating_droplet/baselines are tracked in git and appear in it. Changes under cases_dynamic/capillary_rise_energy_grad/ or the dynCA files may come from another agent working concurrently; mention them but do not count them against the lane unless the lane log claims them.
2. Re-run the fast suite, the slow battery, and the hyperct suite if hyperct changed. Quote the summary lines.
3. Re-measure the headline claims from the lane log by running its reproduce commands (the main claim always, more where cheap). Compare numbers digit by digit where a pin is involved. For every re-pinned number check that old and new values are on record and that the baselines carry the methods block.
4. Check the rules: no pinned number changed without evidence; no loosened tolerance or deleted assertion; new switches registered in _axes.py with a SolverMethods field; no duplicate case-local code where the library could do it; METHODS.md current (the drift test in ddgclib/tests/test_methods.py passes); lane log, debugging_plan.md entry and DEVELOPMENT.md updated; no em dashes in new prose; off-limits directories neither edited nor executed; no stray probe outputs left in the repositories (compare git status --short of both repos with the lane's file list).
5. Read the changed code for correctness bugs: wrong conditions, stale caches, silently ignored arguments, behaviour that differs between 2D and 3D without being stated, tests that cannot fail (check new tests against the pre-lane sources where that is cheap: export the HEAD version of the changed modules with git show HEAD:<path> into a scratch package directory and put it first on PYTHONPATH).
6. Judge whether the lane goal is met in full, in part, or not at all, on the evidence you measured yourself.

verdict = pass only when the tree is green and there is no blocking issue. A lane that honestly documents a part it could not finish can still pass with goals_met = partial; list that part under nonblocking_issues. Blocking issues are: red tests, a pin moved without evidence, a claim you could not reproduce, a correctness bug, a rule violation that can still be repaired, missing documentation.`
}

function fixPrompt(lane, impl, verdict, round) {
  return `${COMMON}
YOU ARE FIXING lane ${lane.id} after independent review (fix round ${round}).

The lane brief was:
${lane.brief}
The implementer reported:
${JSON.stringify(impl, null, 1)}

The reviewer's findings:
${JSON.stringify(verdict, null, 1)}

Resolve every blocking issue at its cause (not by weakening a test), address the non-blocking ones where cheap, bring the suites to green, and update the lane log, debugging_plan.md entry and METHODS.md so they describe the final state. If you disagree with a finding, show the measurement that refutes it. Return the full updated report for the lane (not only the delta).`
}

const COMMIT_SCHEMA = {
  type: 'object',
  properties: {
    ddgclib_sha: { type: 'string' },
    hyperct_sha: { type: 'string', description: 'empty string when the lane did not change hyperct' },
    left_unstaged: { type: 'array', items: { type: 'string' }, description: 'lane files deliberately not committed, with the reason' },
  },
  required: ['ddgclib_sha', 'hyperct_sha', 'left_unstaged'],
}

function commitPrompt(lane, impl) {
  return `YOU ARE THE COMMIT STEP for lane ${lane.id} of the ddgclib campaign. The lane passed independent review with green tests. The user's standing rule is to ship verified work, on master, in both repositories (/home/endres/projects/ddgclib and /home/endres/projects/hyperct). Do not edit any file. Do not run tests.

The implementer's file list and summary:
${JSON.stringify({ files_changed: impl.files_changed, lane_log: impl.lane_log, pins_changed: impl.pins_changed, summary: impl.summary }, null, 1)}

Steps.
1. bash ${SCRATCH}/lane_diff.sh --stat shows the lane's files in both repos. Cross-check with the list above.
2. First run git diff --cached --stat in both repos: the index must be empty. If it is not (a stale staged snapshot from an interrupted step), do NOT commit it: run git reset -q (index only, the working tree is untouched) and check that git diff --cached is now empty. Then stage with explicit paths only: git add <path> <path> ... (this stages the working-tree content). Never use git add -A, -u, . or a directory that contains anything outside the lane. After staging, git diff --cached --stat must list exactly the lane's files and no deletion you did not intend. Baselines under cases_dynamic/oscillating_droplet/baselines are tracked: include them when they changed.
   NEVER stage: .gitignore; anything under cases_mean_flow/, benchmarks/, tutorials/, test_cases/, cases_dynamic/capillary_rise/, cases_dynamic/capillary_rise_energy_grad/; ddgclib/tests/test_integrated_validation.py; generated outputs (results/, fig/, *.log, *.pkl, *.mp4, pytest-of-*); anything untracked that is not a source, test or documentation file of this lane. CLAUDE.md and DEVELOPMENT.md are git-ignored: leave them.
3. Commit hyperct first (if it changed), then ddgclib, both on master (check git branch --show-current; if it is not master, stop and report). Message: first line with the repo's prefix convention ENH: / BUG: / MAINT:, a plain one-line summary, ending with (lane ${lane.id}); a body of 5 to 15 lines saying what changed, the headline measured numbers, and every pin that moved as old -> new; no em dashes; then a blank line and exactly this last line:
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
4. No push, no amend, no rebase, no branch, no tag, no checkout/restore/stash, no other git command that changes history or the working tree (git reset -q without paths or arguments, to clear a stale index, is the one allowed exception).
5. Run git status --short in both repos and report the two commit hashes and any lane file you left unstaged with the reason.`
}

const results = []
for (const lane of LANES) {
  phase(lane.phase)
  let impl = await agent(implPrompt(lane), { label: `${lane.id}:implement`, phase: lane.phase, schema: IMPL_SCHEMA })
  if (!impl) {
    log(`lane ${lane.id}: implement agent returned nothing; stopping the chain`)
    results.push({ lane: lane.id, status: 'implement agent returned nothing' })
    break
  }
  let verdict = null
  const MAX_FIX = 2
  for (let round = 0; round <= MAX_FIX; round++) {
    verdict = await agent(verifyPrompt(lane, impl, round), { label: `${lane.id}:verify${round + 1}`, phase: lane.phase, schema: VERIFY_SCHEMA })
    if (!verdict || verdict.verdict === 'pass' || round === MAX_FIX) break
    log(`lane ${lane.id}: review round ${round + 1} found ${verdict.blocking_issues.length} blocking issue(s); fixing`)
    const fixed = await agent(fixPrompt(lane, impl, verdict, round + 1), { label: `${lane.id}:fix${round + 1}`, phase: lane.phase, schema: IMPL_SCHEMA })
    if (fixed) impl = fixed
  }
  const status = !verdict ? 'review agent returned nothing' : verdict.verdict
  results.push({ lane: lane.id, status, goals_met: verdict ? verdict.goals_met : 'unknown', impl, verdict })
  log(`lane ${lane.id}: ${status}${verdict ? ' (goals ' + verdict.goals_met + ', tree green ' + verdict.tree_green + ')' : ''}`)
  if (!verdict || !verdict.tree_green) {
    log(`lane ${lane.id} left the tree not green or unreviewed; stopping the chain so the next lane does not build on it`)
    break
  }
  if (verdict.verdict !== 'pass') {
    log(`lane ${lane.id} is green but still has blocking review findings after the fix rounds; not committed, stopping the chain`)
    break
  }
  const commit = await agent(commitPrompt(lane, impl), { label: `${lane.id}:commit`, phase: lane.phase, schema: COMMIT_SCHEMA, effort: 'low' })
  results[results.length - 1].commit = commit
  log(`lane ${lane.id}: committed ${commit ? commit.ddgclib_sha + ' ' + commit.hyperct_sha : '(commit agent returned nothing)'}`)
  if (!commit) break
}
return results
