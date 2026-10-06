export const meta = {
  name: 'ddgclib-lanes-4',
  description: 'ddgclib campaign lanes F-review, G, I, X, V, D in sequence: lane F independent review, electrolysis 3D and shearing plate, static capillary rise, broken demos, impenetrability and interface vote, final oscillating-droplet evaluation; each lane reviewed adversarially against analytical solutions',
  phases: [
    { title: 'F-review dam break ledger', detail: 'independent review of the committed lane F, fix at the cause, commit' },
    { title: 'G electrolysis 3D and shearing plate', detail: 'gas phase lost in 3D; shearing-plate rescale deletes poles; periodic path axes' },
    { title: 'I static capillary rise', detail: 'capillary_rise 2D/3D scaffolds on presets, Jurin height; hand-built 3D setups on exact volumes' },
    { title: 'X broken demos', detail: 'every runner under cases_dynamic/ runs or is retired, case matrix honest' },
    { title: 'V impenetrability and interface vote', detail: 'library wall clamp, interface-aware vote, per-phase sliver merge' },
    { title: 'D oscillating droplet with every new method', detail: 'A/B of every new axis on the 2D and 3D droplet, adoption by the flip rule, re-pin' },
  ],
}

const ROOT = '/home/user/ddgclib'
const HYP = '/home/user/hyperct'
const SCRATCH = '/tmp/claude-0/-home-user-ddgclib/091c4829-271f-5fa0-a0eb-eaae3d5b54e0/scratchpad'
const PY = 'PYTHONPATH=/home/user/ddgclib /usr/bin/python'
const BRANCH = 'claude/relaxed-davinci-tkh95t'

const COMMON = `
You are continuing the ddgclib dynamics debugging campaign in a cloud session. ddgclib is a Lagrangian discrete-differential-geometry fluid solver (repo root ${ROOT}). Its mesh backend hyperct lives at ${HYP} and is symlinked into the repo as ./hyperct. The owner's standing instructions: get every relevant dynamic test case working with the oscillating droplet as the main benchmark; always ship a fix into the library as long as the method is separated behind the method wrappers (ddgclib.methods: SolverMethods, AXES, PRESETS) and no code is duplicated; never touch the owner's unrelated work. The campaign's adversarial metric is accuracy against analytical solutions (ddgclib.analytical and the case references): every claim of improvement is a measured number against an analytical or reference solution, before and after, through presets.

READ FIRST (skim what is not relevant to your lane):
- debugging_plan.md: the status entries from 2026-10-01 onward (lanes R, S, P, L, H, T, O, B, Q, M, W, F and anything newer) and the section "Reproducibility protocol for method lanes" (binding rules, including rule 8 on process dependence).
- METHODS.md: every method axis with status and evidence, the presets, and the case matrix (section 4).
- docs_temp/11_dynamics_audit_2026-09-25.md and the reports in docs_temp/audit_2026-09-25/.
- The lane logs in docs_temp/debug_session/ that your brief names.
- docs_temp/NEXT_SESSION_PROMPT.md (the hand-off: state, baselines, open items).

RULES
1. Python: run every command from the repo root ${ROOT} as \`${PY} ...\` (there is no editable install; PYTHONPATH makes ddgclib and the ./hyperct symlink importable). For the hyperct suite: cd ${HYP} && /usr/bin/python -m pytest hyperct/tests -k "not benchmark" -q -p no:cacheprovider. The machine has 4 cores and 15 GB RAM: run at most 3 solver processes at once and never a pytest -n fan-out.
2. Git: never commit, stash, checkout, reset, restore or otherwise use git to change either working tree or history (the only exception is the dedicated commit step that runs after a lane passes review). Read-only git (log, diff, status, show) is fine. ddgclib is on branch ${BRANCH} (the cloud session's designated branch; it started from master at commit b5bd379 and every earlier lane is committed), hyperct is on master at 48d163a, so HEAD is the pre-lane state in both: the lane's changes are exactly what \`bash docs_temp/workflows/lane_diff.sh\` prints (add --stat for the file list).
3. Off limits, never edit AND never execute anything from them, not even from a scratch copy (several scripts chdir to an absolute repo path and overwrite the owner's manuscript files; this happened once): cases_mean_flow/, benchmarks/, tutorials/, test_cases/, manuscript directories. Also off limits for edits: cases_dynamic/capillary_rise_energy_grad/ (another agent owns it); in cases_dynamic/capillary_rise/ do not change the behaviour of the dynCA runners or src/_setup_dynca.py. If a library change of yours can move the numbers of those runners, say so in your lane log.
4. Library over case code: a method a case needs goes into ddgclib (or hyperct) behind a registered axis in ddgclib/methods/_axes.py with a SolverMethods field or builder in ddgclib/methods/_config.py. Cases consume a preset from ddgclib/methods/_presets.py and write methods.json with record_methods. No duplicate hand-rolled loops or closures in case files. Keep changes minimal and surgical, match the existing style, add nothing speculative.
5. Pins: every pinned number in ddgclib/tests must stay bit-identical unless your lane deliberately changes the method behind it. If it does, A/B through presets (preset versus preset.replace(...)), re-pin together with the methods block (baselines under cases_dynamic/oscillating_droplet/baselines carry one), and record old and new values in the lane log. Never loosen a tolerance or delete an assertion to make a test pass. When a lane replaces a default because the old behaviour is a defect, keep the old behaviour reachable as a registered option with status 'broken' when that costs a flag, so earlier results stay reproducible; say so if it is not feasible.
6. Scratch and outputs: do not create files you would need to delete. Probe runs write only under ${SCRATCH}/lane<ID>/ (run case runners from scratch copies or pass an explicit output directory). Anything the lane log needs for reproduction (detector and diagnose scripts) lives in the repository, not in scratch. Only results that are meant to be kept go into the case directory (fig/, results/). A diagnose script worth keeping goes next to its case as diagnose_*.py. Never leave stray outputs in either repository: compare git status --short before you finish.
7. Tests. Fast suite: ${PY} -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider (baseline at the start of this workflow: 1240 passed, 12 skipped, 2 xfailed). Slow pinned battery: ${PY} -m pytest ddgclib/tests -q -m slow -p no:cacheprovider (baseline: 34 passed, 1 xfailed). hyperct, when you touch it: see rule 1 (baseline 340 passed, 38 skipped, 6 xfailed; the bare pytest -q at the hyperct root fails at collection for unrelated reasons). All must be green when you finish; earlier lanes may have raised the counts. Use a generous timeout (the fast suite takes a few minutes, the slow battery longer).
8. Use validation by integrated comparisons (ddgclib.analytical), never point-wise pressure comparisons. Stay in the Lagrangian formalism.
9. Documentation when done: a lane log docs_temp/debug_session/lane<ID>-<slug>.md (what changed and where, measurements with exact numbers and the SolverMethods configuration behind each, measured DO-NOTs, known limits, how to reproduce); a new entry at the TOP of the status log in debugging_plan.md; evidence and status in ddgclib/methods/_axes.py; then ${PY} -m ddgclib.methods --update METHODS.md and edit the case matrix rows in section 4 of METHODS.md by hand. CLAUDE.md and DEVELOPMENT.md are not in this clone (git-ignored); skip their checklists. Never use an em dash in prose.
10. If part of the brief turns out wrong or infeasible, do everything that is achievable and state exactly what is left and why. Report honestly: failing tests with their output, unverified claims marked as unverified. State numbers from process-dependent runs as ranges over several fresh processes.
11. This is a cloud session: if a run needs more than about 45 minutes of wall time on 4 cores, or memory beyond 12 GB, cut it to a smaller mode, say so in the lane log, and list the full run as a reproduce command for the owner's hardware.
`

const LANES = [
  {
    id: 'F-review', phase: 'F-review dam break ledger', reviewOnly: true,
    brief: `
LANE F-review: independent review of the committed lane F (dam-break phase ledger and face closure), whose reviewer was cut off by the end of the last session.

Lane F is committed on master as e82c410 (BUG: Multiphase phase ledger and face closure at presence changes; dam break runs its horizons (lane F)). Read \`git show --stat e82c410\` and \`git show e82c410\` for its full diff (there is no uncommitted lane diff: lane_diff.sh prints nothing until a fix is made), and docs_temp/debug_session/laneF-ledger-and-face-closure.md for its claims.

Headline claims to re-measure: alpha_art 0.2 and 0.1 complete the 1585-step 2D horizon with no vertex outside; the 3D preset completes 793 steps on edge_area_source='p_ij_simplex'; the shipped alpha 0.3 run is bit-identical, digest 952d4544676ca366; the four new pins in ddgclib/tests/test_case_dam_break.py::TestDamBreakPins; the two new axes phase_ledger and face_closure with their defaults flipped and the old values kept as 'broken'; split_method='simplex' is opt-in. Accuracy: compare the surviving 2D run against the case's reference (KE_liq peak and time, front position law) and state the numbers as the lane log does; check that nothing in the lane loosened a tolerance or deleted an assertion.

If something is wrong, the fix step repairs it at the cause and the commit step commits it as lane F-review. If everything holds, the lane produces no commit; say so.
`,
  },
  {
    id: 'G', phase: 'G electrolysis 3D and shearing plate',
    brief: `
LANE G: the 3D electrolysis bubble keeps its gas phase; the shearing-plate droplet survives its setup and its shear.

Evidence: METHODS.md case matrix rows for electrolysis_bubble and shearing_plate_droplet; docs_temp/audit_2026-09-25/cases_caprise_dam_bubble_shear.md; laneB log (shearing plate: the anisotropic rescale (1, 2/3) maps outer vertices at (0, +-0.0075) onto the droplet poles and the case's on_collision='evict' loop deletes both interface poles; the 3D setup crashes on the same collision); laneL log (the periodic path implements neither frozen_set='membership' nor the lane Q axis; SolverMethods raises); laneF-ledger-and-face-closure log (electrolysis_bubble_3D carried 26 stranded (vertex, phase) pairs from its setup, released at the first rebuild under the new phase_ledger default; not re-measured); lane P (delaunay_material).

Goal. Both cases run through presets on library paths, with every new axis available on the periodic path, and each has a pinned smoke test with a physical check against the analytical solution.

Tasks.
1. Electrolysis 3D: reproduce the loss of the gas phase through the preset on the current library; find whether it is the gas injection in the callback, the per-phase redistribution of a one-vertex phase, the wall clamp, or the ledger (lane F's method may already apply). Fix in the library behind the registry. Physical check: a static bubble of the injected volume holds its Laplace pressure (2 sigma / R in 3D, sigma / R in 2D) to the discretisation error, and the injected gas mass is conserved to round-off; report the measured error against the analytical value at two refinement levels.
2. Shearing plate: move the rescale into the builder with hyperct's HC.V.move_all (scale only the vertices outside the droplet ring, or push a rescaled vertex that lands inside R0 + h radially out), so no interface vertex is deleted; make retopologize_multiphase_periodic (ddgclib/methods/_retopo.py) and ddgclib/geometry/periodic.py forward frozen_set, edge_area_source, area_orientation, phase_ledger, face_closure and remap like the non-periodic path, or raise with a clear message for what they cannot do; fix the seam-edge asymmetry of the 2D periodic area vectors that lane O counted (62 of 886 directed edges) if it is on the path. Physical check: a droplet under weak shear keeps its volume and its interface for the full short run, and the periodic seam carries no spurious force (sum of forces on a quiescent droplet at round-off); the quiescent droplet's Laplace pressure against the analytical value.
3. Pins: electrolysis 2D and 3D smoke (mass drift per phase, bubble volume), shearing plate 2D short run (droplet volume, interface vertex count, deformation parameter) and 3D setup plus a few steps.
4. Make setup_shearing_plate_droplet re-entrant (a second call in one process crashes) or document at the function why not.
`,
  },
  {
    id: 'I', phase: 'I static capillary rise',
    brief: `
LANE I: the static capillary-rise scaffolds on presets, and hand-built 3D setups on exact volumes.

Evidence: METHODS.md case matrix rows capillary_rise/capillary_rise_2D.py and _3D.py (hand-rolled, _recompute_duals, static-angle Washburn body force, 'scaffold'); docs_temp/audit_2026-09-25/cases_caprise_dam_bubble_shear.md; laneP log (hydrostatic column on presets, delaunay_material, free-surface flutter and the saddle of the discrete equilibrium, artificial viscosity); laneR log section 3 (free-surface limits of the remap); laneS log section 7 and reviewer note (hand-built 3D complexes that do not come from a builder still read the fan-walk volumes at setup: cube_flow, cube2droplet, Hagen_Poiseuile/src/_geometry.py). OFF LIMITS: cases_dynamic/capillary_rise_energy_grad/ and the dynCA runners and src/_setup_dynca.py in cases_dynamic/capillary_rise/ (another agent); you may add new files in cases_dynamic/capillary_rise/ and edit capillary_rise_2D.py, capillary_rise_3D.py and the non-dynCA src helpers.

Goal. capillary_rise_2D.py and _3D.py run through presets on library integrators (no hand-rolled loop), reach the static Jurin height for their contact angle to the discretisation error, and are pinned; every 3D setup in cases_dynamic/ that builds a Complex by hand gets exact simplex volumes through one library call.

Tasks.
1. Library: one helper (or an argument of the existing builders) that gives a hand-built Complex its simplex cache (rebuild_simplex_cache_2d/_3d from lane S) and apply it to every such setup; test that cache_dual_volumes tiles the domain for each.
2. Capillary rise static: read the two scaffolds; replace the loop by SolverMethods (connectivity: dual_only or delaunay_material + remap='conservative', chosen by measurement as lane P did; walls frozen by membership; the wall and contact-angle condition as a registered BC in the library if the case needs one that does not exist); gravity through body_force; EOS through pressure_model.
3. Validation (the adversarial metric): equilibrium meniscus height against Jurin's law and the meniscus shape against the Young-Laplace solution for the tube width used, with integrated comparisons (ddgclib.analytical), at two refinement levels at least; report the error and its convergence order.
4. Pins: meniscus height and max|u| at the end for 2D (fast) and 3D (slow if needed); presets capillary_rise_static_2D/3D; README and methods.json per the case convention.
5. Report what the static case says about the free-surface items lane P left open (flutter, saddle, artificial viscosity), and what the dynCA work (read-only) would need from the library next.
`,
  },
  {
    id: 'X', phase: 'X broken demos',
    brief: `
LANE X: every runner in cases_dynamic/ either runs or is retired, and the case matrix says which.

Evidence: METHODS.md case matrix (cube2droplet: broken import Cube2droplet; liquid_bridge_approach: broken import; cube_flow stalls after step 0; bc_demo_v2: ImportError _rebuild_nn_from_delaunay; liquid_bridge_equilibrium Case 1 and 5: AttributeError 'vd'; liquid_bridge_cfd_dem: AttributeError 'u'); laneW log (not converted: cube2droplet/*, liquid_bridge_cfd_dem/*, oscillating_droplet_p_ref/scripts/*, the stale oscillating_droplet/diagnose_split_methods.py). Rule 3 applies: run only scripts under cases_dynamic/ (never cases_mean_flow, benchmarks, tutorials, test_cases), from scratch copies or with explicit output directories, never leaving outputs in the case directories except where the case convention keeps them.

Goal. Every runner under cases_dynamic/ (except capillary_rise* and the ones other lanes own) starts, runs its short or smoke mode headlessly, and writes methods.json; or it is retired with one line in the case matrix saying why. No duplicate solver code remains in a demo.

Tasks.
1. Run each runner's shortest mode from a scratch copy with the live hyperct on the path; list every failure with the traceback head.
2. Fix import errors by pointing at the library function that replaced the old one (not by re-adding dead code); convert demos with hand-rolled loops to SolverMethods presets where the library can do it (cube_flow, bc_demo); retire files that duplicate a shipped case or depend on removed features, by stating so in the case matrix and in a short header docstring (do not delete files; list which files the owner should delete).
3. liquid_bridge_*: the frozen-surface-mesh cases (retopologize_fn=False) and the DEM-coupled case: make them run through connectivity='frozen' presets and record methods.json; the semi-implicit liquid_bridge_approach loop is case physics: run it or document precisely why not. Where a revived runner has an analytical reference (a static bridge's Laplace pressure, a Poiseuille profile), measure it and state the error.
4. Add a fast smoke test per revived runner (a few steps, headless) so they cannot break silently again.
5. Case matrix of METHODS.md section 4: one honest row per runner, status from this lane's run, with the preset name.
`,
  },
  {
    id: 'V', phase: 'V impenetrability and interface vote',
    brief: `
LANE V: impenetrability and the interface vote, the two blockers every free-surface case shares.

Evidence: laneL log known limit 1 (a fluid vertex can pass between two wall vertices; HP2D had one 5.1e-3 outside the top wall at t = 30; dam-break refinement 4 and its split_method='simplex' arm end with a vertex through the floor; the case-local WallClampBC of the electrolysis case and the dynCA clamp are the two existing clamps); laneF-ledger-and-face-closure follow-ups (an interface-aware simplex vote so a one-cell liquid tongue survives the collapse, since a tie now goes to the lower phase ID and erases a lone liquid vertex; a mass-conserving per-phase sliver merge or a registered mass floor for the refinement 4 small-cell spike; the 3D dam break on wall half cells); the lane G log of this workflow (what it did to the electrolysis clamp).

Goal. A library wall clamp (planar and, if cheap, general wall: put the vertex back on the wall and zero its normal velocity) behind a registered BC and a method axis for the policy, replacing the two case-local clamps (the dynCA clamp stays untouched: only say what it would take); an interface-aware vote as a registered split_method or vote option; the sliver merge extended to m_phase behind an axis.

Tasks.
1. Reproduce each blocker through its preset on the current library: HP2D at t = 30 (vertex outside the top wall), dam break refinement 4 and the split_method='simplex' arm (vertex through the floor), the lone liquid tongue erased by the tie rule.
2. Implement the three methods in the library behind the registry, each measured one at a time through preset.replace(...) on the dam break (alpha_art 0.1 and refinement 4 through the horizon), on HP2D (the Poiseuille profile error against the analytical solution must not grow) and on the electrolysis case (Laplace pressure error against the analytical value, gas mass drift).
3. Adopt by the flip rule of the protocol; re-pin what moves with old and new on record; keep the old behaviour reachable as a registered option.
4. Pins: one fast regression per adopted method.
`,
  },
  {
    id: 'D', phase: 'D oscillating droplet with every new method',
    brief: `
LANE D: try every new method on the oscillating droplet, 2D and 3D, and adopt what wins.

The owner asked for this explicitly: 'when you finish all lanes please also try your newest methods/fixes on the oscillating droplet too'. The droplet is the campaign's main benchmark. Evidence: cases_dynamic/oscillating_droplet/ (runners, src, baselines with their methods blocks, diagnose_* drivers of lanes J, M, Q and the box_shift driver of lane B), debugging_plan.md status entries of July (lanes C, D, E, G, H: the exact two-fluid reference, the conservative remap, the 2D over-decay attributed to the every-step projection, the 3D bump / over-decay cancellation, the redistribution lever) and of October (lanes R, S, T, O, B, Q, M, W, F and this workflow), METHODS.md presets oscillating_droplet_2D / _2D_dual_only / _2D_bare_delaunay / _2D_projection2 / _3D / _3D_delaunay and the flip rule of the protocol.

Current pins (lane B, full outer mesh): 2D preset oscillating_droplet_2D l2 0.17439096487276182 / tail 0.9998871416222597; projection2 arm l2 0.0388 / tail 1.394; 3D preset oscillating_droplet_3D (dual_only) l2 0.24811443136179492 / tail 0.0841737962816189; 3D delaunay 1.529.

Tasks.
1. Build the arm list from the registry: every explicit axis value that applies to the multiphase droplet and that was added or changed since 2026-09-25, each as preset.replace(...) on the 2D and the 3D preset: frozen_set='membership'; edge_area_source='p_ij_simplex' (3D); area_orientation (default now, legacy as the control); remap='conservative' in 3D now that lanes O, Q and B changed what it reads (the July rejection was measured on the lossy mesh with the fan cache); projection_every in {2, 5} with and without the 3D remap; redistribute_mass=False; phase_ledger and face_closure arms (the droplet runs at P0 = 0 where lane F measured them inert: confirm); split_method='simplex'; curvature_path='csf_dual' as the measured-worse control; connectivity='delaunay_material' if it applies to a closed box (say so if not); lane V's clamp and vote if registered. Then the best combinations (at most eight), chosen from the single-axis results.
2. Score every arm with the case's own score (l2 against the two-fluid reference, tail growth, linf, mass drift, R_max peak, KE max, the quarter-mean error channels of lane G of July) and record methods.json next to each score. Run the 3D arms in parallel processes, at most 3 at once on this 4-core machine (deterministic since lane T; a full 3D run is about 8 min on the owner's machine, measure one first and budget from it; if the full arm list does not fit in about 2 hours of wall time, run the 3D arms in the shortest mode that still scores and list the full runs as reproduce commands).
3. Adopt by the protocol's flip rule (better l2 AND tail against the pin), re-pin the baselines with the methods block when a preset changes, and keep the previous preset as a named preset so the old number stays reproducible. Where a candidate improves one channel and worsens another, say so with the numbers and do not flip.
4. Write the result as a short section at the top of the debugging_plan.md status log titled 'Where the oscillating droplet stands (2026-10)': the pinned numbers before and after this campaign (2D 0.17479 in September), the physics gap that remains against the two-fluid reference with its best current attribution (lane Q showed the 3D pin is a cancellation between an outward bump and an over-decay; lane H of July attributed the 2D over-decay to the every-step projection), and the one or two next experiments the evidence supports.
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
    accuracy: { type: 'array', items: { type: 'object', properties: { case: { type: 'string' }, analytical: { type: 'string' }, before: { type: 'string' }, after: { type: 'string' }, methods: { type: 'string' } }, required: ['case', 'analytical', 'before', 'after', 'methods'] }, description: 'Error against the analytical or reference solution before and after the lane, with the SolverMethods configuration' },
    tests: { type: 'object', properties: { fast: { type: 'string' }, slow: { type: 'string' }, hyperct: { type: 'string' } }, required: ['fast', 'slow', 'hyperct'] },
    case_status: { type: 'array', items: { type: 'object', properties: { case: { type: 'string' }, status: { type: 'string' }, evidence: { type: 'string' } }, required: ['case', 'status', 'evidence'] } },
    not_done: { type: 'array', items: { type: 'string' }, description: 'Parts of the brief left open, with the reason' },
    followups: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'lane_log', 'files_changed', 'pins_changed', 'accuracy', 'tests', 'case_status', 'not_done', 'followups'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['pass', 'fail'] },
    tree_green: { type: 'boolean', description: 'fast, slow and (if touched) hyperct suites all pass and METHODS.md is current' },
    goals_met: { type: 'string', enum: ['full', 'partial', 'none'] },
    tests: { type: 'object', properties: { fast: { type: 'string' }, slow: { type: 'string' }, hyperct: { type: 'string' } }, required: ['fast', 'slow', 'hyperct'] },
    remeasured: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, result: { type: 'string' }, confirmed: { type: 'boolean' } }, required: ['claim', 'result', 'confirmed'] } },
    accuracy_verdict: { type: 'string', description: 'For each analytical comparison the lane claims: the number you measured yourself, whether it matches, and whether any accuracy got worse' },
    blocking_issues: { type: 'array', items: { type: 'string' } },
    nonblocking_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdict', 'tree_green', 'goals_met', 'tests', 'remeasured', 'accuracy_verdict', 'blocking_issues', 'nonblocking_issues'],
}

function implPrompt(lane) {
  return `${COMMON}
${lane.brief}
Finish only when the test suites are green and the documentation of rule 9 is written. Your structured result is read by an adversarial reviewer who will re-run the tests and re-measure your headline numbers and every analytical comparison.`
}

function verifyPrompt(lane, impl, round) {
  return `${COMMON}
YOU ARE THE INDEPENDENT, ADVERSARIAL REVIEWER of lane ${lane.id} (review round ${round + 1}). You did not write it. Do not edit any source, test or documentation file; you may write scratch files under ${SCRATCH}/review_${lane.id}/. Your job is to find what is wrong or unproven. Default to doubt: a number you have not measured yourself is unverified.

The lane brief was:
${lane.brief}
The implementer reported:
${JSON.stringify(impl, null, 1)}

Steps.
1. Read the complete lane diff: bash docs_temp/workflows/lane_diff.sh (add --stat for the file list); re-pinned baselines under cases_dynamic/oscillating_droplet/baselines are tracked in git and appear in it. ${lane.reviewOnly ? 'For this review-only lane the diff to read is the committed one named in the brief (git show).' : ''}
2. Re-run the fast suite, the slow battery, and the hyperct suite if hyperct changed. Quote the summary lines.
3. The adversarial metric: re-measure every accuracy claim against the analytical or reference solution yourself (run the lane's reproduce commands, through the presets, with the methods block the lane names) and compare before (HEAD preset, or the preset the lane kept as the old behaviour) and after. Compare numbers digit by digit where a pin is involved. For every re-pinned number check that old and new values are on record and that the baselines carry the methods block. If any accuracy against an analytical solution got worse without the lane saying so, that is blocking.
4. Check the rules: no pinned number changed without evidence; no loosened tolerance or deleted assertion; new switches registered in _axes.py with a SolverMethods field; no duplicate case-local code where the library could do it; METHODS.md current (the drift test in ddgclib/tests/test_methods.py passes); lane log and debugging_plan.md entry written; no em dashes in new prose; off-limits directories neither edited nor executed; no stray probe outputs left in the repositories (compare git status --short of both repos with the lane's file list).
5. Read the changed code for correctness bugs: wrong conditions, stale caches, silently ignored arguments, behaviour that differs between 2D and 3D without being stated, tests that cannot fail (check new tests against the pre-lane sources where that is cheap: export the HEAD version of the changed modules with git show HEAD:<path> into a scratch package directory and put it first on PYTHONPATH).
6. Judge whether the lane goal is met in full, in part, or not at all, on the evidence you measured yourself.

verdict = pass only when the tree is green and there is no blocking issue. A lane that honestly documents a part it could not finish can still pass with goals_met = partial; list that part under nonblocking_issues. Blocking issues are: red tests, a pin moved without evidence, a claim you could not reproduce, an accuracy regression the lane did not declare, a correctness bug, a rule violation that can still be repaired, missing documentation.`
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
    ddgclib_sha: { type: 'string', description: 'empty string when nothing was committed' },
    hyperct_sha: { type: 'string', description: 'empty string when the lane did not change hyperct' },
    left_unstaged: { type: 'array', items: { type: 'string' }, description: 'lane files deliberately not committed, with the reason' },
  },
  required: ['ddgclib_sha', 'hyperct_sha', 'left_unstaged'],
}

function commitPrompt(lane, impl) {
  return `YOU ARE THE COMMIT STEP for lane ${lane.id} of the ddgclib campaign. The lane passed independent review with green tests. The owner's standing rule is to ship verified work in both repositories (${ROOT} on branch ${BRANCH}, the cloud session's designated branch, and ${HYP} on master). Do not edit any file. Do not run tests.

The implementer's file list and summary:
${JSON.stringify({ files_changed: impl.files_changed, lane_log: impl.lane_log, pins_changed: impl.pins_changed, summary: impl.summary }, null, 1)}

Steps.
1. cd ${ROOT} && bash docs_temp/workflows/lane_diff.sh --stat shows the lane's files in both repos. Cross-check with the list above. If there is nothing to commit (a review-only lane that found nothing wrong), report empty shas and stop.
2. First run git diff --cached --stat in both repos: the index must be empty. If it is not (a stale staged snapshot from an interrupted step), do NOT commit it: run git reset -q (index only, the working tree is untouched) and check that git diff --cached is now empty. Then stage with explicit paths only: git add <path> <path> ... (this stages the working-tree content). Never use git add -A, -u, . or a directory that contains anything outside the lane. After staging, git diff --cached --stat must list exactly the lane's files and no deletion you did not intend. Baselines under cases_dynamic/oscillating_droplet/baselines are tracked: include them when they changed. The workflow script docs_temp/workflows/ddgclib-lanes-4.js is committed by the session owner, not by you.
   NEVER stage: .gitignore; anything under cases_mean_flow/, benchmarks/, tutorials/, test_cases/, cases_dynamic/capillary_rise_energy_grad/; the dynCA runners and src/_setup_dynca.py under cases_dynamic/capillary_rise/ (lane I's own new files and its edits of capillary_rise_2D.py, capillary_rise_3D.py and the non-dynCA src helpers ARE staged); ddgclib/tests/test_integrated_validation.py; generated outputs (results/, fig/, *.log, *.pkl, *.mp4, *.png, pytest-of-*); anything untracked that is not a source, test or documentation file of this lane.
3. Commit hyperct first (if it changed; branch must be master), then ddgclib (branch must be ${BRANCH}; check git branch --show-current; if it is anything else, stop and report). Message: first line with the repo's prefix convention ENH: / BUG: / MAINT:, a plain one-line summary, ending with (lane ${lane.id}); a body of 5 to 15 lines saying what changed, the headline measured numbers, and every pin that moved as old -> new; no em dashes; then a blank line and exactly these two last lines:
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014BjQpYDFMoVyaopnwHSxBq
4. No push, no amend, no rebase, no branch, no tag, no checkout/restore/stash, no other git command that changes history or the working tree (git reset -q without paths or arguments, to clear a stale index, is the one allowed exception).
5. Run git status --short in both repos and report the two commit hashes and any lane file you left unstaged with the reason.`
}

const results = []
for (const lane of LANES) {
  phase(lane.phase)
  let impl
  if (lane.reviewOnly) {
    impl = { summary: 'Committed lane F (e82c410), unreviewed; see docs_temp/debug_session/laneF-ledger-and-face-closure.md', lane_log: 'docs_temp/debug_session/laneF-ledger-and-face-closure.md', files_changed: ['see git show --stat e82c410'], pins_changed: [], accuracy: [], tests: { fast: 'claimed 1240 passed, 12 skipped, 2 xfailed', slow: 'claimed 34 passed, 1 xfailed', hyperct: 'unchanged' }, case_status: [], not_done: [], followups: [] }
  } else {
    impl = await agent(implPrompt(lane), { label: `${lane.id}:implement`, phase: lane.phase, schema: IMPL_SCHEMA })
  }
  if (!impl) {
    log(`lane ${lane.id}: implement agent returned nothing; stopping the chain`)
    results.push({ lane: lane.id, status: 'implement agent returned nothing' })
    break
  }
  let verdict = null
  let fixed_any = false
  const MAX_FIX = 2
  for (let round = 0; round <= MAX_FIX; round++) {
    verdict = await agent(verifyPrompt(lane, impl, round), { label: `${lane.id}:verify${round + 1}`, phase: lane.phase, schema: VERIFY_SCHEMA })
    if (!verdict || verdict.verdict === 'pass' || round === MAX_FIX) break
    log(`lane ${lane.id}: review round ${round + 1} found ${verdict.blocking_issues.length} blocking issue(s); fixing`)
    const fixed = await agent(fixPrompt(lane, impl, verdict, round + 1), { label: `${lane.id}:fix${round + 1}`, phase: lane.phase, schema: IMPL_SCHEMA })
    if (fixed) { impl = fixed; fixed_any = true }
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
  if (lane.reviewOnly && !fixed_any) {
    log(`lane ${lane.id}: review passed without changes; nothing to commit`)
    continue
  }
  const commit = await agent(commitPrompt(lane, impl), { label: `${lane.id}:commit`, phase: lane.phase, schema: COMMIT_SCHEMA, effort: 'low' })
  results[results.length - 1].commit = commit
  log(`lane ${lane.id}: committed ${commit ? commit.ddgclib_sha + ' ' + commit.hyperct_sha : '(commit agent returned nothing)'}`)
  if (!commit) break
}
return results
