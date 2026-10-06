export const meta = {
  name: 'ddgclib-lanes-2b',
  description: 'ddgclib campaign lanes T, O, B, Q, M in sequence: determinism, 2D area-vector orientation, droplet builder shift, 3D simplex edge areas, stokes path',
  phases: [
    { title: 'T deterministic iteration order', detail: 'remove id()-ordered iteration so reconnecting runs are process independent' },
    { title: 'O 2D dual area vector orientation', detail: 'fix the orientation defect behind the strict xfail; re-pin what moves' },
    { title: 'B droplet builder box shift', detail: 'droplet builders lose outer vertices to move collisions; fix and re-pin' },
    { title: 'Q 3D simplex edge areas', detail: 'vectorised p_ij_simplex kernel, explicit edge_area_source axis, 3D A/B' },
    { title: 'M stokes curvature path', detail: 'dynamic A/B, then fix status or remove' },
  ],
}

const SCRATCH = '/tmp/claude-1000/-home-endres-projects-ddgclib/aeffc932-b55d-49f0-9349-f03613d0bde4/scratchpad'
const PY = '/home/endres/anaconda3/envs/ddg/bin/python'

const COMMON = `
You are continuing the ddgclib dynamics debugging campaign. ddgclib is a Lagrangian discrete-differential-geometry fluid solver (repo root /home/endres/projects/ddgclib). Its mesh backend hyperct lives at /home/endres/projects/hyperct and is symlinked into the repo as ./hyperct. The user's standing instructions: get all relevant dynamic test cases working with the oscillating droplet as the main benchmark; always ship a fix into the library as long as the method is separated behind the method wrappers (ddgclib.methods: SolverMethods, AXES, PRESETS) and no code is duplicated.

READ FIRST (skim what is not relevant to your lane):
- debugging_plan.md: the status entries from 2026-10-01 onward (lanes R, S, P, L, H and anything newer) and the section "Reproducibility protocol for method lanes" (binding rules, including rule 8 on process dependence).
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
7. Tests. Fast suite: ${PY} -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider (baseline at the start of this workflow: 1136 passed, 12 skipped, 3 xfailed, about 100 s). Slow pinned battery: ${PY} -m pytest ddgclib/tests -q -m slow -p no:cacheprovider (baseline: 26 passed, 1 xfailed, about 190 s). hyperct, when you touch it: cd /home/endres/projects/hyperct && ${PY} -m pytest hyperct/tests -k "not benchmark" -q -p no:cacheprovider (baseline 316 passed; the bare pytest -q at the hyperct root fails at collection for unrelated reasons). All must be green when you finish; earlier lanes may have raised the counts.
8. Use validation by integrated comparisons (ddgclib.analytical), never point-wise pressure comparisons. Stay in the Lagrangian formalism.
9. Documentation when done: a lane log docs_temp/debug_session/lane<ID>-<slug>.md (what changed and where, measurements with exact numbers and the SolverMethods configuration behind each, measured DO-NOTs, known limits, how to reproduce); a new entry at the TOP of the status log in debugging_plan.md; the DEVELOPMENT.md checklist; evidence and status in ddgclib/methods/_axes.py; then ${PY} -m ddgclib.methods --update METHODS.md and edit the case matrix rows in section 4 of METHODS.md by hand. Never use an em dash in prose.
10. If part of the brief turns out wrong or infeasible, do everything that is achievable and state exactly what is left and why. Report honestly: failing tests with their output, unverified claims marked as unverified. State numbers from process-dependent runs as ranges over several fresh processes.
`

const LANES = [
  {
    id: 'T', phase: 'T deterministic iteration order',
    brief: `
LANE T: deterministic iteration order (no id()-ordered iteration on the force and retopology paths).

Evidence: debugging_plan.md protocol rule 8; docs_temp/debug_session/laneP-hydrostatic-library-integrators.md (repeatability section), laneL-frozen-set-membership.md (shearing-plate vote, iter_triangles_2d, remesh _edge_list), laneH-poiseuille-developing.md section 5.3 and the reviewer's note: the 3D centred-on-cache arm takes l2 0.0811, 0.0821 or 0.1107 depending on the process and on what ran earlier in the same process, deterministic for a given sequence of runs, which points at allocation-address-ordered iteration (id-keyed sets and frozensets in hyperct/ddg/_compute_dual.py, _boundary.py, the fan walk, the p_ij ring), not at random ties. laneS made rebuild_simplex_cache_2d/3d enumerate in HC.V order already.

Problem. Sets of vertex objects and id()-sorted containers iterate in memory-address order. Summation order, tie breaks and the choice among equivalent candidates then differ between interpreters and between runs in one interpreter. 2D runs are bit-identical between processes today; 3D runs that reconnect or read dual face areas are reproducible to two digits only after about ten acoustic times, so 3D pins are restricted to early peak values and every 3D A/B needs several processes.

Goal. A dynamic run is a pure function of its inputs: bit-identical between fresh processes and independent of earlier runs in the same process, in 2D and 3D, for every connectivity value.

HANDOFF. A first attempt at this lane was interrupted by a machine restart after about an hour of work. Its edits are in the working trees and are unreviewed and possibly incomplete: hyperct/ddg/_compute_dual.py, hyperct/remesh/_driver.py, hyperct/remesh/_quality.py, new hyperct/tests/test_deterministic_order.py; ddgclib/geometry/periodic.py, ddgclib/operators/stabilisation.py, ddgclib/tests/test_case_hydrostatic.py (it was adding end-of-run pins to the hydrostatic 3D tests when it stopped), new ddgclib/tests/test_determinism.py, new cases_dynamic/diagnose_determinism.py (the detector). Its scratch outputs are lost. Start by reading that diff critically and running the detector and the suites to see what state it left; keep what is right, repair or finish the rest, and do every task below as if it were yours from the start (the measurements must be yours).

Tasks.
1. Build the detector first: a script that runs a short reconnecting 3D case (the hagen_poiseuille_3D centred arm, the hydrostatic_3D remap arm, the 3D droplet with connectivity='delaunay') in N fresh processes and after different preceding runs, and reports the digest of the final state. Reproduce the documented spread.
2. Find every place on the retopology, dual, force and multiphase paths (hyperct and ddgclib) where the result depends on the iteration order of a set, dict keyed by id, or id()-sorted list: v.nn iteration, neighbour intersections, iter_triangles_2d, hyperct.remesh._driver._edge_list, the fan walk, _dual_area_vector_3d_p_ij, batch_e_star assembly, boundary_from_simplices, the interface extraction. Replace by an order that is a function of the mesh (cache insertion order of HC.V, or coordinates), at the lowest level that fixes it for all callers. Do not sort where an insertion-ordered container does the job; watch the cost (report wall time before and after on the 2D and 3D droplet).
3. Pins. Changing a summation order moves last bits, and a long run amplifies them. For every pinned number that moves: show old and new values, show that the change is of round-off-amplification size (compare with the spread a 1e-15 relative perturbation of the initial positions produces in the same run), and re-pin by rule 5. The 2D and 3D droplet baselines, the static floors, the remap pins, the Hagen-Poiseuille and hydrostatic pins are all in scope. If a pin moves by more than that, it is a different effect: find it.
4. With determinism established, replace the early-peak 3D pins that rule 8 forced by end-of-run values where that is now justified, and rewrite protocol rule 8 to say what is guaranteed.
5. Add a test that runs a short reconnecting 3D case in two fresh interpreters and asserts equal digests, and one that runs it twice in one interpreter after an unrelated run.
6. Note for the other agent: state in the lane log which dynCA-visible hyperct functions changed order.
`,
  },
  {
    id: 'O', phase: 'O 2D dual area vector orientation',
    brief: `
LANE O: orientation of the 2D dual area vector.

Evidence: docs_temp/debug_session/laneH-poiseuille-developing.md (the section on the orientation defect and its follow-up list); the strict xfail test test_2d_dual_area_vectors_close_on_a_sheared_jittered_mesh (find it with grep in ddgclib/tests); ddgclib/operators/stress.py:dual_area_vector, 2D non-periodic branch.

Problem. On sheared or jittered 2D meshes some dual face area vectors A_ij come out with the wrong sign, so the dual cell does not close (sum_j A_ij != 0) and pressure and viscous fluxes on those faces act backwards. Lane H counted 737 / 431 flipped vectors in the lane L Hagen-Poiseuille reproducer, 272 in lane R's bare-Delaunay box and 9 in the material-Delaunay test. Lane H documented it and did not fix it because the fix moves numbers in the tests of lanes L, R and P.

Goal. A_ij is correctly oriented (outward from i, A_ij = -A_ji, closed cells for interior vertices) on every valid 2D mesh, and every result that depended on the defect is re-measured.

HANDOFF. A first attempt at this lane stopped after a short time when the model's usage limit was reached. Its edits are in the working tree, unreviewed: a new reference helper simplex_area_vectors(v, HC, dim) appended to ddgclib/operators/stress.py (exact barycentric dual area vectors of the edges at a vertex from the simplex cache, with no orientation choice; nothing calls it yet) and a new 741-line probe cases_dynamic/diagnose_area_orientation.py. Nothing else changed (git diff HEAD and git status show exactly these two). Read both critically, keep what is right, and do every task below as if it were yours from the start; the measurements must be yours.

Tasks.
1. Reproduce: count flipped vectors and closure residuals on the xfail mesh, on the 2D oscillating droplet (setup and along a run with connectivity='delaunay' + remap), the dam break, the hydrostatic 2D column, the template box. Say for each pinned 2D case whether any integrated vertex was ever affected. Check the 3D paths and the periodic 2D branch for the same defect.
2. Fix at the cause (orientation from the geometry of the shared dual edge relative to the primal edge, not a sign patch that hides a wrong polygon). The xfail must become a passing test; add closure and antisymmetry tests on sheared, jittered and reconnected meshes, including boundary half cells.
3. Re-measure everything that moves, by rule 5: the 2D droplet baseline and floors, static droplet, remap pin, dam break, Hagen-Poiseuille 2D pins and the lane L reproducer counts, hydrostatic 2D pins, lane R's bare-Delaunay instability test (is single-phase bare Delaunay + EOS still unstable without the defect? lane K's mechanism says yes; measure it), the capillary-rise smoke runner (report only). For the droplet apply the comparison on l2 and tail and report both; a correctness fix is adopted regardless, but the numbers must be on record.
4. Keep the legacy orientation reachable as a registered option with status 'broken' if that is one flag (rule 5), so the pre-fix pins can be reproduced.
5. Reconsider pressure_flux for the hagen_poiseuille_2D preset as lane H's reviewer suggested, only on measurement.
`,
  },
  {
    id: 'B', phase: 'B droplet builder box shift',
    brief: `
LANE B: the droplet-in-box builders lose outer vertices.

Evidence: docs_temp/debug_session/laneL-frozen-set-membership.md (follow-ups and the move-collision census): droplet_in_box_2d loses 6 of 145 outer vertices at refinement 3 and droplet_in_box_3d loses 3 of 189 at refinement 2, the (L, L) corner included, because the box shift moves vertices one at a time onto coordinate keys that other vertices still hold (hyperct HC.V.move used to evict silently; since lane L it raises unless on_collision='evict', and lane L introduced HC.V.move_all for whole-mesh transforms). Code: ddgclib/geometry/domains/_multiphase_droplet.py and the setups that call it (cases_dynamic/oscillating_droplet/src/, electrolysis_bubble/src/_setup.py, shearing_plate_droplet/src/_setup.py, dam break if it shares the path).

Problem. Every oscillating-droplet, static-droplet, electrolysis and shearing-plate number in the campaign was computed on an outer mesh with missing vertices, a missing domain corner included. The Delaunay retopology papers over the holes with large cells.

Goal. The builders return the full mesh; the campaign's main benchmark is re-established on it with every number on record.

Tasks.
1. Reproduce the loss (which vertices, at which refinement, in 2D and 3D) and show what the first retopology does with the hole (cell sizes, boundary set, total volume, the outer-phase mass).
2. Fix with move_all in the library builder. Keep the legacy lossy shift reachable by an explicit builder argument (rule 5) so the old pins can be reproduced, and say in the registry / METHODS.md where that switch lives (it is a setup choice, not a solver axis; record it in the methods.json extra block of the runners).
3. Re-measure and re-pin, by rule 5, everything on that path: 2D droplet (l2, tail, mass drift, floors, static droplet, envelope regression at refinement 2/2, remap endurance, projection cadence tests), 3D droplet (l2, tail, floor), a5b long-run pins, electrolysis and shearing-plate smoke numbers, dam break if affected. Report old and new side by side with the preset names. Update the baselines JSON with their methods block.
4. Say what the fix does to the known open physics items: the 2D over-decay (lane H of July, projection_every), the 3D bump / over-decay cancellation (lane G of July), the shearing-plate instability and its 23 setup collisions.
`,
  },
  {
    id: 'Q', phase: 'Q 3D simplex edge areas',
    brief: `
LANE Q: exact simplex-based dual face areas in 3D.

Evidence: docs_temp/debug_session/laneJ-3d-edge-area-source.md (the e_star cache written by batch_e_star at every 3D retopology is not linearly precise: it carries about 79 percent of the 3D static-floor retopology excess and fails dual-face closure by about 1 percent at every interface vertex; the library p_ij path picks a wrong face vertex on 743 of 5193 directed edges; reading the faces from HC._simplices, 'p_ij_simplex', closes to 2e-16 with linear precision 2e-15 but costs 4x in its per-edge Python form); laneH-poiseuille-developing.md (3D centred-on-cache arm drifts radially); METHODS.md reported axis edge_area_source; the lane T log (determinism) written before you.

Goal. A vectorised kernel that computes the oriented barycentric dual face area vectors of all edges from the top-simplex cache in one pass, an explicit method axis to select the 3D edge-area source, and a measured decision on the 3D default.

Tasks.
1. Kernel in hyperct (next to batch_e_star / simplex_dual_volumes): for every edge, A_ij = sum over incident tetrahedra of the two barycentric-subdivision triangles (edge midpoint, face barycentres, cell barycentre), oriented from i to j; vectorised over the simplex array; also 2D for parity if it is one branch. Tests: antisymmetry, closure sum_j A_ij = 0 at interior vertices, linear precision of the integrated gradient to round-off, agreement with the per-edge reference of lane J's driver, boundary half cells.
2. Make edge_area_source an EXPLICIT axis ('e_star_cache' legacy default, 'p_ij' , 'p_ij_simplex') with a SolverMethods field; the retopology fills HC._edge_area_cache from the chosen source; every operator that reads face areas (stress force, multiphase force, velocity Laplacian / viscous flux, simplex_gradient fluxes if they read them) follows the same switch. effective_methods keeps reporting what actually ran.
3. A/B through presets, in several fresh processes if lane T did not establish determinism: 3D static floor (pinned 7.274172e-05 on the cache, lane J measured 6.2839e-05), full 3D droplet (pinned l2 0.24811340819647862, tail 0.08409976059818802; lane J: l2 0.287 on p_ij because a cancellation is exposed, inflation halves), hagen_poiseuille_3D with pressure_flux='centred', hydrostatic_3D, dam_break_3D smoke, 3D wall time per step.
4. Decide the default by the protocol. Lane J's verdict was: do not flip on the l2 score alone, evaluate on the sign-decomposed bump and over-decay channels of July's lane G together with the redistribution lever. Do that evaluation with the new kernel (cheap now) including redistribute_mass and projection_every arms, and state plainly whether the exact areas are adopted as 3D default, with the re-pins if so.
5. Fix or retire the per-edge _dual_area_vector_3d_p_ij face-vertex selection defect (lane J F4b) so that 'p_ij' is either correct or gone.
`,
  },
  {
    id: 'M', phase: 'M stokes curvature path',
    brief: `
LANE M: the 'stokes' curvature path.

Evidence: docs_temp/11_dynamics_audit_2026-09-25.md finding F2; docs_temp/debug_session/laneI-3d-apex-cache.md (the coordinate-keyed map is now cleared at every interface refresh; status moved from broken to experimental without a dynamic A/B); METHODS.md axis curvature_path; debugging_plan.md entry of 2026-05-27 (the integrated-Stokes curvature variant was found mathematically identical to the cotangent form on a static mesh).

Goal. Every value of curvature_path has a status that rests on a dynamic measurement, and values that add nothing are removed with a record, so the axis is small and trustworthy.

Tasks.
1. For each registered value of curvature_path, run the static 2D floor and a full 2D oscillating droplet through PRESETS['oscillating_droplet_2D'].replace(curvature_path=...), and the 3D counterparts where the value supports 3D. Report l2, tail, floors and wall time against the default 'integrated'.
2. If 'stokes' reproduces 'integrated' to round-off on a moving mesh, it is a duplicate: remove the code path and the axis value, keep one line in the registry notes and the lane log saying what it was and why it went. If it differs, explain the difference and set its status by the numbers.
3. Do the same triage for the other non-default values (for example csf_dual): validated, opt-in, measured-worse, or removed. Remove dead helper code that only the removed values used; do not touch unrelated dead code.
4. Tests: one per surviving value on a moving mesh (the caches must follow the mesh), so a stale-cache regression of the kind lane I fixed cannot return unnoticed.
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
2. Stage with explicit paths only: git add <path> <path> ... Never use git add -A, -u, . or a directory that contains anything outside the lane. Baselines under cases_dynamic/oscillating_droplet/baselines are tracked: include them when they changed.
   NEVER stage: .gitignore; anything under cases_mean_flow/, benchmarks/, tutorials/, test_cases/, cases_dynamic/capillary_rise/, cases_dynamic/capillary_rise_energy_grad/; ddgclib/tests/test_integrated_validation.py; generated outputs (results/, fig/, *.log, *.pkl, *.mp4, pytest-of-*); anything untracked that is not a source, test or documentation file of this lane. CLAUDE.md and DEVELOPMENT.md are git-ignored: leave them.
3. Commit hyperct first (if it changed), then ddgclib, both on master (check git branch --show-current; if it is not master, stop and report). Message: first line with the repo's prefix convention ENH: / BUG: / MAINT:, a plain one-line summary, ending with (lane ${lane.id}); a body of 5 to 15 lines saying what changed, the headline measured numbers, and every pin that moved as old -> new; no em dashes; then a blank line and exactly this last line:
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
4. No push, no amend, no rebase, no branch, no tag, no other git command that changes history or the working tree.
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
