export const meta = {
  name: 'ddgclib-lanes-1',
  description: 'ddgclib campaign lanes S, P, L, H in sequence: implement in the library behind the method registry, independently verify, fix',
  phases: [
    { title: 'S exact setup volumes', detail: 'builders populate the simplex cache; 2D fallback volume fixed or retired' },
    { title: 'P hydrostatic on SolverMethods', detail: 'Hydrostatic_column 1D/2D/3D/periodic through presets, pinned' },
    { title: 'L wall-membership freezing', detail: 'frozen set by wall membership, not hull re-tagging; move collisions' },
    { title: 'H Hagen-Poiseuille 2D/3D', detail: 'developed profile through presets, pinned' },
  ],
}

const SCRATCH = '/tmp/claude-1000/-home-endres-projects-ddgclib/aeffc932-b55d-49f0-9349-f03613d0bde4/scratchpad'
const PY = '/home/endres/anaconda3/envs/ddg/bin/python'

const COMMON = `
You are continuing the ddgclib dynamics debugging campaign. ddgclib is a Lagrangian discrete-differential-geometry fluid solver (repo root /home/endres/projects/ddgclib). Its mesh backend hyperct lives at /home/endres/projects/hyperct and is symlinked into the repo as ./hyperct. The user's standing instruction for this phase: get all relevant dynamic test cases working, and prefer implementing code in the library behind the method wrappers (ddgclib.methods: SolverMethods, AXES, PRESETS) over duplicate case-local code.

READ FIRST (skim what is not relevant to your lane):
- debugging_plan.md: the top status entries (2026-10-01 lane R and anything newer, 2026-09-25 audit and lanes I/J/K) and the section "Reproducibility protocol for method lanes" (binding rules).
- METHODS.md: every method axis with status and evidence, the presets, and the case matrix (section 4).
- docs_temp/11_dynamics_audit_2026-09-25.md and the line-anchored reports in docs_temp/audit_2026-09-25/.
- The lane logs in docs_temp/debug_session/ that your brief names.

RULES
1. Python is ${PY}, run from the repo root /home/endres/projects/ddgclib. The shell prints conda start-up noise on every command; ignore it.
2. Git: never commit, stash, checkout, reset, restore or otherwise use git to change either working tree. Both trees hold uncommitted work that must not be lost. Read-only git (log, diff, status, show) is fine. New hyperct edits stay uncommitted in its working tree.
3. Off limits: cases_dynamic/capillary_rise_energy_grad/ (another agent owns it), cases_mean_flow/, benchmarks/, manuscript directories. In cases_dynamic/capillary_rise/ do not change the behaviour of the dynCA runners or src/_setup_dynca.py.
4. Library over case code: a method a case needs goes into ddgclib (or hyperct) behind a registered axis in ddgclib/methods/_axes.py with a SolverMethods field or builder in ddgclib/methods/_config.py. Cases consume a preset from ddgclib/methods/_presets.py and write methods.json with record_methods. Do not leave hand-rolled time loops or retopology closures in case files when the library can do the job. Keep changes minimal and surgical, match the existing style, add nothing speculative. New behaviour is opt-in unless your brief says otherwise, so existing defaults stay bit-identical.
5. Pins: every pinned number in ddgclib/tests must stay bit-identical unless your lane deliberately changes the method behind it. If so, A/B it through presets (preset versus preset.replace(...)), apply the flip rule of the protocol, re-pin together with the methods block, and record old and new values. Never loosen a tolerance or delete an assertion to make a test pass.
6. Scratch files go to ${SCRATCH}. A diagnose script worth keeping goes next to its case as diagnose_*.py. Case outputs go inside the case directory (fig/, results/).
7. Tests. Fast suite: ${PY} -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider (baseline at the start of this workflow: 983 passed, 12 skipped, 2 xfailed, about 80 s). Slow pinned battery: ${PY} -m pytest ddgclib/tests -q -m slow -p no:cacheprovider (baseline: 16 passed, 1 xfailed, about 30 s). hyperct suite, when you touch hyperct: cd /home/endres/projects/hyperct && ${PY} -m pytest -q (baseline 290 passed). All three must be green when you finish; earlier lanes may have raised the counts.
8. Use validation by integrated comparisons (ddgclib.analytical: integrated_pressure_error, integrated_l2_norm, volume_averaged_scalar), never point-wise pressure comparisons. Stay in the Lagrangian formalism.
9. Documentation when done: a lane log docs_temp/debug_session/lane<ID>-<slug>.md (what changed and where, measurements with exact numbers and the SolverMethods configuration behind each, measured DO-NOTs, known limits, how to reproduce); a new entry at the TOP of the status log in debugging_plan.md; the DEVELOPMENT.md checklist; evidence and status in ddgclib/methods/_axes.py; then ${PY} -m ddgclib.methods --update METHODS.md (rewrites the generated sections 2 and 3 in place) and edit the case matrix row in section 4 of METHODS.md by hand. Never use an em dash in prose.
10. If part of the brief turns out wrong or infeasible, do everything that is achievable and state exactly what is left and why. Report honestly: failing tests with their output, unverified claims marked as unverified.
`

const LANES = [
  {
    id: 'S', phase: 'S exact setup volumes',
    brief: `
LANE S: exact setup dual volumes.

Evidence: docs_temp/debug_session/laneK-single-phase-eos-instability.md sections 3.2, 4 and 7 item 3; docs_temp/debug_session/laneR-single-phase-conservative-remap.md section 3.

Problem. On a builder mesh that has no top-simplex cache (HC._simplices is None), ddgclib.operators.stress.cache_dual_volumes falls back to hyperct.ddg.dual_cell_area_2d in 2D (and the v_star fan walk in 3D). In 2D that fallback undercounts the four corner cells of a rectangle 4x (total 0.96875 instead of 1.0) and credits a moving free-surface vertex with a quarter of its own volume change. Masses and EOS pressures read these volumes: Hydrostatic_2D grows exponentially from round-off with zero reconnection on the fallback and decays eight orders with exact simplex volumes. Every retopology after setup uses the exact simplex volumes, so there is also a step-0 jump between volume sources.

Goal. Every 2D and 3D setup path yields simplex-exact barycentric dual volumes (Vol_i = sum over incident top simplices of |T|/(dim+1)) from the first cache_dual_volumes call, without each case having to remember a workaround.

Tasks.
1. Find where builder meshes get duals and volumes: ddgclib/geometry/domains/ (the builders and their shared helpers such as tag_boundaries), case setup helpers under cases_dynamic/*/src/, and the mass ICs in ddgclib/initial_conditions.py. Choose ONE library place that populates the simplex cache for builder connectivity. 2D has hyperct.ddg.rebuild_simplex_cache_2d. For 3D find or add the hyperct equivalent that enumerates the tetrahedra of the EXISTING connectivity without re-triangulating; if that is not feasible, document precisely why and what 3D builders must do instead.
2. Pin safety first: measure which pinned numbers move when builders carry the cache. Bit-identity is the default expectation (the droplet setups run a library retopology before their ICs: verify, do not assume). Any pin that moves is a method change on that pin: handle it by rule 5, or keep that setup path unchanged and say why.
3. The fallback itself: fix hyperct dual_cell_area_2d so the boundary polygon of a boundary vertex includes the vertex's own point (corner and free-surface cells then tile the domain), or retire it if nothing legitimate needs it any more. Add a hyperct test that the 2D dual areas of a rectangle with corners sum to the exact area. If the fix would move a pin that still relies on the fallback, say which and handle by rule 5.
4. Remove the explicit rebuild_simplex_cache_2d workaround in cases_dynamic/template/template.py once the builder does it, and keep ddgclib/tests/test_single_phase_remap.py pins (PIN_KE0, PIN_KE_END) bit-identical.
5. Validate with the lane K driver: ${PY} cases_dynamic/template/diagnose_single_phase_eos.py hydro --variant case and --variant nogravity_seed (see the log section 9 for the options) must now behave like the exact variants; a rectangle built by the builder must have total dual volume equal to its area.
6. Tests in ddgclib/tests: builders return a mesh with a simplex cache; dual volumes tile the domain for the 2D builders (rectangle, l_shape, disk, annulus) and, as far as task 1 allows, the 3D ones (box, cylinder_volume, pipe, ball).
7. Registry: update the reported axis dual_volume in _axes.py (status and evidence of dual_cell_area_2d and simplex_exact).
Not in scope: porting the Hydrostatic_column runners (that is lane P).
`,
  },
  {
    id: 'P', phase: 'P hydrostatic on SolverMethods',
    brief: `
LANE P: Hydrostatic_column on the library integrators.

Evidence: docs_temp/11_dynamics_audit_2026-09-25.md finding F12 and case matrix; docs_temp/audit_2026-09-25/cases_poiseuille_hydrostatic_bridges.md; lane K log sections 4 and 7; lane R log section 3 (known limits); the lane S log written just before you.

State. cases_dynamic/Hydrostatic_column/ has four runners (Hydrostatic_1D.py, Hydrostatic_2D.py, Hydrostatic_3D.py, Hydrostatic_2D_periodic.py) with hand-rolled symplectic loops (_recompute_duals plus cache_dual_volumes, no retagging, HydrostaticEOSMass, artificial viscosity mu_art = 0.5 rho c0 dx). Audit status: 1D decays, 2D unstable, 3D stalls, periodic aborts. Lane K attributed the 2D blow-up to the fallback dual volume, lane S fixed the setup volumes, lane R put the single-phase conservative remap into the library.

Goal. All four runners run through SolverMethods presets on library integrators, settle to hydrostatic equilibrium, and are pinned by tests. No hand-rolled time loop remains in the runners.

Tasks.
1. Read the runners and their src/ helpers. List what the hand-rolled loops do that the library integrators do not (artificial viscosity, damping, gravity, free-surface handling, which vertices are frozen). Gravity goes through methods.dudt_fn(..., body_force=...). Anything else that is really needed goes into the library behind a registered axis, not back into the case.
2. Choose the connectivity for each case by measurement, and report both arms: 'dual_only' (no reconnection; natural for a static column) and 'delaunay' with remap='conservative', redistribute_mass=True. The remap arm must also be stable. If it is not, diagnose it. Lane R section 3 names the prime suspect: the global mass rescale leaves a uniform pressure offset K(s-1) per rebuild, which is not force-free on an open free-surface fan. If that is the cause, implement the fix in the library (for example an overlap-free gauge like the multiphase vol_corr, or a locally conservative variant), registered as an option on the remap axis, with tests.
3. Add presets hydrostatic_1D, hydrostatic_2D, hydrostatic_3D, hydrostatic_2D_periodic to ddgclib/methods/_presets.py. Runners consume them, record methods.json, follow the dynamic case convention in CLAUDE.md (StateHistory with save_dir, outputs in the case directory, README.md with run instructions, non-blocking plotting so the runner can run headless).
4. Validation: integrated comparisons from ddgclib.analytical. Success means max|u| decays (or stays at round-off for an exact discrete equilibrium) over at least 40 acoustic times, and the integrated pressure error is small and converges with refinement. Extend ddgclib/tests/test_case_hydrostatic.py with fast pinned variants built from the presets (mark 3D slow if it needs it).
5. Free surface: lane K measured a 30 to 45 percent mismatch between the discrete pressure force and the energy-consistent normal force on open fans even with exact volumes. Quantify whether it matters for the settled column and for a perturbed column (sloshing seed), and report the numbers. Do not paper over an instability with extra viscosity; if the column needs mu_art to settle, say how much and why.
6. Update the METHODS.md case matrix row for Hydrostatic_column and the audit F12 status line.
`,
  },
  {
    id: 'L', phase: 'L wall-membership freezing',
    brief: `
LANE L: frozen vertices by wall membership, not by hull membership.

Evidence: docs_temp/11_dynamics_audit_2026-09-25.md finding F10 (corner mechanisms C1 to C9, especially C1, C2, C3); docs_temp/audit_2026-09-25/bcs_and_cases.md and bcs_partA.md section A.3; DEVELOPMENT.md, the prescribed_V specification (search for prescribed_V); docs_temp/audit_2026-09-25/integrators.md.

Problem. The integrators rebuild the frozen boundary set bV from the topological hull at every retopology (_retopologize in ddgclib/dynamic_integrators/_integrators_dynamic.py, optionally filtered by boundary_filter). Nothing enforces impenetrability. Once one vertex steps past a straight wall, the collinear wall vertices stop being hull vertices, are unfrozen and start to move: the whole wall collapses (Hagen_Poiseuile_2D at about step 1248) and vertices pile up in corners. Related: PeriodicInletBC coordinate-key collisions and plug injection, and HC.V.move onto an existing coordinate key.

Goal. A library policy in which the set of frozen (wall) vertices is persistent membership, kept separate from the topological tag v.boundary that compute_vd needs for half cells. The legacy hull policy stays the default so every pin is bit-identical; cases that need it select the new policy through SolverMethods.

Tasks.
1. Reproduce C1 with a small fast reproducer on the Hagen-Poiseuille 2D setup (or smaller) and keep it as a test that fails on the legacy policy in the documented way and passes on the new one.
2. Implement the minimal library change. Register it as an explicit axis in ddgclib/methods/_axes.py (for example frozen_set: 'hull' legacy default, 'membership' new) with a SolverMethods field and builder plumbing, so runners select it through a preset. Decide and document what happens to a non-wall hull vertex (inlet and outlet vertices must keep advecting), to a vertex that reaches a wall, and to a wall vertex that the hull no longer contains.
3. Impenetrability and key collisions: audit C2 and C3. Make HC.V.move in hyperct refuse or handle a move onto an occupied coordinate key instead of silently corrupting the cache, with a hyperct test; fix the PeriodicInletBC collision path in ddgclib/_boundary_conditions.py that depends on it.
4. Show that Hagen_Poiseuile_2D runs past the former collapse point with the new policy through the preset hagen_poiseuille_2D (give the runner a headless short mode if blocking plot calls prevent a scripted run). Record wall-vertex positions before and after: they must not move.
5. Check the other cases that freeze by hull membership (dam break, hydrostatic, capillary rise static runners, shearing plate, electrolysis) and say which should switch; switch only those where you measured that it is neutral or better, by rule 5.
Not in scope: validating the developed Poiseuille profile (lane H).
`,
  },
  {
    id: 'H', phase: 'H Hagen-Poiseuille 2D/3D',
    brief: `
LANE H: Hagen-Poiseuille 2D and 3D working and pinned.

Evidence: METHODS.md case matrix rows for Hagen_Poiseuile*; docs_temp/audit_2026-09-25/cases_poiseuille_hydrostatic_bridges.md (including the 3D stall, audit label M2: the inlet cap is frozen, and retopologize_cylinder never updates HC._simplices); docs_temp/audit_2026-09-25/bcs_and_cases.md; the lane L log written just before you.

State. cases_dynamic/Hagen_Poiseuile/Hagen_Poiseuile_2D.py uses preset hagen_poiseuille_2D with PeriodicInletBC, OutletBufferedDeleteBC, PositionalNoSlipWallBC; before lane L it collapsed at about step 1248. cases_dynamic/Hagen_Poiseuile_3D/Hagen_Poiseuile_3D.py uses connectivity='custom' (retopologize_cylinder) and stalls. Hagen_Poiseuile_equilibrium validates the static residual only. ddgclib/tests/test_case_hagen_poiseuille.py and test_hagen_poiseuille_equilibrium.py exist.

Goal. The Lagrangian 2D channel and 3D pipe runs develop the analytical Poiseuille profile from their initial condition, through presets, and a pinned test guards each.

Tasks.
1. 2D: run to the developed state through the preset (with the lane L frozen-set policy if that is what makes it survive). Measure the velocity profile error against the analytical solution with integrated or dual-volume-weighted comparisons, the mass flux at inlet and outlet, and vertex count over time (no pile-up in corners, no depletion). Fix what stops it in the library, behind the registry.
2. 3D: diagnose the stall (audit M2). Replace the custom retopologize_cylinder closure by a library connectivity policy if one can do the job; otherwise move the closure into ddgclib/methods/_retopo.py as a named, registered connectivity value like the former case-local closures. It must keep HC._simplices consistent.
3. Presets updated, runners headless-capable, methods.json recorded, README per case.
4. Tests: extend ddgclib/tests/test_case_hagen_poiseuille.py with a fast pinned 2D run built from the preset and a 3D run (slow marker if needed).
5. Report what remains between the measured profile and the analytical one, with numbers, and which method axis it is attributed to.
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
BEFORE ANY EDIT run: bash ${SCRATCH}/lane_snapshot.sh ${lane.id}
(it mirrors the current sources so a reviewer can diff exactly your lane with: bash ${SCRATCH}/lane_diff.sh ${lane.id}).
${lane.brief}
Finish only when the three test suites are green and the documentation of rule 9 is written. Your structured result is read by a reviewer who will re-run the tests and re-measure your headline numbers.`
}

function verifyPrompt(lane, impl, round) {
  return `${COMMON}
YOU ARE THE INDEPENDENT REVIEWER of lane ${lane.id} (review round ${round + 1}). You did not write it. Do not edit any source, test or documentation file; you may write scratch files under ${SCRATCH}. Your job is to find what is wrong or unproven.

The lane brief was:
${lane.brief}
The implementer reported:
${JSON.stringify(impl, null, 1)}

Steps.
1. Read the complete lane diff: bash ${SCRATCH}/lane_diff.sh ${lane.id} (add --stat for the file list). Changes under cases_dynamic/capillary_rise_energy_grad/ or the dynCA files may come from another agent working concurrently; mention them but do not count them against the lane unless the lane log claims them.
2. Re-run the fast suite, the slow battery, and the hyperct suite if hyperct changed. Quote the summary lines.
3. Re-measure the headline claims from the lane log by running its reproduce commands (at least the main claim, more if cheap). Compare numbers digit by digit where a pin is involved.
4. Check the rules: no pinned number changed without A/B evidence and a recorded old and new value; no loosened tolerance or deleted assertion; new switches registered in _axes.py with a SolverMethods field; no duplicate case-local code where the library could do it; defaults bit-identical unless the brief allowed a change; METHODS.md current (the drift test in ddgclib/tests/test_methods.py passes); lane log, debugging_plan.md entry and DEVELOPMENT.md updated; no em dashes in new prose; off-limits directories untouched by the lane.
5. Read the changed code for correctness bugs: wrong conditions, stale caches, silently ignored arguments, behaviour that differs between 2D and 3D without being stated, tests that cannot fail.
6. Judge whether the lane goal is met in full, in part, or not at all, on the evidence you measured yourself.

verdict = pass only when the tree is green and there is no blocking issue. A lane that honestly documents a part it could not finish can still pass with goals_met = partial; list that part under nonblocking_issues. Blocking issues are: red tests, a pin moved without evidence, a claim you could not reproduce, a correctness bug, a rule violation, missing documentation.`
}

function fixPrompt(lane, impl, verdict, round) {
  return `${COMMON}
YOU ARE FIXING lane ${lane.id} after independent review (fix round ${round}). Do NOT run lane_snapshot.sh again (the lane snapshot must stay as the pre-lane state).

The lane brief was:
${lane.brief}
The implementer reported:
${JSON.stringify(impl, null, 1)}

The reviewer's findings:
${JSON.stringify(verdict, null, 1)}

Resolve every blocking issue at its cause (not by weakening a test), address the non-blocking ones where cheap, bring the three suites to green, and update the lane log, debugging_plan.md entry and METHODS.md so they describe the final state. If you disagree with a finding, show the measurement that refutes it. Return the full updated report for the lane (not only the delta).`
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
}
return results
