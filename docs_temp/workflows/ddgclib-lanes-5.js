export const meta = {
  name: 'ddgclib-lanes-5',
  description: 'ddgclib campaign lanes V-review and D on the workstation: independent review of the merged lane V (wall clamp, simplex vote, ledger-aware merge) with its headline numbers re-measured, then the final oscillating-droplet evaluation of every new method with adoption by the flip rule',
  phases: [
    { title: 'V-review impenetrability and interface vote', detail: 'independent review of the committed lane V (08979e1 plus the static-review fixes of 2026-10-10), re-measure its headline numbers, fix at the cause, commit' },
    { title: 'D oscillating droplet with every new method', detail: 'A/B of every new axis on the 2D and 3D droplet, adoption by the flip rule, re-pin, droplet summary on top of debugging_plan.md' },
  ],
}

// ---------------------------------------------------------------------------
// REPLACE THESE FIVE CONSTANTS FOR THE WORKSTATION BEFORE LAUNCHING.
// ROOT: the ddgclib clone; HYP: the hyperct clone next to it (ROOT/hyperct is
// a symlink to HYP/hyperct); SCRATCH: a scratch directory outside both repos;
// PY: the interpreter of the conda env where ddgclib and hyperct are editable
// installs (pip install -e ROOT; pip install -e HYP); BRANCH: master.
// CORES / MAX_PROCS: the workstation's core count and how many solver
// processes a lane may run at once (3D droplet arms are about 8 min each on a
// fast core; the 2D full run about 4 min).
// ---------------------------------------------------------------------------
const ROOT = '/REPLACE/ME/projects/ddgclib'
const HYP = '/REPLACE/ME/projects/hyperct'
const SCRATCH = '/REPLACE/ME/scratch/ddgclib-lanes-5'
const PY = '/REPLACE/ME/.anaconda3/envs/ddg/bin/python'
const BRANCH = 'master'
const CORES = 32
const MAX_PROCS = 8
// Baseline of the three suites on the machine that merged the cloud work
// (the owner's dev PC, 2026-10-10, Python 3.13.12, numpy 2.4.2, scipy
// 1.17.0, matplotlib 3.10.8, OpenBLAS). Two pre-existing failures there,
// both the cloud session's environment items with the identical values and
// present on the hand-off tree before any cloud work (HANDOFF_2026-10-10.md
// section 2 has the measured cause of (a)):
//   (a) test_single_phase_remap.py::TestBoxDecayWithEOS::test_remap_matches_fixed_connectivity
//       (0.4399043770574842 != 0.43990437705748414: the conservative remap is
//       1 + ulp instead of the identity on an unchanged edge set)
//   (b) test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column
//       (umax peak 0.15269571220608844 against the pin 0.15305813130485327)
// Re-measure once on the workstation before launching and replace the three
// strings with what it prints; say in the first lane log which of (a) / (b)
// reproduce there.
const BASELINE = {
  fast: '1 failed, 1343 passed, 1 skipped, 43 deselected, 2 xfailed (the failure is (a))',
  slow: '1 failed, 41 passed, 1347 deselected, 1 xfailed (the failure is (b))',
  hyperct: '363 passed, 15 skipped, 50 deselected, 6 xfailed',
}

for (const [k, v] of Object.entries({ ROOT, HYP, SCRATCH, PY })) {
  if (v.includes('REPLACE')) throw new Error(`constant ${k} still holds the placeholder; edit the head of this script for the workstation`)
}
for (const [k, v] of Object.entries(BASELINE)) {
  if (v.includes('PLACEHOLDER')) throw new Error(`BASELINE.${k} still holds the placeholder; run the suite once and paste its summary line`)
}

const COMMON = `
You are continuing the ddgclib dynamics debugging campaign on the owner's workstation. ddgclib is a Lagrangian discrete-differential-geometry fluid solver (repo root ${ROOT}). Its mesh backend hyperct lives at ${HYP} and is symlinked into the repo as ./hyperct; both are editable installs of the interpreter ${PY}. The owner's standing instructions: get every relevant dynamic test case working with the oscillating droplet as the main benchmark; always ship a fix into the library as long as the method is separated behind the method wrappers (ddgclib.methods: SolverMethods, AXES, PRESETS) and no code is duplicated; commit verified work on master; never touch the owner's unrelated work. The campaign's adversarial metric is accuracy against analytical solutions (ddgclib.analytical and the case references): every claim of improvement is a measured number against an analytical or reference solution, before and after, through presets.

READ FIRST (skim what is not relevant to your lane):
- docs_temp/HANDOFF_2026-10-10.md (the hand-off of the merge session: state of every lane, what the static review of lane V found and fixed, what is deferred to this machine).
- docs_temp/HANDOFF_2026-10-07.md (the cloud session: lanes F-review, G, I, X; environment baseline there; open items).
- debugging_plan.md: the status entries from 2026-10-01 onward (lanes R, S, P, L, H, T, O, B, Q, M, W, F, G, I, X, V) and the section "Reproducibility protocol for method lanes" (binding rules, including rule 8 on process dependence).
- METHODS.md: every method axis with status and evidence, the presets, and the case matrix (section 4).
- docs_temp/NEXT_SESSION_PROMPT.md (the standing rules of the campaign and the original lane D brief).
- The lane logs in docs_temp/debug_session/ that your brief names.

RULES
1. Python: run every command from the repo root ${ROOT} as \`${PY} ...\` (editable installs: ddgclib and the live hyperct tree import directly; check once with \`${PY} -c "import hyperct; print(hyperct.__file__)"\`, it must point into ${HYP}). For the hyperct suite: cd ${HYP} && ${PY} -m pytest hyperct/tests -k "not benchmark" -q -p no:cacheprovider. The machine has ${CORES} cores: run at most ${MAX_PROCS} solver processes at once and never a pytest -n fan-out (several pins are process-order sensitive by protocol rule 8).
2. Git: never commit, stash, checkout, reset, restore or otherwise use git to change either working tree or history (the only exception is the dedicated commit step that runs after a lane passes review). Read-only git (log, diff, status, show) is fine. ddgclib is on ${BRANCH} and hyperct on master, so HEAD is the pre-lane state in both: the lane's changes are exactly what \`bash docs_temp/workflows/lane_diff.sh\` prints (add --stat for the file list). The owner has unrelated uncommitted work in the tree (benchmarks/, cases_mean_flow/, .gitignore, ddgclib/tests/test_integrated_validation.py, the dynCA files under cases_dynamic/capillary_rise/); lane_diff.sh excludes it; never touch it.
3. Off limits, never edit AND never execute anything from them, not even from a scratch copy (several scripts chdir to an absolute repo path and overwrite the owner's manuscript files; this happened once): cases_mean_flow/, benchmarks/, tutorials/, test_cases/, manuscript directories. Also off limits for edits: cases_dynamic/capillary_rise_energy_grad/ (another agent owns it); in cases_dynamic/capillary_rise/ do not change the behaviour of the dynCA runners, src/_setup_dynca.py or README_dynCA.md. If a library change of yours can move the numbers of those runners, say so in your lane log.
4. Library over case code: a method a case needs goes into ddgclib (or hyperct) behind a registered axis in ddgclib/methods/_axes.py with a SolverMethods field or builder in ddgclib/methods/_config.py. Cases consume a preset from ddgclib/methods/_presets.py and write methods.json with record_methods. No duplicate hand-rolled loops or closures in case files. Keep changes minimal and surgical, match the existing style, add nothing speculative.
5. Pins: every pinned number in ddgclib/tests must stay bit-identical unless your lane deliberately changes the method behind it. If it does, A/B through presets (preset versus preset.replace(...)), re-pin together with the methods block (baselines under cases_dynamic/oscillating_droplet/baselines carry one), and record old and new values in the lane log. Never loosen a tolerance or delete an assertion to make a test pass. When a lane replaces a default because the old behaviour is a defect, keep the old behaviour reachable as a registered option with status 'broken' when that costs a flag, so earlier results stay reproducible; say so if it is not feasible.
6. Scratch and outputs: do not create files you would need to delete. Probe runs write only under ${SCRATCH}/lane<ID>/ (run case runners from scratch copies or pass an explicit output directory). Anything the lane log needs for reproduction (detector and diagnose scripts) lives in the repository, not in scratch. Only results that are meant to be kept go into the case directory (fig/, results/). A diagnose script worth keeping goes next to its case as diagnose_*.py. Never leave stray outputs in either repository: compare git status --short before you finish.
7. Tests. Fast suite: ${PY} -m pytest ddgclib/tests -q -m "not slow" -p no:cacheprovider. Slow pinned battery: ${PY} -m pytest ddgclib/tests -q -m slow -p no:cacheprovider. hyperct, when you touch it: see rule 1 (the bare pytest -q at the hyperct root fails at collection for unrelated reasons). Use a generous timeout. BASELINE measured before this workflow: fast "${BASELINE.fast}"; slow "${BASELINE.slow}"; hyperct "${BASELINE.hyperct}". "Green" means: no failure beyond the ones named in the BASELINE, and those unchanged (a lane that moves one of them must say why). The known failures are (a) test_single_phase_remap.py::TestBoxDecayWithEOS::test_remap_matches_fixed_connectivity (the conservative remap reconstructs the masses to 1 + ulp on a rebuild that reconnected nothing; measured cause in docs_temp/HANDOFF_2026-10-10.md section 2) and (b) test_case_hydrostatic.py::TestColumn3D::test_remap_arm_holds_the_3d_column (umax peak 0.15269571220608844 against the pin 0.15305813130485327; the cospherical mesh is tie-decided at 1e-15). Never fix them by loosening a tolerance: lane V-review decides (a) at the cause (skip the mass reconstruction when the rebuild did not change the edge set, then A/B and re-pin every remap run with evidence) or records it as a follow-up with the reason; a lane that finds the residual tie behind (b) may fix it at the cause and say so. If any other count moves, say which test and why.
8. Use validation by integrated comparisons (ddgclib.analytical), never point-wise pressure comparisons. Stay in the Lagrangian formalism.
9. Documentation when done: a lane log docs_temp/debug_session/lane<ID>-<slug>.md (what changed and where, measurements with exact numbers and the SolverMethods configuration behind each, measured DO-NOTs, known limits, how to reproduce); a new entry at the TOP of the status log in debugging_plan.md; evidence and status in ddgclib/methods/_axes.py; then ${PY} -m ddgclib.methods --update METHODS.md and edit the case matrix rows in section 4 of METHODS.md by hand. Never use an em dash in prose.
10. If part of the brief turns out wrong or infeasible, do everything that is achievable and state exactly what is left and why. Report honestly: failing tests with their output, unverified claims marked as unverified. State numbers from process-dependent runs as ranges over several fresh processes.
`

const LANES = [
  {
    id: 'V-review', phase: 'V-review impenetrability and interface vote', reviewOnly: true,
    brief: `
LANE V-review: independent review of the committed lane V (impenetrability and the interface vote), whose reviewer never ran in the cloud session (the implementer hit a usage limit), with every headline number re-measured on this machine.

Lane V is the commit 08979e1 (ENH: WIP, NOT REVIEWED: library wall clamp, simplex vote and ledger-aware merge behind three axes (lane V)), merged into master on 2026-10-10 (merge commit 556d57d), plus the static-review fixes of the merge session recorded in docs_temp/HANDOFF_2026-10-10.md (section "Lane V static review"; the commit named there, if any, is part of the lane). Read \`git show --stat 08979e1\`, \`git show 08979e1\` and the fix commit for the full diff (there is no uncommitted lane diff: lane_diff.sh prints nothing until a fix is made), and docs_temp/debug_session/laneV-impenetrability-and-vote.md for its claims. The static review already checked the code for correctness bugs, cannot-fail tests, registry and documentation consistency without running a simulation; do not repeat what its section in the hand-off marks as verified, but do re-run the three new test files.

Headline claims to re-measure (section 10 of the lane log has every command; outputs go to ${SCRATCH}/laneV-review/, pass --out): (1) wall_clamp='project' is the identity on every pinned run: refinement 3 dam break at alpha 0.3 / 0.2 / 0.1 bit-identical to .replace(wall_clamp=None) with 0 put-backs (compare digests of the two arms on this machine; the cloud digests differ from this machine's, lane log section 8); (2) the refinement 4 dam break at alpha 0.3 completes its 3170 steps on the preset (gap 0.1 of the wall spacing) and ends at step 2892 with an air vertex under the floor on .replace(wall_clamp=None) (about 16 min each); (3) laneL's HP2D configuration (diagnose_frozen_set.py hp2d --L 15 --dt 0.01 --steps 3000 --arm membership) goes from 54 vertices outside the walls to 0 and its profile l2 against the analytical Poiseuille profile from 0.554 to 0.399; (4) the shipped HP2D developing pin (test_case_hagen_poiseuille.py) and the electrolysis digests (test_case_electrolysis_bubble.py) are unchanged by the preset flips; (5) simplex_vote='mass_fraction' with the preset's neighbour_count split ejects at step 0, and with split_method='simplex' + merge_method='mass_conserving' (merge_cdist 2.5e-3) completes alpha 0.2 with the tongue kept (front 0.0772, KE_liq peak 4.19e-3) but is measured worse at alpha 0.1; (6) merge_method='mass_conserving' never fires at refinement 3 alpha 0.3 / 0.2 and removes the refinement 4 small-cell spike (|u|max 2.59 -> 0.51); (7) the three electrolysis static-bubble arms of section 2.2 (preset bit-identical with the vote; simplex + vote collapses the bubble). Accuracy: the Poiseuille profile l2 and the Laplace jump of the static bubble are the analytical comparisons; state what you measured. Check the rules: no tolerance loosened, no assertion deleted, every new switch registered and recorded in methods.json, METHODS.md current (the drift test in ddgclib/tests/test_methods.py), no em dash in new prose, the dynCA files untouched.

If something is wrong, the fix step repairs it at the cause and the commit step commits it as lane V-review. If everything holds, the lane produces no commit; say so. Either way the lane log gets a section "Review 2026-10 (workstation)" with the re-measured numbers of this machine (the fix step writes it; if there is nothing to fix, the reviewer's structured report is copied into the hand-off by the session owner).
`,
  },
  {
    id: 'D', phase: 'D oscillating droplet with every new method',
    brief: `
LANE D: try every new method on the oscillating droplet, 2D and 3D, and adopt what wins.

The owner asked for this explicitly: 'when you finish all lanes please also try your newest methods/fixes on the oscillating droplet too'. The droplet is the campaign's main benchmark. Evidence: cases_dynamic/oscillating_droplet/ (runners oscillating_droplet_2D.py / _3D.py on PRESETS through _POLICY_PRESETS, src/_setup.py:setup_oscillating_droplet(methods=), src/_metrics.py, the baselines with their methods blocks, diagnose_* drivers of lanes J, M, Q (diagnose_3d_edge_area_source.py has an ARMS dict of preset.replace(...) keywords and a 'dynamic' task that scores arms: reuse its pattern, do not duplicate it) and the box_shift driver of lane B), debugging_plan.md status entries of July (lanes C, D, E, G, H: the exact two-fluid reference, the conservative remap, the 2D over-decay attributed to the every-step projection, the 3D bump / over-decay cancellation, the redistribution lever) and of October (lanes R, S, T, O, B, Q, M, W, F, G, I, X, V), METHODS.md presets oscillating_droplet_2D / _2D_dual_only / _2D_bare_delaunay / _2D_projection2 / _3D / _3D_delaunay and the flip rule of the protocol.

Current pins (lane B, full outer mesh): 2D preset oscillating_droplet_2D l2 0.17439096487276182 / tail 0.9998871416222597; projection2 arm l2 0.0388 / tail 1.394; 3D preset oscillating_droplet_3D (dual_only) l2 0.24811443136179492 / tail 0.0841737962816189 / R_max_peak 0.010790237633926668; 3D delaunay l2 1.529 (measured worse). Lane Q's 3D exact-face arm: l2 0.28653 / tail 0.08825 (cancellation exposed, not adopted).

Tasks.
1. Build the arm list from the registry: every explicit axis value that applies to the multiphase droplet and that was added or changed since 2026-09-25, each as preset.replace(...) on the 2D and the 3D preset: frozen_set='membership'; edge_area_source='p_ij_simplex' (3D); area_orientation (default 'primal_edge' now, 'dual_midpoint' legacy as the control); remap='conservative' in 3D now that lanes O, Q and B changed what it reads (the July rejection was measured on the lossy mesh with the fan cache); projection_every in {2, 5} with and without the 3D remap; redistribute_mass=False; phase_ledger and face_closure arms (the droplet runs at P0 = 0 where lane F measured them inert: confirm); split_method='simplex'; curvature_path='csf_dual' as the measured-worse control; connectivity='delaunay_material' if it applies to a closed box (say so if not); lane V's three axes: wall_clamp='project' (setup_oscillating_droplet does not build a WallClampBC yet: wire methods.wall_clamp_bc(box=...) into it the way cases_dynamic/dam_break/src/_setup.py does, with the clamp gap from wall_vertex_spacing; on the closed box it should be the identity: confirm 0 put-backs and a bit-identical score, and if it is not the identity say where a vertex left the box), simplex_vote='mass_fraction' (only together with split_method='simplex', the lane V DO-NOT; with and without merge_method='mass_conserving' in 2D, where merge_cdist must be set to about 0.1 of the mesh spacing because the preset cannot carry it; the merge does not apply to the 3D dual_only preset: say so), merge_method='mass_conserving' alone in 2D; and the lane X observation that the first Delaunay rebuild halves the interface flag count of a square droplet (trace the interface flag count over the first rebuilds of the round droplet in 2D and 3D and say whether it happens there). Then the best combinations (at most eight), chosen from the single-axis results.
2. Score every arm with the case's own score (l2 against the two-fluid reference, tail growth, linf, mass drift, R_max peak, KE max, the quarter-mean error channels of lane G of July) and record methods.json next to each score, under ${SCRATCH}/laneD/<arm>/ with a results table in the lane log. Run the 3D arms in parallel processes, at most ${MAX_PROCS} at once (deterministic since lane T; a full 3D run is about 8 min on the owner's dev machine: measure one first and budget from it). Run the full horizons: this machine has the compute; only if the whole arm list would exceed about 6 hours of wall time, run the combination arms in the shortest mode that still scores and list the full runs as reproduce commands.
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
4. Check the rules: no pinned number changed without evidence; no loosened tolerance or deleted assertion; new switches registered in _axes.py with a SolverMethods field; no duplicate case-local code where the library could do it; METHODS.md current (the drift test in ddgclib/tests/test_methods.py passes); lane log and debugging_plan.md entry written; no em dashes in new prose; off-limits directories neither edited nor executed; no stray probe outputs left in the repositories (compare git status --short of both repos with the lane's file list and the owner's pre-existing uncommitted work listed in docs_temp/HANDOFF_2026-10-10.md).
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
  return `YOU ARE THE COMMIT STEP for lane ${lane.id} of the ddgclib campaign. The lane passed independent review with green tests. The owner's standing rule is to ship verified work in both repositories (${ROOT} on ${BRANCH} and ${HYP} on master). Do not edit any file. Do not run tests.

The implementer's file list and summary:
${JSON.stringify({ files_changed: impl.files_changed, lane_log: impl.lane_log, pins_changed: impl.pins_changed, summary: impl.summary }, null, 1)}

Steps.
1. cd ${ROOT} && bash docs_temp/workflows/lane_diff.sh --stat shows the lane's files in both repos. Cross-check with the list above. If there is nothing to commit (a review-only lane that found nothing wrong), report empty shas and stop.
2. First run git diff --cached --stat in both repos: the index must be empty. If it is not (a stale staged snapshot from an interrupted step), do NOT commit it: run git reset -q (index only, the working tree is untouched) and check that git diff --cached is now empty. Then stage with explicit paths only: git add <path> <path> ... (this stages the working-tree content). Never use git add -A, -u, . or a directory that contains anything outside the lane. After staging, git diff --cached --stat must list exactly the lane's files and no deletion you did not intend. Baselines under cases_dynamic/oscillating_droplet/baselines are tracked: include them when they changed. The workflow script docs_temp/workflows/ddgclib-lanes-5.js is committed by the session owner, not by you.
   NEVER stage: .gitignore; anything under cases_mean_flow/, benchmarks/, tutorials/, test_cases/, cases_dynamic/capillary_rise_energy_grad/; the dynCA runners, src/_setup_dynca.py and README_dynCA.md under cases_dynamic/capillary_rise/; ddgclib/tests/test_integrated_validation.py; generated outputs (results/, fig/, *.log, *.pkl, *.mp4, *.png, pytest-of-*, *.run.xml); anything untracked that is not a source, test or documentation file of this lane.
3. Commit hyperct first (if it changed; branch must be master), then ddgclib (branch must be ${BRANCH}; check git branch --show-current; if it is anything else, stop and report). Message: first line with the repo's prefix convention ENH: / BUG: / MAINT:, a plain one-line summary, ending with (lane ${lane.id}); a body of 5 to 15 lines saying what changed, the headline measured numbers, and every pin that moved as old -> new; no em dashes; then a blank line and exactly this last line:
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
4. No push, no amend, no rebase, no branch, no tag, no checkout/restore/stash, no other git command that changes history or the working tree (git reset -q without paths or arguments, to clear a stale index, is the one allowed exception).
5. Run git status --short in both repos and report the two commit hashes and any lane file you left unstaged with the reason.`
}

const results = []
for (const lane of LANES) {
  phase(lane.phase)
  let impl
  if (lane.reviewOnly) {
    impl = { summary: 'Committed lane V (08979e1, merged 556d57d) plus the static-review fixes of 2026-10-10 (docs_temp/HANDOFF_2026-10-10.md); never independently reviewed with its numbers re-measured; see docs_temp/debug_session/laneV-impenetrability-and-vote.md', lane_log: 'docs_temp/debug_session/laneV-impenetrability-and-vote.md', files_changed: ['see git show --stat 08979e1 and the fix commit named in docs_temp/HANDOFF_2026-10-10.md'], pins_changed: [], accuracy: [], tests: { fast: 'see docs_temp/HANDOFF_2026-10-10.md', slow: 'see docs_temp/HANDOFF_2026-10-10.md', hyperct: 'unchanged by lane V' }, case_status: [], not_done: [], followups: [] }
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
