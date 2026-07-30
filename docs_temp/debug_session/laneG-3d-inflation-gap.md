# laneG-3d-inflation-gap — close the 3D droplet-inflation physics gap, re-decide the 3D policy

Date: 2026-07-30.  Lane G of the wf6 session.  Task: diagnose the smooth
~0.8%-of-R0 3D droplet inflation behind the dual_only summary 0.24811
(laneB/laneE next-prompt option 3), fix it, re-score, then re-run the 3D
remap A/B and re-decide `retopo_policy_3d`.

Drivers + raw JSONs in scratchpad `wf6/laneG/`
(`probe_discrete_balance.py`, `dyn_probe.py`, `probe_closure.py`,
`probe_variational.py`; results `probe_discrete_balance.json`,
`dyn_*.json`, `probe_variational.json`, `*.log`).  The dynamic driver
was validated by bit-exact reproduction of the pinned 3D baseline
before any new measurement (full 872-step dual_only refine 2/2 replica:
summary 0.24811340819647862, R_max_peak 0.010790236105250779, mass
1.905002320272536e-14 — all identical to
`baselines/baseline_oscillation_3d.json`).

## Verdict up front

**DIAGNOSIS CLOSED, NO DEFAULT FLIPPED, BASELINE UNCHANGED (summary
0.24811340819647862 stands).**  The "smooth ~0.8% droplet inflation"
is now fully attributed and every candidate fix in the lane brief was
measured; all fail the flip rules, several with hard negative results
that future lanes must not retry:

1. The inflation is NOT a volume mode and NOT a preload-scalar
   mismatch: it is the cube-symmetry SHAPE mode of the cube-sphere
   interface (the 6 valence-8 face-center vertices ARE R_max; volume
   conserved to 3e-4) relaxing toward the discrete equilibrium shape,
   plus a per-step mass-redistribution pump along a marginal energy
   valley.
2. Suspect (ii) (YL preload discretisation): REAL in sign (discrete
   LSQ jump 9.1313 vs analytic 10.0 -> measured net outward residual,
   and max|F_net| = the pinned 3D floor 6.0153e-05) but measured a
   NO-OP as a fix — the l=0 preload error self-corrects through the
   EOS volume constraint (trajectories identical to 0.03% R0).
3. Suspect (i) (operator truncation): the task's O(h^2) prediction
   FAILS — the bump is NON-CONVERGENT under droplet refinement
   (+9.1% / +2.0% / +4.1% of R0 at droplet refine 1/2/3, matched
   time), outer refinement is a no-op at the interface, and at refine
   2/3 the discrete energy is PUMPED monotonically (+8.1e-8 J over
   0.095 s) — the "look deeper" branch fired.
4. The pump is the per-step mass redistribution running under
   dual_only (where its reason to exist — Delaunay reconnection —
   is absent): switching it off kills the R_max overshoot entirely
   (R_max_peak 0.010790 -> 0.0105 exact), makes mass drift exactly
   0.0 and tail_growth 0.0101, BUT the pinned l2 gets WORSE (0.26636
   vs 0.24811) because the pinned baseline's l2 is a CANCELLATION of
   the outward bump artifact against a genuine early over-decay (the
   3D sibling of the known 2D solver-side over-decay, laneC).  By the
   better-l2 flip rule and the <=5% regression rule this cannot be
   adopted.
5. Remap A/B re-decision (task step 4): the fix precondition
   ("inflation gap closed") was NOT met, so per laneE's DO-NOT the
   3D remap stays rejected; `retopo_policy_3d = 'dual_only'` and the
   3D baseline stay as pinned.  The scored refine-2/3 legs (below)
   document that both redistribution-bearing configs degrade at
   refine 3 exactly as the eps=0 probes predict.

The real lever, out of scope for a minimal-diff lane: the interface
TRIANGULATION family (cube-sphere projection with 12 valence-4 +
6 valence-8 scale-invariant special vertices).  An isotropic
(icosphere-class) interface mesh is the recorded follow-up, together
with the redistribution-projection rework it shares with the 2D
over-decay suspect list.

## 1. Discrimination: (i) operator truncation vs (ii) preload mismatch

### 1.1 Frozen-droplet discrete force balance (probe 1a)

Frozen epsilon=0 3D droplet, refine 2/2 (98 interface vertices), state
exactly as `setup_oscillating_droplet` returns.  Per interface vertex:
discrete surface-tension force F_st (production 'integrated'
cotan-Heron path), inner-phase dual-face area sum S_in (what a uniform
pressure jump pushes against), implied discrete-consistent jump
dp_i* = (F_st.S_in)/|S_in|^2; plus the ACTUAL t=0 net force under the
shipped analytic preload (2*gamma/R0 = 10 Pa).

| quantity | eps=0 | eps=0.05 (case IC) |
|---|---|---|
| applied preload jump (p_in - p_out) | 10.000000000000108 +- 8.2e-14 | same |
| discrete-consistent dp* (area-weighted LSQ) | **9.131337861740798** (-8.7%) | 9.098917669713172 |
| dp* net-radial-force-neutral | 9.757414187883283 | 9.717693564996548 |
| dp_i* min / mean / max | 7.689 / 10.860 / **23.280** | 7.138 / 10.817 / 24.390 |
| dp_i* by interface valence (n=12 val-4 / 80 val-6 / 6 val-8) | **23.280 / 9.216 / 7.946** | 23.243 / 9.174 / 7.876 |
| net radial force per iface vertex (mean, + = outward) | **+3.075e-06 N** (sum +3.01e-4) | +3.533e-06 N |
| max \|F_net\| | 6.0153201139652905e-05 = the pinned 3D floor step0 | 7.0014e-05 |
| 'stokes' path (integrated_hndA_i_interface) vs 'integrated' | identical to 1e-14 | identical |

Reading: suspect (ii) is REAL in sign — the analytic preload
over-pressurizes the droplet by 0.87 Pa in the l=0 projection, net
outward residual, and the pinned 3D floor force (6.0153e-05) IS this
imbalance.  But the per-vertex spread is 2.5x the mean shift and
valence-structured: the cube-sphere projection's 12 valence-4
interface vertices carry an effective local jump of 23.3 Pa (2.3x),
the 6 valence-8 face-center vertices 7.9 Pa.  The 'stokes'
boundary-integral path is numerically identical to the cotan path on
this mesh — consistent with the earlier probe that closed stencil
variants for the integrated operator; no stencil lever exists.

### 1.2 The scalar discrete-consistent preload does NOT move the outcome

Frozen eps=0 droplet free-run under dual_only (250 steps = t 0.056,
1/3 horizon), shipped analytic preload vs re-preloaded with the
discrete LSQ jump:

| preload | R_max_if(t=0.056) | R_mean_if | R_min_if | droplet dual vol drift |
|---|---|---|---|---|
| analytic 10.0 | 0.010198877615 (+2.0% R0) | 0.009948808746 (-0.5%) | 0.009518373446 (-4.8%) | +0.033% |
| discrete LSQ 9.1313 | 0.010196316901 | 0.009946334009 | 0.009515537906 | -0.024% |

The trajectories are the same to 0.03% of R0.  **Suspect (ii) is
rejected as the driver**: the l=0 preload mismatch self-corrects
through the EOS/redistribution volume constraint (droplet dual volume
stays constant to 3e-4), while the shape keeps deforming.  The
inflation is NOT a volume mode — it is a volume-neutral SHAPE mode:
R_max rides the outward bumps while R_min dives (valence-4 vertices
pulled in at 5x the R_max rate) and the mean barely moves.

### 1.2b The pump: per-step mass redistribution under dual_only

The eps=0 relaxed state has HIGHER discrete energy than the on-sphere
start (par 1.4: dE = +1.0e-9 J while KE dissipates) — the motion runs
along a marginal valley (gamma*dA = dp*dV to 0.4%), so any small
NON-variational per-step force can drive it.  The suspect with a
per-step footprint is `redistribute_mass=True` (the setup default,
introduced for per-step-Delaunay reconnection artifacts, A.5.b):
under dual_only there IS no reconnection, yet every step the per-phase
mass is re-targeted to reproduce the PRE-step pressure field — erasing
the local EOS pressure response to bump growth (a local outward bump
no longer feels a local pressure penalty, only the global scale).

Measured (eps=0, refine 2/2, dual_only, matched t=0.0563):

| redistribute_mass | R_max_if | R_mean_if | R_min_if | mass drift |
|---|---|---|---|---|
| True (shipped) | 0.010198877615 (+2.0%) | 0.009948808746 | 0.009518373446 (-4.8%) | 1.1e-14 |
| **False** | **0.010109546642 (+1.1%)** | 0.009964586773 | 0.009627495144 (-3.7%) | **0.0 exact** |

Turning the redistribution OFF halves the bump at matched time (and
KE is still relaxing — the equilibrium offset shrinks further); the
Lagrangian mass ledger becomes exact.  The remaining bump is the
intrinsic PL special-vertex part (par 1.4b).

### 1.3 Refinement scaling of the bump (probe 1b, matched time t=0.0563)

eps=0 relaxed-bump amplitude (R_max_if - R0)/R0 at matched physical
time, dual_only, analytic preload:

| refine (outer/droplet) | n_iface | bump (t=0.0563) | ratio |
|---|---|---|---|
| 2/1 | 26 | **+9.1%** | 4.6x vs 2/2 |
| 2/2 | 98 | **+2.0%** | — |
| 2/3 | 386 | **+4.1% and still growing** | NON-CONVERGENT |
| 3/3 | 386 iface | static balance identical to 2/3 (lsq 9.4459 vs 9.4444; val-4 23.848 / val-8 6.073 both) | outer refinement is a no-op at the interface |

**The task brief's O(h^2)-truncation prediction FAILS at 2/3**: the
bump does not shrink 4x — it doubles, and turns into a slower
face-scale mode (the "look deeper" branch fires).  Mode shape at 2/2
(relaxed state, cube-symmetry pattern): R_max = exactly the 6
valence-8 cube-face centers (+2.0%, all six equal to 1e-9), R_min =
the 12 valence-4 edge-midpoints (-4.8%), the 8 corner-direction
vertices +1.8%, the ring at 20-30 deg from face centers -2%.  The
val-8 face-center jump deficit WORSENS with droplet refinement
(volumetric dp* 7.95 -> 6.07; the val-6 ring flips sign 9.22 ->
10.61), i.e. the volumetric-dual pressure measure at the interface is
not converging to the interface geometry the curvature operator
integrates over.

The discrete LSQ jump deficit scales 44% (2/1) -> 8.7% (2/2, ratio
5.1x) -> 5.6% (2/3, ratio only 1.56x) — the residual deficit at finer
refinement is carried almost entirely by 18 SCALE-INVARIANT special
vertices of the cube-sphere projection (12 cube-edge-midpoint
valence-4 vertices: dp* 23.28 at 2/2 and 23.85 at 2/3; 6 face-center
valence-8: 7.95 / 6.07), while the regular valence-6 ring converges
(9.22 -> 10.60 around 10).  The special-vertex mismatch is O(1) in Pa
but acts on an O(h^2) area share, so the induced displacement bump
still shrinks ~4x per level (confirmed dynamically below).

### 1.4 Variational probe: the bump is intrinsic to the discrete energy

The 'integrated' cotan operator is the exact PL-area gradient, so if
the coupled discretization is variationally consistent the eps=0
relaxation must run downhill in E = gamma*A_PL - dp*V.  Measured over
450 steps (refine 2/2): A_PL 1.2120330042e-03 -> 1.2064264459e-03
(dA = -5.6066e-06, -0.46%), V_enc(PL) 3.9061358820e-06 ->
3.8780038110e-06 (dV = -2.8132e-08), and gamma*dA = 2.803e-07 J vs
dp*dV = 2.813e-07 J — **equal to 0.4%**: the relaxation slides along a
near-marginal valley of the discrete energy toward the bumpy shape
(E net change +1e-9 J, ~5e-5 relative).  The on-sphere state is not
the discrete equilibrium; the bumpy shape is — this is genuine
truncation of the coarse polyhedral interface (12 valence-4 / 6
valence-8 cube-sphere vertices), not a force-side inconsistency bug.

### 1.4b No consistent pairing puts the coarse sphere in equilibrium

Sharper test of the "consistency between the pressure side and the
integrated operator" lever: pair the cotan operator (the exact
PL-area gradient) with the exact PL-VOLUME gradient
``gradV_i = (1/6) sum_{tri in star(i)} (x_j x x_k)`` — the most
consistent discrete pressure measure possible.  Measured on the
on-sphere eps=0 config:

- **Direction: exactly consistent already** — cos(F_st, gradV_i) =
  -1.00000000 at every interface vertex, both refinements.  There is
  no direction-mismatch defect to fix.
- Magnitude ratio dp*_PL_i = -(F_st.gradV)/|gradV|^2: valence-6 ring
  converges cleanly to the analytic jump (10.154 +- 0.76 at 2/2 ->
  10.049 +- 0.31 at 2/3) but the special vertices are scale-invariant
  (valence-4: 15.0055 / 15.0003; valence-8: 8.196 / 7.715).

So even the perfectly consistent PL pairing leaves +-50% per-vertex
imbalance at the 18 special vertices: the inscribed polyhedron
GENUINELY concentrates discrete curvature at the cube-edge ridge
vertices and depletes it at the face centers.  The bumpy equilibrium
is intrinsic to the coarse cube-sphere interface; no preload or
pressure-side pairing can hold the coarse sphere still.  The lever
that remains is interface resolution (the O(h^2) displacement
response), optionally a more isotropic interface triangulation
(icosphere-class) as a future mesh-quality lane.

### 1.5 Dual-face closure defect (recorded, secondary)

Interior dual cells must satisfy sum_j A_ij = 0.  Measured (refine
2/2): bulk droplet vertices close to 5.8e-17 (machine), but interface
vertices leave |sum A_ij| up to 2.3% of sum|A_ij| (up to 10.7% of
|S_in|; eps=0.05: up to 6.4% / 24.6%) and bulk outer up to 2.6% — the
`_dual_area_vector_3d_p_ij` ring-walk/face-matching degrades on the
distorted interface-adjacent tets.  This contaminates the pressure
side at the few-percent level per vertex — an order of magnitude
smaller than the valence-driven dp* spread (+-50-130%), so it is NOT
the driver; recorded as a future hygiene item.

## 2. Full-horizon decomposition: the pinned l2 is a cancellation

Full 872-step scored runs at refine 2/2, dual_only, quarter-horizon
error means (normalized by eps*R0; + = above envelope):

| config | q1 | q2 | q3 | q4 | l2 | tail | mass | R_max_peak |
|---|---|---|---|---|---|---|---|---|
| redistribute ON (pinned baseline) | **+0.319** | +0.262 | -0.056 | -0.139 | **0.24811340819647862** | 0.08410 | 1.9e-14 | 0.010790 (overshoot) |
| redistribute OFF | **-0.380** | -0.210 | -0.208 | -0.122 | **0.26636475750102395** | **0.01009** | **0.0 exact** | **0.0105 (no overshoot)** |

Reading: the shipped baseline's l2 is the sum of TWO opposite-signed
errors — the redistribution-pumped outward bump (+0.32 early) partially
cancels a genuine early over-decay (-0.38 in the clean config; the 3D
sibling of the known 2D solver-side over-decay, laneC).  Removing the
pump exposes the over-decay one-signed (worst -0.60 at t=0.025,
converging to -0.068 at t_end) and scores slightly WORSE on the pinned
l2 while every physics channel improves: KE tail 8.3x cleaner, mass
drift exactly zero, zero R_max overshoot, final-time error halved.
The early over-decay driver at 2/2: the static discrete imbalance
forces (up to 6e-5 N/vertex, par 1.1) are ~5x the physical l=2
restoring force per vertex at this resolution, and the apex R_max
vertex IS a special (face-center) vertex of the cube-sphere mesh.

### 2.1 Refinement + pump interaction (eps=0, matched t=0.0563)

| refine | redistribute ON | redistribute OFF |
|---|---|---|
| 2/2 | +2.0% (plateaued) | +1.1% |
| 2/3 | +4.1% (growing) | +3.2% (growing) |
| 3/3 | +4.07% (= 2/3 to 3 digits) | — |

Variational probe at 2/3 (933 steps): E = gamma*A_PL - dp*V rises
MONOTONICALLY after an initial dip, dE = +8.1097e-08 J over 0.095 s
with redistribution ON (~80x the 2/2 net drift) AND **+5.6629e-08 J
with redistribution OFF** (where the PL area itself INCREASES,
dA = +6.62e-07) — at refine 2/3 the dominant energy pump is NOT the
redistribution but the volumetric-dual pressure side itself (the
non-convergent val-8/val-6 S_in bias where fine inner tets meet the
coarse outer at the interface; par 1.3/1.4b/1.5).  Mode shape at 2/3
(t=0.051): the 6 face centers at +3.9% with a smooth 10-20 deg halo
at +1.1%, corners/valence-4 pulled in (-1.4% / -1.8%).

## 2b. Fix decision

Decision rules: better-l2-AND-tail for any default flip; <=5%
regression tolerance on pinned channels; WIN target summary < 0.1.

| candidate (full-horizon scored runs, 872 steps at 2/2, 2651 at 2/3) | l2 | tail | mass | R_max_peak | KE_max [J] | verdict |
|---|---|---|---|---|---|---|
| 2/2 redis dual_only (pinned) | **0.24811340819647862** | 0.08410 | 1.9e-14 | 0.010790 | 1.7432e-06 | **stands** |
| 2/2 noredis dual_only | 0.26636475750102395 | 0.01009 | 0.0 | 0.0105 | 6.5233e-06 | REJECT (l2 +7.4%) |
| 2/3 redis dual_only | 0.46645348342458526 | 0.01266 | 6.9e-14 | 0.010775 | 1.1059e-06 | REJECT (l2 +88%) |
| 2/3 noredis dual_only | 0.37204702775924850 | 0.00177 | 0.0 | 0.010544 | 8.0195e-06 | REJECT (l2 +50%; R_max ends +4.2% R0) |
| 2/3 redis delaunay_remap | (detached leg still integrating at close; JSON lands in scratchpad `wf6/laneG/dyn_full_rd3_remap.json`) | | | | | REJECT regardless (flip precondition failed; laneE DO-NOT stands) |
| scalar discrete preload (any refine) | trajectory unchanged (par 1.2) | | | | | NO-OP |

**No adoption.**  Runner reverted to refine 2/2 behavior-identical
(comment block + self-describing `refinement_*`/`retopo_policy` keys
in score.json only); `_setup.py` gains a comment at the YL preload
recording the negative scalar-preload result.  No solver, operator,
params, baseline, or hyperct changes.

## 3. Remap A/B re-decision (task step 4)

Precondition "inflation gap closed" NOT met -> per the laneE DO-NOT
("do not apply the remap in 3D until the droplet-inflation gap is
closed") the 3D remap stays rejected without a new flip evaluation;
`retopo_policy_3d = 'dual_only'` unchanged, `baseline_oscillation_3d.json`
unchanged.  The laneD par 2.6.5 bookkeeping caveat was re-observed in
passing: the remap legs run with `vol_corr` recorded per frame (the
dual_only legs hold [1.0, 1.0] exactly for the whole horizon).

## 4. Measurement battery (ddg env, repo root; changes are
comments + score self-description only, so bit-identity is the gate)

| item | result |
|---|---|
| floor battery (`test_case_oscillating_droplet.py -m ""`) | **18 passed** in 25.37s (also 18 P pre-change) — all pins untouched |
| fast suite (`-m "not slow" -q`) | **876 passed, 0 failed**, 12 skipped, 17 deselected, 2 xfailed in 70.13s (identical pre/post) |
| static_droplet_2D | summary **1.1847162859108737e-03** (interface_radius_drift same, mass_drift 0.0) — bit-identical |
| oscillating_droplet_2D | l2 **0.17479361640597058** / linf 0.32364245955409165 / tail **0.9998967874595965** / mass 2.4056717879332966e-14 — bit-identical to `baseline_oscillation.json` |
| oscillating_droplet_3D (official runner, post-edit) | l2 **0.24811340819647862** / linf 0.5992701359956998 / tail 0.08409976059818802 / mass 1.905002320272536e-14 / R_max_peak 0.010790236105250779 — **bit-identical on every numeric key** to `baseline_oscillation_3d.json`; score.json now carries `refinement_outer=2 / refinement_droplet=2 / retopo_policy='dual_only'` |
| hyperct (`pytest hyperct/tests -q`) | **336 passed**, 40 skipped, 6 xfailed, 39 errors in 3.93s — identical to laneA/laneE (pre-existing benchmark-fixture errors; no hyperct edits this lane) |

Driver validation: the probe driver reproduced the pinned 3D baseline
bit-identically (summary 0.24811340819647862, R_max_peak
0.010790236105250779, mass 1.905002320272536e-14) before any new
measurement was trusted.

## 5. Changed files

- `cases_dynamic/oscillating_droplet/oscillating_droplet_3D.py` —
  NOTE comment (do-not-refine + measured no-ops) and three
  self-describing score keys; behavior verified bit-identical.
- `cases_dynamic/oscillating_droplet/src/_setup.py` — comment at the
  YL preload recording the negative scalar-preload measurement.
- `docs_temp/debug_session/laneG-3d-inflation-gap.md` (this log);
  dated entry at the top of `debugging_plan.md`'s Status log.
- No solver, operator, params, baseline, test, or hyperct changes.

## 6. Guidance for future lanes (measured, binding)

1. **Do not flip the 3D droplet runner to refinement_droplet=3** (or
   outer 3) expecting the O(h^2) shrink — the bump is non-convergent
   on the cube-sphere family and the redistribution pump grows with
   refinement.
2. **Do not re-try scalar YL preloads** (LSQ / radial-neutral /
   analytic variants) against the 3D shape drift — measured
   trajectory-neutral to 0.03% R0.
3. **Do not read the 3D l2 0.24811 as a pure inflation gap** — it is
   bump(+) cancelling over-decay(−); a change that removes the bump
   can RAISE l2 (measured: redistribution OFF gives l2 0.26636 with
   every physics channel cleaner: tail 0.0101, mass 0.0, no
   overshoot).  Score-side conclusions need the sign decomposition.
4. The real levers, in order: (a) isotropic interface triangulation
   (icosphere-class droplet surface — removes the 18 scale-invariant
   special vertices; would re-pin the 3D floors, a full lane of its
   own); (b) the per-step mass-redistribution projection rework
   (shared suspect with the 2D over-decay, lane5/laneC) — under
   dual_only it is a pump with no reconnection to fix (it carries
   ~half the 2/2 bump), but switching it off must be co-evaluated
   with the l2 cancellation above, and at 2/3 the DOMINANT pump is
   the volumetric-dual pressure side itself (dE +5.66e-8 J with
   redistribution OFF), so (b) alone cannot fix refine 3.
5. The dual-face closure defect (par 1.5, up to 2.6% non-closure at
   interface/outer vertices; bulk droplet machine-clean) and the
   50/50 interface-edge `edge_phase_area_fractions` heuristic (whose
   bias flips sign with refinement: val-6 dp* 9.22 at 2/2 -> 10.61 at
   2/3 while the exact-PL pairing converges 10.15 -> 10.05) are the
   concrete hygiene targets on the pressure side if (b) is attempted.
