# Audit: DynamicSimulation dim/HC kwarg collision and dead `params.rho`

> Sources checked | Written 2026-07-02 by physics-audit workflow

**Item key:** `simulation-dim-collision`
**Files:** `/home/endres/projects/ddgclib/ddgclib/dynamic_integrators/_simulation.py`,
`/home/endres/projects/ddgclib/ddgclib/dynamic_integrators/_integrators_dynamic.py`
**Sources read:** `docs_temp/code_map/integrators_and_bcs.md` (:64-66, :100-106, :172),
`docs_temp/02_physics_foundations.md` (:36, :146), `docs_temp/04_solver_pipeline.md` (:96)

## What the physics/API contract requires

The acceleration function (`dudt_fn`, canonically `stress.dudt_i` /
`stress_acceleration`) must be evaluated with the simulation's spatial
dimension and mesh: `dudt_fn(v, dim=<sim dim>, mu=..., HC=<mesh>)`
(`docs_temp/02_physics_foundations.md:36`). Evaluating it with the wrong
`dim` changes the physics operator (3D dual-face flux sums on a 2D mesh);
evaluating with `HC=None` breaks dual-mesh access entirely.

`SimulationParams.dudt_kwargs` explicitly promises delivery to `dudt_fn`:

- `_simulation.py:61-65` — docstring `"""Build keyword arguments for dudt_fn
  from params."""`; returns `{'dim': self.dim, 'mu': self.mu, **extra}`.
- `_simulation.py:11-18` — class usage example instructs
  `sim.set_acceleration_fn(acceleration)` with the **raw** (un-partialed)
  `ddgclib.operators.gradient.acceleration`.
- `SimulationParams.rho` is declared at `_simulation.py:56` and documented
  at `:43-44` ("Density.").

## What the code actually does

`DynamicSimulation.run()` (`_simulation.py:158-170`) builds one flat kwargs
dict containing `'dim': p.dim` (`:165`) and then
`integrator_kwargs.update(p.dudt_kwargs)` (`:170`) before calling
`self._integrator(**integrator_kwargs)` (`:182`).

Every dynamic integrator has `dim` as a **named parameter**
(`_integrators_dynamic.py:652`, `:756`, `:846`, `:991`, `:1068`), so Python
binds `dim` to the integrator itself; only the leftovers (`mu` + non-colliding
`extra` keys) land in the integrator's `**dudt_kwargs`, which is what actually
reaches `dudt_fn` via `_compute_accel(dudt_fn, verts, workers, **dudt_kwargs)`
(`_integrators_dynamic.py:543`, `:575`, `:588`). Consequences:

1. `dudt_fn` never receives `dim` — it silently falls back to its own default,
   which is `dim=3` for `gradient.acceleration` (`gradient.py:104`) and
   `stress_acceleration`/`dudt_i` (`stress.py:778-780`, alias `:829`).
2. `dudt_fn` never receives `HC` either — `HC` is also a named integrator
   parameter; supplying `extra={'HC': HC}` does **not** raise (the dict
   `.update()` merely overwrites the integrator's `HC` key) — it is silently
   swallowed too.
3. A dim-3 fallback acceleration does not even crash downstream: the velocity
   update truncates it, `v.u[:dim] += dt * a[:dim]`
   (`_integrators_dynamic.py:1055-1056`).
4. `params.rho` appears nowhere outside its declaration (`grep rho
   _simulation.py` → only `:43`, `:56`) — defined but never forwarded.

## Probe

Script: `/tmp/claude-1000/-home-endres-projects-ddgclib/1b66bdb7-f777-4a6f-a12c-a369d7b87764/scratchpad/audit/simulation-dim-collision/probe_dim_collision.py`
Run with `/home/endres/anaconda3/envs/ddg/bin/python` (env `ddg`,
`PYTHONPATH=/home/endres/projects/ddgclib`).

Design: tiny 2D `rectangle(L=1, h=1, refinement=1)` mesh; spy
`def spy(v, dim=3, mu=None, HC=None, rho=None, **kw)` records what arrives;
`SimulationParams(dt=1e-4, n_steps=1, dim=2, mu=0.123, rho=999.0)`;
integrator `euler_velocity_only`. Plus: raw `stress.dudt_i` run, the
`extra={'HC': HC}` variant, and a `functools.partial` control.

### Probe OUTPUT (verbatim)

```
PART 1: spy dudt_fn via DynamicSimulation(dim=2, mu=0.123, rho=999)
params.dim = 2, params.mu = 0.123, params.rho = 999.0
params.dudt_kwargs = {'dim': 2, 'mu': 0.123}
run() completed, t = 0.0001
Spy call count: 5
Spy received: dim=3  mu=0.123  HC_is_None=True  rho=None  extra_kwargs=[]
--> dim requested via SimulationParams : 2
--> dim actually received by dudt_fn   : 3
--> rho requested via SimulationParams : 999.0
--> rho actually received by dudt_fn   : None
*** BUG CONFIRMED: dim swallowed by integrator, dudt_fn fell back to its own default (3) ***
*** CONFIRMED: params.rho never forwarded ***

PART 2: raw stress dudt_i via DynamicSimulation (the documented API)
run() RAISED: AttributeError: 'NoneType' object has no attribute 'Vd'
      File ".../hyperct/ddg/_operators.py", line 58, in e_star
        vc_12 = HC.Vd[tuple(vc_12)]

PART 3: extra={'HC': HC} (attempt to supply HC via params.extra)
run() completed, t = 0.0001        <- silently swallowed, no TypeError

PART 4: control — functools.partial binding (CLAUDE.md pattern)
Spy received via partial: dim=2  mu=0.123  HC_is_None=False  rho=999.0
```

So: the sim asked for `dim=2`, the acceleration function got `dim=3`,
`HC=None`, no `rho` — and `run()` finished with no warning. The class's own
docstring recipe (raw `acceleration` / `dudt_i`) crashes with
`AttributeError: 'NoneType' object has no attribute 'Vd'`. The `partial`
control confirms the documented workaround delivers everything correctly.

### Independent verification re-run (same day, second pass)

A second, independently written probe
(`.../scratchpad/audit/simulation-dim-collision/probe_dim_collision.py`,
run with `PYTHONPATH=/home/endres/projects/ddgclib
/home/endres/anaconda3/envs/ddg/bin/python`) reproduces all findings and
adds one **new hazard**:

```
PROBE A: spy dudt_fn inside DynamicSimulation(dim=2, mu=0.123, rho=777)
params.dudt_kwargs = {'dim': 2, 'mu': 0.123}
spy received: dim=3  mu=0.123  HC_is_None=True  rho=None  extra_keys=[]
--> params.dim was 2; dudt_fn got dim=3  (FALLBACK TO DEFAULT dim=3 -- BUG CONFIRMED)
--> params.rho was 777.0; dudt_fn got rho=None  (NEVER FORWARDED)

PROBE B: partial-bound dudt_fn (canonical CLAUDE.md pattern)
partial bound (dim=2, mu=0.5); spy received dim=2 mu=0.00089
--> dim survives via partial: True; partial's mu=0.5 SILENTLY OVERRIDDEN by params.mu=8.9e-4

PROBE C: extra={'HC': HC} -- does HC reach dudt_fn?
params.dudt_kwargs keys = ['HC', 'dim', 'mu']
spy received: dim=3  HC_is_None=True
--> extra['HC'] is ALSO swallowed by the integrator's own HC parameter

PROBE D: documented usage -- set_acceleration_fn(acceleration) raw
RAISED AttributeError: 'NoneType' object has no attribute 'Vd'
```

**New hazard (PROBE B):** the "safe" `functools.partial` workaround is only
half-safe inside `DynamicSimulation`. `partial`-bound `dim`/`HC` survive
(the colliding kwargs are absorbed by the integrator's named parameters),
but `mu` is **not** an integrator parameter, so `SimulationParams.mu` always
reaches `dudt_fn` through `**dudt_kwargs` — and call-time keywords override
`partial`-bound ones. A user who binds `mu=0.5` via `partial` but leaves
`SimulationParams.mu` at its default gets viscosity `8.9e-4` with no
warning. The same silent override applies to any `extra` key that the user
also bound via `partial`.

Corroborating test run: `pytest ddgclib/tests/test_dynamic_integrators.py
-k "SimulationParams or DynamicSimulation" -q` → `8 passed` (all masked by
matching defaults, as analysed above).

## Usage survey (severity input)

`grep -rn DynamicSimulation` over the repo:

- **No production physics case uses it.** All of `cases_dynamic/` (including
  `oscillating_droplet/`, `oscillating_droplet_p_ref/`,
  `electrolysis_bubble/`, `shearing_plate_droplet/`) call integrators
  directly with `functools.partial`-bound `dudt_fn` — zero hits outside
  `cases_dynamic/template/`.
- `cases_dynamic/template/example_features_demo.py:243-261` — live demo, but
  `channel_accel(v, dim=2, ...)` (`:147`) happens to default to `dim=2`,
  masking the bug by coincidence.
- `cases_dynamic/template/template.py:176-186` — Option B, commented out;
  its `dudt_fn(v, dim=2, mu=0.1, **kwargs)` (`:147`) also masks by default.
- `ddgclib/tests/test_dynamic_integrators.py:284-401` — all
  `TestDynamicSimulation` tests use `dim=1` params with accel fns whose
  default is `dim=1` (`zero_accel`/`constant_accel` `:50-58`) — masked.
  `TestSimulationParams.test_custom` (`:297-302`) even asserts `'dim'` and
  `'HC'` are in `dudt_kwargs`, codifying the delivery intent that `run()`
  breaks. All 8 tests pass (`8 passed, 16 deselected in 0.45s`), i.e. the
  suite cannot detect the bug.
- Advertised as public API: `dynamic_integrators/__init__.py`, `README.md:16`,
  `LIBRARY_AUDIT.md:62`. Internal knowledge docs already flag the trap
  (`docs_temp/code_map/integrators_and_bcs.md:106`,
  `docs_temp/04_solver_pipeline.md:96`) — known, but not fixed and not
  reflected in the class's own docstring.

## Verdict: CONFIRMED_BUG (severity: medium)

`DynamicSimulation.run()` violates its own documented contract:
`SimulationParams.dudt_kwargs` ("keyword arguments for dudt_fn",
`_simulation.py:62`) never delivers `dim` (or `HC` via `extra`) to
`dudt_fn`, silently substituting the physics operator's `dim=3` default in a
2D simulation, and `params.rho` is dead code. The class docstring's own usage
example crashes. Mitigation: no production case currently uses the runner,
and stress-based `dudt_fn`s fail loudly on `HC=None` — but any custom
acceleration callable with defaulted `dim`/`HC` (the pattern used in the
template itself) runs silently with wrong dimensionality.

## Droplet/bubble impact

None. `oscillating_droplet`, `oscillating_droplet_p_ref`,
`electrolysis_bubble`, `shearing_plate_droplet` (and every other case in
`cases_dynamic/`) invoke `euler`/`symplectic_euler`/`euler_velocity_only`
directly with `partial`-bound `dudt_fn` and never import `DynamicSimulation`.

## Suggested fix

In `run()` (`_simulation.py:158-182`), stop flat-merging: bind the physics
kwargs into the function instead of the integrator call —

```python
import functools
dudt_fn = functools.partial(self._dudt_fn, **{**p.dudt_kwargs, 'HC': self.HC})
integrator_kwargs = {'HC': self.HC, 'bV': self.bV, 'dudt_fn': dudt_fn,
                     'dt': p.dt, 'dim': p.dim, 'callback': callback,
                     'bc_set': self._bc_set,
                     'skip_triangulation': p.skip_triangulation}
```

(or validate that `p.extra` contains no keys colliding with integrator
parameter names and raise). Either forward `rho` in `dudt_kwargs` for
acceleration functions that accept it, or delete the field; update
`TestDynamicSimulation` to use a spy asserting the received `dim`/`HC`
(the current dim=1-default tests cannot catch regressions). Also fix the
class docstring example (`_simulation.py:11-18`), which crashes as written.

Caveat on the fix above: `partial`-of-`partial` keeps the *outer* keyword,
so blindly re-binding `p.dudt_kwargs` would still clobber values the user
already bound via `functools.partial` (the PROBE B `mu` hazard). Robust
version: inspect `self._dudt_fn` (`functools.partial` → `.keywords`) and
only bind keys the user has not already supplied, or raise on conflict
instead of silently choosing a winner.
