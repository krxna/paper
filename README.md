# Energy-Aware UAV Fog Simulation

A two-stage UAV-fog task-distribution simulator: Stage 1 elects a Master Fog Head
via a rank-reversal-free MCDM pipeline (MEREC → Kalman → KL gate → SPOTIS); Stage 2
routes each task with an attention-encoder + PPO actor-critic. The code is split
into 8 focused modules in dependency order.

## Tiering and Stage-1 criteria (current design)

**Tiers are computed, not assigned.** Each UAV draws an intrinsic per-node hardware
profile from a single latent quality `q ∈ [0,1]` (clock, MIPS, compute-weight,
RAM/storage, battery, failure rates all interpolate across the envelope in
`config.py §1.2`). Its **computational efficiency** `CE_eff = fr_hz / (ςⱼ·C_mipsⱼ·1e6)`
(higher = better; the reciprocal of the paper's compute-cost `CE`) is computed, the
swarm is ranked **descending** by `CE_eff`, and tiers are assigned by rank position:
top 4 → Tier 1, next 10 → Tier 2, last 6 → Tier 3 (`TIER_COUNTS`). `CE_eff` is
strictly monotone in `q`, so tiers are clean nested capability classes. The 20
sampled `q` values are affinely rescaled so the clock envelope endpoints
(1.5 / 3.0 GHz — Stackelberg §IV-A's [1.5, 3] GHz, exact) are pinned every episode,
keeping the task-feasibility guarantee
exact. `CE_eff` is used **only** to form tiers — it is no longer a Stage-1 criterion.

**Stage-1 capability criterion is memory efficiency (ME), replacing processing
capability (PC).** `ME_j = clip([CAP_ALPHA·(MP_tot−MP_occ) + CAP_BETA·(MS_tot−MS_occ)] / ME_NORM_GB, 0, 1)`
captures both *capacity* (per-node `MP_tot`/`MS_tot` differ, so ME discriminates even
at zero load) and *occupancy* (`MP_occ`/`MS_occ` are recomputed each tick from the
tasks resident in a node's queues via `update_memory_occupancy`, so ME falls under
load and recovers as queues drain). The five fixed criteria are now
`[E_res(+), ME(+), R(+), D(−), WL(−)]`.

> Note: results are not byte-identical to any earlier fixed-tier version — the model
> changed by design. Runs remain fully reproducible from `SEED = 42`.

## UAV mobility (UAV_Mobility_Design.pdf)

Three movement modes, selected via `MOVE_MODE` in `config.py` or the
`UAV_MOVE_MODE` environment variable:

- `event_driven` (default) — each UAV is deployed at, and tracks, a distinct slot
  on a 250 m standoff ring around its assigned IoT demand hotspot (5 UAVs per
  hotspot; the hotspot centroid drifts at 2 m/s). Co-assigned UAVs also occupy
  distinct altitude layers in 80–150 m, so collisions are impossible by design.
- `random` — random-waypoint motion at 12 m/s cruise, demand-agnostic ablation.
- `static` — no motion, z = 0: reproduces the original fixed-position simulation
  exactly (P(0) = P_HOVER, 2-D geometry).

Each control tick, step 1 moves the swarm (`world.step_mobility`) and charges
Zeng19 `P(V)·Δt` propulsion energy (`physics.drain_flight_energy` /
`physics.aero_power_w` — hover is the V = 0 point of the same curve); step 2
recomputes all 3-D slant ranges (`physics.slant_distance_m`) so data rates,
delays, reliabilities, and comm energies re-price after movement.

## Paper figures

`conda run -n venv python3 experiments.py` runs the instrumented ablation matrix
(head election, mobility modes, dispatch policies) and writes all results
figures to `figures/`; `conda run -n venv python3 arch_diagram.py` renders the
system-architecture diagram (`figures/fig8_architecture.*`).

Figure 7 uses a separate event-audited load sweep because assigned queue demand
is not processor utilisation. Run `event_training.py` followed by
`utilisation_study.py`; the study exports per-seed and per-UAV CSVs, a compressed
event audit, a signed manifest, and `fig7_system_utilisation.{png,pdf}`. Its
definitions, controller-cost sensitivities, and statistical protocol are in
`SYSTEM_UTILISATION_METHODOLOGY.md`. `experiments.py --plot-only` will only
render Figure 7 when those audited CSVs exist; it will not recreate the removed
clipped assigned-cycle plot.

`conda run -n venv python3 fog_head_experiments.py` runs the unified Stage-1
comparison against FU-Serve, 2DP-FHS, and 3D-POS. It holds the simulated horizon
fixed, sweeps Poisson arrival rate, and writes trial summaries, compressed
per-task/per-tick observations, paired-bootstrap plots, tail latency, switching,
control-cost, and selected-head-quality figures to `fog_head_figures/`. The
elected head now carries physical admission-request and executor-assignment
control traffic; task payloads remain direct IoT-to-executor. Tests run with
`conda run -n venv python3 test_head_baselines.py` and
`conda run -n venv python3 test_fog_head_study.py`; implementation decisions and
paper ambiguities are recorded in
`baselines/FOG_HEAD_BASELINE_DEVIATIONS_LOG.md`.

## How to run

```bash
conda run -n venv python3 main.py
```

`main.py` is the entry point and prompts for the number of live episodes; it runs
offline pre-training first, then the live episode(s). Importing the modules in
dependency order prints the startup banners (`[SEED]`, `[CONFIG]`, `[WORLD]`).

To run the built-in self-test instead of the simulation, set
`RUN_SELF_TEST = True` in `config.py` and run `main.py`.

## Module layout (top of the dependency chain at the bottom)

| File            | Sections | Responsibility |
|-----------------|----------|----------------|
| `config.py`     | 0–4      | Constants, global seed, hardware envelope, `CE_eff` scorer, ME model constants. Seeds RNGs and prints `[CONFIG]` at import. |
| `models.py`     | 5        | Dataclasses: `Task`, `FogNode`, `IoTDevice`. |
| `world.py`      | 6–7      | World builders. `build_fog_swarm` runs hardware→CE_eff→rank→tier; builds globals and prints `[WORLD]`. |
| `physics.py`    | 8–10     | Delay/energy models; `memory_efficiency`, `update_memory_occupancy`, `computational_efficiency`, workload, reliability. |
| `broker.py`     | 11       | Stage 1 — `BrokerSelector` (MEREC → Kalman → KL gate → SPOTIS). |
| `head_baselines.py` | —    | Pure FU-Serve, 2DP-FHS, and 3D-POS fog-head selectors. |
| `neural.py`     | 12–14    | Stage 2 — attention encoder, actor, critic, reward, GAE, PPO update. |
| `simulation.py` | 15–16    | Executor (`dispatch_task`) and `run_episode` control loop. |
| `main.py`       | 17       | Self-test, offline pre-training, summary printing, `main()` driver, entry point. |
| `fog_head_experiments.py` | — | Fixed-horizon paired arrival-rate study, detailed result export, and journal figures. |


## Dependency flow

```
config  ->  models  ->  world  ->  physics  ->  broker  ┐
                                              ->  neural ┤->  simulation  ->  main
```

Each module only imports from modules above it, so the chain has no cycles.
