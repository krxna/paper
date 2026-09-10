"""Re-run the paper's task-distribution ablation matrix (experiments.py Figs. 1-6)
on Modal, evaluated through the FINAL 6-criterion Stage-1 broker
(``controller_placement="head"`` -> ``BrokerSelector(final_fix=True)``), so both
studies in the paper share one selector. See paper_plan.md Phase 1.

Design: train once per training seed (BC warm start + N_PRETRAIN offline PPO
episodes on ``simulation.run_episode``, identical to ``experiments.py main()``),
checkpoint the encoder/actor/critic to the results volume, then fan out
evaluation containers that load the frozen checkpoint and each run ONE of the
six paired arms at one evaluation seed via ``experiments.run_exp``.

Only the three RL arms (A1 optimal-head, B1 static-mobility ablation, B2
random-mobility ablation) depend on the trained nets, so only those are swept
over BOTH training seed and evaluation seed. The three non-RL baselines (C1
headless/random, D1 FODAS, E1 ReLIEF) do not depend on the trained nets at all
-- sweeping them over training seed would be pseudo-replication, not an
independent sample -- so they are swept over evaluation seed only.

Writes to a NEW directory (``figures_head_final/`` locally, via ``modal volume
get``) -- ``figures/`` and ``fog_head_v3_modal/`` are never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

app = modal.App("uav-fog-taskdist")
results = modal.Volume.from_name(
    "uav-fog-taskdist-results", create_if_missing=True, version=2)
source = Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy==2.4.4", "torch==2.12.0", "matplotlib==3.10.9")
    .env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg"})
    .add_local_dir(
        source,
        remote_path="/workspace",
        ignore=~modal.FilePatternMatcher(
            "*.py", "baselines/*.py", "profiles/*.json"),
    )
)

FUNC = dict(image=image, cpu=4.0, memory=8192, volumes={"/results": results})

# (key, label, move_mode, head_mode, dispatch, coordinated) -- identical to the
# six eval_specs in experiments.py main() (Study 1, Figs. 1-6).
RL_ARMS = [
    ("A1", "optimal-head/event/rl", "event_driven", "spotis", "rl", True),
    ("B1", "static/spotis/rl", "static", "spotis", "rl", True),
    ("B2", "random-move/spotis/rl", "random", "spotis", "rl", True),
]
BASELINE_ARMS = [
    ("C1", "headless/random", "event_driven", "none", "random", False),
    ("D1", "fodas/edf-heuristic", "event_driven", "none", "fodas", False),
    ("E1", "relief/q-learning", "event_driven", "none", "relief", False),
]
ARM_KEYS = [a[0] for a in RL_ARMS] + [a[0] for a in BASELINE_ARMS]


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'")
    return value


def _arm_specs(run_id: str, n_tasks: int,
               train_seeds: list[int], eval_seeds: list[int]) -> list[dict]:
    primary_ts, primary_es = train_seeds[0], eval_seeds[0]
    specs = []
    for arm in RL_ARMS:
        for train_seed in train_seeds:
            for eval_seed in eval_seeds:
                specs.append({
                    "run_id": run_id, "train_seed": train_seed,
                    "eval_seed": eval_seed, "arm": arm, "n_tasks": n_tasks,
                    "primary": train_seed == primary_ts and eval_seed == primary_es,
                })
    for arm in BASELINE_ARMS:
        for eval_seed in eval_seeds:
            specs.append({
                "run_id": run_id, "train_seed": None, "eval_seed": eval_seed,
                "arm": arm, "n_tasks": n_tasks, "primary": eval_seed == primary_es,
            })
    return specs


# ------------------------------- pure-Python summary helpers (no numpy/torch
# at module level -- these must be importable in the local entrypoint without
# the heavy deps being installed there; mirrors modal_stage1.py's convention).

def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _percentile(xs, p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return float(s[0])
    k = (n - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, n - 1)
    return float(s[f]) if f == c else float(s[f] + (s[c] - s[f]) * (k - f))


def _jain(xs) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan")
    s, s2, n = sum(xs), sum(x * x for x in xs), len(xs)
    return (s * s) / (n * s2) if s2 > 0 else float("nan")


def _per_device(rec) -> tuple[list[float], list[float]]:
    by_dev: dict = {}
    for dv, d, s in zip(rec["dev_id"], rec["delay_ms"], rec["success"]):
        by_dev.setdefault(int(dv), []).append((d, s))
    mean_delay, succ = [], []
    for rows in by_dev.values():
        ds = [d for d, s in rows if s]
        if ds:
            mean_delay.append(sum(ds) / len(ds))
        succ.append(sum(s for _, s in rows) / len(rows))
    return mean_delay, succ


def _summarize(out: dict, arm_key: str, train_seed, eval_seed: int,
               whash: str) -> dict:
    rec, energy = out["rec"], out["E"]
    delays, successes = rec["delay_ms"], rec["success"]
    n_tasks = len(delays)
    delivered = max(sum(successes), 1)
    sys_j = (energy["prop"] + energy["comp"] + energy["net"]
             + energy["control"] + energy["signal"] + energy["edge"])
    md, sc = _per_device(rec)
    return {
        "arm_key": arm_key, "label": out["label"],
        "train_seed": train_seed if train_seed is not None else "",
        "eval_seed": eval_seed, "workload_hash": whash,
        "n_tasks": n_tasks,
        "mean_delay_ms": _mean(delays),
        "p50_delay_ms": _percentile(delays, 50),
        "p95_delay_ms": _percentile(delays, 95),
        "p99_delay_ms": _percentile(delays, 99),
        "success_rate": _mean(successes) if n_tasks else 0.0,
        "deadline_miss_rate": 1.0 - _mean(successes) if n_tasks else 1.0,
        "broker_ms_mean": _mean(rec["broker_ms"]),
        "up_ms_mean": _mean(rec["up_ms"]),
        "queue_ms_mean": _mean(rec["queue_ms"]),
        "exec_ms_mean": _mean(rec["exec_ms"]),
        "prop_j": energy["prop"], "comp_j": energy["comp"],
        "net_j": energy["net"], "control_j": energy["control"],
        "signal_j": energy["signal"], "edge_j": energy["edge"],
        "sys_j_per_delivered": sys_j / delivered,
        "sys_j_per_attempted": sys_j / max(n_tasks, 1),
        "head_changes": out["head_changes"],
        "jain_delay": _jain(md),
        "jain_success": _jain([s + 1e-9 for s in sc]),
    }


def _enter():
    """Import the simulation with threads pinned and the profile verified."""
    import os
    import sys
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    import torch
    torch.set_num_threads(1)
    profile_path = "profiles/simulated_controller_profile.json"
    assert os.path.exists(profile_path), (
        "controller profile missing -> require_profile would silently "
        "re-benchmark on Modal hardware and every container would get a "
        "different Stage-1 price table")
    from controller_profile import require_profile, workload_hash
    return require_profile(), workload_hash()


@app.function(timeout=7_200, retries=5, **FUNC)
def train(spec: dict) -> dict:
    """BC warm start + N_PRETRAIN offline PPO episodes for one training seed.

    Mirrors experiments.py main()'s training loop exactly (behavior_clone_fodas
    then simulation.run_episode(..., train=True) per episode), parameterised by
    train_seed instead of the fixed module-level SEED.
    """
    import torch

    profile, whash = _enter()
    from config import set_global_seed, LEARNING_RATE
    from neural import AttentionEncoder, Actor, Critic, behavior_clone_fodas
    from broker import BrokerSelector
    import simulation

    train_seed = spec["train_seed"]
    n_pretrain = spec["ppo_episodes"]
    run_id = spec["run_id"]
    signature = {"workload_hash": whash, "train_seed": train_seed,
                 "ppo_episodes": n_pretrain}

    out = Path("/results") / _safe_name(run_id) / f"train_{train_seed}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"

    encoder, actor, critic = AttentionEncoder(), Actor(), Critic()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(actor.parameters())
        + list(critic.parameters()), lr=LEARNING_RATE)

    completed = 0
    resuming = False
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        signature_matches = {k: saved.get(k) for k in signature} == signature
        if signature_matches and saved.get("complete") is True:
            print(f"[TRAIN seed={train_seed}] already complete from a prior "
                  f"attempt -> {out}", flush=True)
            return {"train_seed": train_seed, "path": str(out)}
        if signature_matches:
            encoder.load_state_dict(saved["encoder"])
            actor.load_state_dict(saved["actor"])
            critic.load_state_dict(saved["critic"])
            optimizer.load_state_dict(saved["optimizer"])
            encoder.running_norm.mu = saved["running_norm"]["mu"].clone()
            encoder.running_norm.sigma = saved["running_norm"]["sigma"].clone()
            encoder.running_norm._first = bool(saved["running_norm"]["first"])
            completed = int(saved["completed_episodes"])
            resuming = True
            print(f"[TRAIN seed={train_seed}] resuming from a prior attempt's "
                  f"checkpoint at episode {completed}/{n_pretrain} -- a "
                  "preemption or retry only costs episodes since then, not "
                  "the whole run", flush=True)

    if not resuming:
        set_global_seed(train_seed)
        print(f"[TRAIN seed={train_seed}] BC warm start ...", flush=True)
        bc = behavior_clone_fodas(encoder, actor)
        print(f"[TRAIN seed={train_seed}] BC loss={bc['bc_loss']:.3f}", flush=True)

    def _save(completed_episodes: int, complete: bool) -> None:
        torch.save({
            **signature, "complete": complete,
            "completed_episodes": completed_episodes,
            "encoder": encoder.state_dict(), "actor": actor.state_dict(),
            "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
            "running_norm": {
                "mu": encoder.running_norm.mu,
                "sigma": encoder.running_norm.sigma,
                "first": encoder.running_norm._first,
            },
        }, ckpt_path)
        results.commit()

    for i in range(completed, n_pretrain):
        set_global_seed(train_seed * 100_000 + 1000 + i)
        st = simulation.run_episode(
            i + 1, actor, critic, BrokerSelector(), encoder, optimizer,
            train=True, controller_profile=profile)
        _save(i + 1, complete=(i + 1 == n_pretrain))
        if (i + 1) % max(n_pretrain // 10, 1) == 0 or i + 1 == n_pretrain:
            print(f"[TRAIN seed={train_seed}] PPO {i + 1}/{n_pretrain} "
                  f"drop={st['drop_rate'] * 100:.2f}% "
                  f"loss={st['total_loss']:.3f}", flush=True)

    print(f"[TRAIN seed={train_seed}] checkpoint saved -> {out}", flush=True)
    return {"train_seed": train_seed, "path": str(out)}


@app.function(timeout=7_200, retries=5, **FUNC)
def evaluate(spec: dict) -> dict:
    """Run ONE (arm, train_seed, eval_seed) evaluation through the final
    6-criterion broker (``controller_placement="head"``) and write a compact
    JSON payload -- full per-task/per-tick arrays only for the designated
    "primary" replicate of each arm, to keep payload size in line with the
    original 6-run figures/_runs_cache.pkl.
    """
    import json

    import torch

    profile, whash = _enter()
    from neural import AttentionEncoder, Actor
    import experiments as exp

    run_id = spec["run_id"]
    train_seed = spec["train_seed"]
    eval_seed = spec["eval_seed"]
    arm_key, label, move_mode, head_mode, dispatch, coordinated = spec["arm"]
    n_tasks = spec["n_tasks"]
    primary = spec["primary"]

    result_dir = Path("/results") / _safe_name(run_id) / "eval"
    ts_tag = train_seed if train_seed is not None else "na"
    fname = f"{arm_key}_t{ts_tag}_e{eval_seed}.json"
    out_path = result_dir / fname
    if out_path.exists():
        print(f"[EVAL] {arm_key} train={ts_tag} eval={eval_seed} already "
              "done from a prior attempt -- skipping", flush=True)
        return {"file": fname}

    nets = None
    if train_seed is not None:
        ckpt_path = (Path("/results") / _safe_name(run_id)
                     / f"train_{train_seed}" / "ckpt.pt")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert saved["workload_hash"] == whash, (
            "this container's workload hash disagrees with the training "
            "container's -- Stage-1 price table or physics constants drifted "
            "between the train and eval images")
        encoder, actor = AttentionEncoder(), Actor()
        encoder.load_state_dict(saved["encoder"])
        actor.load_state_dict(saved["actor"])
        encoder.running_norm.mu = saved["running_norm"]["mu"].clone()
        encoder.running_norm.sigma = saved["running_norm"]["sigma"].clone()
        encoder.running_norm._first = bool(saved["running_norm"]["first"])
        encoder.eval()
        actor.eval()
        nets = (encoder, actor)

    out = exp.run_exp(
        label, move_mode, head_mode, dispatch, coordinated, nets, n_tasks,
        seed=eval_seed, controller_placement="head")

    summary = _summarize(out, arm_key, train_seed, eval_seed, whash)
    payload = {"summary": summary}
    if primary:
        payload["full"] = {
            "label": out["label"], "rec": out["rec"], "series": out["series"],
            "E": out["E"], "succ_rate": out["succ_rate"],
            "head_changes": out["head_changes"], "dur": out["dur"],
            "task_arrival_rate": out["task_arrival_rate"],
        }

    result_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    results.commit()
    print(f"[EVAL] {arm_key} train={ts_tag} eval={eval_seed} "
          f"succ={summary['success_rate'] * 100:.1f}% "
          f"meanD={summary['mean_delay_ms']:.1f}ms", flush=True)
    return {"file": fname}


@app.function(timeout=3_600, **FUNC)
def merge(spec: dict) -> str:
    """Aggregate all eval payloads: summary CSV, seed-level bootstrap-CI CSV
    (same protocol as fog_head_experiments._bootstrap_mean_ci), and a
    runs_cache.pkl of the six "primary" full-detail runs for plotting."""
    import csv
    import glob
    import json
    import pickle

    import numpy as np

    _enter()
    from fog_head_experiments import _bootstrap_mean_ci

    run_id = spec["run_id"]
    eval_dir = Path("/results") / _safe_name(run_id) / "eval"
    files = sorted(glob.glob(str(eval_dir / "*.json")))
    assert files, f"no eval payloads under {eval_dir}"
    payloads = [json.loads(Path(f).read_text()) for f in files]

    hashes = {p["summary"]["workload_hash"] for p in payloads}
    assert len(hashes) == 1, (
        f"containers disagree on the workload hash: {hashes}")

    out_dir = Path("/results") / _safe_name(run_id) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [p["summary"] for p in payloads]
    fieldnames = list(summary_rows[0].keys())
    with (out_dir / "taskdist_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    rng = np.random.default_rng(0)
    metrics = ["success_rate", "mean_delay_ms", "p95_delay_ms",
               "deadline_miss_rate", "sys_j_per_delivered", "jain_delay",
               "queue_ms_mean"]
    ci_rows = []
    for arm_key in ARM_KEYS:
        rows_a = [r for r in summary_rows if r["arm_key"] == arm_key]
        assert rows_a, f"no rows for arm {arm_key}"
        row = {"arm_key": arm_key, "label": rows_a[0]["label"],
               "n_runs": len(rows_a)}
        for metric in metrics:
            mean, lo, hi = _bootstrap_mean_ci(
                [r[metric] for r in rows_a], 2000, rng)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        ci_rows.append(row)
    with (out_dir / "taskdist_ci_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ci_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ci_rows)

    reps = {p["summary"]["arm_key"]: p["full"] for p in payloads if "full" in p}
    assert set(reps) == set(ARM_KEYS), (
        f"representatives incomplete: {sorted(reps)} -- the primary replicate "
        "of some arm did not complete, so its full rec/series/E is missing")
    with (out_dir / "runs_cache.pkl").open("wb") as fh:
        pickle.dump(reps, fh)

    results.commit()
    print(f"[merge] {len(summary_rows)} rows, {len(reps)} representative "
          f"runs -> {out_dir}", flush=True)
    return str(out_dir)


@app.function(timeout=14_400, retries=2, **FUNC)
def driver(spec: dict) -> str:
    """Fan out training, then evaluation, then merge -- entirely on Modal so a
    disconnected client cannot strand the merge (mirrors modal_stage1.driver).

    The three non-RL baseline arms (C1/D1/E1) do not depend on the trained
    checkpoint at all, so they are spawned immediately, concurrently with the
    (much longer) sequential PPO training -- rather than waiting idle for
    training to finish before they even start. Their results are gathered
    just before the RL-dependent evaluations run (by which point they have
    almost certainly already completed in the background).
    """
    eval_specs = _arm_specs(
        spec["run_id"], spec["n_tasks"], spec["train_seeds"], spec["eval_seeds"])
    baseline_specs = [s for s in eval_specs if s["train_seed"] is None]
    rl_specs = [s for s in eval_specs if s["train_seed"] is not None]

    baseline_calls = [evaluate.spawn(s) for s in baseline_specs]
    print(f"{len(baseline_calls)} baseline evals spawned concurrently with "
          "training", flush=True)

    train_specs = [
        {"run_id": spec["run_id"], "train_seed": s,
         "ppo_episodes": spec["ppo_episodes"]}
        for s in spec["train_seeds"]
    ]
    trained = list(train.map(train_specs))
    print(f"{len(trained)} training runs complete", flush=True)

    rl_done = list(evaluate.map(rl_specs))
    print(f"{len(rl_done)} RL-dependent eval runs complete", flush=True)

    baseline_done = [call.get() for call in baseline_calls]
    print(f"{len(baseline_done)} baseline eval runs complete", flush=True)

    out = merge.remote(spec)
    print(f"[driver] merged -> {out}", flush=True)
    return out


PRESETS = {
    # tiny end-to-end check: 1 train seed, 2 eval seeds, all 6 arms, few tasks
    "smoke": dict(train_seeds=[42], eval_seeds=[42, 43],
                  ppo_episodes=2, n_tasks=500),
    # the paper matrix: 3 train seeds x 11 eval seeds for the 3 RL arms
    # (33 replicates/arm -> 99 eval containers, ~100-container budget) and
    # 11 eval seeds for the 3 non-RL baselines (11 replicates/arm, spawned
    # concurrently with training since they don't depend on the checkpoint)
    "full": dict(train_seeds=[42, 43, 44],
                 eval_seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
                 ppo_episodes=300, n_tasks=25_000),
}


@app.local_entrypoint()
def main(mode: str = "smoke", run_id: str = "taskdist-final-broker") -> None:
    """mode=smoke validates the pipeline; mode=full runs the paper matrix.

    Always spawns the driver and returns immediately -- a full run is far too
    long for a blocking local client to hold open (an earlier version made
    that mistake: a later local disconnect surfaced as a driver.remote()
    ConnectionError, and separately, killing a blocking local client
    propagates a cancellation to the remote job, unlike a network drop, which
    the platform's own ``modal run --detach`` app-level flag survives).
    Recover a spawned run's status/results at any time with
    ``modal app list`` / ``modal volume ls uav-fog-taskdist-results <run_id>``
    -- no need to keep this process alive or reattach to it.
    """
    if mode not in PRESETS:
        raise ValueError(f"mode must be one of {sorted(PRESETS)}")
    preset = PRESETS[mode]
    spec = {"run_id": f"{run_id}-{mode}", **preset}

    call = driver.spawn(spec)
    print(f"Spawned driver call {call.object_id} for {spec['run_id']}.")
    print("Safe to close this terminal and shut down. Recover with:")
    print("  modal app list        # find the app")
    print(f"  modal volume ls uav-fog-taskdist-results {spec['run_id']}")
