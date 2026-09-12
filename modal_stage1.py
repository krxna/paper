"""Run the paper's four Stage-1 selectors on Modal with shared Attention+PPO.

The parallel unit is the TRIAL.  ``_shared_nets`` trains one encoder/actor per
trial seed and all four Stage-1 policies in that trial share those exact nets,
so all four must run in the same container or the paired design is destroyed.
Splitting trials across containers is safe: every paired delta is computed from
four runs that shared one machine and one set of nets.
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

app = modal.App("uav-fog-stage1")
results = modal.Volume.from_name(
    "uav-fog-stage1-results", create_if_missing=True, version=2)
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

SEED0 = 42
FUNC = dict(image=image, cpu=4.0, memory=8192, volumes={"/results": results})


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'")
    return value


def _sweep(fog_sizes: str, arrival_rates: str, episode_seconds: float,
           ppo_episodes: int, trials: int, failure_s: float | None) -> dict:
    from experiments import HANDOVER_SETUP_MS
    return dict(
        arrival_rates=[float(v) for v in arrival_rates.split(",") if v],
        fog_sizes=[int(v) for v in fog_sizes.split(",")],
        scale_arrival_rate=180.0, episode_seconds=episode_seconds,
        trials=trials, seed0=SEED0, ppo_episodes=ppo_episodes,
        handover_ms=HANDOVER_SETUP_MS, save_raw=False,
        controller_placement="head", dispatch="rl", head_failure_s=failure_s)


def _metadata(sweep: dict) -> dict:
    return {"arrival_rates_tasks_per_s": sweep["arrival_rates"],
            "fog_sizes": sweep["fog_sizes"],
            "scale_arrival_rate": sweep["scale_arrival_rate"],
            "episode_seconds": sweep["episode_seconds"],
            "trials": sweep["trials"], "seed0": sweep["seed0"],
            "ppo_episodes": sweep["ppo_episodes"],
            "handover_setup_ms": sweep["handover_ms"],
            "raw_details_enabled": sweep["save_raw"],
            "controller_placement": "head", "dispatch": "rl",
            "head_failure_s": sweep["head_failure_s"]}


def _enter():
    """Import the simulation with threads pinned and the profile verified."""
    import os
    import shutil
    import sys
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    import torch
    torch.set_num_threads(1)
    profile_path = "profiles/simulated_controller_profile_5101520.json"
    assert os.path.exists(profile_path), (
        "controller profile missing -> require_profile would silently "
        "re-benchmark on Modal hardware and every container would get a "
        "different Stage-1 price table")
    shutil.copyfile(profile_path, "profiles/simulated_controller_profile.json")
    from controller_profile import require_profile, workload_hash
    return require_profile(), workload_hash()


@app.function(timeout=7_200, retries=1, **FUNC)
def run_trial(spec: dict) -> dict:
    import json
    import os

    profile, whash = _enter()
    import fog_head_experiments as fhe

    index = spec["index"]
    sweep = _sweep(**spec["sweep"])
    out = Path("/results") / _safe_name(spec["run_id"]) / f"trial_{index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    fhe.OUT_DIR = str(out)          # module global; main() rebinds it the same way

    rows, reps, _ = fhe.run_sweep(start_trial=index, trials=index + 1, **{
        k: v for k, v in sweep.items() if k != "trials"})

    payload = {
        "index": index, "workload_hash": whash,
        "prices": {k: v.get("by_n_fog")
                   for k, v in profile["algorithms"].items()},
        "rows": rows,
        # run_sweep hands back RAW result dicts here: huge and not JSON-safe.
        # _representative_payload is the trimmed form make_figures consumes.
        "representatives": fhe._representative_payload(reps),
    }
    (out / "trial.json").write_text(json.dumps(payload))
    results.commit()
    print(f"[trial {index:02d}] {len(rows)} rows, "
          f"{len(payload['representatives'])} representatives")
    return {"index": index, "rows": len(rows), "path": str(out)}


@app.function(timeout=3_600, **FUNC)
def merge(spec: dict) -> str:
    import glob
    import json
    import os

    _enter()
    import fog_head_experiments as fhe

    run_dir = Path("/results") / _safe_name(spec["run_id"])
    payloads = [json.loads(Path(p).read_text())
                for p in sorted(glob.glob(str(run_dir / "trial_*/trial.json")))]
    assert payloads, f"no trial payloads under {run_dir}"

    hashes = {p["workload_hash"] for p in payloads}
    prices = {json.dumps(p["prices"], sort_keys=True) for p in payloads}
    assert len(hashes) == 1 and len(prices) == 1, (
        f"containers disagree on the Stage-1 price table: "
        f"{len(hashes)} workload hashes, {len(prices)} price tables")

    rows = [r for p in payloads for r in p["rows"]]
    # Only the container running trial 0 populates representatives
    # (fog_head_experiments.py gates on `trial == 0`); the rest return {}.
    reps = {k: v for p in payloads for k, v in p["representatives"].items()}
    sweep = _sweep(**spec["sweep"])
    if sweep["arrival_rates"]:
        assert set(reps) == set(fhe.POLICIES), (
            f"representatives incomplete: {sorted(reps)} -- fh_fig6/fh_fig7 would be "
            "SILENTLY SKIPPED. Trial 0's container must have succeeded.")
    meta = _metadata(sweep)
    meta["completed_trials"] = len(payloads)
    meta["status"] = "complete"

    fhe.OUT_DIR = str(run_dir / "final")
    os.makedirs(fhe.OUT_DIR, exist_ok=True)
    fhe._write_summary(rows, meta)
    figs = []
    if sweep["arrival_rates"]:
        fhe._write_json_atomic(
            os.path.join(fhe.OUT_DIR, "fog_head_representatives.json"), reps)
        fhe.make_figures(rows, reps, spec.get("bootstrap_samples", 2000))
        figs = sorted(os.path.basename(f)
                      for f in glob.glob(os.path.join(fhe.OUT_DIR, "fh_fig*.png")))
        expected = 8 if sweep["head_failure_s"] is not None else 7
        assert len(figs) == expected, f"expected {expected} figures, got {figs}"

    n_cond = (len(sweep["fog_sizes"]) + len(sweep["arrival_rates"])
              + int(sweep["head_failure_s"] is not None))
    want = len(payloads) * n_cond * len(fhe.POLICIES)
    assert len(rows) == want, f"expected {want} rows, got {len(rows)}"

    results.commit()
    print(f"[merge] {len(rows)} rows, {len(figs)} figures -> {fhe.OUT_DIR}")
    return fhe.OUT_DIR


@app.function(timeout=3_600, **FUNC)
def run_host_timing(run_id: str) -> str:
    import subprocess

    output = Path("/results") / _safe_name(run_id)
    output.mkdir(parents=True, exist_ok=True)
    timing_path = output / "modal_host_controller_paths.md"
    with timing_path.open("w") as stream:
        subprocess.run(
            ["python", "/workspace/bench_controller_paths.py",
             "--fog-sizes", "5,10,15,20",
             "--warmups", "100", "--samples", "1000"],
            cwd="/workspace", stdout=stream, check=True)
    results.commit()
    return str(timing_path)


@app.function(timeout=14_400, **FUNC)
def driver(spec: dict) -> str:
    """Fan out the trials and merge, entirely on Modal.

    The orchestration has to live remotely: if map/merge were driven from the
    local entrypoint, disconnecting the client would strand the merge.
    """
    n = spec["sweep"]["trials"]
    specs = [{**spec, "index": i} for i in range(n)]
    done = list(run_trial.map(specs))
    print(f"{len(done)} trials complete: {sum(d['rows'] for d in done)} rows")
    out = merge.remote(spec)
    print(f"[driver] merged -> {out}")
    return out


PRESETS = {
    # tiny end-to-end check: exercises all three study axes and all 8 figures
    "smoke": dict(trials=2, ppo_episodes=1, episode_seconds=2.0,
                  fog_sizes="10,20", arrival_rates="60,120", failure_s=1.0),
    # the paper matrix
    "full": dict(trials=20, ppo_episodes=50, episode_seconds=10.0,
                 fog_sizes="10,20,40,80",
                 arrival_rates="60,120,180,240,300", failure_s=5.0),
    "endurance": dict(trials=20, ppo_episodes=50, episode_seconds=200.0,
                      fog_sizes="20", arrival_rates="60", failure_s=None),
    "fig3_5101520": dict(trials=20, ppo_episodes=50, episode_seconds=10.0,
                          fog_sizes="5,10,15,20", arrival_rates="",
                          failure_s=None),
}


@app.local_entrypoint()
def main(mode: str = "smoke", run_id: str = "stage1-attn-ppo",
         detach: bool = False) -> None:
    """mode=smoke validates the pipeline; mode=full runs the 20-seed matrix.

    With --detach (the modal flag) plus detach=True, the driver is spawned on
    Modal and this process can exit immediately; the run continues without a
    local client.
    """
    if mode not in PRESETS:
        raise ValueError(f"mode must be one of {sorted(PRESETS)}")
    sweep = PRESETS[mode]
    spec = {"run_id": f"{run_id}-{mode}", "sweep": sweep}

    if mode == "full":
        run_host_timing.spawn(spec["run_id"])

    if detach:
        call = driver.spawn(spec)
        print(f"Spawned driver call {call.object_id} for {spec['run_id']}.")
        print("Safe to close this terminal and shut down. Recover with:")
        print(f"  modal app list        # find the app")
        print(f"  modal volume ls uav-fog-stage1-results {spec['run_id']}")
        return

    print(f"Merged output: {driver.remote(spec)}")
