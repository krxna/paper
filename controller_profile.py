"""Controller-cost models and optional hardware profiling.

The default research workflow is simulation-only and uses explicit deterministic
cost assumptions from ``config.py``. Optional hardware profiles remain supported
but are never required or misrepresented as simulated results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping

import numpy as np
import torch

from execution_engine import EXECUTION_MODEL_VERSION


PROFILE_SCHEMA_VERSION = "controller-cost-profile-v2"
REQUIRED_ALGORITHMS = (
    "stage1_without_spotis",
    "stage1_with_spotis",
    "stage1_fu_serve",
    "stage1_2dp_fhs",
    "stage1_3d_pos",
    "attention_ppo",
    "fodas",
    "relief",
    "random",
)
STAGE1_ALGORITHMS = REQUIRED_ALGORITHMS[:5]
DEFAULT_PROFILE_PATH = Path("profiles/simulated_controller_profile.json")

# One-thread process-time measurements from the scalability design.
# N_IoT is 10*N_FOG. Only these rows are valid; no curve is fitted.
STAGE1_MEASURED_MS = {
    "stage1_without_spotis": {
        10: 0.0607, 20: 0.0644, 40: 0.0737, 80: 0.0870, 160: 0.1184},
    "stage1_fu_serve": {
        10: 0.0655, 20: 0.1996, 40: 0.6927, 80: 2.4991, 160: 9.9184},
    "stage1_2dp_fhs": {
        10: 0.3697, 20: 1.2958, 40: 4.7414, 80: 18.068, 160: 70.849},
    "stage1_3d_pos": {
        10: 0.5473, 20: 1.9728, 40: 7.3642, 80: 28.618, 160: 112.32},
    "stage1_with_spotis": {
        10: 0.061914, 20: 0.065688, 40: 0.075174,
        80: 0.08874, 160: 0.120768},
}


class ProfileValidationError(ValueError):
    pass


def workload_hash() -> str:
    """Hash all inputs that define the representative profiling workload."""
    import config
    payload = {
        "schema": PROFILE_SCHEMA_VERSION,
        "execution_model": EXECUTION_MODEL_VERSION,
        "n_fog": config.N_FOG,
        "n_iot": config.N_IOT,
        "task_size_bounds_kb": [
            config.TASK_SIZE_MIN_KB, config.TASK_SIZE_MAX_KB],
        "cycles_per_byte": [
            config.CYCLES_PER_BYTE_MIN, config.CYCLES_PER_BYTE_MAX],
        "feature_dim": getattr(config, "N_NODE_FEATURES", 16),
        "profile_tasks": 32,
        "seed": 73191,
        "algorithms": REQUIRED_ALGORITHMS,
        "profile_fog_sizes": config.PROFILE_FOG_SIZES,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def canonical_profile_hash(profile: Mapping) -> str:
    payload = dict(profile)
    payload.pop("profile_hash", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def scale_profile(profile: Mapping, scale: float) -> Dict:
    """Return a signed sensitivity profile with every service demand scaled."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("controller-cost scale must be positive")
    scaled = copy.deepcopy(dict(profile))
    for stats in scaled["algorithms"].values():
        for key in ("mean_cpu_s", "median_cpu_s", "p95_cpu_s",
                    "normalized_cycles"):
            stats[key] = float(stats[key]) * float(scale)
        for sized_stats in stats.get("by_n_fog", {}).values():
            for key in ("mean_cpu_s", "median_cpu_s", "p95_cpu_s",
                        "normalized_cycles"):
                sized_stats[key] = float(sized_stats[key]) * float(scale)
    scaled.setdefault("assumptions", {})["sensitivity_scale"] = float(scale)
    scaled["profile_kind"] = "simulated_model"
    scaled["profile_hash"] = canonical_profile_hash(scaled)
    return scaled


def create_simulated_profile(
    output: Path = DEFAULT_PROFILE_PATH,
    reference_frequency_hz: float | None = None,
) -> Dict:
    """Write the deterministic controller model used by simulation runs."""
    import config
    ref_hz = float(reference_frequency_hz or config.CONTROLLER_REFERENCE_HZ)
    times_ms = {
        "stage1_without_spotis":
            config.T_MEREC_MS + config.T_KALMAN_MS + config.T_KL_MS,
        "stage1_with_spotis":
            config.T_MEREC_MS + config.T_KALMAN_MS + config.T_KL_MS
            + config.T_SPOTIS_MS,
        "stage1_fu_serve": config.T_FU_SERVE_MS,
        "stage1_2dp_fhs": config.T_2DP_FHS_MS,
        "stage1_3d_pos": config.T_3D_POS_MS,
        "attention_ppo": config.T_PPO_INFER_MS,
        "fodas": config.T_FODAS_DECISION_MS,
        "relief": config.T_RELIEF_DECISION_MS,
        "random": config.T_RANDOM_DECISION_MS,
    }
    algorithms = {}
    for name, milliseconds in times_ms.items():
        seconds = float(milliseconds) / 1000.0
        algorithms[name] = {
            "samples": 0,
            "warmups": 0,
            "mean_cpu_s": seconds,
            "median_cpu_s": seconds,
            "p95_cpu_s": seconds,
            "normalized_cycles": seconds * ref_hz,
        }
        if name in STAGE1_ALGORITHMS:
            algorithms[name]["by_n_fog"] = {
                str(n): {
                    "samples": 0,
                    "warmups": 0,
                    "mean_cpu_s": STAGE1_MEASURED_MS[name][n] / 1000.0,
                    "median_cpu_s": STAGE1_MEASURED_MS[name][n] / 1000.0,
                    "p95_cpu_s": STAGE1_MEASURED_MS[name][n] / 1000.0,
                    "normalized_cycles": (
                        STAGE1_MEASURED_MS[name][n] / 1000.0 * ref_hz),
                }
                for n in config.PROFILE_FOG_SIZES
            }
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_kind": "simulated_model",
        "execution_model_version": EXECUTION_MODEL_VERSION,
        "workload_hash": workload_hash(),
        "created_utc": "not_applicable_deterministic_model",
        "target": {
            "model": "simulated heterogeneous fog-controller CPU",
            "machine": "simulation",
            "power_mode": "not_applicable",
            "clocks": "service scales as cycles / selected UAV frequency",
            "jetpack": "not_applicable",
            "reference_frequency_hz": ref_hz,
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "thread_settings": {
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "omp_num_threads": "",
            "mkl_num_threads": "",
        },
        "measurement": {
            "clock": "configured_deterministic_model",
            "warmups_per_algorithm": 0,
            "samples_per_algorithm": 0,
        },
        "assumptions": {
            "units": "milliseconds converted to cycles at reference_frequency_hz",
            "source": "declared simulation parameters in config.py",
            "empirical_hardware_measurement": False,
            "service_time_ms": times_ms,
            "stage1_service_time_ms_by_n_fog": {
                name: {str(n): STAGE1_MEASURED_MS[name][n]
                       for n in config.PROFILE_FOG_SIZES}
                for name in STAGE1_ALGORITHMS
            },
        },
        "algorithms": algorithms,
    }
    profile["profile_hash"] = canonical_profile_hash(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def _percentile(values: Iterable[float], p: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.percentile(arr, p))


def _measure(
    fn: Callable[[], object], *, warmups: int, samples: int,
    reference_frequency_hz: float,
) -> Dict[str, float]:
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(samples):
        start = time.process_time_ns()
        fn()
        values.append((time.process_time_ns() - start) / 1e9)
    mean = float(statistics.fmean(values))
    return {
        "samples": samples,
        "warmups": warmups,
        "mean_cpu_s": mean,
        "median_cpu_s": float(statistics.median(values)),
        "p95_cpu_s": _percentile(values, 95),
        "normalized_cycles": mean * reference_frequency_hz,
    }


def _benchmark_paths(n_fog: int, n_iot: int) -> Dict[str, Callable[[], object]]:
    """Build complete, deterministic controller decision paths."""
    import random
    from baselines.fodas_baseline import select_node_for_task
    from baselines.relief_baseline import (
        ReLIEFConfig, ReLIEFAgent, discretize_state)
    from broker import BrokerSelector
    from head_baselines import FUServeSelector, ThreeDPOSSelector, TwoDPFHSSelector
    from neural import (
        Actor, AttentionEncoder, build_candidate_features, filter_candidates)
    from physics import (
        base_reliability, memory_efficiency, node_workload_cycles,
        slant_distance_m, data_rate_bps, total_delay_ms,
        update_memory_occupancy)
    from world import build_fog_swarm, build_iot_devices, assign_mobility
    from models import Task
    from config import CYCLES_PER_BYTE, deadline_for_size_ms

    if n_fog < 2 or n_iot < 1:
        raise ValueError("benchmark sizing requires n_fog>=2 and n_iot>=1")

    random.seed(73191)
    np.random.seed(73191)
    torch.manual_seed(73191)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    fog = build_fog_swarm(n_fog)
    iot, centres = build_iot_devices(n_iot)
    assign_mobility(fog, centres)
    task = Task(
        task_id=0, size_kb=180.0,
        cycles=180.0 * 1024.0 * CYCLES_PER_BYTE,
        deadline_ms=deadline_for_size_ms(180.0), arrival_s=0.0, is_small=True,
    )
    dev = iot[0]
    update_memory_occupancy(fog)
    me = memory_efficiency(fog)

    def metrics():
        result = {}
        for node in fog:
            dist = slant_distance_m(node, dev.x, dev.y)
            rate = data_rate_bps(dist, dev.p_tx_w)
            result[node.node_id] = {
                "me": me[node.node_id],
                "r0": base_reliability(
                    node, task.cycles, task.size_kb, rate),
                "d_ms": total_delay_ms(
                    task, node, dist, dev.p_tx_w, False)["total_ms"],
                "wl": node_workload_cycles(node),
            }
        return result

    per_node = metrics()
    stage1_no = BrokerSelector(final_fix=True)
    stage1_no.select_head(fog, per_node, 0, 1)
    stage1_yes = BrokerSelector(final_fix=True)
    stage1_yes.select_head(fog, per_node, 0, 1)
    fu_serve_selector = FUServeSelector()
    two_dp_selector = TwoDPFHSSelector()
    three_d_selector = ThreeDPOSSelector()
    encoder = AttentionEncoder()
    actor = Actor()
    encoder.eval()
    actor.eval()
    relief = ReLIEFAgent(ReLIEFConfig())
    relief_state = discretize_state(1.0, 0.0, 1.0, relief.cfg)
    random_rng = random.Random(73191)

    def stage1_without_spotis():
        return stage1_no.select_head(fog, per_node, 0, 2)

    def stage1_with_spotis():
        # Preserve measurement state while forcing the first-election SPOTIS path.
        stage1_yes.prev_head = None
        stage1_yes.prev_spotis_ranking = None
        return stage1_yes.select_head(fog, per_node, 0, 2)

    def stage1_fu_serve():
        return fu_serve_selector.select_head(
            fog, per_node, 0, 1, iot_devices=iot)

    def stage1_2dp_fhs():
        return two_dp_selector.select_head(
            fog, per_node, 0, 1, iot_devices=iot)

    def stage1_3d_pos():
        return three_d_selector.select_head(
            fog, per_node, 0, 1, iot_devices=iot)

    def attention_ppo():
        x = build_candidate_features(fog, task, dev, me)
        mask, scores = filter_candidates(fog, task, dev)
        with torch.no_grad():
            h, c = encoder(x, update_stats=False)
            soc = torch.tensor([n.soc for n in fog], dtype=torch.float32)
            return actor.select_action(
                h, c, soc, greedy=True, candidate_mask=mask,
                heuristic_scores=scores)

    def fodas():
        return select_node_for_task(task, dev, fog, task.arrival_s)

    def relief_path():
        return relief.select_pair(fog, task, dev, relief_state, greedy=True)

    def random_path():
        primary = random_rng.randrange(n_fog)
        backup = random_rng.randrange(n_fog - 1)
        if backup >= primary:
            backup += 1
        return primary, backup

    return {
        "stage1_without_spotis": stage1_without_spotis,
        "stage1_with_spotis": stage1_with_spotis,
        "stage1_fu_serve": stage1_fu_serve,
        "stage1_2dp_fhs": stage1_2dp_fhs,
        "stage1_3d_pos": stage1_3d_pos,
        "attention_ppo": attention_ppo,
        "fodas": fodas,
        "relief": relief_path,
        "random": random_path,
    }


def create_profile(
    *, output: Path, power_mode: str, clocks: str, jetpack: str,
    reference_frequency_hz: float, warmups: int, samples: int,
) -> Dict:
    import config
    paths = _benchmark_paths(config.N_FOG, config.N_IOT)
    algorithms = {
        name: _measure(
            fn, warmups=warmups, samples=samples,
            reference_frequency_hz=reference_frequency_hz)
        for name, fn in paths.items()
    }
    for name in STAGE1_ALGORITHMS:
        algorithms[name]["by_n_fog"] = {}
    for n_fog in config.PROFILE_FOG_SIZES:
        sized_paths = (paths if n_fog == config.N_FOG
                       else _benchmark_paths(n_fog, 10 * n_fog))
        for name in STAGE1_ALGORITHMS:
            algorithms[name]["by_n_fog"][str(n_fog)] = (
                {key: value for key, value in algorithms[name].items()
                 if key != "by_n_fog"} if n_fog == config.N_FOG else _measure(
                    sized_paths[name], warmups=warmups, samples=samples,
                    reference_frequency_hz=reference_frequency_hz))
    machine = platform.machine()
    model_text = ""
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        model_text = model_path.read_bytes().replace(b"\x00", b"").decode(
            errors="replace")
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_kind": "hardware_measurement",
        "execution_model_version": EXECUTION_MODEL_VERSION,
        "workload_hash": workload_hash(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": {
            "model": model_text or platform.platform(),
            "machine": machine,
            "power_mode": power_mode,
            "clocks": clocks,
            "jetpack": jetpack,
            "reference_frequency_hz": float(reference_frequency_hz),
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "thread_settings": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "measurement": {
            "clock": "time.process_time_ns",
            "warmups_per_algorithm": int(warmups),
            "samples_per_algorithm": int(samples),
        },
        "algorithms": algorithms,
    }
    profile["profile_hash"] = canonical_profile_hash(profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def validate_profile(profile: Mapping, *, allow_test_profile: bool = False) -> Dict:
    required_top = {
        "schema_version", "execution_model_version", "workload_hash",
        "created_utc", "target", "runtime", "thread_settings",
        "measurement", "algorithms", "profile_hash",
    }
    missing = required_top - set(profile)
    if missing:
        raise ProfileValidationError(
            f"controller profile missing fields: {sorted(missing)}")
    if profile["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ProfileValidationError("unsupported controller profile schema")
    if profile["execution_model_version"] != EXECUTION_MODEL_VERSION:
        raise ProfileValidationError("execution-model version mismatch")
    if profile["workload_hash"] != workload_hash():
        raise ProfileValidationError("profiling workload hash mismatch")
    if profile["profile_hash"] != canonical_profile_hash(profile):
        raise ProfileValidationError("controller profile hash mismatch")

    kind = profile.get("profile_kind", "hardware_measurement")
    if kind not in {"simulated_model", "hardware_measurement"}:
        raise ProfileValidationError("unknown controller profile kind")
    target = profile["target"]
    for key in (
        "model", "machine", "power_mode", "clocks", "jetpack",
        "reference_frequency_hz",
    ):
        if key not in target or target[key] in ("", None):
            raise ProfileValidationError(f"target metadata missing {key}")
    model = str(target["model"]).lower()
    machine = str(target["machine"]).lower()
    is_orin = "jetson orin nano" in model and machine in {"aarch64", "arm64"}
    if kind == "hardware_measurement" and not is_orin and not allow_test_profile:
        raise ProfileValidationError(
            "final profile must be measured on an aarch64 Jetson Orin Nano")
    ref_hz = float(target["reference_frequency_hz"])
    if not math_is_finite_positive(ref_hz):
        raise ProfileValidationError("invalid reference frequency")
    if kind == "hardware_measurement" and not allow_test_profile:
        for key in ("power_mode", "clocks", "jetpack"):
            value = str(target[key]).strip().lower()
            if value in {"test", "unknown", "none", "n/a"}:
                raise ProfileValidationError(
                    f"target metadata {key} is provisional")

    runtime = profile["runtime"]
    for key in ("python", "pytorch", "numpy"):
        if not str(runtime.get(key, "")).strip():
            raise ProfileValidationError(f"runtime metadata missing {key}")
    threads = profile["thread_settings"]
    if int(threads.get("torch_num_threads", 0)) != 1:
        raise ProfileValidationError("torch_num_threads must equal 1")
    if int(threads.get("torch_num_interop_threads", 0)) != 1:
        raise ProfileValidationError("torch_num_interop_threads must equal 1")
    for key in ("omp_num_threads", "mkl_num_threads"):
        value = str(threads.get(key, "")).strip()
        if value not in {"", "1"}:
            raise ProfileValidationError(f"{key} must be 1 when set")
    measurement = profile["measurement"]
    expected_clock = (
        "configured_deterministic_model"
        if kind == "simulated_model" else "time.process_time_ns")
    if measurement.get("clock") != expected_clock:
        raise ProfileValidationError("controller profile clock/source mismatch")
    if (int(measurement.get("warmups_per_algorithm", 0)) < 10
            or int(measurement.get("samples_per_algorithm", 0)) < 100):
        if kind == "hardware_measurement" and not allow_test_profile:
            raise ProfileValidationError(
                "profile-level warmup/sample counts are insufficient")

    algorithms = profile["algorithms"]
    if set(algorithms) != set(REQUIRED_ALGORITHMS):
        raise ProfileValidationError(
            "profile must contain exactly the required controller paths")
    for name, stats in algorithms.items():
        for key in (
            "samples", "warmups", "mean_cpu_s", "median_cpu_s",
            "p95_cpu_s", "normalized_cycles",
        ):
            if key not in stats:
                raise ProfileValidationError(f"{name} missing {key}")
        if int(stats["samples"]) < 100 or int(stats["warmups"]) < 10:
            if kind == "hardware_measurement" and not allow_test_profile:
                raise ProfileValidationError(
                    f"{name} has insufficient warmup/sample counts")
        for key in ("mean_cpu_s", "median_cpu_s", "p95_cpu_s",
                    "normalized_cycles"):
            if not math_is_finite_positive(float(stats[key]), allow_zero=True):
                raise ProfileValidationError(f"{name} has invalid {key}")
        expected = float(stats["mean_cpu_s"]) * ref_hz
        if not np.isclose(
            expected, float(stats["normalized_cycles"]), rtol=1e-9, atol=1e-6
        ):
            raise ProfileValidationError(f"{name} normalized cycles mismatch")
        if name in STAGE1_ALGORITHMS:
            from config import PROFILE_FOG_SIZES
            by_n_fog = stats.get("by_n_fog")
            expected_sizes = {str(n) for n in PROFILE_FOG_SIZES}
            if (not isinstance(by_n_fog, Mapping)
                    or not expected_sizes.issubset(by_n_fog)):
                raise ProfileValidationError(
                    f"{name} by_n_fog must cover at least {sorted(expected_sizes)}")
            for n_fog, sized_stats in by_n_fog.items():
                for key in (
                    "samples", "warmups", "mean_cpu_s", "median_cpu_s",
                    "p95_cpu_s", "normalized_cycles",
                ):
                    if key not in sized_stats:
                        raise ProfileValidationError(
                            f"{name} by_n_fog[{n_fog}] missing {key}")
                for key in ("mean_cpu_s", "median_cpu_s", "p95_cpu_s",
                            "normalized_cycles"):
                    if not math_is_finite_positive(
                        float(sized_stats[key]), allow_zero=True):
                        raise ProfileValidationError(
                            f"{name} by_n_fog[{n_fog}] has invalid {key}")
                expected = float(sized_stats["mean_cpu_s"]) * ref_hz
                if not np.isclose(
                    expected, float(sized_stats["normalized_cycles"]),
                    rtol=1e-9, atol=1e-6,
                ):
                    raise ProfileValidationError(
                        f"{name} by_n_fog[{n_fog}] normalized cycles mismatch")
    return dict(profile)


def math_is_finite_positive(value: float, allow_zero: bool = False) -> bool:
    return bool(np.isfinite(value) and (value >= 0 if allow_zero else value > 0))


def load_profile(
    path: os.PathLike | str, *, allow_test_profile: bool = False
) -> Dict:
    try:
        profile = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileValidationError(
            f"cannot read controller profile {path}: {exc}") from exc
    return validate_profile(profile, allow_test_profile=allow_test_profile)


def require_profile(path: os.PathLike | str = DEFAULT_PROFILE_PATH) -> Dict:
    path = Path(path)
    if not path.exists():
        if path == DEFAULT_PROFILE_PATH:
            create_simulated_profile(path)
        else:
            raise ProfileValidationError(f"controller profile missing: {path}")
    return load_profile(path, allow_test_profile=False)


def controller_cycles(
    profile: Mapping, algorithm: str, n_fog: int | None = None,
) -> float:
    if algorithm not in REQUIRED_ALGORITHMS:
        raise KeyError(algorithm)
    stats = profile["algorithms"][algorithm]
    by_n_fog = stats.get("by_n_fog")
    if by_n_fog is not None:
        key = None if n_fog is None else str(n_fog)
        if key not in by_n_fog:
            raise ProfileValidationError(
                f"{algorithm} has no measured price for n_fog={n_fog}")
        stats = by_n_fog[key]
    return float(stats["normalized_cycles"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    simulated = sub.add_parser(
        "simulate", help="write the deterministic simulation controller model")
    simulated.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    run = sub.add_parser("profile", help="measure all controller paths")
    run.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    run.add_argument("--power-mode", required=True)
    run.add_argument("--clocks", required=True)
    run.add_argument("--jetpack", required=True)
    run.add_argument("--reference-frequency-hz", type=float, required=True)
    run.add_argument("--warmups", type=int, default=100)
    run.add_argument("--samples", type=int, default=2000)
    check = sub.add_parser("validate", help="strictly validate an imported profile")
    check.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "simulate":
        profile = create_simulated_profile(args.output)
        validate_profile(profile)
        print(f"validated simulated controller model: {args.output}")
        print(f"profile hash: {profile['profile_hash']}")
        return 0
    if args.command == "profile":
        if args.warmups < 10 or args.samples < 100:
            parser.error("use at least 10 warmups and 100 samples")
        profile = create_profile(
            output=args.output, power_mode=args.power_mode,
            clocks=args.clocks, jetpack=args.jetpack,
            reference_frequency_hz=args.reference_frequency_hz,
            warmups=args.warmups, samples=args.samples,
        )
        try:
            validate_profile(profile)
        except ProfileValidationError as exc:
            print(f"PROFILE WRITTEN BUT NOT FINAL-COMPATIBLE: {exc}", file=sys.stderr)
            return 2
        print(f"validated profile: {args.output}")
        print(f"profile hash: {profile['profile_hash']}")
        return 0
    profile = load_profile(args.path)
    print(f"valid {profile.get('profile_kind', 'hardware')} profile: "
          f"{profile['profile_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
