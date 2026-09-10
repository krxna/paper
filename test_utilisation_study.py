import math
from dataclasses import replace

from controller_profile import require_profile
from baselines.relief_baseline import ReLIEFAgent, ReLIEFConfig
from config import MAX_QUEUE_DEPTH, W_LINK_ENERGY, deadline_for_size_ms
from execution_engine import ExecutionEngine
from models import Task
from neural import expected_wait_ms, filter_candidates
from scenario import ScenarioConfig, _world_for_seed, make_policy, run_scenario
from utilisation_study import (
    recompute_capacity_from_audit, relief_random_differences)


PROFILE = require_profile()


def run(policy, *, placement="native"):
    cfg = ScenarioConfig(
        seed=17, target_load=0.5, policy=policy,
        controller_placement=placement,
        warmup_s=0.2, measurement_s=1.0)
    return run_scenario(cfg, make_policy(policy, cfg.seed), PROFILE)


def test_policies_share_identical_exogenous_inputs_and_realized_load():
    random_result = run("random")
    fodas_result = run("FODAS")
    assert random_result.input_fingerprint == fodas_result.input_fingerprint
    assert math.isclose(random_result.realized_load, fodas_result.realized_load)


def test_propulsion_and_communication_energy_are_accounted():
    result = run("random")
    energy = result.energy_totals
    assert energy["propulsion_j"] > 0.0
    assert energy["uav_communication_j"] > 0.0
    assert math.isclose(
        energy["uav_total_j"],
        energy["propulsion_j"] + energy["task_compute_j"]
        + energy["control_compute_j"] + energy["uav_communication_j"])


def test_equal_controller_charges_random_decision_cpu():
    native = run("random", placement="native")
    equal = run("random", placement="equal_controller")
    assert native.capacity.control_cycles == 0.0
    assert equal.capacity.control_cycles > 0.0
    assert native.external_control_cycles["random"] > 0.0


def test_exact_edf_wait_and_doomed_task_admission_gate():
    fog, iot, _ = _world_for_seed(42)
    engine = ExecutionEngine(fog)
    engine.submit_control(
        control_id="queued-control", node_id=0, release_s=0.0,
        cycles=0.02 * fog[0].fr_avg_hz(), algorithm="test")
    task = Task(
        task_id=1, size_kb=100.0, cycles=3.0e9,
        deadline_ms=1.0, arrival_s=0.0, is_small=True)
    assert math.isclose(expected_wait_ms(fog[0], task), 20.0)
    mask, _ = filter_candidates(fog, task, iot[0])
    assert not mask.any()


def test_candidate_score_penalizes_farther_link_energy():
    fog, iot, _ = _world_for_seed(42)
    dev, base = iot[0], fog[0]
    near = replace(base, node_id=0, x=dev.x + 500.0, y=dev.y, z=0.0)
    far = replace(base, node_id=1, x=dev.x + 1000.0, y=dev.y, z=0.0)
    task = Task(
        task_id=1, size_kb=100.0, cycles=100.0 * 1024.0 * 1700.0,
        deadline_ms=deadline_for_size_ms(100.0), arrival_s=0.0, is_small=True)
    mask, scores = filter_candidates([near, far], task, dev)
    assert mask.all()
    assert scores[0] - scores[1] > 0.9 * W_LINK_ENERGY


def test_event_audit_reconstructs_every_capacity_category():
    result = run("FODAS", placement="equal_controller")
    frequencies = {
        node_id: values["provisioned_cycles"] / result.config.measurement_s
        for node_id, values in result.capacity.per_node.items()}
    rebuilt = recompute_capacity_from_audit(
        result.audit, frequencies,
        result.config.measurement_start_s,
        result.config.measurement_end_s)
    for node_id, expected in result.capacity.per_node.items():
        for key in (
            "productive_cycles", "wasted_cycles", "control_cycles",
            "idle_cycles",
        ):
            assert math.isclose(
                rebuilt[node_id][key], expected[key],
                rel_tol=2e-9, abs_tol=1e-4)


def test_relief_updates_online_in_shared_scenario():
    cfg = ScenarioConfig(
        seed=17, target_load=0.5, policy="ReLIEF",
        warmup_s=0.0, measurement_s=0.2)
    adapter = make_policy("ReLIEF", cfg.seed, relief_pretrain=False)
    run_scenario(cfg, adapter, PROFILE)
    assert adapter.agent.steps > 0
    assert adapter.log10_reliability_ema < 0.0


def test_relief_feasibility_uses_live_queue_depth_and_availability():
    fog, _iot, _centres = _world_for_seed(17)

    class FullEngine:
        def queue_depth(self, _node_id):
            return MAX_QUEUE_DEPTH

    fog[0].cpu_state = type("CpuState", (), {"_engine": FullEngine()})()
    fog[1].available = False
    feasible = ReLIEFAgent(ReLIEFConfig())._feasible_nodes(fog, 0.0)
    assert 0 not in feasible
    assert 1 not in feasible


def test_relief_random_delta_is_paired_by_seed():
    rows = []
    for seed, random_value, relief_value in ((0, 0.10, 0.12), (1, 0.20, 0.19)):
        for policy, value in (("random", random_value), ("ReLIEF", relief_value)):
            rows.append({
                "controller_placement": "native", "target_load": 1.0,
                "seed": seed, "policy": policy, "productive": value})
    native = [
        row for row in relief_random_differences(rows)
        if row["controller_placement"] == "native"]
    assert len(native) == 1
    assert native[0]["n_pairs"] == 2
    assert math.isclose(native[0]["mean_paired_difference"], 0.5)


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().copy().items()
             if name.startswith("test_") and callable(obj)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} utilization-study tests passed")
