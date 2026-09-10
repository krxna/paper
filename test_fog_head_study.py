import math

import numpy as np

from broker import BrokerSelector, residual_coordination_time_s
from config import (HEAD_AVAILABILITY_HORIZON_S, HEAD_COMM_RANGE_M,
                    HEAD_MIN_TENURE_TICKS, KALMAN_M_Q, KALMAN_M_R,
                    HW_FR_STRONG_GHZ, HW_FR_WEAK_GHZ, N_CRITERIA,
                    set_global_seed)
from execution_engine import ExecutionEngine
from controller_profile import require_profile
from experiments import control_plane_cost
from models import FogNode, IoTDevice
from physics import head_operational_availability
from scenario import (AttentionPPOPolicyAdapter, ScenarioConfig,
                      _world_for_seed, make_policy, run_scenario)
from world import (assign_mobility, build_fog_swarm, build_iot_devices,
                   generate_task_stream, generate_task_stream_for_duration)


def node(node_id, *, x=0.0, y=0.0):
    return FogNode(node_id=node_id, tier=1, fr_avg_ghz=2.0,
                   MP_tot_gb=4.0, MS_tot_gb=4.0, MP_occ_gb=0.0,
                   MS_occ_gb=0.0, E_initial_j=1000.0, E_res_j=1000.0,
                   lambda_fail=0.01, mu_fail=0.01, varsigma=4.0,
                   C_mips=16000.0, x=x, y=y, z=100.0)


def test_fixed_horizon_stream_is_reproducible_and_bounded():
    set_global_seed(123)
    first = generate_task_stream_for_duration(2.0, 100.0)
    set_global_seed(123)
    second = generate_task_stream_for_duration(2.0, 100.0)
    assert [t.arrival_s for t in first] == [t.arrival_s for t in second]
    assert first
    assert all(0.0 < t.arrival_s < 2.0 for t in first)
    assert all(a.arrival_s < b.arrival_s for a, b in zip(first, first[1:]))


def test_explicit_arrival_rate_changes_time_scale():
    set_global_seed(321)
    slow = generate_task_stream(100, arrival_rate=50.0)
    set_global_seed(321)
    fast = generate_task_stream(100, arrival_rate=200.0)
    assert math.isclose(slow[-1].arrival_s / fast[-1].arrival_s, 4.0)


def test_control_plane_cost_depends_on_elected_head_geometry():
    dev = IoTDevice(dev_id=0, x=0.0, y=0.0, p_tx_w=2.51,
                    is_high_power=False)
    near_head = node(0, x=0.0)
    far_head = node(1, x=4000.0)
    executor = node(2, x=1000.0)
    near = control_plane_cost(dev, near_head, executor)
    far = control_plane_cost(dev, far_head, executor)
    assert near["delay_ms"] < far["delay_ms"]
    assert near["swarm_energy_j"] < far["swarm_energy_j"]
    assert near["edge_energy_j"] < far["edge_energy_j"]


def test_headless_control_path_has_zero_cost():
    dev = IoTDevice(dev_id=0, x=0.0, y=0.0, p_tx_w=2.51,
                    is_high_power=False)
    out = control_plane_cost(dev, None, node(0))
    assert all(value == 0.0 for value in out.values())


def test_head_availability_uses_declared_mission_horizon():
    candidate = node(0)
    expected = math.exp(-(candidate.lambda_fail + candidate.mu_fail) * 10.0)
    assert math.isclose(head_operational_availability(candidate, 10.0), expected)
    assert 0.0 < expected < 1.0


def test_measurement_filter_damps_a_single_candidate_state_step():
    fog = [node(0), node(1)]
    selector = BrokerSelector(final_fix=True)
    good = {"me": 1.0, "r0": 0.99, "d_ms": 10.0, "wl": 0.0}
    poor = {"me": 0.2, "r0": 0.90, "d_ms": 400.0, "wl": 1e12}
    first = selector.select_head(fog, {0: good, 1: poor}, 0, 1)
    assert first["head_id"] == 0

    selector.head_tenure = HEAD_MIN_TENURE_TICKS
    changed = selector.select_head(fog, {0: poor, 1: good}, 0, 2)
    assert changed["kl_trigger"]
    assert changed["run_spotis"]
    assert not changed["state_trigger"]
    assert not changed["head_changed"]
    assert changed["head_id"] == 0


def test_measurement_kalman_step_and_node_independence():
    selector = BrokerSelector(final_fix=True)
    baseline = np.tile(selector.S_min, (2, 1))
    assert np.array_equal(
        selector.kalman_smooth(baseline, [10, 20]), baseline)
    q, r = KALMAN_M_Q, KALMAN_M_R
    x = (q + math.sqrt(q * q + 4.0 * q * r)) / 2.0
    gain = x / (x + r)
    steps = math.ceil(-1.0 / math.log1p(-gain))
    observed = baseline
    for _ in range(steps):
        step = np.vstack((selector.S_max, selector.S_min))
        observed = selector.kalman_smooth(step, [10, 20])
    expected = selector.S_min + (selector.S_max - selector.S_min) * (
        1.0 - (1.0 - gain) ** steps)
    assert np.allclose(observed[0], expected)
    assert np.array_equal(observed[1], baseline[1])
    selector.kalman_smooth(selector.S_max[None, :], [20])
    assert set(selector.measurement_state) == {20}


def test_predictive_criteria_shape_and_spotis_rank_reversal_freedom():
    fog = [node(0), node(1, x=100.0), node(2, x=200.0)]
    metrics = {
        candidate.node_id: {"me": 0.9, "r0": 0.99,
                            "d_ms": 1.0 + candidate.node_id,
                            "wl": 100.0 * candidate.node_id}
        for candidate in fog}
    selector = BrokerSelector(final_fix=True)
    matrix = selector.build_decision_matrix(fog, metrics)
    assert matrix.shape == (3, N_CRITERIA) == (3, 6)
    weights = np.full(N_CRITERIA, 1.0 / N_CRITERIA)
    full_scores = selector._spotis_distance(matrix, weights)
    reduced_scores = selector._spotis_distance(matrix[:2], weights)
    assert np.array_equal(full_scores[:2], reduced_scores)
    fixed_bounds = selector.S_min.copy(), selector.S_max.copy()
    fog[0].available = False
    selected = selector.select_head(fog, metrics, 0, 0)
    assert selected["head_node"].available
    assert all(candidate.available for candidate in selected["ranking"])
    assert np.array_equal(selector.S_min, fixed_bounds[0])
    assert np.array_equal(selector.S_max, fixed_bounds[1])


def test_rct_stationary_horizon_and_relative_speed():
    head = node(0)
    peer = node(1, x=HEAD_COMM_RANGE_M - 1.0)
    assert residual_coordination_time_s(
        head, [head, peer]) == HEAD_AVAILABILITY_HORIZON_S
    peer.vx_ms = 10.0
    assert residual_coordination_time_s(head, [head, peer]) < 1.0


def test_hypothetical_control_projection_matches_submitted_empty_queue():
    candidate = node(0)
    engine = ExecutionEngine([candidate])
    cycles = 2.5e6
    hypothetical = engine.projected_control_service_s(candidate.node_id, cycles)
    engine.submit_control(
        control_id="probe", node_id=candidate.node_id, release_s=0.0,
        cycles=cycles, algorithm="test")
    submitted = engine.projected_control_completion_s("probe") - engine.now_s
    assert math.isclose(hypothetical, submitted)


def test_challenger_inherits_incumbent_control_backlog_in_dctrl():
    fog = [node(0), node(1)]
    engine = ExecutionEngine(fog)
    engine.submit_control(
        control_id="incumbent-backlog", node_id=0, release_s=0.0,
        cycles=100_000_000.0, algorithm="test")
    adapter = AttentionPPOPolicyAdapter(
        None, None, head_mode="spotis", stage2_mode="fodas",
        final_head_selection=True)
    adapter.head = fog[0]
    captured = {}

    def select_head(_fog, metrics, _episode, _tick):
        captured.update(metrics)
        return {"head_node": fog[0]}

    adapter.broker.select_head = select_head
    adapter.elect_head(
        fog, [IoTDevice(0, 0.0, 0.0, 2.51, False)],
        type("TaskLike", (), {"cycles": 1e6, "size_kb": 1.0})(), 0)
    assert math.isclose(captured[0]["d_ms"], captured[1]["d_ms"])


def test_initial_soc_is_spread_seeded_and_keeps_clock_feasibility_envelope():
    first, _iot, _centres = _world_for_seed(91)
    second, _iot, _centres = _world_for_seed(91)
    first_soc = [candidate.soc for candidate in first]
    assert first_soc == [candidate.soc for candidate in second]
    assert max(first_soc) > min(first_soc)
    clocks = [candidate.fr_avg_ghz for candidate in first]
    assert min(clocks) == HW_FR_WEAK_GHZ
    assert max(clocks) == HW_FR_STRONG_GHZ


def test_induced_dead_head_recovers_without_new_control_arrivals_on_dead_node():
    for head_mode in ("spotis", "fu_serve", "2dp_fhs", "3d_pos"):
        adapter = make_policy(
            "head_fodas", 31, head_mode=head_mode)
        result = run_scenario(
            ScenarioConfig(
                seed=31, target_load=0.5, policy="head_fodas",
                controller_placement="head", warmup_s=0.0,
                measurement_s=1.0, arrival_rate_override=60.0,
                head_failure_s=0.3),
            adapter, require_profile())
        failure = next(
            row for row in result.audit
            if row.get("event") == "node_availability"
            and row.get("reason") == "induced_head_failure")
        dead_id, failed_at = failure["node_id"], failure["time_s"]
        assert math.isfinite(result.head_recovery_ticks)
        assert adapter.head is not None and adapter.head.available
        assert adapter.head.node_id != dead_id
        assert not any(
            row.get("event") == "control_arrival"
            and row.get("node_id") == dead_id
            and row.get("time_s", -math.inf) >= failed_at
            for row in result.audit)


def test_runtime_swarm_sizing_scales_world_without_global_mutation():
    set_global_seed(7)
    fog = build_fog_swarm(80)
    iot, centres = build_iot_devices(800)
    assign_mobility(fog, centres)
    assert len(fog) == 80 and len(iot) == 800
    assert sum(candidate.tier == 1 for candidate in fog) == 40
    assert sum(device.is_high_power for device in iot) == 200


def test_live_decision_power_gate_uses_all_fixed_bound_criteria():
    adapter = make_policy(
        "head_fodas", 7, head_mode="spotis", final_head_selection=True)
    result = run_scenario(
        ScenarioConfig(
            seed=7, target_load=0.96, policy="head_fodas",
            controller_placement="head", warmup_s=0.0, measurement_s=20.0,
            arrival_rate_override=180.0),
        adapter, require_profile())
    power = np.nanmean([
        [tick[f"decision_power_{index}"] for index in range(N_CRITERIA)]
        for tick in result.tick_records], axis=0)
    assert power[3] + power[4] > 0.45
    assert np.all(power > 0.0)


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().copy().items()
             if name.startswith("test_") and callable(obj)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} fog-head study tests passed")
