import math

from execution_engine import AttemptStatus, BackupPlan, ExecutionEngine
from models import FogNode


def node(node_id=0, ghz=1.0):
    return FogNode(
        node_id=node_id, tier=1, fr_avg_ghz=ghz,
        MP_tot_gb=4.0, MS_tot_gb=8.0,
        MP_occ_gb=0.0, MS_occ_gb=0.0,
        E_initial_j=1_000_000.0, E_res_j=1_000_000.0,
        lambda_fail=0.0, mu_fail=0.0, varsigma=4.0,
        C_mips=16_000.0, x=0.0, y=0.0, z=100.0,
    )


def submit(engine, task_id, cycles, deadline_s, **kwargs):
    return engine.submit_task(
        task_id=task_id, arrival_s=0.0, deadline_s=deadline_s,
        cycles=cycles, size_kb=100.0, primary_node_id=0, **kwargs)


def test_on_time_service_is_productive_and_conserves_capacity():
    engine = ExecutionEngine([node()])
    submit(engine, 1, 100_000_000.0, 1.0)
    engine.run(1.0)
    engine.drain_relevant(1.0)
    result = engine.capacity_result(0.0, 1.0)
    util = result.utilization()
    assert math.isclose(util["productive"], 0.1)
    assert math.isclose(util["wasted"], 0.0)
    assert math.isclose(sum(util[k] for k in (
        "productive", "wasted", "control", "idle")), 1.0)


def test_running_late_attempt_is_wholly_wasted():
    engine = ExecutionEngine([node()])
    submit(engine, 1, 100_000_000.0, 0.05)
    engine.run()
    attempt = engine.attempts["task-1:primary"]
    assert attempt.status == AttemptStatus.LATE
    assert math.isclose(attempt.service_cycles, 100_000_000.0)
    result = engine.capacity_result(0.0, 0.2)
    assert math.isclose(result.wasted_cycles, 100_000_000.0)
    assert result.productive_cycles == 0.0


def test_control_preempted_started_task_resumes_after_deadline():
    engine = ExecutionEngine([node()])
    submit(engine, 1, 200_000_000.0, 0.15)
    engine.run(0.05)
    engine.submit_control(
        control_id="ctrl", node_id=0, release_s=0.05,
        cycles=200_000_000.0, algorithm="test")
    assert math.isclose(
        engine.attempts["task-1:primary"].remaining_cycles,
        150_000_000.0)
    engine.run()
    attempt = engine.attempts["task-1:primary"]
    assert attempt.status == AttemptStatus.LATE
    assert math.isclose(attempt.service_cycles, 200_000_000.0)
    assert any(row.get("event") == "task_resume" for row in engine.audit_rows())


def test_queued_expiry_consumes_no_cpu():
    engine = ExecutionEngine([node()])
    submit(engine, 1, 200_000_000.0, 1.0)
    submit(engine, 2, 100_000_000.0, 0.05)
    engine.run()
    queued = engine.attempts["task-2:primary"]
    assert queued.status == AttemptStatus.EXPIRED
    assert queued.service_cycles == 0.0


def test_controller_projection_is_a_causal_fifo_dependency():
    engine = ExecutionEngine([node()])
    engine.submit_control(
        control_id="a", node_id=0, release_s=0.0,
        cycles=100_000_000.0, algorithm="test")
    assert math.isclose(engine.projected_control_completion_s("a"), 0.1)
    engine.run(0.02)
    engine.submit_control(
        control_id="b", node_id=0, release_s=0.02,
        cycles=50_000_000.0, algorithm="test")
    assert math.isclose(engine.projected_control_completion_s("b"), 0.15)
    assert math.isclose(engine.control_backlog_cycles(0), 130_000_000.0)
    assert math.isclose(
        engine.projected_control_service_s(0, 20_000_000.0), 0.15)


def test_transmission_energy_uses_actual_service_duration():
    n = node()
    engine = ExecutionEngine([n])
    submit(
        engine, 1, 10_000_000.0, 1.0,
        primary_upload_s=0.1, primary_rx_power_w=2.0)
    engine.run()
    assert math.isclose(engine.nodes[0].communication_energy_j, 0.2)
    assert math.isclose(n.E_res_j, 1_000_000.0 - 0.2 - 0.1)


def test_node_depletion_fails_running_primary_and_releases_backup():
    primary, backup = node(0), node(1)
    engine = ExecutionEngine([primary, backup])
    submit(
        engine, 1, 200_000_000.0, 1.0,
        backup=BackupPlan(node_id=1, fresh_upload_s=0.0, forward_s=0.0))
    engine.run(0.05)
    engine.set_node_available(0, False, reason="test_depletion")
    engine.run()
    assert engine.attempts["task-1:primary"].status == AttemptStatus.FAILED
    assert engine.outcomes[1].backup_attempt_id == "task-1:backup"
    assert engine.outcomes[1].success


def test_service_slices_never_overlap_on_one_cpu():
    engine = ExecutionEngine([node()])
    submit(engine, 1, 100_000_000.0, 1.0)
    submit(engine, 2, 100_000_000.0, 1.0)
    engine.run()
    slices = sorted(engine.service_slices, key=lambda row: row.start_s)
    assert all(a.end_s <= b.start_s + 1e-12 for a, b in zip(slices, slices[1:]))


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().copy().items()
             if name.startswith("test_") and callable(obj)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} execution-engine tests passed")
