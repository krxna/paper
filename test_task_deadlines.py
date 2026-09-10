"""Regression test for the task-deadline configuration.

Guards the inconsistency found in the 2026-09 audit, where the manuscript's
workload model had drifted to a 300-ms small-task deadline while every reported
Stage-1 and Stage-2 run had actually been generated with 150 ms.  The rule lives
in exactly one place -- config.deadline_for_size_ms -- and these tests assert the
real generators resolve to it rather than re-stating the constants.
"""

import random

import config
from models import Task
from scenario import KeyedStreams, generate_paired_tasks


def test_threshold_boundary_uses_the_configured_deadlines():
    below = config.SMALL_TASK_THRESHOLD_KB - 1e-9
    assert config.deadline_for_size_ms(below) == config.DEADLINE_SMALL_MS
    assert config.deadline_for_size_ms(
        config.SMALL_TASK_THRESHOLD_KB) == config.DEADLINE_LARGE_MS
    assert config.deadline_for_size_ms(
        config.TASK_SIZE_MAX_KB) == config.DEADLINE_LARGE_MS


def test_reported_experiments_ran_at_150_over_550_ms():
    """The published Stage-1/Stage-2 numbers were produced under this rule.

    figures_head_final/final/runs_cache.pkl carries only 150.0/550.0 ms task
    deadlines, and fog_head_v3_modal/fog_head_representatives.json imputes
    missed tasks at exactly those two values.  Changing either constant
    invalidates every reported delivery rate and deadline-imputed delay.
    """
    assert config.DEADLINE_SMALL_MS == 150.0
    assert config.DEADLINE_LARGE_MS == 550.0
    assert config.SMALL_TASK_THRESHOLD_KB == 250.0


def test_admission_margin_is_separate_from_the_deadline():
    """The 25-ms buffer is applied at admission, never folded into the SLA."""
    assert config.ADMISSION_MARGIN_MS == 25.0
    for size in (60.0, 100.0, 249.0, 250.0, 800.0):
        deadline = config.deadline_for_size_ms(size)
        assert deadline in (config.DEADLINE_SMALL_MS, config.DEADLINE_LARGE_MS)
        assert deadline != config.DEADLINE_SMALL_MS - config.ADMISSION_MARGIN_MS
        assert deadline != config.DEADLINE_LARGE_MS - config.ADMISSION_MARGIN_MS


def test_training_and_evaluation_generators_agree():
    """models.Task.generate (training) and scenario.generate_paired_tasks
    (evaluation) must resolve the same rule -- no per-stage override exists."""
    random.seed(20260906)
    training = [Task.generate(task_id=i, arrival_s=0.001 * i) for i in range(2000)]
    evaluation = generate_paired_tasks(
        KeyedStreams(20260906), arrival_rate=180.0, end_s=12.0)
    assert training and evaluation

    for task in training + evaluation:
        assert task.deadline_ms == config.deadline_for_size_ms(task.size_kb)
        assert task.is_small == (task.size_kb < config.SMALL_TASK_THRESHOLD_KB)

    assert {t.deadline_ms for t in training} == {
        config.DEADLINE_SMALL_MS, config.DEADLINE_LARGE_MS}
    assert ({t.deadline_ms for t in evaluation}
            == {t.deadline_ms for t in training})


def test_no_stale_deadline_literals_in_task_generation():
    """The size->deadline branch exists once, in config."""
    import inspect
    for module_fn in (Task.generate, generate_paired_tasks):
        src = inspect.getsource(module_fn)
        assert 'deadline_for_size_ms' in src
        assert '150' not in src and '300' not in src and '550' not in src


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name)
