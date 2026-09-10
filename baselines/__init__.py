"""
baselines/__init__.py
=====================
Makes the baselines folder importable as a package.

Currently contains:
  - fodas_baseline  : deterministic EDF/completion-time FODAS scheduler
                      (heuristic_fodas variant; PPO-RNN variant is future work)

Usage inside experiments.py:
    from baselines.fodas_baseline import select_node_for_task
"""
