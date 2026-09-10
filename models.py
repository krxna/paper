"""
models.py
=========
Core data structures: Task, FogNode, and IoTDevice.

These are plain dataclasses describing the simulated entities. Task.generate
samples a new task; FogNode carries per-UAV state (energy, queues, position).
"""

from dataclasses import dataclass, field
import random

from config import *

# SECTION 5 — DATACLASSES
# ==============================================================================

# ------------------------------------------------------------------------------
# §5.1  Task
#
# FEASIBILITY ARGUMENT (why deadlines are achievable):
#   Compute time T_e = cycles / fr_hz = (size_kb * 1024 * c) / (fr_avg * 1e9),
#   where c ~ Uniform[CYCLES_PER_BYTE_MIN, CYCLES_PER_BYTE_MAX] = [1600, 1800] cyc/byte
#   ([200, 225] cycles/bit, the feasible lower segment of Stackelberg §IV-A's [200,1400]).
#   Worst case is always c = 1800 (the range maximum):
#   For the smallest task on the slowest tier (min intensity):
#     T_e = (60 * 1024 * 1600) / (1.5e9) ≈ 65.5 ms
#   For the largest task on the slowest tier (max intensity):
#     T_e = (800 * 1024 * 1800) / (1.5e9) ≈ 983 ms  — exceeds any deadline => must route to Tier 1
#   For the largest task on the fastest tier (Tier 1, 3.0 GHz, max intensity):
#     T_e = (800 * 1024 * 1800) / (3.0e9) ≈ 491 ms  < DEADLINE_LARGE_MS (550 ms) ✓
#   With wireless + processing overhead budget Omega = 50 ms added:
#     Small tasks (<250 KB) carry DEADLINE_SMALL_MS = 150 ms. At the top of the intensity
#     range a near-threshold small task needs (250*1024*1800)/(3.0e9) ~= 154 ms even on the
#     FASTEST node, so the tightest small tasks are deliberately infeasible: the 150 ms SLA
#     is a hard routing pressure, not a guarantee, and the reported miss rates reflect that.
#     Typical small tasks are comfortably feasible, e.g. 100 KB at 1700 cyc/byte on a 2.0 GHz
#     node takes ~87 ms.
#     Large tasks need Tier 1 or Tier 2: routing pressure is non-trivial but feasible [Mao17].
#   This ensures every task CAN be served (no infeasible task) while hard routing choices remain.
#   NOTE: tiers are now EMERGENT (assigned by ranking nodes on computational efficiency;
#   see config §1.2 / world.build_fog_swarm), but build_fog_swarm pins the clock envelope
#   endpoints (1.5 GHz slowest, 3.0 GHz fastest) every episode, so the bounds above —
#   and therefore this feasibility guarantee — remain exact.
# ------------------------------------------------------------------------------
@dataclass
class Task:
    task_id:     int          # unique task identifier (sequential within episode)
    size_kb:     float        # payload size, KB (bounded Pareto sample)
    cycles:      float        # total CPU cycles required: size_kb * 1024 * c_i, c_i ~ U[MIN,MAX] cyc/byte
    deadline_ms: float        # service deadline in ms from arrival
    arrival_s:   float        # absolute arrival time within episode, seconds
    is_small:    bool         # True if size_kb < SMALL_TASK_THRESHOLD_KB

    @classmethod
    def generate(cls, task_id: int, arrival_s: float) -> "Task":
        """
        Sample a new task.  Size drawn from bounded Pareto distribution.

        Pareto gives heavy-tailed payloads: most tasks are small, but a few are very large.
        This matches empirical IoT traffic distributions [Mao17 §III-A].
        Rejection sampling keeps every task in [TASK_SIZE_MIN_KB, TASK_SIZE_MAX_KB] so the
        compute-time feasibility argument above always holds.
        """
        # Pareto sample: x_m * (1 - u)^(-1/alpha) where u ~ Uniform(0,1).
        # x_m = TASK_SIZE_MIN_KB is the scale (minimum) of the distribution.
        # We reject samples above the max bound (rare; Pareto tail probability is small for alpha=1.5).
        while True:
            u = random.random()                          # uniform draw in (0, 1)
            raw = TASK_SIZE_MIN_KB * (1.0 - u) ** (-1.0 / PARETO_ALPHA)  # Pareto variate
            if raw <= TASK_SIZE_MAX_KB:                  # accept only in-bounds samples
                size_kb = raw
                break

        # Per-task computational intensity: sampled uniformly from the feasible lower
        # segment [CYCLES_PER_BYTE_MIN, CYCLES_PER_BYTE_MAX] of Stackelberg §IV-A's
        # [200, 1400] cycles/bit range (heterogeneous application mix).
        cpb = random.uniform(CYCLES_PER_BYTE_MIN, CYCLES_PER_BYTE_MAX)
        cycles = size_kb * 1024.0 * cpb                 # total CPU cycles

        is_small    = size_kb < SMALL_TASK_THRESHOLD_KB  # classify task size
        deadline_ms = deadline_for_size_ms(size_kb)

        return cls(
            task_id=task_id,
            size_kb=size_kb,
            cycles=cycles,
            deadline_ms=deadline_ms,
            arrival_s=arrival_s,
            is_small=is_small,
        )


# ------------------------------------------------------------------------------
# §5.2  FogNode  — one UAV acting as a fog compute node
# ------------------------------------------------------------------------------
@dataclass
class FogNode:
    node_id:     int          # unique node index (0-based, across all tiers)
    tier:        int          # hardware tier: 1 (best) .. 3 (lightest)
    fr_avg_ghz:  float        # mean CPU frequency, GHz  (fr_avg = (fr_min+fr_max)/2)
    MP_tot_gb:   float        # total primary memory (RAM), GB
    MS_tot_gb:   float        # total secondary storage, GB
    MP_occ_gb:   float        # currently occupied primary memory, GB
    MS_occ_gb:   float        # currently occupied storage, GB
    E_initial_j: float        # full-charge battery energy, J
    E_res_j:     float        # residual (remaining) battery energy, J
    lambda_fail: float        # compute failure rate, 1/s (higher = less reliable)
    mu_fail:     float        # link failure rate, 1/s
    varsigma:    float        # compute weight (lower = higher compute capability)
    C_mips:      float         # CPU throughput, MIPS
    x:           float        # position x-coordinate, m  (within [0, GRID_M])
    y:           float        # position y-coordinate, m
    # --- Mobility state (UAV_Mobility_Design.pdf; world.assign_mobility / step_mobility) ---
    z:           float = 0.0  # fixed flight altitude, m (0 in 'static' mode = original sim;
                              # an 80–150 m layer in moving modes; enters 3-D slant range only)
    speed_ms:    float = 0.0  # current horizontal speed V, m/s (sets Zeng19 propulsion power P(V))
    vx_ms:       float = 0.0  # horizontal velocity x component, m/s
    vy_ms:       float = 0.0  # horizontal velocity y component, m/s
    hotspot_idx: int   = -1   # assigned IoT demand hotspot (event_driven target; -1 = unassigned)
    slot_idx:    int   = -1   # slot on the hotspot standoff ring / altitude layer (0..4)
    wp_x:        float = 0.0  # current random-waypoint target x, m ('random' mode only)
    wp_y:        float = 0.0  # current random-waypoint target y, m
    # Each queue entry is a tuple (Task, deadline_abs_ms) where deadline_abs_ms is the
    # wall-clock deadline of the task expressed as ms from episode start.
    queue_primary: list = field(default_factory=list)   # tasks routed here as PRIMARY
    queue_backup:  list = field(default_factory=list)   # tasks routed here as BACKUP
    # Attached by execution_engine.ExecutionEngine. Kept out of equality/repr so
    # old serialized worlds remain loadable.
    cpu_state: object = field(default=None, repr=False, compare=False)
    available: bool = True

    @property
    def soc(self) -> float:
        """State of charge: residual / initial energy, clamped to [0, 1]."""
        return max(0.0, min(1.0, self.E_res_j / self.E_initial_j))

    def fr_avg_hz(self) -> float:
        """CPU frequency in Hz (used in delay/energy formulae)."""
        return self.fr_avg_ghz * 1e9   # GHz -> Hz


# ------------------------------------------------------------------------------
# §5.3  IoTDevice  — a ground sensor or terminal generating tasks
# ------------------------------------------------------------------------------
@dataclass
class IoTDevice:
    dev_id:       int     # unique device index (0-based)
    x:            float   # position x-coordinate, m
    y:            float   # position y-coordinate, m
    p_tx_w:       float   # uplink transmit power, W  (2.51 W low / 31.62 W high)
    is_high_power: bool   # True = 45 dBm terminal; False = 34 dBm sensor
