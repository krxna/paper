"""
world.py
========
World builders and world instantiation.

build_fog_swarm / build_iot_devices / generate_task_stream construct the
simulated entities. As in the original single-file script, importing this
module immediately builds the global FOG_SWARM, IOT_DEVICES, and TASK_STREAM
and prints the [WORLD] banner, so the world state is available to every other
module that imports it.
"""

import math
import random
from typing import List, Tuple

from config import *
from models import Task, FogNode, IoTDevice

# SECTION 6 — WORLD BUILDERS
# ==============================================================================

def build_fog_swarm(n_fog: int = N_FOG) -> List[FogNode]:
    """
    Build the 20-UAV swarm with the INVERTED pipeline:
        per-node hardware  ->  computational efficiency  ->  rank  ->  tier.

    Step 1 — Sample a latent quality q_j in [0,1] for each UAV.
    Step 2 — Affinely rescale the 20 q-values so the empirical min maps to 0 and
             the max to 1.  This PINS the hardware envelope endpoints every episode
             (one node at fr_avg = HW_FR_STRONG_GHZ, one at HW_FR_WEAK_GHZ), which
             keeps the task-feasibility guarantee in models.py exact.
    Step 3 — Interpolate every spec linearly on q (WEAK at q=0 .. STRONG at q=1).
             All specs are monotone in q, so a higher-quality drone is better
             all-round (fast clock, more memory, bigger battery, lower failure
             rates) — preserving the paper's "tiers of decreasing capability" story.
    Step 4 — Compute CE_eff per node, rank DESCENDING, and slice TIER_COUNTS
             (10 / 10) into Tier 1 / 2.  Tier is thus an EMERGENT label.

    Each UAV is placed at a random (x, y) uniform in [0, GRID_M]^2.
    Occupancy starts at zero; initial SOC is sampled from the configured range.
    """
    # --- Step 1: sample latent quality per node ---
    if n_fog < 2:
        raise ValueError("n_fog must be at least 2")
    raw_q = [random.random() for _ in range(n_fog)]

    # --- Step 2: affine rescale so min->0, max->1 (pins envelope endpoints) ---
    q_lo, q_hi = min(raw_q), max(raw_q)
    q_span = (q_hi - q_lo) if (q_hi - q_lo) > 1e-12 else 1.0
    q = [(v - q_lo) / q_span for v in raw_q]

    def _lerp(weak: float, strong: float, t: float) -> float:
        return weak + t * (strong - weak)

    # --- Step 3: interpolate hardware specs from q ---
    specs: List[dict] = []
    for j in range(n_fog):
        t = q[j]
        fr_avg   = _lerp(HW_FR_WEAK_GHZ,   HW_FR_STRONG_GHZ,   t)   # GHz
        c_mips   = _lerp(HW_MIPS_WEAK,     HW_MIPS_STRONG,     t)   # MIPS
        varsigma = _lerp(HW_VARSIGMA_WEAK, HW_VARSIGMA_STRONG, t)   # compute weight (lower=better)
        mp_tot   = _lerp(HW_MP_WEAK_GB,    HW_MP_STRONG_GB,    t)   # GB RAM
        ms_tot   = _lerp(HW_MS_WEAK_GB,    HW_MS_STRONG_GB,    t)   # GB storage
        e_init   = _lerp(HW_E_WEAK_J,      HW_E_STRONG_J,      t)   # J
        lam      = _lerp(HW_LAMBDA_WEAK,   HW_LAMBDA_STRONG,   t)   # 1/s
        mu       = _lerp(HW_MU_WEAK,       HW_MU_STRONG,       t)   # 1/s

        ce_eff = computational_efficiency_score(fr_avg, varsigma, c_mips)  # higher = better

        specs.append({
            "fr_avg": fr_avg, "c_mips": c_mips, "varsigma": varsigma,
            "mp_tot": mp_tot, "ms_tot": ms_tot, "e_init": e_init,
            "lambda_fail": lam, "mu_fail": mu, "ce_eff": ce_eff,
        })

    # A fixed one-point range consumes no RNG so SOC_INIT_MIN=SOC_INIT_MAX=1
    # reproduces the pre-heterogeneity world exactly.
    initial_soc = ([SOC_INIT_MIN] * n_fog if SOC_INIT_MIN == SOC_INIT_MAX else
                   [random.uniform(SOC_INIT_MIN, SOC_INIT_MAX)
                    for _ in range(n_fog)])

    # --- Step 4: rank by CE_eff (descending) and assign tiers by rank position ---
    # Ties broken by node index for determinism (CE_eff ties are measure-zero here).
    order = sorted(range(n_fog), key=lambda j: (specs[j]["ce_eff"], -j), reverse=True)
    tier_of: dict = {}
    pos = 0
    tier_counts = ((n_fog + 1) // 2, n_fog // 2)
    for tier_idx, count in enumerate(tier_counts):     # tier_idx 0,1 -> tier 1,2
        for _ in range(count):
            tier_of[order[pos]] = tier_idx + 1
            pos += 1

    # --- Step 5: instantiate FogNodes in node_id order ---
    nodes: List[FogNode] = []
    for j in range(n_fog):
        s = specs[j]
        x = random.uniform(0.0, GRID_M)              # random patrol position, m
        y = random.uniform(0.0, GRID_M)

        nodes.append(FogNode(
            node_id     = j,
            tier        = tier_of[j],                 # EMERGENT: from CE_eff rank
            fr_avg_ghz  = s["fr_avg"],
            MP_tot_gb   = s["mp_tot"],
            MS_tot_gb   = s["ms_tot"],
            MP_occ_gb   = 0.0,                        # no memory occupied at start
            MS_occ_gb   = 0.0,
            E_initial_j = s["e_init"],
            E_res_j     = s["e_init"] * initial_soc[j],
            lambda_fail = s["lambda_fail"],
            mu_fail     = s["mu_fail"],
            varsigma    = s["varsigma"],
            C_mips      = s["c_mips"],
            x           = x,
            y           = y,
        ))

    return nodes


def build_iot_devices(n_iot: int = N_IOT) -> Tuple[List[IoTDevice], List[List[float]]]:
    """
    Place 200 IoT devices using a 4-component Gaussian mixture model.

    4 random cluster centres simulate spatial hotspots (e.g., factory floors, warehouses).
    Each device is drawn from a uniformly-chosen cluster with std=400 m, then clipped to grid.
    Power assignment: first 150 = low-power sensors (2.51 W), last 50 = high-power terminals.

    Returns (devices, centres).  The centres are the demand hotspots the mobility
    layer tracks (UAV_Mobility_Design.pdf: event_driven UAVs blanket their assigned
    hotspot on a standoff ring); they are returned as MUTABLE [x, y] lists because
    step_mobility drifts them in place each control tick.
    """
    if n_iot < 1:
        raise ValueError("n_iot must be positive")
    std_m = 400.0   # cluster spread, m — 400 m radius hotspot is a realistic indoor/campus scale

    # Sample N_IOT_CLUSTERS random centres inside the grid (well inside, so devices stay in-grid).
    centres = [
        [random.uniform(500.0, GRID_M - 500.0),   # x-centre (500 m margin avoids edge clipping)
         random.uniform(500.0, GRID_M - 500.0)]    # y-centre (mutable: drifts in step_mobility)
        for _ in range(N_IOT_CLUSTERS)
    ]

    devices: List[IoTDevice] = []

    n_low_power = round(n_iot * N_LOW_POWER_IOT / N_IOT)
    for dev_id in range(n_iot):
        # Pick cluster uniformly at random for this device.
        cx, cy = random.choice(centres)

        # Sample position from Gaussian centred on cluster, then clip to [0, GRID_M].
        x = float(np.clip(np.random.normal(cx, std_m), 0.0, GRID_M))
        y = float(np.clip(np.random.normal(cy, std_m), 0.0, GRID_M))

        # First 150 devices are low-power; remaining 50 are high-power (matches §1.3).
        if dev_id < n_low_power:
            p_tx_w       = P_TX_LOW_W    # ~2.51 W
            is_high_power = False
        else:
            p_tx_w       = P_TX_HIGH_W   # ~31.62 W
            is_high_power = True

        devices.append(IoTDevice(
            dev_id       = dev_id,
            x            = x,
            y            = y,
            p_tx_w       = p_tx_w,
            is_high_power = is_high_power,
        ))

    return devices, centres


# ==============================================================================
# SECTION 6m — UAV MOBILITY  (UAV_Mobility_Design.pdf)
#
# Three modes (MOVE_MODE / UAV_MOVE_MODE env var):
#   event_driven — each UAV flies to a distinct slot on a HOTSPOT_RING_M standoff
#                  ring around its assigned (slowly drifting) demand hotspot and
#                  station-keeps there (FU-Serve / Con-Fog demand-following).
#   random       — random-waypoint motion at cruise speed, demand-agnostic
#                  ablation baseline (Con-Fog §V-B); never stops moving.
#   static       — no motion, z = 0: reproduces the original simulation exactly.
#
# Collision-free by design (PDF §3): co-assigned UAVs are separated in x-y (ring
# slots) AND in altitude (distinct layers in [ALT_MIN_M, ALT_MAX_M]), so no
# avoidance controller is needed.  Motion is horizontal only; z never changes.
# step_mobility sets each node's speed_ms, which the episode loop feeds into the
# Zeng19 power curve (physics.aero_power_w) to drain propulsion energy per tick.
# ==============================================================================

def _ring_slot_offset(slot: int, slots_per_hotspot: int = UAVS_PER_HOTSPOT) -> Tuple[float, float]:
    """(dx, dy) of ring slot `slot` on the HOTSPOT_RING_M standoff circle."""
    theta = 2.0 * math.pi * slot / slots_per_hotspot
    return HOTSPOT_RING_M * math.cos(theta), HOTSPOT_RING_M * math.sin(theta)


def assign_mobility(fog_nodes: List[FogNode], centres: List[List[float]]) -> None:
    """
    Initialise per-node mobility state for the current MOVE_MODE.

    Node j is assigned hotspot j // UAVS_PER_HOTSPOT and ring slot j % UAVS_PER_HOTSPOT,
    giving each of the N_IOT_CLUSTERS hotspots exactly UAVS_PER_HOTSPOT UAVs.  Each
    slot maps to a distinct altitude layer z in [ALT_MIN_M, ALT_MAX_M] (moving modes)
    so co-assigned UAVs are vertically separated.  In 'static' mode z stays 0 and
    nothing moves — byte-for-byte the original fixed-position simulation.
    Random-waypoint targets start at the node's own position so the first
    step_mobility call draws a fresh waypoint.
    """
    slots_per_hotspot = max(1, math.ceil(len(fog_nodes) / len(centres)))
    for node in fog_nodes:
        node.hotspot_idx = min(node.node_id // slots_per_hotspot, len(centres) - 1)
        node.slot_idx    = node.node_id % slots_per_hotspot
        node.speed_ms    = 0.0
        node.vx_ms = node.vy_ms = 0.0
        node.wp_x, node.wp_y = node.x, node.y
        if MOVE_MODE == "static":
            node.z = 0.0                       # original 2-D geometry preserved
        else:
            node.z = ALT_MIN_M + node.slot_idx * (ALT_MAX_M - ALT_MIN_M) / max(slots_per_hotspot - 1, 1)
        if MOVE_MODE == "event_driven":
            # Demand-following UAVs are DEPLOYED at their assigned ring station
            # (plus a small placement jitter), then track the drifting hotspot —
            # matching UAV_Mobility_Design.pdf Fig. 6, where the mean UAV-to-
            # hotspot distance starts near the ring radius, not km-scale away.
            cx, cy = centres[node.hotspot_idx]
            ox, oy = _ring_slot_offset(node.slot_idx, slots_per_hotspot)
            node.x = min(max(cx + ox + random.gauss(0.0, 200.0), 0.0), GRID_M)
            node.y = min(max(cy + oy + random.gauss(0.0, 200.0), 0.0), GRID_M)


def _step_towards(node: FogNode, tx: float, ty: float, dt_s: float) -> None:
    """Move node horizontally towards (tx, ty) at up to UAV_CRUISE_MS; set speed_ms."""
    dx, dy = tx - node.x, ty - node.y
    dist   = math.hypot(dx, dy)
    step   = UAV_CRUISE_MS * dt_s
    if dist <= step:                            # arrive within this tick
        node.x, node.y = tx, ty
        node.vx_ms = dx / dt_s
        node.vy_ms = dy / dt_s
        node.speed_ms = dist / dt_s             # partial-tick speed (< cruise)
    else:
        node.vx_ms = dx / dist * UAV_CRUISE_MS
        node.vy_ms = dy / dist * UAV_CRUISE_MS
        node.x += dx / dist * step
        node.y += dy / dist * step
        node.speed_ms = UAV_CRUISE_MS


def step_mobility(fog_nodes: List[FogNode], centres: List[List[float]], dt_s: float) -> None:
    """
    Advance mobility by one control tick (step 1 of the tick, PDF Fig. 2):
    drift the demand hotspots, then move each UAV according to MOVE_MODE.

    Mutates node.x / node.y / node.speed_ms and the hotspot centres in place.
    Propulsion energy is NOT drained here — the episode loop charges
    aero_power_w(node.speed_ms) * dt_s right after (physics.drain_flight_energy),
    so hover (V=0) and flight share one Zeng19 power curve.
    """
    if MOVE_MODE == "static":
        for node in fog_nodes:
            node.speed_ms = 0.0                 # P(0) = P_HOVER — original energy model
            node.vx_ms = node.vy_ms = 0.0
        return

    if MOVE_MODE == "event_driven":
        # Demand centroids drift as a slow bounded random walk (2 m/s).
        for c in centres:
            ang  = random.uniform(0.0, 2.0 * math.pi)
            c[0] = min(max(c[0] + HOTSPOT_DRIFT_MS * dt_s * math.cos(ang), 500.0), GRID_M - 500.0)
            c[1] = min(max(c[1] + HOTSPOT_DRIFT_MS * dt_s * math.sin(ang), 500.0), GRID_M - 500.0)
        # Each UAV tracks its ring slot around the assigned (drifting) hotspot.
        for node in fog_nodes:
            cx, cy = centres[node.hotspot_idx]
            ox, oy = _ring_slot_offset(node.slot_idx)
            _step_towards(node, cx + ox, cy + oy, dt_s)
        return

    # MOVE_MODE == "random": random-waypoint, always at cruise speed.
    for node in fog_nodes:
        if math.hypot(node.wp_x - node.x, node.wp_y - node.y) <= WAYPOINT_EPS_M:
            node.wp_x = random.uniform(0.0, GRID_M)   # draw a fresh waypoint
            node.wp_y = random.uniform(0.0, GRID_M)
        _step_towards(node, node.wp_x, node.wp_y, dt_s)


def generate_task_stream(n_tasks: int,
                         arrival_rate: float = TASK_ARRIVAL_RATE) -> List[Task]:
    """
    Generate a list of ``n_tasks`` Tasks with Poisson arrival times.

    Inter-arrival gaps are i.i.d. Exponential(rate=``arrival_rate``).
    Cumulative sum gives absolute arrival times in seconds.
    This is the standard inverse-transform method for a Poisson process.
    """
    if n_tasks < 0:
        raise ValueError("n_tasks must be non-negative")
    if arrival_rate <= 0.0:
        raise ValueError("arrival_rate must be positive")

    tasks: List[Task] = []
    arrival_s = 0.0   # running clock, seconds

    for task_id in range(n_tasks):
        # Exponential inter-arrival gap: -ln(U) / rate, where U ~ Uniform(0,1).
        gap = -math.log(random.random()) / arrival_rate
        arrival_s += gap

        tasks.append(Task.generate(task_id=task_id, arrival_s=arrival_s))

    return tasks


def generate_task_stream_for_duration(
        duration_s: float,
        arrival_rate: float = TASK_ARRIVAL_RATE) -> List[Task]:
    """Generate a Poisson task stream inside a fixed simulation horizon.

    Unlike varying the number of tasks, varying ``arrival_rate`` here changes
    offered load while keeping mobility and propulsion exposure fixed.  A task
    whose sampled arrival would fall after ``duration_s`` is not included.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if arrival_rate <= 0.0:
        raise ValueError("arrival_rate must be positive")

    tasks: List[Task] = []
    arrival_s = 0.0
    while True:
        arrival_s += -math.log(random.random()) / arrival_rate
        if arrival_s >= duration_s:
            break
        tasks.append(Task.generate(task_id=len(tasks), arrival_s=arrival_s))
    return tasks


# ==============================================================================
# SECTION 7 — WORLD INSTANTIATION  (runs at module load for quick sanity check)
# ==============================================================================

# Build swarm, devices, and a single-episode task stream so world state is available globally.
FOG_SWARM:    List[FogNode]   = build_fog_swarm()
IOT_DEVICES:  List[IoTDevice]
IOT_CENTRES:  List[List[float]]
IOT_DEVICES, IOT_CENTRES = build_iot_devices()
assign_mobility(FOG_SWARM, IOT_CENTRES)         # initialise z / hotspot / ring-slot state
TASK_STREAM:  List[Task]      = generate_task_stream(TASKS_PER_EPISODE)

# Compute tier breakdown for the print banner.
_tier_counts = {1: 0, 2: 0}
for _n in FOG_SWARM:
    _tier_counts[_n.tier] += 1

_low_count  = sum(1 for d in IOT_DEVICES if not d.is_high_power)
_high_count = sum(1 for d in IOT_DEVICES if     d.is_high_power)
_ep_duration_s = TASK_STREAM[-1].arrival_s   # simulated time from first to last task arrival

print(
    f"[WORLD] Built — "
    f"Mobility: {MOVE_MODE}, "
    f"UAVs: T1={_tier_counts[1]} T2={_tier_counts[2]}, "
    f"IoT: {_low_count} low-pwr / {_high_count} high-pwr, "
    f"Episode duration (simulated): {_ep_duration_s:.2f} s "
    f"({TASKS_PER_EPISODE} tasks at {TASK_ARRIVAL_RATE} tasks/s)"
)
