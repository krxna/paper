"""
config.py
=========
All module-level configuration constants, the global seed, and the small
validation helpers for the Energy-Aware UAV Fog Simulation.

This module is import-pure for constants, and (exactly as in the original
single-file script) it seeds all RNGs and prints the [CONFIG] banner at import
time. Import this module before any other simulation module.

See the package docstring in __init__.py / main.py for the full system overview.
"""

# SECTION 0 — IMPORTS
# ==============================================================================
import math
import os
import time
import random
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SECTION 1 — REPRODUCIBILITY
# ==============================================================================

# Fixed global seed — printed at startup so every run is fully reproducible.
SEED: int = 42


def set_global_seed(seed: int) -> None:
    """Seed Python built-ins, NumPy, and PyTorch with the same value."""
    random.seed(seed)           # seed Python's built-in RNG
    np.random.seed(seed)        # seed NumPy's global RNG
    torch.manual_seed(seed)     # seed PyTorch CPU RNG (CPU-only sim; no .cuda() needed)
    print(f"[SEED] Global seed set to {seed} — all RNGs reproducible.")


# ==============================================================================
# SECTION 2 — CONFIGURATION  (all module-level constants, grouped by subsystem)
# ==============================================================================

# ------------------------------------------------------------------------------
# §1  Global scenario parameters
# ------------------------------------------------------------------------------

TASKS_PER_EPISODE: int   = 2500    # tasks that must arrive before an episode ends (spec §1.1)
GRID_M:           float  = 5000.0  # side length of the square service area, meters
                                   # 5 km × 5 km grid is a common UAV-fog deployment scale
                                   # [Zeng19] considers service areas of 1–10 km radius

N_FOG:            int    = 20      # total fog UAVs in the swarm
N_IOT:            int    = 200     # total IoT client devices
N_IOT_CLUSTERS:   int    = 4       # IoT positions sampled from a 4-component Gaussian mixture
                                   # (4 "hotspots" model realistic uneven workload distributions)

# ------------------------------------------------------------------------------
# §1.2  UAV hardware model  —  per-node specs that DRIVE the tier assignment
#
# DESIGN (inverted pipeline):  hardware  ->  computational efficiency  ->  rank
#   ->  tier.  The twenty drones are NO LONGER assigned a tier a priori and then
# given that tier's hardware.  Instead each drone receives its own intrinsic
# hardware drawn from a single latent "quality" q in [0,1] (build_fog_swarm),
# its computational efficiency CE_eff is computed from that hardware, the swarm
# is ranked by CE_eff (descending), and the tier label is assigned by rank
# position:  the top TIER_COUNTS[0] become Tier 1, the next TIER_COUNTS[1] Tier 2,
# the rest Tier 2.  This makes "tier" an EMERGENT capability class rather than a
# fixed input, while preserving the paper's narrative (Tier-1 drones are the most
# capable all-round, so a large urgent task may only be serviceable by Tier 1).
#
# Every spec interpolates linearly between a WEAK endpoint (q=0) and a STRONG
# endpoint (q=1).  build_fog_swarm affinely rescales the 20 sampled q-values so
# the empirical minimum maps to 0 and the maximum to 1; this PINS the envelope
# endpoints every episode, guaranteeing at least one node at fr_avg = 3.0 GHz
# and one at 1.5 GHz so the task-feasibility argument in models.py stays exact.
#
# Clock endpoints match Stackelberg §IV-A body text exactly: O_j^CPU = [1.5, 3] GHz.
# All other envelope bounds reproduce the original three-tier table (former
# Tier-3 = WEAK, former Tier-1 = STRONG):
#   clock         1.5 .. 3.0  GHz   (Stackelberg §IV-A [1.5, 3] GHz — exact)
#   MIPS        12 000 .. 24 000
#   varsigma         6 .. 2         (compute weight; LOWER = more capable)
#   RAM            2.0 .. 8.0  GB
#   storage       4.0 .. 8.0  GB
#   battery    162 000 .. 234 000 J (~45 Wh .. ~65 Wh; 6S LiPo packs [Zeng19])
#   lambda_fail  0.025 .. 0.005 1/s (compute-failure rate; lower = better hardware)
#   mu_fail      0.010 .. 0.001 1/s (link-failure rate)
# ------------------------------------------------------------------------------

# Tier sizes, applied to the CE_eff-ranked swarm: top 10 -> T1, last 10 -> T2.
TIER_COUNTS: List[int] = [10, 10]
assert sum(TIER_COUNTS) == N_FOG, "TIER_COUNTS must sum to N_FOG"

# Hardware envelope endpoints (WEAK = q0 = least capable; STRONG = q1 = most capable).
HW_FR_WEAK_GHZ:    float = 1.5        # GHz — slowest clock (Stackelberg §IV-A lower bound)
HW_FR_STRONG_GHZ:  float = 3.0        # GHz — fastest clock (Stackelberg §IV-A upper bound)
HW_MIPS_WEAK:      float = 12_000.0   # MIPS
HW_MIPS_STRONG:    float = 24_000.0   # MIPS
HW_VARSIGMA_WEAK:  float = 6.0        # compute weight at q=0 (least capable)
HW_VARSIGMA_STRONG:float = 2.0        # compute weight at q=1 (most capable; lower = better)
HW_MP_WEAK_GB:     float = 2.0        # GB primary memory (RAM)
HW_MP_STRONG_GB:   float = 8.0        # GB
HW_MS_WEAK_GB:     float = 4.0        # GB secondary storage
HW_MS_STRONG_GB:   float = 8.0        # GB
HW_E_WEAK_J:       float = 162_000.0  # J  (~45 Wh)
HW_E_STRONG_J:     float = 234_000.0  # J  (~65 Wh)
SOC_INIT_MIN:      float = 0.35       # initial residual-energy fraction
SOC_INIT_MAX:      float = 1.00
assert 0 <= SOC_INIT_MIN <= SOC_INIT_MAX <= 1
HW_LAMBDA_WEAK:    float = 0.025      # 1/s compute-failure rate (weakest hardware)
HW_LAMBDA_STRONG:  float = 0.005      # 1/s (strongest hardware)
HW_MU_WEAK:        float = 0.010      # 1/s link-failure rate (weakest)
HW_MU_STRONG:      float = 0.001      # 1/s (strongest)


def computational_efficiency_score(fr_avg_ghz: float, varsigma: float, c_mips: float) -> float:
    """
    Computational efficiency CE_eff of a single node — the tiering criterion.

      CE_eff = fr_hz / (varsigma * C_mips * 1e6)

    This is the RECIPROCAL of the processing-capability "compute cost" CE used in
    the original paper (CE = varsigma*C_mips*1e6 / fr_hz, a demand-to-clock ratio
    where lower was better).  Inverting it yields an efficiency where HIGHER is
    better — clock available per unit of compute demand — so that ranking the swarm
    in DESCENDING CE_eff places the most capable drones in Tier 1.  CE_eff is used
    ONLY to form tiers; it is no longer a Stage-1 MCDM criterion.

    Returns CE_eff (dimensionless, higher = more computationally efficient).
    """
    fr_hz = fr_avg_ghz * 1e9
    return fr_hz / (varsigma * c_mips * 1e6)

# ------------------------------------------------------------------------------
# §1.3  IoT device transmit-power groups
#
# 150 low-power sensors  at p =  2.51 W  (34 dBm — typical LoRa/NB-IoT device max)
#  50 high-power terminals at p = 31.62 W (45 dBm — typical industrial 4G CPE max)
# Conversion: P_watts = 10 ** ((dBm - 30) / 10)
# ------------------------------------------------------------------------------
N_LOW_POWER_IOT:  int   = 150      # low-power sensor count
N_HIGH_POWER_IOT: int   = 50       # high-power terminal count
P_TX_LOW_DBM:     float = 34.0     # dBm — 34 dBm is near the LoRa regulatory ceiling
P_TX_HIGH_DBM:    float = 45.0     # dBm — 45 dBm matches industrial 4G CPE uplink power

# Precomputed watts (used everywhere; avoids repeated exponentiation in hot loops)
P_TX_LOW_W:  float = 10 ** ((P_TX_LOW_DBM  - 30) / 10)   # ~2.51 W
P_TX_HIGH_W: float = 10 ** ((P_TX_HIGH_DBM - 30) / 10)   # ~31.62 W

# ------------------------------------------------------------------------------
# §1.4  Task generation parameters
# ------------------------------------------------------------------------------
TASK_ARRIVAL_RATE: float = 180.0    # Poisson lambda, tasks/second => mean inter-arrival 20 ms
                                    # 180 tasks/s is a moderate IoT density for fog offloading [Mao17]

PARETO_ALPHA: float = 1.5          # Pareto shape for task payload size (heavy-tailed IoT traffic)
                                    # alpha = 1.5 => infinite variance; matches empirical IoT traces

TASK_SIZE_MIN_KB:  float = 60.0    # minimum payload,  KB (spec §1.4)
TASK_SIZE_MAX_KB:  float = 800.0   # maximum payload,  KB  (capped to bound memory; [Mao17])

CYCLES_PER_BYTE_MIN: int = 1_600   # cyc/byte (= 200 cyc/bit) — floor of Stackelberg §IV-A's
CYCLES_PER_BYTE_MAX: int = 1_800   # cyc/byte (= 225 cyc/bit)   [200, 1400] cycles/bit range.
                                    # Upper truncation at 225 cyc/bit is a deadline-feasibility
                                    # ceiling: the largest task (800 KB) on the fastest node
                                    # (3.0 GHz) needs <= ~229 cyc/bit to meet the 550 ms deadline
                                    # with a 50 ms overhead budget. Every sampled value lies
                                    # strictly inside Stackelberg's stated range.
CYCLES_PER_BYTE:     int = 1_700   # NOMINAL midpoint — used ONLY for deterministic
                                    # representative tasks (self-tests, WL previews); live tasks
                                    # sample c ~ Uniform[MIN, MAX] per task in models.Task.generate.
# Derived: l_i (cycles) = s_i[KB] * 1024 * c,  c ~ Uniform[CYCLES_PER_BYTE_MIN, CYCLES_PER_BYTE_MAX]

SMALL_TASK_THRESHOLD_KB: float = 250.0  # tasks < 250 KB are "small"; rest are "large"
DEADLINE_SMALL_MS:       float = 150.0  # deadline for small tasks, ms (tight latency SLA).
                                         # 150 ms is the value every reported Stage-1 and
                                         # Stage-2 run was generated under; see
                                         # test_task_deadlines.py.
DEADLINE_LARGE_MS:       float = 550.0  # deadline for large tasks, ms (relaxed SLA).
                                         # 550 ms is the value the CYCLES_PER_BYTE_MAX
                                         # truncation and the SPOTIS delay bound are derived
                                         # from: the largest task (800 KB at 225 cyc/bit) on
                                         # the fastest node (3.0 GHz) takes ~491 ms, feasible
                                         # within 550 ms with a ~50 ms overhead budget — but
                                         # ONLY on a fast, lightly loaded node.

def deadline_for_size_ms(size_kb: float) -> float:
    """Relative service deadline (ms) for a payload of `size_kb` KiB.

    Single authoritative definition of the size->deadline rule. Every task
    generator (models.Task.generate, scenario.generate_paired_tasks) and every
    synthetic/representative task must go through here so the mathematical
    workload model, the code and the manuscript cannot drift apart again.
    The ADMISSION_MARGIN_MS buffer is applied separately at admission time
    (neural.filter_candidates) and is deliberately NOT folded in here.
    """
    return (DEADLINE_SMALL_MS if size_kb < SMALL_TASK_THRESHOLD_KB
            else DEADLINE_LARGE_MS)


# ------------------------------------------------------------------------------
# §2.1  Wireless channel model  —  UAV air-to-ground line-of-sight (LoS) link
#
# The IoT->UAV link is modelled with the same algebraic form used by ReLIEF
# [Siyadatzadeh23, Eq. 3],  g(d) = upsilon_1 * d^(-upsilon_2),  but the constants
# are RE-CALIBRATED for the air-to-ground deployment scale used here instead of
# the short single-hop broker<->fog distances assumed in ReLIEF's terrestrial
# setting.  UAV-to-ground channels are dominated by line-of-sight propagation,
# for which the path-loss exponent is 2.0-2.5 (we use 2.3), NOT the terrestrial
# non-LoS value of 4.0 that ReLIEF adopts for ground-level fog nodes.  Directly
# importing ReLIEF's (upsilon_1=1e-3, upsilon_2=4, N=1e-10) constants makes the
# received SNR collapse at km-scale links (rate -> a few bps), which is
# unphysical for an aerial relay; the calibration below restores a realistic
# 30-500 Mbps envelope across a 5 km cell.
#
# Link budget (carrier 2 GHz, 30 MHz bandwidth):
#   * Reference gain at d0 = 1 m is the free-space value (lambda/4*pi)^2.
#   * Path loss grows as d^(-2.3) (LoS air-to-ground, [Al-Hourani14]-style).
#   * Noise floor = k*T*B (pure thermal, receiver noise figure = 0 dB), i.e.
#     the -174 dBm/Hz Johnson-Nyquist floor stated by Stackelberg §IV-A.
# This yields SNR ~= 45 (16.6 dB) at the 2.5 km mid-cell and ~= 9 (9.6 dB) at
# the 5 km edge for a 2.51 W sensor — i.e. coverage holds across the whole cell.
# ------------------------------------------------------------------------------
CARRIER_HZ:        float = 2.0e9   # carrier frequency f_c, Hz (2 GHz, sub-6 band)
SPEED_OF_LIGHT:    float = 3.0e8   # c, m/s
_WAVELENGTH_M:     float = SPEED_OF_LIGHT / CARRIER_HZ   # lambda = c / f_c (= 0.15 m)

BANDWIDTH_HZ:      float = 30e6    # channel bandwidth omega, Hz (30 MHz)
                                    # [Stackelberg Table I / ReLIEF Table III]: 30 MHz / 50 Mb/s class

NOISE_FIGURE_DB:   float = 0.0     # receiver noise figure, dB — 0 dB so the floor is the pure
                                    # -174 dBm/Hz thermal term stated by Stackelberg §IV-A
_BOLTZMANN:        float = 1.380649e-23  # k, J/K
_TEMPERATURE_K:    float = 290.0          # T, K (standard reference temperature)
# Noise power N = k*T*B * 10^(NF/10).  At 30 MHz, NF = 0 dB  =>  N ~= 1.2013e-13 W
# (= -174 dBm/Hz + 10*log10(30e6 Hz) ~= -99.2 dBm — exact match to the paper's floor).
NOISE_W:           float = (_BOLTZMANN * _TEMPERATURE_K * BANDWIDTH_HZ
                            * 10.0 ** (NOISE_FIGURE_DB / 10.0))

CHANNEL_UPSILON2:  float = 2.3     # path-loss exponent upsilon_2 (LoS air-to-ground, range 2.0-2.5)
# Reference channel gain at d0 = 1 m: free-space gain (lambda / 4*pi)^2.
CHANNEL_UPSILON1:  float = (_WAVELENGTH_M / (4.0 * math.pi)) ** 2   # ~= 1.42e-4

# ------------------------------------------------------------------------------
# §2.1e  Broker per-stage processing overheads (added to decision latency budget)
# ------------------------------------------------------------------------------
T_MEREC_MS:   float = 0.50   # ms — MEREC weight computation overhead
T_KALMAN_MS:  float = 0.10   # ms — Kalman smoother update overhead
T_KL_MS:      float = 0.05   # ms — KL divergence gate test overhead
T_SPOTIS_MS:  float = 1.00   # ms — SPOTIS full re-rank overhead (only charged when gate fires)
T_PPO_INFER_MS: float = 0.80 # ms — PPO actor inference overhead per task
# Deterministic controller-cost assumptions for the simulation-only study.
# These are model parameters, not claims of physical hardware measurements.
T_FODAS_DECISION_MS: float = 0.60
T_RELIEF_DECISION_MS: float = 0.40
T_RANDOM_DECISION_MS: float = 0.01
# Stage-1 baseline prices preserve the measured ratios against our 0.65 ms
# gated MEREC+Kalman+KL path; Stage-2 prices above remain unchanged.
T_FU_SERVE_MS: float = 1.55
T_2DP_FHS_MS:  float = 10.44
T_3D_POS_MS:   float = 16.51
CONTROLLER_REFERENCE_HZ: float = 2.25e9
CONTROL_TICK_S: float = 0.100   # control-loop period, seconds (10 Hz)
PROFILE_FOG_SIZES: Tuple[int, ...] = (10, 20, 40, 80, 160)

# ------------------------------------------------------------------------------
# §2.2  Aerodynamic / power model  (rotary-wing UAV, hover operating state)
#       All constants from [Zeng19] Table I / Equations (1)–(3).
# ------------------------------------------------------------------------------
W_UAV_N:          float = 100.0    # UAV total weight, N  (≈10.2 kg)
AIR_DENSITY:      float = 1.225    # air density rho, kg/m³  (ISA sea level)
ROTOR_RADIUS_M:   float = 0.5      # rotor radius R_rot, m  (50 cm; typical hex/octocopter)
ROTOR_DISC_AREA:  float = math.pi * 0.5 ** 2   # disc area A = pi*R^2 ≈ 0.7854 m²

U_TIP:            float = 200.0    # rotor blade tip speed, m/s  [Zeng19] uses 120–200 m/s
ROTOR_SOLIDITY:   float = 0.05     # rotor solidity sigma_s (ratio of blade area to disc area)
                                    # Typical quadrotor: 0.02–0.10 [Zeng19]
DRAG_RATIO:       float = 0.3      # fuselage drag ratio d_0  [Zeng19] nominal 0.3
MEAN_ROTOR_VELOCITY: float = 7.2   # induced velocity in hover v_0, m/s  [Zeng19] Eq.(3)

P_BLADE:    float = 79.85    # W — blade profile power P_0 = (sigma_s * C_d0 / 8) * rho * A * Utip^3
                              #   ≈ (0.05 * 0.3 / 8) * 1.225 * 0.7854 * 200^3 ≈ 79.85 W [Zeng19]
P_INDUCED:  float = 790.67   # W — induced (hovering) power P_i = (W^3/(2*rho*A))^0.5
                              #   = (100^3 / (2 * 1.225 * 0.7854))^0.5 ≈ 790.67 W [Zeng19]
P_HOVER:    float = 870.52   # W — total hover power = P_0 + P_i = 79.85 + 790.67 [Zeng19 Eq.(3)]

# ------------------------------------------------------------------------------
# §2.2m  UAV mobility model  (UAV_Mobility_Design.pdf — FU-Serve / Con-Fog / Zeng19)
#
# Three movement modes (PDF p.1 design table):
#   'event_driven' — each UAV tracks its assigned IoT demand hotspot (default;
#                    FU-Serve / Con-Fog demand-following repositioning)
#   'random'       — random-waypoint motion, demand-agnostic ablation (Con-Fog §V-B)
#   'static'       — no motion; reproduces the original fixed-position simulation
#                    exactly (z = 0, V = 0 => P(0) = P_HOVER)
#
# Motion is HORIZONTAL ONLY at a fixed per-UAV altitude z (PDF: Zeng19 Eq. (3)
# governs power; z appears in the 3-D slant range, FU-Serve Eq. (10), but never
# changes).  Each control tick a UAV is charged P(V) * CONTROL_TICK_S of
# propulsion energy at its current speed via physics.aero_power_w — hover is just
# the V = 0 point of the same curve, so there is no discontinuity.
#
# Collision-free by construction (PDF §3): the UAVS_PER_HOTSPOT UAVs sharing one
# hotspot occupy distinct slots on a HOTSPOT_RING_M standoff ring (x-y separation)
# AND distinct altitude layers in [ALT_MIN_M, ALT_MAX_M] (vertical separation).
# ------------------------------------------------------------------------------
MOVE_MODE: str = os.environ.get("UAV_MOVE_MODE", "event_driven")
assert MOVE_MODE in ("event_driven", "random", "static"), f"bad MOVE_MODE {MOVE_MODE!r}"

UAV_CRUISE_MS:    float = 12.0    # m/s — cruise speed; P(12) ≈ 779 W < P_HOVER (induced-power dip)
HOTSPOT_RING_M:   float = 250.0   # m — standoff-ring radius around the hotspot centroid
ALT_MIN_M:        float = 80.0    # m — lowest altitude layer
ALT_MAX_M:        float = 150.0   # m — highest altitude layer
HOTSPOT_DRIFT_MS: float = 2.0     # m/s — slow random-walk drift of each demand centroid
UAVS_PER_HOTSPOT: int   = N_FOG // N_IOT_CLUSTERS   # 5 UAVs share each hotspot ring
WAYPOINT_EPS_M:   float = 25.0    # m — random-waypoint arrival tolerance (new waypoint drawn)

P_PROC:  float = 10.0    # W — on-board compute power while processing a task [Mao17]
P_COMM:  float =  2.0    # W — radio-transceiver active power during uplink reception
P_IDLE:  float =  3.0    # W — microcontroller / avionics idle power (no task)
P_UAV_TX_W: float = 10.0 # W — inter-UAV forwarding/signalling transmit power
SIGNAL_KB: float = 2.0   # state synchronized to every peer after a head handover
HANDOVER_SETUP_MS: float = 25.0

# ------------------------------------------------------------------------------
# §2.3  Memory-efficiency (ME) model  — the Stage-1 capability criterion
#
# ME replaces the former processing-capability (PC) criterion entirely.  The
# compute half of PC (CE) is gone from Stage 1 (it now forms tiers); ME is the
# sole capability criterion the broker sees.  ME captures BOTH:
#   * capacity   — MP_tot / MS_tot now vary per node, so ME discriminates across
#                  nodes even at zero load; and
#   * occupancy  — MP_occ / MS_occ are recomputed each tick from the tasks
#                  currently resident in a node's queues (update_memory_occupancy),
#                  so ME falls as a node fills up and rises as its queue drains.
#
#   ME_j = clip( [ CAP_ALPHA*(MP_tot - MP_occ) + CAP_BETA*(MS_tot - MS_occ) ]
#                / ME_NORM_GB ,  0, 1 )                                     [unitless, in [0,1]]
#
# ME_NORM_GB normalises by the maximum attainable free memory (strongest node,
# empty queues), keeping ME in [0,1] so the SPOTIS col-1 bound stays [0,1].
# ------------------------------------------------------------------------------
CAP_ALPHA:  float = 0.5    # weight for primary-memory (RAM) term in free-memory score
CAP_BETA:   float = 0.5    # weight for secondary-storage term in free-memory score

# Normaliser: max free memory = strongest node with empty queues.
ME_NORM_GB: float = CAP_ALPHA * HW_MP_STRONG_GB + CAP_BETA * HW_MS_STRONG_GB   # = 8.0 GB

# Per-task memory footprint while the task is resident in a node's queue.
# A queued task runs in an isolated runtime/container: its memory footprint is
# dominated by the decompressed working set + resident framework image, not the
# raw payload, so footprints are ~hundreds of MB-scale and scale with payload via
# documented inflation factors with a per-task runtime floor.  Calibrated so a
# busy node reaches ~10-20% RAM occupancy at realistic queue depths — a clear
# load-coupling signal that modulates ME without dominating the capacity spread
# or saturating against physical capacity.
MEM_RAM_GB_PER_KB:     float = 2.5e-4   # GB RAM per KB payload    (800 KB -> 0.20 GB ≈ 200 MB)
MEM_STORAGE_GB_PER_KB: float = 5.0e-4   # GB storage per KB payload (800 KB -> 0.40 GB ≈ 400 MB)
MEM_RAM_FLOOR_GB:      float = 0.010    # GB minimum RAM held per resident task (runtime baseline)
MEM_STORAGE_FLOOR_GB:  float = 0.020    # GB minimum storage held per resident task

# ------------------------------------------------------------------------------
# §3  MCDM pipeline parameters
# ------------------------------------------------------------------------------

# Six MCDM criteria (order is fixed everywhere in the code):
#   index 0: E_res   — residual energy           (beneficial: higher is better)
#   index 1: ME      — memory efficiency          (beneficial)  [replaces former PC]
#   index 2: R       — reliability                (beneficial)
#   index 3: D       — projected controller delay (non-beneficial: lower is better)
#   index 4: WL      — current workload           (non-beneficial)
#   index 5: ECT     — energy coverage time        (beneficial)
#
# A control-queue-growth criterion (CQG) sat at index 5 until 2026-08-24.  It was
# a 100 ms finite difference of the same queue D already projects, so it carried no
# information D lacks: zero spread at every load below N=80/lambda=720 (tripping
# assert_live_criteria), and above it a sparse impulse — 0 at the 99th percentile,
# spiking to 51 ms/s against a 6 ms/s bound — which MEREC's variance-driven removal
# effect then rewarded with 70% of the decision.  Dropping it left mean delay
# unchanged and raised D+WL from 13.9% to 47.0% of decision power.
N_CRITERIA: int = 6
CRITERIA_BENEFICIAL: List[bool] = [True, True, True, False, False, True]

MCDM_INIT_WEIGHTS: List[float] = [1.0 / N_CRITERIA] * N_CRITERIA

# Fixed predictive-criterion bounds. The aerodynamic lower bound is the
# deterministic minimum of the configured Zeng19 curve over 0..UAV_CRUISE_MS.
HEAD_AVAILABILITY_HORIZON_S: float = 10.0
MIN_AERO_POWER_W: float = 737.0
ECT_MAX_S: float = HW_E_STRONG_J / MIN_AERO_POWER_W
R_HEAD_FLOOR: float = 0.95
CRITERION_SPREAD_FLOOR: float = 1e-4

# Kalman filter scalars — applied as  Q = KALMAN_Q * I5,  R_noise = KALMAN_R * I5
KALMAN_Q:  float = 1e-4   # process noise: how fast we believe weights can drift tick-to-tick
                           # small value = trust the model; range 1e-5..1e-3 is typical [Kal60]
KALMAN_R:  float = 1e-2   # measurement noise: uncertainty in each MEREC weight observation
                           # 100× larger than Q => weights evolve slowly, measurement noisy [Kal60]
KALMAN_P0: float = 1.0    # initial error covariance diagonal (uninformed prior = 1.0) [Kal60]

# Measurement-level smoother used by the seven-criterion fog-head path.  The
# steady-state scalar gain is exactly 0.20 for this pair.
KALMAN_M_Q: float = 5e-4
KALMAN_M_R: float = 1e-2

KL_THRESHOLD: float = 0.05  # theta_KL — only re-run (expensive) SPOTIS when the KL divergence
                              # between old and new weight vectors exceeds this value;
                              # 0.05 nats is a common "significant change" threshold in practice

# ------------------------------------------------------------------------------
# SPOTIS fixed bounds  [Dez20 §III Steps 1–2]
#
# SPOTIS requires a fixed Ideal Solution Profile (ISP) and per-criterion
# [S_min_j, S_max_j] bounds that are chosen A PRIORI, independently of the
# live candidate set.  This is what makes SPOTIS rank-reversal-free: adding or
# removing a UAV cannot shift the bounds and thereby reorder surviving candidates.
# Using live min/max (as in TOPSIS) would violate this guarantee.
#
# Bound choices:
#   col 0  E_res     [0 J, 234 000 J]   — 0 = fully depleted; 234 kJ = strongest full charge
#   col 1  ME        [0, 1]             — memory efficiency; always in unit interval
#   col 2  R         [0, 1]             — reliability; always in unit interval
#   col 3  D         [0 ms, 550 ms]     — 0 = instant; DEADLINE_LARGE_MS = worst allowed delay
#   col 4  WL        [0, WL_MAX_CYCLES] — 0 = empty queue; WL_MAX_CYCLES = fixed design ceiling
#
# Reference: Dezert, Tchamova, Han, Tacnet — "The SPOTIS Method," IEEE FUSION 2020 §III.
# ------------------------------------------------------------------------------

# Maximum queue depth per UAV (design bound, not a live measurement).
# One maximum-size resident task is the reachable per-dispatch operating unit;
# deeper queues are already outside the envelope and clip as worst.
MAX_QUEUE_DEPTH: int = 1

# Worst-case workload: every queued slot holds the largest possible task.
WL_MAX_CYCLES: float = (TASK_SIZE_MAX_KB * 1024 * CYCLES_PER_BYTE_MAX) * MAX_QUEUE_DEPTH

# A priori operating envelope for a full controller FIFO on the slowest UAV.
# It is candidate-independent, so fixed-bound SPOTIS remains rank-reversal-free.
MAX_CONTROL_QUEUE_DEPTH: int = 1
T_CTRL_IDEAL_MS: float = (
    T_PPO_INFER_MS * CONTROLLER_REFERENCE_HZ / (HW_FR_STRONG_GHZ * 1e9))
T_CTRL_WORST_MS: float = (
    MAX_CONTROL_QUEUE_DEPTH * T_PPO_INFER_MS
    * CONTROLLER_REFERENCE_HZ / (HW_FR_WEAK_GHZ * 1e9))
T_CTRL_ENVELOPE_MS: float = T_CTRL_WORST_MS - T_CTRL_IDEAL_MS

# Fixed per-criterion [min, max] bounds for SPOTIS scoring.
# Indices match [E_res, ME, R, D_ctrl, WL, ECT].
SPOTIS_S_MIN: List[float] = [0.0, 0.0, R_HEAD_FLOOR, 0.0, 0.0, 0.0]
SPOTIS_S_MAX: List[float] = [
    HW_E_STRONG_J, 1.0, 1.0, T_CTRL_ENVELOPE_MS, WL_MAX_CYCLES, ECT_MAX_S,
]
LEGACY_SPOTIS_S_MIN: List[float] = [0.0] * 5
LEGACY_SPOTIS_S_MAX: List[float] = [
    HW_E_STRONG_J, 1.0, 1.0, DEADLINE_LARGE_MS,
    (TASK_SIZE_MAX_KB * 1024 * CYCLES_PER_BYTE_MAX) * 64,
]
assert len(CRITERIA_BENEFICIAL) == len(MCDM_INIT_WEIGHTS) == N_CRITERIA
assert len(SPOTIS_S_MIN) == len(SPOTIS_S_MAX) == N_CRITERIA
# 234_000 J is HW_E_STRONG_J (the strongest node's full charge) — the maximum
# any node can ever hold, making the energy bound tight and physically grounded.

# ------------------------------------------------------------------------------
# §4  Self-Attention + PPO hyperparameters
# ------------------------------------------------------------------------------

# Transformer encoder architecture [Vas17]
D_MODEL:      int = 128   # embedding dimension  (Vas17 uses 512; 128 is right-sized for 20 UAVs)
N_HEADS:      int = 8     # attention heads  (D_MODEL must be divisible by N_HEADS: 128/8=16 ✓)
N_ENC_LAYERS: int = 3     # stacked encoder layers  (3 is minimal-but-expressive for small inputs)
D_FF:         int = 256   # feed-forward hidden dim (typically 2–4× D_MODEL [Vas17])

# Reward shaping coefficients rho_1..rho_4  (must sum to 1.0)
#   rho_1 (dWL)    — workload reduction: credit for spreading load
#   rho_2 (dD)     — delay reduction: credit for meeting deadline
#   rho_3 (dR)     — reliability increase: credit for picking reliable UAV
#   rho_4 (energy) — energy cost: penalty for high energy draw
REWARD_RHO: List[float] = [0.15, 0.35, 0.35, 0.15]
assert abs(sum(REWARD_RHO) - 1.0) < 1e-9, "Reward weights must sum to 1.0"

GAMMA:       float = 0.99   # PPO discount factor [Schul17] — standard value for episodic tasks
GAE_LAMBDA:  float = 0.95   # GAE lambda for advantage estimation [Schul17] — standard
PPO_CLIP:    float = 0.20   # PPO clipping epsilon [Schul17] — standard value 0.1–0.3
C1_VALUE:    float = 0.50   # value-loss coefficient in combined loss [Schul17]
C2_ENTROPY:  float = 0.01   # entropy bonus coefficient — encourages exploration [Schul17]

CRITICAL_PENALTY: float = 100.0  # 𝓜: large negative reward when primary UAV SOC < RTH floor
                                   # magnitude chosen to dominate normal reward range (which is O(1))

RELIABILITY_FLOOR: float = 0.95   # R_min — task must have at least 95 % reliability to be feasible

RELIABILITY_LOG_CLAMP: float = 1e-12   # per-task floor used ONLY inside the log-domain Eq.(12)
                                        # aggregate (ReLIEF R = prod_i R_i, accumulated as
                                        # sum_i log10 R_i).  A dropped task records R_i = 0.0,
                                        # whose log is -inf; clamping at 1e-12 turns each outright
                                        # failure into a finite -12 contribution to log10(R_sys) —
                                        # a large, calibrated penalty instead of a metric-destroying
                                        # infinity.  Code-level numerical guard, not a paper value.
SOC_PRIMARY_MIN:   float = 0.30   # primary UAV must have ≥ 30 % state-of-charge to be selected
SOC_BACKUP_MIN:    float = 0.15   # backup UAV must have ≥ 15 % SOC (Return-To-Home floor)

PPO_INNER_EPOCHS: int   = 4       # number of gradient passes per collected batch [Schul17]
LEARNING_RATE:    float = 3e-4    # Adam optimizer LR (PPO best-practice range: 1e-4–3e-4 [Schul17])

# ------------------------------------------------------------------------------
# §4.6  Model-Guided Task-Aware Dispatch constants  (methodology_improvement_plan.md)
# ------------------------------------------------------------------------------

# Physics-guided candidate filtering: prune to the strongest candidates before PPO.
DISPATCH_TOP_K: int = 3
W_LINK_ENERGY: float = 3.0
ADMISSION_MARGIN_MS: float = 25.0

# Evaluation is deterministic; exploration belongs in PPO rollout collection.
EVAL_EPSILON: float = 0.0

# Backup parity: match FODAS by selecting a backup whenever a SOC-valid second node exists.
ALWAYS_SELECT_BACKUP: bool = True
BACKUP_RELIABILITY_TRIGGER: float = 0.985
BACKUP_SLACK_TRIGGER_RATIO:  float = 0.20

# Head-election safety floor; the lookahead cost test governs normal switching.
#   HEAD_REEVALUATION_TICKS — maximum interval between complete rankings even
#                             when the MEREC weight vector remains stable
HEAD_MIN_TENURE_TICKS: int   = 1
HEAD_REEVALUATION_TICKS: int = 10
# Compatibility values used only by equal-controller task-distribution runs.
LEGACY_HEAD_MIN_TENURE_TICKS: int = 10
LEGACY_HEAD_SWITCH_MARGIN: float = 0.03

# Outcome-centred reward coefficients  (replace the delta-based REWARD_RHO formulation)
#   REWARD_SUCCESS  — bonus for delivering a task before its deadline
#   REWARD_MISS     — penalty for missing deadline or dropping
#   REWARD_SLACK_W        — weight for normalized deadline slack bonus
#   REWARD_RELIABILITY_W  — weight for per-task reliability bonus
#   REWARD_ENERGY_W       — weight for normalized energy cost penalty
#   REWARD_QUEUE_EXTERNALITY_W — weight for marginal queue damage penalty
#   REWARD_OVERLOAD_W     — weight for hotspot / overload penalty
REWARD_SUCCESS:             float = 2.0
REWARD_MISS:                float = -3.0
REWARD_SLACK_W:             float = 1.50
REWARD_RELIABILITY_W:       float = 0.25
REWARD_ENERGY_W:            float = 0.75
REWARD_QUEUE_EXTERNALITY_W: float = 0.75
REWARD_UPLOAD_W:            float = 0.25
REWARD_OVERLOAD_W:          float = 1.00

# FODAS-anchored warm start: imitate the deterministic FODAS scheduler before PPO.
BC_PRETRAIN_STEPS: int = 4_000
BC_BATCH_SIZE:     int = 64
BC_EPOCHS:         int = 3
BC_LEARNING_RATE:  float = 3e-4

# ------------------------------------------------------------------------------
# §5  Simulation clock & print policy
# ------------------------------------------------------------------------------
# Energy integrates as E += Power * CONTROL_TICK_S each tick.
                                  # 100 ms is standard for UAV avionics control loops.

# Maximum range that can carry one state-sync message inside one control tick.
# This physics-derived range is intentionally non-binding in the 5 km study grid;
# it is not shortened to manufacture coordinator link failures.
_SIGNAL_RATE_BPS: float = SIGNAL_KB * 1024.0 * 8.0 / CONTROL_TICK_S
_SIGNAL_SNR_MIN: float = math.expm1(
    math.log(2.0) * _SIGNAL_RATE_BPS / BANDWIDTH_HZ)
HEAD_COMM_RANGE_M: float = (
    CHANNEL_UPSILON1 * P_UAV_TX_W / (NOISE_W * _SIGNAL_SNR_MIN)
) ** (1.0 / CHANNEL_UPSILON2)

PRINT_EVERY: int  = 25    # print a full tick block every Nth tick; forced-print exceptions:
                           #   tick 0, final tick, broker-head change, any dropped task

VERBOSE: bool = False     # when True, additionally print every individual task decision
                           # (very noisy for long runs; useful for debugging)

RUN_SELF_TEST: bool = False  # when True, run built-in self-test checks and exit (no simulation)
                               # tests: feature dimensions, SOC mask thresholds, reward signs, PPO ratios

# ==============================================================================
# SECTION 3 — SMALL UTILITY HELPERS  (used to validate CONFIG constants)
# ==============================================================================

def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to Watts.  P_W = 10 ** ((dBm - 30) / 10).
    Used only to document and cross-check IoT transmit-power constants."""
    return 10.0 ** ((dbm - 30.0) / 10.0)


# Sanity-check: P_HOVER must equal P_BLADE + P_INDUCED (rounded to 2 decimal places).
# This guards against accidental edits that break the aerodynamic power budget [Zeng19].
assert P_HOVER == round(P_BLADE + P_INDUCED, 2), (
    f"Power budget inconsistency: P_HOVER={P_HOVER} != P_BLADE+P_INDUCED="
    f"{round(P_BLADE + P_INDUCED, 2)}"
)

# ==============================================================================
# SECTION 4 — STARTUP  (run immediately on import/execution; seeds RNGs, prints summary)
# ==============================================================================

# Seed all RNGs with the fixed global seed.
set_global_seed(SEED)

# Compute a quick derived quantity for the startup banner.
_mean_inter_arrival_ms: float = 1000.0 / TASK_ARRIVAL_RATE  # ms between consecutive tasks on avg

print(
    f"[CONFIG] Loaded — "
    f"UAVs: {N_FOG} (tiers: {TIER_COUNTS}), "
    f"IoT devices: {N_IOT} ({N_LOW_POWER_IOT} low-pwr / {N_HIGH_POWER_IOT} high-pwr), "
    f"Tasks/episode: {TASKS_PER_EPISODE}, "
    f"Mean inter-arrival: {_mean_inter_arrival_ms:.1f} ms, "
    f"Tick: {int(CONTROL_TICK_S*1000)} ms, "
    f"Grid: {GRID_M/1000:.0f} km × {GRID_M/1000:.0f} km"
)
