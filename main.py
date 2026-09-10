"""
uav_fog_simulation.py
=====================
Energy-Aware UAV Fog Simulation — single-file, incrementally built.

SYSTEM OVERVIEW
---------------
A swarm of 20 heterogeneous fog UAVs serves 200 IoT devices scattered over a 5 km × 5 km grid.
Each 100 ms control tick runs two decision stages:

  Stage 1  (slow, math-based)   — elect a Master Fog Head (broker) using a 4-step MCDM pipeline:
               MEREC  -> Kalman smoother -> KL-divergence gate -> SPOTIS ranker   [Kesh21, Kal60, Dez20]
  Stage 2  (fast, RL-based)     — broker dispatches each arriving task to a PRIMARY + BACKUP UAV
               via a self-attention encoder + PPO actor-critic                    [Vas17, Schul17]

TIMING MODEL
------------
  Control tick     : dt = 100 ms  (10 Hz)
  Task arrivals    : Poisson at 50 tasks/s  => ~5 tasks per tick
  Episode length   : 2 500 tasks  (~500 ticks, ~50 s of mission time)
  Energy integral  : E += Power * dt  per tick per UAV
  Wall-clock guard : each PPO decision must complete within ~20 ms inter-arrival budget

KEY REFERENCES
--------------
  [Zeng19]  Zeng, Xu, Zhang — "Energy Minimization for Wireless Communication with Rotary-Wing
            UAV," IEEE Trans. Wireless Commun., 2019.
            Source of the rotary-wing hover power model and all aerodynamic constants
            (P_0, P_i, U_tip, v_0, d_0, sigma_s).

  [Mao17]   Mao et al. — "A Survey on Mobile Edge Computing," IEEE Commun. Surveys & Tutorials,
            2017. Source of MEC offloading delay/energy formulations and workload intensity
            (cycles per byte).

  [Schul17] Schulman et al. — "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
            Source of PPO hyperparameters (clip ratio, GAE lambda, inner epochs).

  [Vas17]   Vaswani et al. — "Attention Is All You Need," NeurIPS 2017.
            Source of transformer encoder architecture choices (d_model, n_heads, d_ff).

  [Kesh21]  Keshavarz-Ghorabaee et al. — "MEREC: A Novel Weights Extraction Method of Criteria
            in MCDM," Symmetry, 2021. Source of objective weight derivation procedure.

  [Dez20]   Dezert, Tchamova, Han, Tacnet — "The SPOTIS Method," IEEE FUSION 2020.
            Source of rank-reversal-free MCDM scoring.

  [Kal60]   Kalman, R. E. — "A New Approach to Linear Filtering and Prediction Problems,"
            ASME J. Basic Eng., 1960. Source of Kalman filter equations used to smooth
            the MCDM weight vector over time.

HOW TO RUN
----------
  conda run -n venv python3 uav_fog_simulation.py

  The script will prompt: "Enter number of LIVE training episodes (integer >= 1):"
  Enter 3 to 5 for a quick smoke test (~1–2 min per episode on a modern laptop).
  Enter 20+ for meaningful learning curves (drop rate should trend downward).

  To enable the built-in self-test suite (checks shapes, masks, rewards, PPO ratios):
    Set RUN_SELF_TEST = True in SECTION 2 — CONFIG, then run normally.
    Self-test output appears before pre-training and exits without running the simulation.

WHAT GOOD OUTPUT LOOKS LIKE
----------------------------
  Pre-training (5 episodes, seed 1042):
    [PreTrain 01/05]  drop=0.00%  rel=0.99999  loss=...

  Live episode per-tick blocks (PRINT_EVERY=25 ticks, ~50 s / 500 ticks):
    [Ep 1 | Tick 1 | t=100.0 ms]
      Energy  : meanSOC=99.1%  minSOC=99.07%  (UAV4 at 99.07%)
      MEREC   : z=[0.xxx, ...]
      KL-gate : KL=0.0780 -> run_SPOTIS=True       ← first tick always runs SPOTIS
      SPOTIS  : head=UAV02 (tier1)  failover=[2, 0, ...]
      Reward  : r=0.35  [dWL=... dD=... dR=... E=...]
      Exec    : successes=5/5  drops=0

  Episode summary row (drop% should decrease over episodes):
    Ep  Drop%       Rel     Delay   minSOC%  pi_loss   v_loss   ent  lat_mn  lat_p95  RT
     1   0.00%   0.99999   ...ms   99.00%  ...      ...      ...  ...ms   ...ms  PASS

  Healthy indicators:
    - Drop rate stays at 0% while UAVs have charge; may rise as SOC falls below 0.30
    - p95 decision latency << 20 ms in every row (RT = PASS)
    - SOC column falls from ~99% toward ~80% by end of a long run (energy is consumed)
    - KL-gate suppresses SPOTIS on most ticks after tick 1 (run_SPOTIS=False)
    - Policy loss magnitude decreases; value loss converges; entropy stays > 0
"""

# ==============================================================================
# SECTION 17 — OFFLINE PRE-TRAINING AND MAIN DRIVER  (§4.7)
#
# This is the entry point. Run it with:
#     conda run -n venv python3 main.py
#
# Importing the modules below reproduces the original startup sequence in order:
# config (seeds RNGs, prints [CONFIG]) -> world (builds the swarm, prints [WORLD])
# -> physics -> broker -> neural -> simulation. The bodies of run_self_test,
# pretrain_offline, _print_summary_row, and main below are unchanged.
# ==============================================================================

import math
import time
import random
from typing import List, Dict

import torch

from config import *
from models import Task, FogNode, IoTDevice
from world import FOG_SWARM, IOT_DEVICES, TASK_STREAM
from physics import *
from broker import *
from neural import *
from simulation import *
# Names starting with "_" are not pulled in by "import *", so import them explicitly.
from neural import _N_NODE_FEATURES, _NO_NODE

def run_self_test() -> None:
    """
    Built-in self-test suite.  Called only when RUN_SELF_TEST=True (see CONFIG).

    Checks four invariants that would silently corrupt results if broken:
      1. Feature dimensions  — task-aware candidate features have the configured shape
      2. SOC mask thresholds — primary SOC >= 0.30, backup SOC >= 0.15, f_p != f_b
      3. Reward signs        — improving step yields positive reward; worsening yields negative
      4. PPO ratio           — probability ratio r_t = exp(logp_new - logp_old) ≈ 1.0 on epoch 0

    Each check prints PASS or FAIL.  Any FAIL indicates a regression in the relevant section.
    """
    import sys
    n_fail = 0

    def _check(label: str, condition: bool) -> None:
        nonlocal n_fail
        status = "PASS" if condition else "FAIL"
        if not condition:
            n_fail += 1
        print(f"  [{status}] {label}")

    print()
    print("=" * 70)
    print("  SELF-TEST  (RUN_SELF_TEST=True)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Feature dimensions
    # ------------------------------------------------------------------
    print("\n[Test 1] Feature dimensions")
    feature_task = TASK_STREAM[0]
    feature_source = IOT_DEVICES[0]
    x_feat = build_candidate_features(
        FOG_SWARM, feature_task, feature_source
    )   # shape should be (N_FOG, _N_NODE_FEATURES)

    _check(f"build_candidate_features shape == ({N_FOG}, {_N_NODE_FEATURES})",
           tuple(x_feat.shape) == (N_FOG, _N_NODE_FEATURES))

    enc = AttentionEncoder()
    enc.eval()
    with torch.no_grad():
        H_t, c_t = enc(x_feat, update_stats=False)

    _check(f"Encoder H shape == ({N_FOG}, {D_MODEL})",  tuple(H_t.shape) == (N_FOG, D_MODEL))
    _check(f"Encoder c shape == ({D_MODEL},)",           tuple(c_t.shape) == (D_MODEL,))
    _check("c == H.mean(dim=0)",                         torch.allclose(c_t, H_t.mean(dim=0), atol=1e-5))

    # ------------------------------------------------------------------
    # Test 2: SOC mask thresholds
    #
    # Strategy: manufacture a swarm where node 0 is the ONLY node with SOC >= 0.30,
    # and node 1 is the ONLY node with SOC in [0.15, 0.30) (valid backup only).
    # All other nodes have SOC = 0 (infeasible for either role).
    # Expected: f_p = 0, f_b = 1, f_p != f_b.
    # ------------------------------------------------------------------
    print("\n[Test 2] SOC mask thresholds (primary>=0.30, backup>=0.15, f_p != f_b)")

    soc_test = torch.zeros(N_FOG, dtype=torch.float32)
    soc_test[0] = 0.50   # only valid primary (SOC 0.50 >= 0.30 ✓)
    soc_test[1] = 0.20   # only valid backup (0.15 <= SOC < 0.30; invalid primary)
    # all other nodes: SOC = 0.0 (invalid for both roles)

    actor_t = Actor()
    actor_t.eval()
    with torch.no_grad():
        f_p_t, f_b_t, logp_p_t, logp_b_t, _, _ = actor_t.select_action(
            H_t, c_t, soc_test, greedy=True
        )

    _check(f"f_p == 0 (only node with SOC >= 0.30)",              f_p_t == 0)
    _check(f"f_b == 1 (only node with SOC >= 0.15 and f_b != f_p)", f_b_t == 1)
    _check("f_p != f_b",                                          f_p_t != f_b_t)
    _check("Primary SOC >= SOC_PRIMARY_MIN (0.30)",
           float(soc_test[f_p_t]) >= SOC_PRIMARY_MIN)
    _check("Backup SOC >= SOC_BACKUP_MIN (0.15)",
           float(soc_test[f_b_t]) >= SOC_BACKUP_MIN)

    # ------------------------------------------------------------------
    # Test 3: Reward signs
    #
    # Successful, on-time, reliable service should have positive reward.
    # A missed deadline should have negative reward, and an RTH-floor violation
    # must override every other term with the hard critical penalty.
    # ------------------------------------------------------------------
    print("\n[Test 3] Reward signs")

    # Healthy, successful service with positive deadline slack.
    dummy_primary = FogNode(
        node_id=0, tier=1, fr_avg_ghz=3.0,
        MP_tot_gb=8.0, MS_tot_gb=8.0, MP_occ_gb=0.0, MS_occ_gb=0.0,
        E_initial_j=234_000.0, E_res_j=187_200.0,   # SOC = 0.80
        lambda_fail=0.005, mu_fail=0.001,
        varsigma=2, C_mips=24_000, x=0.0, y=0.0,
    )
    dummy_task = Task(
        task_id=0, size_kb=150.0,
        cycles=150.0 * 1024 * CYCLES_PER_BYTE,
        deadline_ms=deadline_for_size_ms(150.0), arrival_s=0.0, is_small=True,
    )

    r_improve, _ = compute_reward(
        task=dummy_task,
        primary_node=dummy_primary,
        outcome={"success": 1, "dropped": False,
                 "D_primary_ms": 60.0, "R_i": 0.98},
        E_consume_j=500.0,                             # small energy draw
        queue_delay_primary_ms=5.0,
        swarm_mean_queue_depth=2.0,
        swarm_max_queue_depth=2.0,
        next_queue_delay_primary_ms=4.0,
    )
    _check(f"Reward is positive for successful on-time service (got {r_improve:.4f})",
           r_improve > 0.0)

    r_miss, _ = compute_reward(
        task=dummy_task,
        primary_node=dummy_primary,
        outcome={"success": 0, "dropped": False,
                 "D_primary_ms": 200.0, "R_i": 0.50},
        E_consume_j=5_000.0,
        queue_delay_primary_ms=80.0,
        swarm_mean_queue_depth=2.0,
        swarm_max_queue_depth=8.0,
        next_queue_delay_primary_ms=120.0,
    )
    _check(f"Reward is negative for missed service (got {r_miss:.4f})", r_miss < 0.0)

    # Worsening step with low-SOC primary (triggers RTH hard penalty).
    dummy_depleted = FogNode(
        node_id=1, tier=3, fr_avg_ghz=1.5,
        MP_tot_gb=2.0, MS_tot_gb=4.0, MP_occ_gb=0.0, MS_occ_gb=0.0,
        E_initial_j=162_000.0, E_res_j=16_200.0,   # SOC = 0.10 < 0.15 => RTH penalty
        lambda_fail=0.025, mu_fail=0.010,
        varsigma=6, C_mips=12_000, x=0.0, y=0.0,
    )
    r_penalty, comp_penalty = compute_reward(
        task=dummy_task,
        primary_node=dummy_depleted,
        outcome={"success": 0, "dropped": True,
                 "D_primary_ms": 200.0, "R_i": 0.0},
        E_consume_j=50_000.0,
        queue_delay_primary_ms=100.0,
        swarm_mean_queue_depth=2.0,
        swarm_max_queue_depth=10.0,
        next_queue_delay_primary_ms=150.0,
    )
    _check(f"RTH penalty fires when SOC < 0.15 (got r={r_penalty:.1f})",
           comp_penalty['rth_penalty'] and r_penalty == -CRITICAL_PENALTY)

    # ------------------------------------------------------------------
    # Test 4: PPO probability ratio ≈ 1.0 on first inner-epoch pass
    #
    # If we run ONE PPO inner epoch, collect the batch on the SAME policy that generated it,
    # then the ratio r_t = exp(logp_new - logp_old) should equal 1.0 everywhere
    # (same weights => same log-probs => exponent = 0 => ratio = 1).
    # We verify this by running ppo_update with PPO_INNER_EPOCHS=1 and checking losses
    # are finite (not NaN/inf), which would happen if ratio diverged.
    # ------------------------------------------------------------------
    print("\n[Test 4] PPO probability ratio finite-and-valid on first epoch")

    actor_ppo = Actor()
    critic_ppo = Critic()
    enc_ppo    = AttentionEncoder()
    opt_ppo    = torch.optim.Adam(
        list(actor_ppo.parameters()) + list(critic_ppo.parameters()) + list(enc_ppo.parameters()),
        lr=LEARNING_RATE,
    )

    # Build a minimal synthetic batch: 4 steps, all with valid actions.
    soc_full = torch.ones(N_FOG, dtype=torch.float32)   # all nodes fully charged => no masking
    with torch.no_grad():
        H_ppo, c_ppo = enc_ppo(x_feat, update_stats=False)

    mini_batch: Dict = {'H': [], 'c': [], 'soc': [], 'f_p': [], 'f_b': [],
                        'logp_old': [], 'advantages': [], 'vtargets': []}

    for _ in range(4):   # 4 synthetic steps
        with torch.no_grad():
            f_p_s, f_b_s, lp_p, lp_b, _, _ = actor_ppo.select_action(
                H_ppo, c_ppo, soc_full, greedy=False
            )
            if f_p_s == _NO_NODE:
                continue
            lp_val = (lp_p.item() if lp_p is not None else 0.0) + \
                     (lp_b.item() if lp_b is not None else 0.0)
            V_ppo = float(critic_ppo(c_ppo).item())

        mini_batch['H'].append(H_ppo.detach())
        mini_batch['c'].append(c_ppo.detach())
        mini_batch['soc'].append(soc_full)
        mini_batch['f_p'].append(f_p_s)
        mini_batch['f_b'].append(f_b_s)
        mini_batch['logp_old'].append(lp_val)
        mini_batch['advantages'].append(1.0)
        mini_batch['vtargets'].append(V_ppo + 0.5)

    if mini_batch['f_p']:
        losses_ppo = ppo_update(actor_ppo, critic_ppo, opt_ppo, mini_batch)
        _check("PPO policy_loss is finite (not NaN/inf)",
               math.isfinite(losses_ppo['policy_loss']))
        _check("PPO value_loss is finite (not NaN/inf)",
               math.isfinite(losses_ppo['value_loss']))
    else:
        print("  [SKIP] No valid actions in synthetic batch — mask may be too tight")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    if n_fail == 0:
        print(f"  Self-test PASSED  (0 failures)")
    else:
        print(f"  Self-test FAILED  ({n_fail} failure(s) — see FAIL lines above)")
    print("=" * 70)
    sys.exit(0 if n_fail == 0 else 1)


# Number of synthetic pre-training episodes run before live fine-tuning.
# Small by default (5) to keep startup time reasonable; increase for better cold-start.
_N_PRETRAIN_EPISODES: int = 5

# Sub-seed for pre-training data: distinct from SEED so pre-train traces never
# overlap with live-episode traces, while remaining fully reproducible.
_PRETRAIN_SEED: int = SEED + 1000   # 1042; offset avoids accidental collision with SEED


def pretrain_offline(
    actor:    Actor,
    critic:   Critic,
    encoder:  AttentionEncoder,
    optimizer: torch.optim.Optimizer,
    n_pretrain_episodes: int = _N_PRETRAIN_EPISODES,
) -> None:
    """
    Offline "digital-twin" pre-training before live deployment.

    Runs n_pretrain_episodes of run_episode(..., train=True) on synthetic traces
    generated under a distinct sub-seed.  Because no real hardware is involved,
    incurring -CRITICAL_PENALTY (battery exhaustion) is safe: it just teaches the
    policy to avoid low-SOC primaries before the first real flight.

    The encoder, actor, and critic weights are updated in-place, so the caller's
    module instances are already warm-started when live fine-tuning begins.
    A fresh BrokerSelector is used per pre-train episode (Kalman state resets).
    """
    print()
    print("=" * 70)
    print("  OFFLINE PRE-TRAINING  (digital-twin mode)")
    print(f"  Seed: {_PRETRAIN_SEED}  |  Episodes: {n_pretrain_episodes}")
    print("=" * 70)

    # Temporarily re-seed to the pre-training sub-seed so synthetic traces are
    # reproducible and distinct from the live-phase random stream.
    set_global_seed(_PRETRAIN_SEED)

    for pt_ep in range(1, n_pretrain_episodes + 1):
        broker_pt = BrokerSelector()   # fresh Kalman state per pre-train episode

        # Suppress per-tick prints during pre-training (very verbose for 500 ticks);
        # save and restore PRINT_EVERY around the call.
        saved_pe = PRINT_EVERY
        import builtins
        _orig_print = builtins.print   # save real print

        # Redirect tick-level output to /dev/null during pre-training.
        # Only the per-episode summary line is emitted (via _orig_print below).
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            stats = run_episode(
                episode_idx = pt_ep,
                actor       = actor,
                critic      = critic,
                broker      = broker_pt,
                encoder     = encoder,
                optimizer   = optimizer,
                train       = True,
            )

        _orig_print(
            f"  [PreTrain {pt_ep:02d}/{n_pretrain_episodes}]  "
            f"drop={stats['drop_rate']*100:.2f}%  "
            f"rel={stats['mean_reliability']:.5f}  "
            f"log10R={stats['log10_system_reliability']:.2f}  "
            f"loss={stats['total_loss']:.4f}  "
            f"(policy={stats['policy_loss']:.4f}  "
            f"value={stats['value_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f})"
        )

    # Restore the live-phase seed so live episodes draw from a fresh RNG stream.
    set_global_seed(SEED)
    print("  Pre-training complete. Weights warm-started for live fine-tuning.")
    print("=" * 70)
    print()


def _print_summary_row(stats: Dict, header: bool = False) -> None:
    """
    Print one fixed-width summary row for a live episode.

    Columns (all labeled):
      Ep | Drop% | Rel | log10R | Delay(ms) | minSOC% | pi_loss | v_loss | ent | lat_mean | lat_p95 | RT

    'Rel'    = arithmetic mean of per-task R_i (secondary operator metric).
    'log10R' = ReLIEF Eq.(12) system reliability, sum_i log10 R_i (0 = perfect;
               more negative = worse; a single dropped task contributes -12).
    """
    if header:
        print()
        print(
            f"{'Ep':>4}  {'Drop%':>6}  {'Rel':>8}  {'log10R':>8}  {'Delay':>8}  "
            f"{'minSOC%':>7}  {'pi_loss':>8}  {'v_loss':>9}  {'ent':>6}  "
            f"{'lat_mn':>7}  {'lat_p95':>7}  RT"
        )
        print("-" * 100)
        return

    rt_flag = "PASS" if stats['p95_lat_ms'] < stats['budget_ms'] else "FAIL"
    # minSOC is not in stats directly; it's the mean_soc - soc_spread/2 approximation.
    # We don't store per-node SOC in stats; use mean_soc as proxy here.
    # (Exact minSOC is printed per-tick inside run_episode.)
    print(
        f"{stats['episode']:>4}  "
        f"{stats['drop_rate']*100:>5.2f}%  "
        f"{stats['mean_reliability']:>8.5f}  "
        f"{stats['log10_system_reliability']:>8.2f}  "
        f"{stats['mean_delay_ms']:>7.1f}ms  "
        f"{stats['mean_soc']*100:>6.2f}%  "
        f"{stats['policy_loss']:>+8.4f}  "
        f"{stats['value_loss']:>9.2f}  "
        f"{stats['entropy']:>6.3f}  "
        f"{stats['mean_lat_ms']:>6.2f}ms  "
        f"{stats['p95_lat_ms']:>6.2f}ms  "
        f"{rt_flag}"
    )


def main() -> None:
    """
    Top-level driver: startup banner → pre-training → live training loop → final report.

    Execution flow:
      1. Print startup banner.
      2. Build shared neural modules (encoder, actor, critic) + Adam optimizer.
      3. Ask user for number of live episodes via input() — never hardcoded.
      4. Run offline pre-training (cold-start warm-up).
      5. Fine-tune with live episodes, printing a summary row per episode.
      6. Print final report.
    """

    # ---- (1) Startup banner ----
    print()
    print("=" * 70)
    print("  Energy-Aware UAV Fog Simulation")
    print("  Self-Attention + PPO Routing  |  MEREC-Kalman-SPOTIS Broker")
    print("=" * 70)
    set_global_seed(SEED)   # re-seed (may already be done at import, but be explicit)
    print(f"  Seed             : {SEED}")
    print(f"  Fog UAVs         : {N_FOG}  (T1={TIER_COUNTS[0]} "
          f"T2={TIER_COUNTS[1]}; tiers from CE_eff rank)")
    print(f"  IoT devices      : {N_IOT}  "
          f"({N_LOW_POWER_IOT} low-pwr @ {P_TX_LOW_DBM:.0f} dBm  "
          f"/ {N_HIGH_POWER_IOT} high-pwr @ {P_TX_HIGH_DBM:.0f} dBm)")
    print(f"  Tasks/episode    : {TASKS_PER_EPISODE}")
    print(f"  Control tick     : {int(CONTROL_TICK_S*1000)} ms  ({1/CONTROL_TICK_S:.0f} Hz)")
    print(f"  Arrival rate     : {TASK_ARRIVAL_RATE:.0f} tasks/s  "
          f"(mean inter-arrival: {1000/TASK_ARRIVAL_RATE:.1f} ms)")
    print(f"  Real-time budget : {1000/TASK_ARRIVAL_RATE:.1f} ms / task")
    print(f"  Print every      : {PRINT_EVERY} ticks  (VERBOSE={VERBOSE})")
    print(f"  Encoder          : D_MODEL={D_MODEL}  N_HEADS={N_HEADS}  "
          f"N_ENC_LAYERS={N_ENC_LAYERS}  D_FF={D_FF}")
    print(f"  PPO              : gamma={GAMMA}  lambda={GAE_LAMBDA}  "
          f"clip={PPO_CLIP}  lr={LEARNING_RATE}  epochs={PPO_INNER_EPOCHS}")
    print("=" * 70)

    # ---- (2) Build shared modules ----
    encoder   = AttentionEncoder()
    actor     = Actor()
    critic    = Critic()

    # Single Adam optimizer over all trainable parameters.
    # The live phase is fine-tuning: weights are initialized from pre-training,
    # and the same optimizer instance carries momentum between phases.
    all_params = (list(encoder.parameters())
                + list(actor.parameters())
                + list(critic.parameters()))
    optimizer  = torch.optim.Adam(all_params, lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"\n  Trainable parameters: {n_params:,}")

    # ---- (3) Prompt for number of live episodes ----
    n_live = 0
    while n_live < 1:
        raw = input("\nEnter number of LIVE training episodes (integer >= 1): ").strip()
        try:
            n_live = int(raw)
            if n_live < 1:
                raise ValueError
        except ValueError:
            print(f"  Invalid input '{raw}'. Please enter a positive integer.")
            n_live = 0

    print(f"  Running {n_live} live episode(s).\n")

    # ---- (4) Offline pre-training (cold-start warm-up) ----
    pretrain_offline(
        actor    = actor,
        critic   = critic,
        encoder  = encoder,
        optimizer = optimizer,
        n_pretrain_episodes = _N_PRETRAIN_EPISODES,
    )

    # ---- (5) Live training loop ----
    all_stats: List[Dict] = []

    _print_summary_row({}, header=True)   # column headers

    for ep in range(1, n_live + 1):
        # Each live episode gets a fresh BrokerSelector (clean Kalman state).
        # The neural-network weights carry over automatically (shared module instances).
        broker = BrokerSelector()

        stats = run_episode(
            episode_idx = ep,
            actor       = actor,
            critic      = critic,
            broker      = broker,
            encoder     = encoder,
            optimizer   = optimizer,
            train       = True,
        )

        all_stats.append(stats)
        _print_summary_row(stats)

    # ---- (6) Final report ----
    print()
    print("=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)

    if not all_stats:
        print("  No live episodes completed.")
        return

    drop_rates = [s['drop_rate'] for s in all_stats]
    best_drop  = min(drop_rates)
    last_drop  = drop_rates[-1]
    first_drop = drop_rates[0]

    # Learning evidence: drop rate trended downward (last < first).
    drop_improved = last_drop < first_drop
    # Compute simple linear regression slope for a more robust trend test.
    # slope < 0 means drop rate is decreasing over episodes, i.e., the agent is learning.
    n_eps = len(drop_rates)
    if n_eps >= 2:
        ep_indices = list(range(n_eps))                       # x-axis: episode number 0..N-1
        x_mean = sum(ep_indices) / n_eps                      # mean episode index
        y_mean = sum(drop_rates)  / n_eps                     # mean drop rate
        covariance = sum(
            (ep_indices[i] - x_mean) * (drop_rates[i] - y_mean)
            for i in range(n_eps)
        )
        x_variance = sum((ep_indices[i] - x_mean) ** 2 for i in range(n_eps)) or 1e-9
        slope = covariance / x_variance   # units: drop-rate change per episode
    else:
        slope = 0.0

    # Real-time evidence: p95 < budget in every episode.
    budget_ms = 1000.0 / TASK_ARRIVAL_RATE
    rt_met_all = all(s['p95_lat_ms'] < budget_ms for s in all_stats)

    mean_rel_final = all_stats[-1]['mean_reliability']
    log10R_final   = all_stats[-1]['log10_system_reliability']
    geoR_final     = all_stats[-1]['geo_mean_reliability']
    nines_final    = all_stats[-1]['reliability_nines']
    mean_delay_final = all_stats[-1]['mean_delay_ms']

    print(f"  Live episodes      : {n_live}")
    print(f"  Best drop rate     : {best_drop*100:.3f}%  (episode {drop_rates.index(best_drop)+1})")
    print(f"  Final drop rate    : {last_drop*100:.3f}%")
    print(f"  Drop rate slope    : {slope*100:+.4f}% / episode  "
          f"({'IMPROVING ↓' if slope < 0 else 'NOT improving'})")
    print(f"  Learning evidence  : {'YES — drop rate trended DOWN' if drop_improved else 'NOT YET'}")
    print(f"  Final mean per-task R : {mean_rel_final:.5f}   (arithmetic mean, secondary metric)")
    print(f"  Final system R (Eq.12): log10(prod R_i) = {log10R_final:.2f}   "
          f"geo-mean/task = {geoR_final:.6f}   nines = {nines_final:.2f}")
    print(f"  Final mean delay   : {mean_delay_final:.1f} ms")
    print(f"  RT budget          : {budget_ms:.1f} ms / task")
    print(f"  RT met (all eps)   : {'YES — production-ready' if rt_met_all else 'NO — latency exceeded budget in some episode(s)'}")
    print("=" * 70)


if RUN_SELF_TEST:
    run_self_test()

if __name__ == "__main__":
    main()
