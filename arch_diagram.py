"""
arch_diagram.py
===============
Renders the full end-to-end system architecture / methodology diagram
(figures/fig8_architecture.{png,pdf}) — every subsystem, constant, and data
path of the two-stage energy-aware UAV-fog pipeline, including the mobility
layer from UAV_Mobility_Design.pdf.

Usage:  conda run -n venv python3 arch_diagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# palette
C_EDGE  = "#fff3d6"; E_EDGE  = "#c8963e"   # IoT / edge
C_CHAN  = "#eeeeee"; E_CHAN  = "#888888"   # channel
C_MOB   = "#ddeaf7"; E_MOB   = "#2c6fad"   # mobility (new, blue as in PDF)
C_FOG   = "#e4f2e4"; E_FOG   = "#3e8e4e"   # fog layer
C_S1    = "#fde4e1"; E_S1    = "#c0392b"   # stage 1
C_S2    = "#e9e2f5"; E_S2    = "#6c3fa0"   # stage 2
C_EXE   = "#e0f4f4"; E_EXE   = "#1f8a8a"   # executor
C_MET   = "#f5f5dc"; E_MET   = "#7a7a3a"   # metrics

fig, ax = plt.subplots(figsize=(15, 19))
ax.set_xlim(0, 100); ax.set_ylim(12, 133)
ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec, title_fs=10.5, fs=8.2, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=ec, zorder=3)
    ax.text(x + 1.2, y + h - 4.2, "\n".join(lines), ha="left", va="top",
            fontsize=fs, zorder=3, linespacing=1.45)


def arrow(x1, y1, x2, y2, label="", color="#333333", lw=1.8, style="-|>",
          ls="-", fs=8, dx=0.8, connstyle=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw,
                                 linestyle=ls, zorder=4,
                                 connectionstyle=connstyle or "arc3,rad=0"))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=fs,
                color=color, ha="left", va="center", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


# ============================== TITLE ==============================
ax.text(50, 130.5, "Energy-Aware UAV-Fog Task Distribution — System Architecture",
        ha="center", fontsize=16, fontweight="bold")
ax.text(50, 128.4, "3-D mobile UAV fog layer  ·  Stage-1 rank-reversal-free MCDM fog-head election  ·  "
                   "Stage-2 attention + PPO task routing  ·  delayed-backup execution",
        ha="center", fontsize=10, color="#444444")

# ====================== EDGE / IoT LAYER (bottom-up data path drawn top-down) ======================
box(2, 112, 45, 14, "EDGE LAYER — IoT ground devices (z = 0)",
    ["• 200 devices on 5 km × 5 km grid; 4-hotspot Gaussian mixture (σ = 400 m)",
     "• 150 low-power sensors @ 34 dBm (2.51 W) · 50 terminals @ 45 dBm (31.6 W)",
     "• Task arrivals: Poisson λ = 50 tasks/s (mean inter-arrival 20 ms)",
     "• Payload: bounded Pareto α = 1.5, 60–800 KB;  intensity c ~ U[1600, 1800] cyc/B",
     "• Deadlines: small (<250 KB) 150 ms · large 550 ms  →  lᵢ = sᵢ·1024·c cycles"],
    C_EDGE, E_EDGE)

box(53, 112, 45, 14, "WIRELESS CHANNEL — LoS air-to-ground uplink",
    ["• 3-D slant range d = √(Δx² + Δy² + z²)   [FU-Serve Eq. 10]",
     "• Path loss g(d) = υ₁·d^(−2.3),  υ₁ = (λ/4π)² @ f_c = 2 GHz",
     "• Shannon rate TR = B·log₂(1 + g·P_tx/N₀),  B = 30 MHz",
     "• Noise floor N₀ = kTB (−174 dBm/Hz, NF = 0 dB)",
     "• T_u = 8·1024·sᵢ / TR   —  movement re-prices every link each tick",
     "• Inter-UAV relay leg (head → executor): same model @ P_tx = 10 W"],
    C_CHAN, E_CHAN)

arrow(47, 119, 53, 119, color=E_EDGE)
ax.text(50, 120.6, "task offload (uplink)", fontsize=7.5, ha="center",
        color=E_EDGE, bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))

# ====================== MOBILITY ======================
box(2, 92, 45, 17, "UAV MOBILITY LAYER  (per 100 ms tick — step 1)",
    ["Modes (MOVE_MODE):",
     "  • event_driven — track assigned drifting demand hotspot (FU-Serve/Con-Fog)",
     "  • random — waypoint wander @ cruise, demand-agnostic ablation (Con-Fog §V-B)",
     "  • static — V = 0, z = 0: reproduces fixed-position baseline exactly",
     "Collision-free by design: 5 UAVs/hotspot on 250 m standoff ring (x-y)",
     "  + distinct altitude layers 80–150 m (vertical separation)",
     "Zeng19 rotary-wing power  P(V) = P₀(1+3V²/U²ₜ) + Pᵢ·induced + ½d₀ρAV³",
     "  hover P(0) = 870.5 W  ·  cruise P(12 m/s) ≈ 779 W (induced-power dip)",
     "Per tick: drift hotspots (2 m/s) → move UAVs → drain P(V)·Δt from battery"],
    C_MOB, E_MOB)

box(53, 92, 45, 17, "UAV FOG LAYER — 20 heterogeneous fog UAVs",
    ["Hardware from latent quality q ∈ [0,1] (envelope pinned each episode):",
     "  clock 1.5–3.0 GHz · 12–24k MIPS · ς 6→2 · RAM 2–8 GB · store 4–8 GB",
     "  battery 162–234 kJ · λ_fail 0.025→0.005 /s · μ_fail 0.010→0.001 /s",
     "Tiering (emergent): CE_eff = f/(ς·C_MIPS·10⁶) → rank ↓ → T1 (10) / T2 (10)",
     "Per-node live state: E_res (SOC), MP/MS occupancy, EDF queues, position (x,y,z)",
     "Energy budget/tick:  E += P(V)·Δt + P_proc·t_exec + 2P_comm·t_comm + P_idle·t_idle"],
    C_FOG, E_FOG)

arrow(24, 112, 24, 109.4, "positions feed geometry", color=E_MOB)
arrow(47, 100.5, 53, 100.5, color=E_MOB)
ax.text(50, 102.1, "P(V)·Δt drain + (x,y,z)", fontsize=7.2, ha="center",
        color=E_MOB, bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))
arrow(75, 112, 75, 109.4, color=E_CHAN)
ax.text(76, 110.7, "slant ranges → data-rates (recomputed after movement — step 2)",
        fontsize=7.2, ha="left", va="center", color="#666666",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))

# ====================== STAGE 1 ======================
box(2, 66, 96, 22, "STAGE 1 — MASTER FOG-HEAD ELECTION  (per 100 ms tick — rank-reversal-free MCDM, step 3)",
    [""], C_S1, E_S1, title_fs=11.5)

box(4, 68, 21, 15.5, "Criteria matrix  X (20×5)",
    ["E_res ↑  residual energy (J)",
     "ME ↑   memory efficiency",
     "  = [α(MPᵗ−MPᵒ)+β(MSᵗ−MSᵒ)]/8 GB",
     "R ↑    reliability e^(−λt_e−μt_c)",
     "D ↓    expected delay (ms)",
     "WL ↓   queued cycles"],
    "white", E_S1, title_fs=9)
box(27, 68, 16, 15.5, "MEREC",
    ["objective weights",
     "from criterion",
     "removal effects",
     "z ∈ ℝ⁵ per tick",
     "(+0.50 ms)"],
    "white", E_S1, title_fs=9)
box(45, 68, 17, 15.5, "Kalman smoother",
    ["w ← w + K(z − w)",
     "Q = 10⁻⁴·I₅",
     "R = 10⁻²·I₅",
     "damps weight jitter",
     "(+0.10 ms)"],
    "white", E_S1, title_fs=9)
box(64, 68, 15, 15.5, "KL gate",
    ["D_KL(w_new‖w_old)",
     "> θ = 0.05 ?",
     "→ re-rank only on",
     "  significant drift",
     "(+0.05 ms)"],
    "white", E_S1, title_fs=9)
box(81, 68, 15, 15.5, "SPOTIS",
    ["fixed a-priori bounds",
     "[S_min, S_max] per",
     "criterion ⇒ NO rank",
     "reversal [Dez20]",
     "→ head + failover",
     "(+1.00 ms if run)"],
    "white", E_S1, title_fs=9)
arrow(25, 76.5, 27, 76.5, color=E_S1); arrow(43, 76.5, 45, 76.5, color=E_S1)
arrow(62, 76.5, 64, 76.5, color=E_S1); arrow(79, 76.5, 81, 76.5, color=E_S1)
ax.text(50, 67.1, "Elected head = single entry point: brokers every uplink, runs Stage-2 dispatch, "
                  "relays tasks to executors  ·  on handover: 2 KB state-sync to all UAVs + 25 ms "
                  "re-election setup blocks the head radio",
        ha="center", va="center", fontsize=7.5, color=E_S1, zorder=3)

arrow(75, 92, 75, 88.4, "per-node metrics (post-move geometry)", color=E_FOG)

# ====================== STAGE 2 ======================
box(2, 40, 96, 22, "STAGE 2 — PER-TASK ROUTING  (attention encoder + PPO actor-critic, step 4)",
    [""], C_S2, E_S2, title_fs=11.5)

box(4, 42, 20, 15.5, "Node features (20×5)",
    ["SOC · free-mem ratio",
     "queue cycles · TR̂ · CE_eff",
     "RunningNorm (EMA μ, σ)"],
    "white", E_S2, title_fs=9)
box(26, 42, 20, 15.5, "Attention encoder",
    ["3 × TransformerEncoder",
     "d_model = 128 · 8 heads",
     "FF 256  [Vas17]",
     "H (20×128); c = mean(H)"],
    "white", E_S2, title_fs=9)
box(48, 42, 22, 15.5, "Actor (pointer head)",
    ["SOC feasibility masks:",
     "  primary ≥ 0.30 · backup ≥ 0.15",
     "  f_b ≠ f_p  (independence)",
     "→ primary f_p + backup f_b",
     "greedy @ eval · sample @ train"],
    "white", E_S2, title_fs=9)
box(72, 42, 24, 15.5, "Critic + PPO update",
    ["V_φ(c) baseline · GAE(γ=0.99, λ=0.95)",
     "clip ε = 0.2 · 4 inner epochs · Adam 3e-4",
     "r = 0.15ΔWL + 0.35ΔD + 0.35ΔR − 0.15E",
     "SOC < RTH floor ⇒ −100 penalty",
     "offline pre-train → live fine-tune"],
    "white", E_S2, title_fs=9)
arrow(24, 50.5, 26, 50.5, color=E_S2); arrow(46, 50.5, 48, 50.5, color=E_S2)
arrow(70, 50.5, 72, 50.5, color=E_S2)

arrow(50, 66, 50, 62.4, "elected head brokers each task", color=E_S1)

# ====================== EXECUTOR ======================
box(2, 14, 63, 22, "EXECUTOR — delayed-backup dispatch  (step 5)",
    ["1. Task relayed:  IoT → fog head (uplink) → primary f_p (inter-UAV link, 10 W)",
     "2. Head's single radio serializes all transfers — unfinished air-time carries",
     "   over as backlog (M/G/1-style);  handover setup (25 ms) blocks the pipeline",
     "3. Primary → EDF queue;  D = T_broker + T_u + T_relay + T_radio + T_Q + T_e",
     "4. Survival draw  P = e^(−(λ+μ)·t_exposure)  couples R-metric to outcomes",
     "5. Backup slot HELD, dispatched ONLY if primary fails / misses deadline",
     "   and remaining budget covers T_u(f_b)+T_Q(f_b)+T_e(f_b)  (≈99% never sent)",
     "6. Dual-queue EDF: backup queue has absolute non-preemptive priority",
     "7. Energy charged to executing node(s): E_proc + E_comm;  queues pruned",
     "   of deadline-expired tasks each tick  (step 6: cleanup + logging)"],
    C_EXE, E_EXE)

box(69, 14, 29, 22, "EPISODE METRICS  (paper figures)",
    ["• delay CDF + mean decomposition",
     "  (broker/up/relay/radio/queue/exec)",
     "• comm overhead: air-time · MB·hop",
     "  · energy useful/wasted/signalling",
     "• propagation delay by mobility mode",
     "• energy split (propulsion P(V) /",
     "  compute / comms) · SOC trajectory",
     "• per-device QoS fairness — Jain over",
     "  per-device mean delay + success",
     "• per-UAV CPU util · memory occupancy",
     "• success rate · J per delivered task"],
    C_MET, E_MET)

arrow(33, 40, 33, 36.4, color=E_S2)
ax.text(34, 38.2, "f_p, f_b per task", fontsize=7.5, color=E_S2, ha="left",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))
arrow(65, 25, 69, 25, color=E_EXE)
ax.text(67, 22.9, "telemetry", fontsize=7, ha="center", color=E_EXE,
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))

# ---- control-tick cycle annotation (right margin) ----
arrow(98.5, 17, 98.5, 105, color="#999999", lw=1.4, style="-|>",
      connstyle="arc3,rad=0")
ax.text(99.2, 61, "next 100 ms control tick  (① mobility → ② geometry → ③ Stage-1 → "
                  "④ Stage-2 → ⑤ execute → ⑥ cleanup)",
        rotation=90, va="center", fontsize=8.5, color="#666666")

# ---- reward feedback dashed ----
arrow(20, 36, 20, 40, color=E_EXE, ls="--", lw=1.2)
ax.text(19.2, 37.9, "realised ΔWL, ΔD, ΔR, E → reward", fontsize=7.5,
        color=E_EXE, ha="right",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"fig8_architecture.{ext}"),
                bbox_inches="tight", dpi=300)
print("fig8_architecture.png / .pdf written")
