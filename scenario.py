"""Shared scenario runner for training, evaluation, and dispatch baselines."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import config
import world
from baselines.fodas_baseline import select_node_for_task as fodas_select
from baselines.relief_baseline import (
    WL_TARGET_CYCLES, _estimate_pair, discretize_state, pretrain_agent,
    update_reliability_log_ema,
)
from broker import BrokerSelector
from controller_profile import controller_cycles
from execution_engine import (
    BackupPlan, CapacityResult, ExecutionEngine, EXECUTION_MODEL_VERSION,
)
from models import FogNode, IoTDevice, Task
from neural import (
    Actor, AttentionEncoder, Critic, _NO_NODE, build_candidate_features,
    filter_candidates,
)
from physics import (
    aero_power_w, base_reliability, computational_efficiency, data_rate_bps,
    drain_flight_energy, memory_efficiency, slant_distance_m,
    system_workload_imbalance,
    system_reliability_nines,
    total_delay_ms, transmission_delay_ms, uav_uav_distance_m,
    update_memory_occupancy,
)
from world import (
    assign_mobility, build_fog_swarm, build_iot_devices, step_mobility,
)


LOAD_TARGETS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
DEFAULT_EVAL_SEEDS = tuple(range(10))


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class KeyedStreams:
    """Policy-independent deterministic random values indexed by semantic keys."""

    def __init__(self, seed: int):
        self.seed = int(seed)

    def rng(self, key: str) -> random.Random:
        return random.Random(_stable_seed(self.seed, key))

    def uniform(self, key: str, low: float = 0.0, high: float = 1.0) -> float:
        return self.rng(key).uniform(low, high)

    def exponential(self, key: str, rate: float) -> float:
        if rate <= 0:
            return math.inf
        return self.rng(key).expovariate(rate)

    def integer(self, key: str, stop: int) -> int:
        return self.rng(key).randrange(stop)


@dataclass(frozen=True)
class ScenarioConfig:
    seed: int
    target_load: float
    policy: str
    controller_placement: str = "native"
    warmup_s: float = 10.0
    measurement_s: float = 120.0
    control_tick_s: float = config.CONTROL_TICK_S
    move_mode: str = "event_driven"
    training: bool = False
    arrival_rate_override: Optional[float] = None
    handover_setup_ms: float = config.HANDOVER_SETUP_MS
    n_fog: int = config.N_FOG
    n_iot: int = config.N_IOT
    head_failure_s: Optional[float] = None

    def __post_init__(self):
        if self.target_load <= 0:
            raise ValueError("target_load must be positive")
        if self.warmup_s < 0 or self.measurement_s <= 0:
            raise ValueError("invalid measurement protocol")
        if self.controller_placement not in {
            "native", "equal_controller", "head"
        }:
            raise ValueError("invalid controller placement")
        if (self.arrival_rate_override is not None
                and self.arrival_rate_override <= 0):
            raise ValueError("arrival_rate_override must be positive")
        if self.handover_setup_ms < 0:
            raise ValueError("handover_setup_ms cannot be negative")
        if self.n_fog < 2 or self.n_iot < 1:
            raise ValueError("n_fog must be at least 2 and n_iot must be positive")
        if self.head_failure_s is not None and self.head_failure_s < 0:
            raise ValueError("head_failure_s cannot be negative")

    @property
    def measurement_start_s(self) -> float:
        return self.warmup_s

    @property
    def measurement_end_s(self) -> float:
        return self.warmup_s + self.measurement_s


@dataclass
class ScenarioResult:
    config: ScenarioConfig
    capacity: CapacityResult
    target_load: float
    realized_load: float
    arrival_rate: float
    success_rate: float
    drop_rate: float
    mean_queue_delay_ms: float
    p95_queue_delay_ms: float
    primary_attempts: int
    backup_attempts: int
    external_control_cycles: Dict[str, float]
    energy_totals: Dict[str, float]
    task_records: List[Dict]
    audit: List[Dict]
    input_fingerprint: str
    head_recovery_ticks: float
    tasks_dropped_in_gap: int
    delay_spike_ms: float
    control_jobs_lost: int
    rollout: List[Dict] = field(default_factory=list, repr=False)
    tick_records: List[Dict] = field(default_factory=list)

    def summary_row(self) -> Dict:
        util = self.capacity.utilization()
        return {
            "execution_model_version": EXECUTION_MODEL_VERSION,
            "policy": self.config.policy,
            "controller_placement": self.config.controller_placement,
            "seed": self.config.seed,
            "target_load": self.target_load,
            "realized_load": self.realized_load,
            "arrival_rate": self.arrival_rate,
            **util,
            "success_rate": self.success_rate,
            "drop_rate": self.drop_rate,
            "mean_queue_delay_ms": self.mean_queue_delay_ms,
            "p95_queue_delay_ms": self.p95_queue_delay_ms,
            "primary_attempts": self.primary_attempts,
            "backup_attempts": self.backup_attempts,
            "external_control_cycles": sum(
                self.external_control_cycles.values()),
            "external_control_cycles_by_algorithm": dict(
                self.external_control_cycles),
            "energy_totals": dict(self.energy_totals),
            "input_fingerprint": self.input_fingerprint,
            "head_recovery_ticks": self.head_recovery_ticks,
            "tasks_dropped_in_gap": self.tasks_dropped_in_gap,
            "delay_spike_ms": self.delay_spike_ms,
            "control_jobs_lost": self.control_jobs_lost,
        }


class PolicyAdapter:
    name = "base"
    uses_stage1 = False
    native_control_external = True
    controller_algorithm = "random"

    def decide(
        self, task: Task, dev: IoTDevice, fog: List[FogNode],
        sim_time_s: float, me: Mapping[int, float],
    ) -> Tuple[int, int]:
        raise NotImplementedError


class RandomPolicyAdapter(PolicyAdapter):
    name = "random"
    controller_algorithm = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(_stable_seed(seed, "random-policy"))

    def decide(self, task, dev, fog, sim_time_s, me):
        feasible = [i for i, n in enumerate(fog)
                    if n.soc >= config.SOC_PRIMARY_MIN and n.available]
        if not feasible:
            return _NO_NODE, _NO_NODE
        primary = self.rng.choice(feasible)
        backups = [i for i, n in enumerate(fog)
                   if i != primary and n.soc >= config.SOC_BACKUP_MIN
                   and n.available]
        return primary, self.rng.choice(backups) if backups else _NO_NODE


class FODASPolicyAdapter(PolicyAdapter):
    name = "FODAS"
    controller_algorithm = "fodas"

    def decide(self, task, dev, fog, sim_time_s, me):
        return fodas_select(task, dev, fog, sim_time_s)


class ReLIEFPolicyAdapter(PolicyAdapter):
    name = "ReLIEF"
    controller_algorithm = "relief"

    def __init__(self, seed: int, *, pretrain: bool = True):
        self.seed = seed
        self.agent = None
        if not pretrain:
            from baselines.relief_baseline import ReLIEFConfig, ReLIEFAgent
            self.agent = ReLIEFAgent(ReLIEFConfig())
        self.log10_reliability_ema = 0.0
        self._pending_update = None

    def prepare(self, fog, iot, centres):
        if self.agent is None:
            self.agent = pretrain_agent(
                self.seed, world_state=copy.deepcopy((fog, iot, centres)))

    def decide(self, task, dev, fog, sim_time_s, me):
        if self.agent is None:
            raise RuntimeError("ReLIEF adapter was not prepared")
        per_node, imbalance = system_workload_imbalance(fog)
        state = discretize_state(
            system_reliability_nines(self.log10_reliability_ema),
            imbalance, WL_TARGET_CYCLES, self.agent.cfg)
        action = self.agent.select_pair(fog, task, dev, state, greedy=True)
        self._pending_update = None
        if action[0] != _NO_NODE:
            delay_ms, reliability = _estimate_pair(
                task, dev, fog, action[0], action[1])
            self._pending_update = (
                state, action, task, delay_ms, reliability, per_node)
        return action

    def observe_assignment(self, fog):
        if self._pending_update is None:
            return
        state, action, task, delay_ms, reliability, per_node = (
            self._pending_update)
        self._pending_update = None
        self.log10_reliability_ema = update_reliability_log_ema(
            self.log10_reliability_ema, reliability, self.agent.cfg)
        per_node[fog[action[0]].node_id] += task.cycles
        mean_work = sum(per_node.values()) / len(per_node)
        imbalance = sum(abs(work - mean_work) for work in per_node.values())
        next_state = discretize_state(
            system_reliability_nines(self.log10_reliability_ema),
            imbalance, WL_TARGET_CYCLES, self.agent.cfg)
        reward = self.agent.reward(
            imbalance, WL_TARGET_CYCLES, delay_ms,
            task.deadline_ms, reliability)
        self.agent.observe(state, action, reward, next_state, fog)


class AttentionPPOPolicyAdapter(PolicyAdapter):
    name = "Attention+PPO"
    uses_stage1 = True
    native_control_external = False
    controller_algorithm = "attention_ppo"

    def __init__(
        self, encoder: Optional[AttentionEncoder], actor: Optional[Actor],
        critic: Optional[Critic] = None,
        head_mode: str = "spotis",
        stage2_mode: str = "rl",
        final_head_selection: bool = False,
    ):
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.head_mode = head_mode
        self.stage2_mode = stage2_mode
        self.final_head_selection = final_head_selection
        self.controller_algorithm = (
            "fodas" if stage2_mode == "fodas" else "attention_ppo")
        if head_mode == "spotis":
            self.broker = BrokerSelector(final_fix=final_head_selection)
        elif head_mode in {"fu_serve", "2dp_fhs", "3d_pos"}:
            from head_baselines import make_head_selector
            self.broker = make_head_selector(head_mode)
        else:
            self.broker = BrokerSelector()
        self.head: Optional[FogNode] = None
        self.pending_head: Optional[FogNode] = None
        self.last_stage1: Optional[Dict] = None
        self.transitions: Dict[int, Dict] = {}

    def elect_head(
        self, fog: List[FogNode], iot: List[IoTDevice],
        task: Task, tick: int,
    ) -> Dict:
        update_memory_occupancy(fog)
        me = memory_efficiency(fog)
        workload, _ = system_workload_imbalance(fog)
        mean_power = float(np.mean([d.p_tx_w for d in iot]))
        metrics = {}
        stage2_cycles = (config.T_PPO_INFER_MS / 1000.0
                         * config.CONTROLLER_REFERENCE_HZ)
        engine = fog[0].cpu_state._engine
        head_id = (self.head.node_id if self.head is not None
                   and self.head.available else None)
        inherited = (engine.control_backlog_cycles(head_id)
                     if head_id is not None else 0.0)
        for node in fog:
            mean_dist = max(float(np.mean([
                slant_distance_m(node, d.x, d.y) for d in iot
            ])), 1.0)
            rate = data_rate_bps(mean_dist, mean_power)
            extra_s = (0.0 if node.node_id == head_id else
                       inherited / node.fr_avg_hz())
            control_service_ms = 1000.0 * (
                engine.projected_control_service_s(
                    node.node_id, stage2_cycles) + extra_s)
            metrics[node.node_id] = {
                "me": me[node.node_id],
                "r0": base_reliability(
                    node, task.cycles, task.size_kb, rate),
                "d_ms": (max(0.0, control_service_ms - config.T_CTRL_IDEAL_MS)
                         if self.final_head_selection else total_delay_ms(
                             task, node, mean_dist, mean_power, False)["total_ms"]),
                "wl": workload[node.node_id],
            }
        if self.head_mode == "spotis":
            self.last_stage1 = self.broker.select_head(
                fog, metrics, 0, tick)
        else:
            self.last_stage1 = self.broker.select_head(
                fog, metrics, 0, tick, iot_devices=iot)
        self.pending_head = self.last_stage1["head_node"]
        return self.last_stage1

    def activate_pending_head(self) -> Optional[FogNode]:
        if self.pending_head is not None:
            self.head = self.pending_head
            self.pending_head = None
        return self.head

    def decide(self, task, dev, fog, sim_time_s, me):
        if self.stage2_mode == "fodas":
            return fodas_select(task, dev, fog, sim_time_s)
        if self.encoder is None or self.actor is None:
            raise RuntimeError("RL dispatch requires encoder and actor")
        features = build_candidate_features(fog, task, dev, me)
        mask, scores = filter_candidates(fog, task, dev)
        with torch.no_grad():
            h, c = self.encoder(features, update_stats=self.encoder.training)
            soc = torch.tensor([n.soc for n in fog], dtype=torch.float32)
            primary, backup, logp_p, logp_b, *_ = self.actor.select_action(
                h, c, soc, greedy=not self.encoder.training,
                candidate_mask=mask, heuristic_scores=scores)
            value = float(self.critic(c).item()) if self.critic is not None else 0.0
        logp = (
            float(logp_p.item()) if logp_p is not None else 0.0)
        if logp_b is not None and backup != _NO_NODE:
            logp += float(logp_b.item())
        if self.critic is not None and primary != _NO_NODE:
            self.transitions[task.task_id] = {
                "H": h.detach(), "c": c.detach(), "soc": soc,
                "cand_mask": mask, "f_p": primary, "f_b": backup,
                "logp_old": logp, "value": value,
            }
        return primary, backup


def make_policy(
    policy: str, seed: int, nets: Optional[Tuple[AttentionEncoder, Actor]] = None,
    *, relief_pretrain: bool = True, critic: Optional[Critic] = None,
    head_mode: str = "spotis",
    final_head_selection: bool = False,
) -> PolicyAdapter:
    key = policy.lower().replace("+", "").replace("-", "").replace("_", "")
    if key in {"attentionppo", "rl", "ppo"}:
        if nets is None:
            raise ValueError("Attention+PPO requires (encoder, actor)")
        return AttentionPPOPolicyAdapter(
            *nets, critic=critic, head_mode=head_mode,
            final_head_selection=final_head_selection)
    if key in {"headfodas", "fodashead"}:
        return AttentionPPOPolicyAdapter(
            None, None, head_mode=head_mode, stage2_mode="fodas",
            final_head_selection=final_head_selection)
    if key == "random":
        return RandomPolicyAdapter(seed)
    if key == "fodas":
        return FODASPolicyAdapter()
    if key == "relief":
        return ReLIEFPolicyAdapter(seed, pretrain=relief_pretrain)
    raise ValueError(f"unknown policy {policy}")


def expected_task_cycles() -> float:
    alpha = config.PARETO_ALPHA
    low = config.TASK_SIZE_MIN_KB
    high = config.TASK_SIZE_MAX_KB
    norm = 1.0 - (low / high) ** alpha
    if math.isclose(alpha, 1.0):
        mean_kb = alpha * low ** alpha * math.log(high / low) / norm
    else:
        mean_kb = (
            alpha * low ** alpha
            * (high ** (1.0 - alpha) - low ** (1.0 - alpha))
            / ((1.0 - alpha) * norm)
        )
    mean_cpb = 0.5 * (
        config.CYCLES_PER_BYTE_MIN + config.CYCLES_PER_BYTE_MAX)
    return mean_kb * 1024.0 * mean_cpb


def arrival_rate_for_load(target_load: float, fog: Sequence[FogNode]) -> float:
    return target_load * sum(n.fr_avg_hz() for n in fog) / expected_task_cycles()


def generate_paired_tasks(
    streams: KeyedStreams, arrival_rate: float, end_s: float,
) -> List[Task]:
    rng = streams.rng("task-stream")
    tasks: List[Task] = []
    arrival = 0.0
    while True:
        arrival += rng.expovariate(arrival_rate)
        if arrival >= end_s:
            break
        while True:
            u = rng.random()
            size = config.TASK_SIZE_MIN_KB * (1.0 - u) ** (
                -1.0 / config.PARETO_ALPHA)
            if size <= config.TASK_SIZE_MAX_KB:
                break
        cpb = rng.uniform(
            config.CYCLES_PER_BYTE_MIN, config.CYCLES_PER_BYTE_MAX)
        small = size < config.SMALL_TASK_THRESHOLD_KB
        tasks.append(Task(
            task_id=len(tasks), size_kb=size,
            cycles=size * 1024.0 * cpb,
            deadline_ms=config.deadline_for_size_ms(size),
            arrival_s=arrival, is_small=small,
        ))
    return tasks


def _world_for_seed(
    seed: int, n_fog: int = config.N_FOG, n_iot: int = config.N_IOT,
):
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(_stable_seed(seed, "initial-world"))
        np.random.seed(_stable_seed(seed, "initial-world-numpy") & 0xFFFFFFFF)
        fog = build_fog_swarm(n_fog)
        iot, centres = build_iot_devices(n_iot)
        assign_mobility(fog, centres)
        return fog, iot, centres
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _step_mobility_paired(
    fog: List[FogNode], centres: List[List[float]], dt_s: float,
    mobility_rng: random.Random,
) -> None:
    state = random.getstate()
    try:
        random.setstate(mobility_rng.getstate())
        step_mobility(fog, centres, dt_s)
        mobility_rng.setstate(random.getstate())
    finally:
        random.setstate(state)


def _input_fingerprint(
    tasks: Sequence[Task], fog: Sequence[FogNode],
    iot: Sequence[IoTDevice], centres: Sequence[Sequence[float]],
) -> str:
    payload = [
        [(round(t.arrival_s, 12), round(t.size_kb, 9),
          round(t.cycles, 3), t.deadline_ms) for t in tasks],
        [(n.node_id, round(n.fr_avg_ghz, 9), round(n.E_res_j, 6),
          round(n.x, 6), round(n.y, 6),
          round(n.lambda_fail, 12), round(n.mu_fail, 12)) for n in fog],
        [(d.dev_id, round(d.x, 6), round(d.y, 6), round(d.p_tx_w, 9))
         for d in iot],
        [[round(value, 6) for value in centre] for centre in centres],
        {
            "control_tick_s": config.CONTROL_TICK_S,
            "move_mode": world.MOVE_MODE,
            "execution_model": EXECUTION_MODEL_VERSION,
        },
    ]
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def run_scenario(
    scenario: ScenarioConfig,
    adapter: PolicyAdapter,
    controller_profile: Mapping,
) -> ScenarioResult:
    """Run one policy/load/seed cell through the shared execution engine."""
    world.MOVE_MODE = scenario.move_mode
    streams = KeyedStreams(scenario.seed)
    fog, iot, centres = _world_for_seed(
        scenario.seed, scenario.n_fog, scenario.n_iot)
    if isinstance(adapter, ReLIEFPolicyAdapter):
        adapter.prepare(fog, iot, centres)
    engine = ExecutionEngine(
        fog, rng=streams.rng("engine"), task_power_w=config.P_PROC,
        control_power_w=config.P_PROC)
    control_cycles = lambda algorithm: controller_cycles(
        controller_profile, algorithm, n_fog=len(fog))
    engine.schedule_measurement_boundaries(
        scenario.measurement_start_s, scenario.measurement_end_s)
    rate = (
        scenario.arrival_rate_override
        if scenario.arrival_rate_override is not None
        else arrival_rate_for_load(scenario.target_load, fog))
    tasks = generate_paired_tasks(streams, rate, scenario.measurement_end_s)
    fingerprint = _input_fingerprint(tasks, fog, iot, centres)
    mobility_rng = streams.rng("mobility")
    efficiency = computational_efficiency(fog)
    efficient_ids = sorted(efficiency, key=efficiency.get, reverse=True)

    nominal = Task(
        task_id=-1, size_kb=150.0,
        cycles=150.0 * 1024.0 * config.CYCLES_PER_BYTE,
        deadline_ms=config.deadline_for_size_ms(150.0), arrival_s=0.0,
        is_small=True)
    next_tick = 0.0
    tick = 0
    tick_records: List[Dict] = []
    task_context: Dict[int, Dict] = {}
    previous_head_id = -1
    pending_head_updates: List[Tuple[float, FogNode]] = []
    propulsion_energy_measurement_j = 0.0
    handover_communication_j = 0.0
    handover_count = 0
    failed_head_id: Optional[int] = None
    head_failure_at_s: Optional[float] = None
    head_recovery_at_s: Optional[float] = None
    control_jobs_lost = 0

    def activate_completed_heads(at_s: float) -> None:
        nonlocal handover_communication_j, handover_count, head_recovery_at_s
        if not isinstance(adapter, AttentionPPOPolicyAdapter):
            return
        ready = sorted(
            (item for item in pending_head_updates if item[0] <= at_s + 1e-12),
            key=lambda item: item[0])
        for _completed_s, node in ready:
            pending_head_updates.remove((_completed_s, node))
            if not node.available:
                continue
            previous = adapter.head
            adapter.pending_head = node
            adapter.activate_pending_head()
            if (failed_head_id is not None and head_recovery_at_s is None
                    and node.node_id != failed_head_id):
                head_recovery_at_s = at_s
            if (scenario.controller_placement != "head" or previous is None
                    or previous.node_id == node.node_id):
                continue
            peers = [peer for peer in fog if peer.node_id != node.node_id]
            mean_distance = float(np.mean([
                uav_uav_distance_m(node, peer) for peer in peers]))
            per_peer_s = (config.SIGNAL_KB * 1024.0 * 8.0
                          / data_rate_bps(mean_distance, config.P_UAV_TX_W))
            total_s = per_peer_s * len(peers)
            node.E_res_j = max(
                0.0, node.E_res_j - config.P_UAV_TX_W * total_s)
            for peer in peers:
                peer.E_res_j = max(
                    0.0, peer.E_res_j - config.P_COMM * per_peer_s)
            if scenario.measurement_start_s <= at_s < scenario.measurement_end_s:
                handover_communication_j += (
                    config.P_UAV_TX_W + config.P_COMM) * total_s
            handover_count += 1
            engine.submit_control(
                control_id=f"handover:{handover_count}",
                node_id=node.node_id,
                release_s=engine.now_s,
                cycles=(scenario.handover_setup_ms / 1000.0
                        * node.fr_avg_hz()),
                algorithm="handover",
                external=False,
            )

    def available_efficient_id() -> int:
        try:
            return next(node_id for node_id in efficient_ids
                        if fog[node_id].available)
        except StopIteration as exc:
            raise RuntimeError("no available fog controller") from exc

    def fog_controller_id() -> Optional[int]:
        fallback = available_efficient_id()
        if scenario.controller_placement == "equal_controller":
            return fallback
        if scenario.controller_placement == "head":
            return (adapter.head.node_id
                    if isinstance(adapter, AttentionPPOPolicyAdapter)
                    and adapter.head is not None
                    and adapter.head.available else fallback)
        if adapter.native_control_external:
            return None
        if (isinstance(adapter, AttentionPPOPolicyAdapter) and adapter.head
                and adapter.head.available):
            return adapter.head.node_id
        return fallback

    def maybe_induce_head_failure(at_s: float) -> None:
        nonlocal failed_head_id, head_failure_at_s, control_jobs_lost
        if (scenario.head_failure_s is None or failed_head_id is not None
                or at_s + 1e-12 < scenario.head_failure_s
                or not isinstance(adapter, AttentionPPOPolicyAdapter)
                or adapter.head is None or not adapter.head.available):
            return
        failed_head_id = adapter.head.node_id
        head_failure_at_s = at_s
        state = engine.nodes[failed_head_id]
        control_jobs_lost = (int(state.running_control_id is not None)
                             + len(state.control_queue))
        engine.set_node_available(
            failed_head_id, False, when_s=at_s,
            reason="induced_head_failure")

    def headless_tick_record(at_s: float) -> Dict:
        return {
            "t": at_s,
            "prop_power_w": float(np.mean([
                aero_power_w(node.speed_ms) for node in fog])),
            "soc": float(np.mean([node.soc for node in fog])),
            "head_id": -1, "head_changed": 0, "selector_ran": 0,
            "candidate_head_id": -1, "head_activation_s": math.nan,
            "kl_value": math.nan, "switch_block_reason": "",
            "head_soc": math.nan, "head_workload_cycles": math.nan,
            "head_iot_distance_m": math.nan,
            "head_fog_distance_m": math.nan,
            "head_availability": math.nan,
            "head_control_service_ms": math.nan, "selector_us": 0.0,
        }

    def stage1_algorithm(st1: Mapping) -> str:
        if adapter.head_mode == "spotis":
            return ("stage1_with_spotis" if st1["run_spotis"]
                    else "stage1_without_spotis")
        return f"stage1_{adapter.head_mode}"

    for task in tasks:
        while next_tick <= task.arrival_s + 1e-12:
            engine.run(next_tick)
            activate_completed_heads(next_tick)
            if tick > 0:
                _step_mobility_paired(
                    fog, centres, scenario.control_tick_s, mobility_rng)
                interval_start = next_tick - scenario.control_tick_s
                overlap = max(
                    0.0,
                    min(next_tick, scenario.measurement_end_s)
                    - max(interval_start, scenario.measurement_start_s))
                for node in fog:
                    before = node.E_res_j
                    drain_flight_energy(node, scenario.control_tick_s)
                    if overlap > 0:
                        propulsion_energy_measurement_j += (
                            before - node.E_res_j) * (
                                overlap / scenario.control_tick_s)
                    if node.E_res_j <= 0.0 and node.available:
                        engine.set_node_available(
                            node.node_id, False, when_s=next_tick,
                            reason="propulsion_energy_depletion")
            maybe_induce_head_failure(next_tick)
            if adapter.uses_stage1:
                controller_before = fog_controller_id()
                st1 = adapter.elect_head(fog, iot, nominal, tick)
                algorithm = stage1_algorithm(st1)
                stage1_id = f"stage1:{tick}"
                engine.submit_control(
                    control_id=stage1_id,
                    node_id=controller_before,
                    release_s=next_tick,
                    cycles=control_cycles(algorithm),
                    algorithm=algorithm,
                    external=False,
                )
                completed_s = engine.projected_control_completion_s(stage1_id)
                candidate_head = st1["head_node"]
                pending_head_updates.append((completed_s, candidate_head))
                head = adapter.head
                head_id = -1 if head is None else head.node_id
                tick_records.append({
                    "t": next_tick,
                    "prop_power_w": float(np.mean([
                        aero_power_w(node.speed_ms) for node in fog])),
                    "soc": float(np.mean([node.soc for node in fog])),
                    "head_id": head_id,
                    "head_changed": int(
                        previous_head_id >= 0 and head_id != previous_head_id),
                    "selector_ran": int(st1.get("run_spotis", False)),
                    "candidate_head_id": candidate_head.node_id,
                    "head_activation_s": completed_s,
                    "kl_value": st1.get("kl_value", math.nan),
                    "switch_block_reason": st1.get("switch_block_reason", ""),
                    "head_soc": math.nan if head is None else head.soc,
                    "head_workload_cycles": (
                        math.nan if head is None
                        else engine.remaining_workload(head_id)),
                    "head_iot_distance_m": (
                        math.nan if head is None else float(np.mean([
                            slant_distance_m(head, dev.x, dev.y) for dev in iot]))),
                    "head_fog_distance_m": (
                        math.nan if head is None else float(np.mean([
                            uav_uav_distance_m(head, node) for node in fog
                            if node.node_id != head.node_id]))),
                    "head_availability": (
                        math.nan if head is None else math.exp(
                            -(head.lambda_fail + head.mu_fail)
                            * config.HEAD_AVAILABILITY_HORIZON_S)),
                    "head_control_service_ms": (
                        math.nan if head is None else 1000.0
                        * engine.projected_control_service_s(
                            head_id, control_cycles(
                                adapter.controller_algorithm))),
                    "selector_us": 1.0e6 * control_cycles(algorithm)
                        / controller_profile["target"]["reference_frequency_hz"],
                    **({f"decision_power_{index}": float(value)
                        for index, value in enumerate(st1["decision_power"])}
                       if "decision_power" in st1 else {}),
                })
                previous_head_id = head_id
            else:
                tick_records.append(headless_tick_record(next_tick))
            tick += 1
            next_tick = tick * scenario.control_tick_s

        engine.run(task.arrival_s)
        activate_completed_heads(task.arrival_s)
        update_memory_occupancy(fog)
        me = memory_efficiency(fog)
        dev = iot[streams.integer(f"task:{task.task_id}:device", len(iot))]
        primary, backup_id = adapter.decide(
            task, dev, fog, task.arrival_s, me)
        ctrl_id = fog_controller_id()
        ctrl_cycles = control_cycles(adapter.controller_algorithm)
        external = (
            scenario.controller_placement == "native"
            and adapter.native_control_external)
        dispatch_id = f"dispatch:{task.task_id}"
        engine.submit_control(
            control_id=dispatch_id,
            node_id=ctrl_id,
            release_s=task.arrival_s,
            cycles=ctrl_cycles,
            algorithm=adapter.controller_algorithm,
            external=external,
        )
        if primary == _NO_NODE:
            task_context[task.task_id] = {
                "dev_id": dev.dev_id, "head_id": (
                    adapter.head.node_id
                    if isinstance(adapter, AttentionPPOPolicyAdapter)
                    and adapter.head is not None else -1),
                "primary_id": -1, "upload_ms": math.nan,
            }
            continue
        primary_node = fog[primary]
        primary_dist = slant_distance_m(
            primary_node, dev.x, dev.y)
        primary_rate = data_rate_bps(primary_dist, dev.p_tx_w)
        upload_s = transmission_delay_ms(
            task.size_kb, primary_rate) / 1000.0
        task_context[task.task_id] = {
            "dev_id": dev.dev_id,
            "head_id": (
                adapter.head.node_id
                if isinstance(adapter, AttentionPPOPolicyAdapter)
                and adapter.head is not None else -1),
            "primary_id": primary,
            "upload_ms": 1000.0 * upload_s,
            "prop_us": primary_dist / 3.0e8 * 1.0e6,
            "exec_ms": 1000.0 * task.cycles / primary_node.fr_avg_hz(),
            "edge_j": dev.p_tx_w * upload_s,
            "net_rx_j": config.P_COMM * upload_s,
        }
        decision_complete_s = engine.projected_control_completion_s(dispatch_id)
        decision_s = max(0.0, decision_complete_s - task.arrival_s)
        task_context[task.task_id]["decision_ms"] = 1000.0 * decision_s
        task_context[task.task_id]["selector_ms"] = (
            1000.0 * ctrl_cycles
            / (controller_profile["target"]["reference_frequency_hz"]
               if external or ctrl_id is None
               else engine.nodes[ctrl_id].frequency_hz))

        backup = None
        if backup_id != _NO_NODE and backup_id != primary:
            backup_node = fog[backup_id]
            backup_dist = slant_distance_m(
                backup_node, dev.x, dev.y)
            backup_rate = data_rate_bps(backup_dist, dev.p_tx_w)
            fresh_s = transmission_delay_ms(
                task.size_kb, backup_rate) / 1000.0
            forward_rate = data_rate_bps(
                uav_uav_distance_m(primary_node, backup_node),
                config.P_UAV_TX_W)
            forward_s = transmission_delay_ms(
                task.size_kb, forward_rate) / 1000.0
            backup = BackupPlan(
                node_id=backup_id,
                fresh_upload_s=fresh_s,
                forward_s=forward_s,
                fresh_link_failure_s=streams.exponential(
                    f"task:{task.task_id}:backup:fresh-link",
                    backup_node.mu_fail),
                forward_link_failure_s=streams.exponential(
                    f"task:{task.task_id}:backup:forward-link",
                    backup_node.mu_fail),
                compute_failure_service_s=streams.exponential(
                    f"task:{task.task_id}:node:{backup_id}:compute",
                    backup_node.lambda_fail),
                fresh_rx_power_w=config.P_COMM,
                forward_rx_power_w=config.P_COMM,
                forward_tx_node_id=primary,
                forward_tx_power_w=config.P_UAV_TX_W,
            )
        engine.submit_task(
            task_id=task.task_id,
            arrival_s=task.arrival_s,
            deadline_s=task.arrival_s + task.deadline_ms / 1000.0,
            cycles=task.cycles,
            size_kb=task.size_kb,
            primary_node_id=primary,
            primary_release_delay_s=decision_s,
            primary_upload_s=upload_s,
            primary_link_failure_s=streams.exponential(
                f"task:{task.task_id}:primary:link",
                primary_node.mu_fail),
            primary_compute_failure_service_s=streams.exponential(
                f"task:{task.task_id}:node:{primary}:compute",
                primary_node.lambda_fail),
            primary_rx_power_w=config.P_COMM,
            backup=backup,
        )
        if isinstance(adapter, ReLIEFPolicyAdapter):
            adapter.observe_assignment(fog)

    while next_tick <= scenario.measurement_end_s + 1e-12:
        engine.run(next_tick)
        activate_completed_heads(next_tick)
        if tick > 0:
            _step_mobility_paired(
                fog, centres, scenario.control_tick_s, mobility_rng)
            interval_start = next_tick - scenario.control_tick_s
            overlap = max(
                0.0,
                min(next_tick, scenario.measurement_end_s)
                - max(interval_start, scenario.measurement_start_s))
            for node in fog:
                before = node.E_res_j
                drain_flight_energy(node, scenario.control_tick_s)
                if overlap > 0:
                    propulsion_energy_measurement_j += (
                        before - node.E_res_j) * (
                            overlap / scenario.control_tick_s)
                if node.E_res_j <= 0.0 and node.available:
                    engine.set_node_available(
                        node.node_id, False, when_s=next_tick,
                        reason="propulsion_energy_depletion")
        maybe_induce_head_failure(next_tick)
        if adapter.uses_stage1:
            controller_before = fog_controller_id()
            st1 = adapter.elect_head(fog, iot, nominal, tick)
            algorithm = stage1_algorithm(st1)
            stage1_id = f"stage1:{tick}"
            engine.submit_control(
                control_id=stage1_id,
                node_id=controller_before,
                release_s=next_tick,
                cycles=control_cycles(algorithm),
                algorithm=algorithm, external=False)
            completed_s = engine.projected_control_completion_s(stage1_id)
            candidate_head = st1["head_node"]
            pending_head_updates.append((completed_s, candidate_head))
            head = adapter.head
            head_id = -1 if head is None else head.node_id
            tick_records.append({
                "t": next_tick,
                "prop_power_w": float(np.mean([
                    aero_power_w(node.speed_ms) for node in fog])),
                "soc": float(np.mean([node.soc for node in fog])),
                "head_id": head_id,
                "head_changed": int(
                    previous_head_id >= 0 and head_id != previous_head_id),
                "selector_ran": int(st1.get("run_spotis", False)),
                "candidate_head_id": candidate_head.node_id,
                "head_activation_s": completed_s,
                "kl_value": st1.get("kl_value", math.nan),
                "switch_block_reason": st1.get("switch_block_reason", ""),
                "head_soc": math.nan if head is None else head.soc,
                "head_workload_cycles": (
                    math.nan if head is None
                    else engine.remaining_workload(head_id)),
                "head_iot_distance_m": (
                    math.nan if head is None else float(np.mean([
                        slant_distance_m(head, dev.x, dev.y) for dev in iot]))),
                "head_fog_distance_m": (
                    math.nan if head is None else float(np.mean([
                        uav_uav_distance_m(head, node) for node in fog
                        if node.node_id != head.node_id]))),
                "head_availability": (
                    math.nan if head is None else math.exp(
                        -(head.lambda_fail + head.mu_fail)
                        * config.HEAD_AVAILABILITY_HORIZON_S)),
                "head_control_service_ms": (
                    math.nan if head is None else 1000.0
                    * engine.projected_control_service_s(
                        head_id, control_cycles(
                            adapter.controller_algorithm))),
                "selector_us": 1.0e6 * control_cycles(algorithm)
                    / controller_profile["target"]["reference_frequency_hz"],
                **({f"decision_power_{index}": float(value)
                    for index, value in enumerate(st1["decision_power"])}
                   if "decision_power" in st1 else {}),
            })
            previous_head_id = head_id
        else:
            tick_records.append(headless_tick_record(next_tick))
        tick += 1
        next_tick = tick * scenario.control_tick_s
    engine.run(scenario.measurement_end_s)
    activate_completed_heads(scenario.measurement_end_s)
    engine.drain_relevant(scenario.measurement_end_s)
    capacity = engine.capacity_result(
        scenario.measurement_start_s, scenario.measurement_end_s)

    measured_tasks = [
        task for task in tasks
        if scenario.measurement_start_s <= task.arrival_s
        < scenario.measurement_end_s
    ]
    task_records = []
    queue_delays = []
    successes = []
    for task in measured_tasks:
        outcome = engine.outcomes.get(task.task_id)
        if outcome is None:
            context = task_context.get(task.task_id, {})
            task_records.append({
                "task_id": task.task_id, "success": 0,
                "arrival_s": task.arrival_s,
                "drop_reason": "no_feasible_primary",
                "queue_delay_ms": math.nan,
                "latency_ms": math.nan,
                "deadline_ms": task.deadline_ms,
                "is_small": int(task.is_small),
                "size_kb": task.size_kb,
                "cycles": task.cycles,
                **context,
            })
            successes.append(0)
            continue
        attempts = [engine.attempts[outcome.primary_attempt_id]]
        if outcome.backup_attempt_id:
            attempts.append(engine.attempts[outcome.backup_attempt_id])
        delays = [
            1000.0 * (a.started_s - a.ready_s)
            for a in attempts if a.started_s is not None
        ]
        queue_delay = min(delays) if delays else math.nan
        if delays:
            queue_delays.extend(delays)
        successes.append(int(outcome.success))
        task_records.append({
            "task_id": task.task_id,
            "arrival_s": task.arrival_s,
            "deadline_s": outcome.deadline_s,
            "success": int(outcome.success),
            "drop_reason": outcome.drop_reason,
            "winner_attempt_id": outcome.winner_attempt_id or "",
            "queue_delay_ms": queue_delay,
            "primary_attempt_id": outcome.primary_attempt_id,
            "backup_attempt_id": outcome.backup_attempt_id or "",
            "latency_ms": (
                1000.0 * (outcome.delivered_s - outcome.arrival_s)
                if outcome.delivered_s is not None else math.nan),
            "deadline_ms": task.deadline_ms,
            "is_small": int(task.is_small),
            "size_kb": task.size_kb,
            "cycles": task.cycles,
            **task_context.get(task.task_id, {}),
        })
    offered = sum(t.cycles for t in measured_tasks)
    provisioned = (
        scenario.measurement_s * sum(n.fr_avg_hz() for n in fog))
    realized = offered / provisioned
    task_compute_energy_j = sum(
        (values["productive_cycles"] + values["wasted_cycles"])
        / engine.nodes[node_id].frequency_hz * config.P_PROC
        for node_id, values in capacity.per_node.items())
    control_compute_energy_j = sum(
        values["control_cycles"] / engine.nodes[node_id].frequency_hz
        * config.P_PROC
        for node_id, values in capacity.per_node.items())
    communication_energy_j = 0.0
    for attempt in engine.attempts.values():
        if (attempt.transmission_start_s is None
                or attempt.transmission_end_s is None
                or attempt.transmission_end_s <= attempt.transmission_start_s):
            continue
        overlap = max(
            0.0,
            min(scenario.measurement_end_s, attempt.transmission_end_s)
            - max(scenario.measurement_start_s, attempt.transmission_start_s))
        communication_energy_j += overlap * (
            attempt.rx_power_w
            + (attempt.tx_power_w if attempt.tx_node_id is not None else 0.0))
    communication_energy_j += handover_communication_j
    energy_totals = {
        "propulsion_j": propulsion_energy_measurement_j,
        "task_compute_j": task_compute_energy_j,
        "control_compute_j": control_compute_energy_j,
        "uav_communication_j": communication_energy_j,
        "handover_communication_j": handover_communication_j,
        "uav_total_j": (
            propulsion_energy_measurement_j + task_compute_energy_j
            + control_compute_energy_j + communication_energy_j),
    }
    if (isinstance(adapter, AttentionPPOPolicyAdapter)
            and adapter.final_head_selection
            and adapter.head_mode == "spotis"):
        adapter.broker.assert_live_criteria()
    if head_failure_at_s is None:
        head_recovery_ticks = math.nan
        tasks_dropped_in_gap = 0
        delay_spike_ms = math.nan
    else:
        gap_end = (head_recovery_at_s if head_recovery_at_s is not None
                   else scenario.measurement_end_s)
        head_recovery_ticks = (
            max(1, math.ceil((gap_end - head_failure_at_s)
                             / scenario.control_tick_s - 1e-12))
            if head_recovery_at_s is not None else math.inf)
        tasks_dropped_in_gap = sum(
            not row["success"] for row in task_records
            if head_failure_at_s <= float(row.get("arrival_s", -math.inf))
            < gap_end)
        pre = [float(row["latency_ms"]) for row in task_records
               if head_failure_at_s - 2.0 <= float(
                   row.get("arrival_s", -math.inf)) < head_failure_at_s
               and math.isfinite(float(row["latency_ms"]))]
        post_bins: Dict[int, List[float]] = {}
        for row in task_records:
            arrival = float(row.get("arrival_s", -math.inf))
            latency = float(row["latency_ms"])
            if (head_failure_at_s <= arrival < head_failure_at_s + 2.0
                    and math.isfinite(latency)):
                index = int((arrival - head_failure_at_s)
                            / scenario.control_tick_s)
                post_bins.setdefault(index, []).append(latency)
        delay_spike_ms = (
            max(0.0, max(map(np.mean, post_bins.values())) - np.mean(pre))
            if pre and post_bins else math.nan)
    return ScenarioResult(
        config=scenario,
        capacity=capacity,
        target_load=scenario.target_load,
        realized_load=realized,
        arrival_rate=rate,
        success_rate=float(np.mean(successes)) if successes else 0.0,
        drop_rate=1.0 - float(np.mean(successes)) if successes else 1.0,
        mean_queue_delay_ms=(
            float(np.mean(queue_delays)) if queue_delays else math.nan),
        p95_queue_delay_ms=(
            float(np.percentile(queue_delays, 95)) if queue_delays else math.nan),
        primary_attempts=sum(
            a.kind.value == "primary" for a in engine.attempts.values()),
        backup_attempts=sum(
            a.kind.value == "backup" for a in engine.attempts.values()),
        external_control_cycles=dict(engine.external_control_cycles),
        energy_totals=energy_totals,
        task_records=task_records,
        audit=engine.audit_rows(),
        input_fingerprint=fingerprint,
        head_recovery_ticks=head_recovery_ticks,
        tasks_dropped_in_gap=tasks_dropped_in_gap,
        delay_spike_ms=delay_spike_ms,
        control_jobs_lost=control_jobs_lost,
        rollout=_build_rollout(adapter, task_records, engine),
        tick_records=tick_records,
    )


def _build_rollout(
    adapter: PolicyAdapter, task_records: Sequence[Mapping],
    engine: ExecutionEngine,
) -> List[Dict]:
    if not isinstance(adapter, AttentionPPOPolicyAdapter):
        return []
    by_task = {int(r["task_id"]): r for r in task_records}
    rollout = []
    for task_id in sorted(adapter.transitions):
        if task_id not in by_task:
            continue
        transition = dict(adapter.transitions[task_id])
        record = by_task[task_id]
        outcome = engine.outcomes.get(task_id)
        latency_ms = math.inf
        wasted_fraction = 0.0
        if outcome is not None:
            if outcome.delivered_s is not None:
                latency_ms = 1000.0 * (
                    outcome.delivered_s - outcome.arrival_s)
            attempts = [
                engine.attempts[outcome.primary_attempt_id]]
            if outcome.backup_attempt_id:
                attempts.append(engine.attempts[outcome.backup_attempt_id])
            total = sum(a.service_cycles for a in attempts)
            wasted = sum(a.service_cycles for a in attempts if not a.useful)
            wasted_fraction = wasted / max(total, 1.0)
        success = bool(record["success"])
        deadline_ms = (
            1000.0 * (outcome.deadline_s - outcome.arrival_s)
            if outcome is not None else config.DEADLINE_LARGE_MS)
        slack_score = float(np.clip(
            1.0 - latency_ms / max(deadline_ms, 1.0), -1.0, 1.0
        )) if math.isfinite(latency_ms) else -1.0
        queue_ms = float(record.get("queue_delay_ms", 0.0))
        if not math.isfinite(queue_ms):
            queue_ms = deadline_ms
        upload_ms = float(record.get("upload_ms", 0.0))
        if not math.isfinite(upload_ms):
            upload_ms = 0.0
        reward = (
            (config.REWARD_SUCCESS if success else config.REWARD_MISS)
            + config.REWARD_SLACK_W * slack_score
            - config.REWARD_QUEUE_EXTERNALITY_W
            * min(queue_ms / max(deadline_ms, 1.0), 1.0)
            - config.REWARD_ENERGY_W * wasted_fraction
            - config.REWARD_UPLOAD_W
            * min(upload_ms / max(deadline_ms, 1.0), 1.0)
        )
        transition.update({
            "reward": float(reward),
            "success": int(success),
            "latency_ms": latency_ms,
        })
        rollout.append(transition)
    return rollout
