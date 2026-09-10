"""Paper-derived fog-head selectors for the unified Stage-1 testbed.

The selectors in this module reuse the simulator's node state, queues, energy
accounting, and 3-D geometry.  They intentionally contain no simulation loop so
the same implementation can be unit-tested and used by ``experiments.run_exp``.
All ties are resolved by the lowest node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import (GRID_M, P_COMM, P_PROC, SOC_PRIMARY_MIN,
                    SPEED_OF_LIGHT)
from models import FogNode, IoTDevice
from physics import data_rate_bps, queue_delay_ms, uav_uav_distance_m

_EPS = 1e-12


def _available(fog_nodes: List[FogNode]) -> List[FogNode]:
    candidates = [node for node in fog_nodes if node.available]
    if not candidates:
        raise ValueError("no available fog-head candidates")
    return candidates


def _pending(node: FogNode) -> Tuple[float, float]:
    """Return pending payload bytes and CPU cycles across both node queues."""
    entries = node.queue_backup + node.queue_primary
    return (sum(t.size_kb * 1024.0 for t, _ in entries),
            sum(t.cycles for t, _ in entries))


def _iot_distance(node: FogNode, dev: IoTDevice) -> float:
    return max(math.sqrt((node.x - dev.x) ** 2 + (node.y - dev.y) ** 2 + node.z ** 2), 1.0)


def _dominates(a: Sequence[float], b: Sequence[float], beneficial: Sequence[bool]) -> bool:
    no_worse = all(x >= y if benefit else x <= y
                   for x, y, benefit in zip(a, b, beneficial))
    strict = any(x > y if benefit else x < y
                 for x, y, benefit in zip(a, b, beneficial))
    return no_worse and strict


def pareto_indices(values: Sequence[Sequence[float]], beneficial: Sequence[bool]) -> List[int]:
    """Indices not dominated by another row, retaining input order."""
    return [i for i, row in enumerate(values)
            if not any(j != i and _dominates(other, row, beneficial)
                       for j, other in enumerate(values))]


def _sum_norm(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if abs(total) <= _EPS:
        return np.full(len(arr), 1.0 / max(len(arr), 1))
    return arr / total


def _result(head: FogNode, ranking: List[FogNode], scores: Dict[int, float],
            backup: Optional[FogNode] = None, **extra) -> Dict:
    return {"head_id": head.node_id, "head_node": head, "ranking": ranking,
            "backup_id": None if backup is None else backup.node_id,
            "backup_node": backup, "scores": scores, "run_spotis": False,
            "head_changed": False, "candidate_head_id": head.node_id,
            "switch_block_reason": "", **extra}


@dataclass
class FUServeSelector:
    """FU-Serve fitness maximization (PCF + CCF), solved by exact argmax."""

    sensing_range_m: float = 1000.0
    rer_threshold: float = SOC_PRIMARY_MIN
    service_count: Dict[int, int] = field(default_factory=dict)
    tenure_loads: Dict[int, List[float]] = field(default_factory=dict)
    _current_head_id: Optional[int] = None
    _current_load_samples: List[float] = field(default_factory=list)

    def _close_tenure(self) -> None:
        if self._current_head_id is not None and self._current_load_samples:
            self.tenure_loads.setdefault(self._current_head_id, []).append(
                float(np.mean(self._current_load_samples)))
        self._current_load_samples = []

    def _history(self, node_id: int) -> Tuple[float, float]:
        loads = self.tenure_loads.get(node_id, [])
        return float(self.service_count.get(node_id, 0)), float(np.mean(loads)) if loads else 0.0

    def select_head(self, fog_nodes: List[FogNode], per_node_metrics: Dict[int, Dict],
                    episode: int = 0, tick: int = 0, **_: object) -> Dict:
        del per_node_metrics, episode, tick
        fog_nodes = _available(fog_nodes)
        rows: Dict[int, Dict[str, float]] = {}
        for node in fog_nodes:
            distances = [uav_uav_distance_m(node, other) for other in fog_nodes
                         if other.node_id != node.node_id
                         and uav_uav_distance_m(node, other) <= self.sensing_range_m]
            dc = len(distances)
            ad = float(np.mean(distances)) if distances else math.inf
            rer = node.soc
            service, avg_load = self._history(node.node_id)
            pcf = service + avg_load
            # PCF is dimensionless, so adapt the paper's geometry term with a
            # dimensionless distance too. Using AD directly in metres made CCF
            # about 1e-2 in this 5-km environment; the first service increment
            # (=1) then permanently locked in the initial winner.
            ad_normalized = ad / self.sensing_range_m if math.isfinite(ad) else math.inf
            ccf = rer * dc / ad_normalized if dc and ad_normalized > 0.0 else 0.0
            rows[node.node_id] = {"rer": rer, "dc": float(dc), "ad": ad,
                                  "ad_normalized": ad_normalized,
                                  "service_index": service, "avg_load": avg_load,
                                  "fitness": pcf + ccf}

        eligible = [n for n in fog_nodes if rows[n.node_id]["rer"] >= self.rer_threshold
                    and rows[n.node_id]["dc"] >= 1 and rows[n.node_id]["ad"] >= 1]
        if eligible:
            ranking = sorted(eligible, key=lambda n: (-rows[n.node_id]["fitness"], n.node_id))
            ranking += sorted((n for n in fog_nodes if n not in eligible),
                              key=lambda n: (-n.soc, n.node_id))
        else:
            ranking = sorted(fog_nodes, key=lambda n: (-n.soc, n.node_id))
        head = ranking[0]

        if head.node_id != self._current_head_id:
            self._close_tenure()
            self._current_head_id = head.node_id
            self.service_count[head.node_id] = self.service_count.get(head.node_id, 0) + 1
        self._current_load_samples.append(rows[head.node_id]["dc"])
        return _result(head, ranking,
                       {i: row["fitness"] for i, row in rows.items()},
                       ranking[1] if len(ranking) > 1 else None,
                       criteria=rows)


@dataclass
class TwoDPFHSSelector:
    """2DP-FHS: delay/performance indices, Pareto set, and utopia ranking."""

    alpha: float = 0.5
    beta: float = 0.5

    def tradeoff_order(self, fog_nodes: List[FogNode], fdi: Sequence[float],
                       fpi: Sequence[float]) -> List[FogNode]:
        """Eq. 13 with benefit direction corrected: minimize alpha*FDI-beta*FPI."""
        score = self.alpha * _sum_norm(fdi) - self.beta * _sum_norm(fpi)
        order = sorted(range(len(fog_nodes)), key=lambda i: (score[i], fog_nodes[i].node_id))
        return [fog_nodes[i] for i in order]

    def rank_objectives(self, fog_nodes: List[FogNode], fdi: Sequence[float],
                        fpi: Sequence[float]) -> Tuple[List[FogNode], List[int]]:
        nd = pareto_indices(list(zip(fdi, fpi)), (False, True))
        # The paper calls Table 3 an "absent NSS" case because none of its
        # alternatives dominates another.  Preserve that explicit workflow
        # condition even though, under the conventional definition, every row
        # would be non-dominated.
        has_dominance = any(_dominates(a, b, (False, True))
                            for i, a in enumerate(zip(fdi, fpi))
                            for j, b in enumerate(zip(fdi, fpi)) if i != j)
        if not has_dominance:
            return self.tradeoff_order(fog_nodes, fdi, fpi), []
        nfdi, nfpi = _sum_norm(fdi), _sum_norm(fpi)
        ideal = (min(nfdi[i] for i in nd), max(nfpi[i] for i in nd))
        distance = {i: math.hypot(nfdi[i] - ideal[0], nfpi[i] - ideal[1]) for i in nd}
        tradeoff = self.alpha * nfdi - self.beta * nfpi  # minimize
        front = sorted(nd, key=lambda i: (distance[i], fog_nodes[i].node_id))
        rest = sorted((i for i in range(len(fog_nodes)) if i not in nd),
                      key=lambda i: (tradeoff[i], fog_nodes[i].node_id))
        return [fog_nodes[i] for i in front + rest], nd

    def select_head(self, fog_nodes: List[FogNode], per_node_metrics: Dict[int, Dict],
                    episode: int = 0, tick: int = 0,
                    iot_devices: Optional[List[IoTDevice]] = None, **_: object) -> Dict:
        del per_node_metrics, episode, tick
        fog_nodes = _available(fog_nodes)
        iot_devices = iot_devices or []
        raw_delay, fpi, rows = [], [], {}
        for node in fog_nodes:
            avg_iot = float(np.mean([_iot_distance(node, d) for d in iot_devices])) if iot_devices else 0.0
            avg_fog = float(np.mean([uav_uav_distance_m(node, o) for o in fog_nodes
                                     if o.node_id != node.node_id])) if len(fog_nodes) > 1 else 0.0
            pd = (avg_iot + avg_fog) / SPEED_OF_LIGHT
            pending_bytes, _ = _pending(node)
            prd = pending_bytes / max(node.fr_avg_hz(), _EPS)
            aqd = queue_delay_ms(node) / 1000.0
            delay = pd + prd + aqd
            ram_bytes = node.MP_tot_gb * (1024.0 ** 3)
            ipm = (ram_bytes - pending_bytes) / max(ram_bytes, _EPS)
            ce = node.C_mips * 1e6 / max(node.fr_avg_hz(), _EPS)
            perf = 0.5 * ipm + 0.5 * ce
            raw_delay.append(delay); fpi.append(perf)
            rows[node.node_id] = {"pd": pd, "prd": prd, "aqd": aqd,
                                  "ipm": ipm, "ce": ce}
        d_net = max(sum(raw_delay), _EPS)
        fdi = [v / d_net for v in raw_delay]
        for node, d, p in zip(fog_nodes, fdi, fpi):
            rows[node.node_id].update(fdi=d, fpi=p)
        ranking, nd = self.rank_objectives(fog_nodes, fdi, fpi)
        scores = {n.node_id: self.alpha * _sum_norm(fdi)[i] - self.beta * _sum_norm(fpi)[i]
                  for i, n in enumerate(fog_nodes)}
        return _result(ranking[0], ranking, scores,
                       ranking[1] if len(ranking) > 1 else None,
                       pareto_ids=[fog_nodes[i].node_id for i in nd], criteria=rows)


@dataclass
class ThreeDPOSSelector:
    """3D-POS: proximity, energy efficiency, and compute-capacity Pareto selection."""

    dcc_alpha: float = 0.5
    dcc_beta: float = 0.5
    tau_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    max_backup_distance_m: float = math.sqrt(2.0) * GRID_M + 200.0
    min_soc: float = SOC_PRIMARY_MIN
    cores_by_tier: Dict[int, int] = field(default_factory=lambda: {1: 2, 2: 1})

    def rank_objectives(self, fog_nodes: List[FogNode], dpf: Sequence[float],
                        dee: Sequence[float], dcc: Sequence[float]) -> Tuple[List[FogNode], List[int]]:
        nd = pareto_indices(list(zip(dpf, dee, dcc)), (False, True, True))
        ndpf, ndee, ndcc = _sum_norm(dpf), _sum_norm(dee), _sum_norm(dcc)
        ideal = (min(ndpf[i] for i in nd), max(ndee[i] for i in nd), max(ndcc[i] for i in nd))
        distance = {i: math.sqrt((ndpf[i] - ideal[0]) ** 2 +
                                 (ndee[i] - ideal[1]) ** 2 +
                                 (ndcc[i] - ideal[2]) ** 2) for i in nd}
        a, b, c = self.tau_weights
        tradeoff = a * ndpf - b * ndee - c * ndcc  # minimize
        front = sorted(nd, key=lambda i: (distance[i], fog_nodes[i].node_id))
        rest = sorted((i for i in range(len(fog_nodes)) if i not in nd),
                      key=lambda i: (tradeoff[i], fog_nodes[i].node_id))
        return [fog_nodes[i] for i in front + rest], nd

    def select_head(self, fog_nodes: List[FogNode], per_node_metrics: Dict[int, Dict],
                    episode: int = 0, tick: int = 0,
                    iot_devices: Optional[List[IoTDevice]] = None, **_: object) -> Dict:
        del per_node_metrics, episode, tick
        fog_nodes = _available(fog_nodes)
        iot_devices = iot_devices or []
        rows, candidates = {}, []
        for node in fog_nodes:
            iot_ds = [_iot_distance(node, d) for d in iot_devices]
            fog_ds = [uav_uav_distance_m(node, o) for o in fog_nodes if o.node_id != node.node_id]
            epf = float(np.mean(iot_ds)) if iot_ds else 0.0
            fpf = float(np.mean(fog_ds)) if fog_ds else 0.0
            cf = max(sum(iot_ds) + sum(fog_ds), _EPS)
            dpf = (epf + fpf) / cf
            pending_bytes, pending_cycles = _pending(node)
            mean_rate = float(np.mean([data_rate_bps(d, P_COMM) for d in iot_ds])) if iot_ds else 1.0
            doe = P_COMM * (pending_bytes * 8.0) / max(mean_rate, _EPS)
            dce = P_PROC * pending_cycles / max(node.fr_avg_hz(), _EPS)
            dee = (node.E_initial_j - (doe + dce)) / max(node.E_initial_j, _EPS)
            cores = self.cores_by_tier.get(node.tier, 1)
            cpi = cores * node.fr_avg_hz() / max(node.C_mips * 1e6, _EPS)
            muf = (node.MP_tot_gb - node.MP_occ_gb) / max(node.MP_tot_gb, _EPS)
            dcc = self.dcc_alpha * cpi + self.dcc_beta * muf
            rows[node.node_id] = {"epf": epf, "fpf": fpf, "cf": cf, "dpf": dpf,
                                  "doe": doe, "dce": dce, "dee": dee,
                                  "cpi": cpi, "muf": muf, "dcc": dcc}
            has_backup = any(o.node_id != node.node_id and
                             uav_uav_distance_m(node, o) <= self.max_backup_distance_m and
                             o.soc >= self.min_soc for o in fog_nodes)
            if node.soc >= self.min_soc and has_backup:
                candidates.append(node)
        if not candidates:  # documented safety fallback when C2/C3 exclude everybody
            candidates = sorted(fog_nodes, key=lambda n: (-n.soc, n.node_id))
        dpf = [rows[n.node_id]["dpf"] for n in candidates]
        dee = [rows[n.node_id]["dee"] for n in candidates]
        dcc = [rows[n.node_id]["dcc"] for n in candidates]
        ranking, nd = self.rank_objectives(candidates, dpf, dee, dcc)
        # C2: backup must be within D_max; choose the best valid runner-up.
        head = ranking[0]
        backup = next((n for n in ranking[1:]
                       if uav_uav_distance_m(head, n) <= self.max_backup_distance_m), None)
        a, b, c = self.tau_weights
        scores_arr = a * _sum_norm(dpf) - b * _sum_norm(dee) - c * _sum_norm(dcc)
        scores = {n.node_id: float(scores_arr[i]) for i, n in enumerate(candidates)}
        return _result(head, ranking, scores, backup,
                       pareto_ids=[candidates[i].node_id for i in nd], criteria=rows)


HEAD_POLICY_FACTORIES = {
    "fu_serve": FUServeSelector,
    "2dp_fhs": TwoDPFHSSelector,
    "3d_pos": ThreeDPOSSelector,
}


def make_head_selector(policy: str):
    try:
        return HEAD_POLICY_FACTORIES[policy]()
    except KeyError as exc:
        raise ValueError(f"unknown baseline head policy {policy!r}") from exc
