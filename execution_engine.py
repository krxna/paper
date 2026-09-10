"""Deterministic event-driven CPU execution and capacity accounting.

This module is the single source of truth for executor CPU state.  It deliberately
contains no policy logic: training, evaluation, and baseline adapters submit the
same typed attempts and controller jobs to :class:`ExecutionEngine`.

Times are seconds, CPU demand is cycles, and deadlines are absolute seconds.
Task scheduling is non-preemptive EDF with backup-queue priority.  Controller
jobs may preempt a task and the exact remaining task cycles are preserved.
"""

from __future__ import annotations

import dataclasses
import enum
import heapq
import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


EXECUTION_MODEL_VERSION = "event-cpu-v2.0.0"
_EPS = 1e-10


class AttemptKind(str, enum.Enum):
    PRIMARY = "primary"
    BACKUP = "backup"


class AttemptStatus(str, enum.Enum):
    HELD = "held"
    TRANSMITTING = "transmitting"
    QUEUED = "queued"
    RUNNING = "running"
    PREEMPTED = "preempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    LINK_FAILED = "link_failed"
    EXPIRED = "expired"
    LATE = "late"
    SUPERSEDED = "superseded"


class EventType(str, enum.Enum):
    MEASUREMENT_START = "measurement_start"
    MEASUREMENT_END = "measurement_end"
    COMPUTE_FAILURE = "compute_failure"
    SERVICE_COMPLETION = "service_completion"
    CONTROL_ARRIVAL = "control_arrival"
    LINK_FAILURE = "link_failure"
    TASK_READY = "task_ready"
    DEADLINE = "deadline"
    BACKUP_RELEASE = "backup_release"


# At equal timestamps: a service failure wins over completion; completion at the
# exact deadline is on time; arrivals and readiness are processed before expiry.
EVENT_PRIORITY: Mapping[EventType, int] = {
    EventType.MEASUREMENT_START: 0,
    EventType.MEASUREMENT_END: 1,
    EventType.COMPUTE_FAILURE: 10,
    EventType.SERVICE_COMPLETION: 20,
    EventType.CONTROL_ARRIVAL: 30,
    EventType.LINK_FAILURE: 40,
    EventType.TASK_READY: 50,
    EventType.BACKUP_RELEASE: 55,
    EventType.DEADLINE: 60,
}


@dataclass
class BackupPlan:
    """A held delayed backup.  Held metadata consumes no payload memory."""

    node_id: int
    fresh_upload_s: float
    forward_s: float
    fresh_link_failure_s: Optional[float] = None
    forward_link_failure_s: Optional[float] = None
    compute_failure_service_s: Optional[float] = None
    fresh_rx_power_w: float = 0.0
    forward_rx_power_w: float = 0.0
    forward_tx_node_id: Optional[int] = None
    forward_tx_power_w: float = 0.0


@dataclass
class TaskAttempt:
    attempt_id: str
    task_id: int
    node_id: int
    kind: AttemptKind
    arrival_s: float
    ready_s: float
    deadline_s: float
    total_cycles: float
    remaining_cycles: float
    size_kb: float
    status: AttemptStatus = AttemptStatus.HELD
    started_s: Optional[float] = None
    completed_s: Optional[float] = None
    failure_after_cycles: Optional[float] = None
    service_cycles: float = 0.0
    useful: bool = False
    release_reason: str = ""
    generation: int = 0
    transmission_start_s: Optional[float] = None
    transmission_end_s: Optional[float] = None
    rx_power_w: float = 0.0
    tx_node_id: Optional[int] = None
    tx_power_w: float = 0.0
    transmission_energy_charged: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.FAILED,
            AttemptStatus.LINK_FAILED,
            AttemptStatus.EXPIRED,
            AttemptStatus.LATE,
            AttemptStatus.SUPERSEDED,
        }


@dataclass
class ControlJob:
    control_id: str
    node_id: int
    release_s: float
    total_cycles: float
    remaining_cycles: float
    algorithm: str
    external: bool = False
    started_s: Optional[float] = None
    completed_s: Optional[float] = None
    generation: int = 0


@dataclass
class ServiceSlice:
    node_id: int
    start_s: float
    end_s: float
    cycles: float
    category: str
    work_id: str
    task_id: Optional[int] = None
    attempt_kind: Optional[str] = None


@dataclass
class TaskOutcome:
    task_id: int
    arrival_s: float
    deadline_s: float
    primary_attempt_id: str
    backup_plan: Optional[BackupPlan] = None
    backup_attempt_id: Optional[str] = None
    winner_attempt_id: Optional[str] = None
    delivered_s: Optional[float] = None
    terminal: bool = False
    drop_reason: str = ""

    @property
    def success(self) -> bool:
        return self.winner_attempt_id is not None


@dataclass
class NodeCpuState:
    node_id: int
    frequency_hz: float
    ram_total_gb: float = math.inf
    storage_total_gb: float = math.inf
    available: bool = True
    primary_queue: List[Tuple[float, int, str]] = field(default_factory=list)
    backup_queue: List[Tuple[float, int, str]] = field(default_factory=list)
    control_queue: List[Tuple[int, str]] = field(default_factory=list)
    running_attempt_id: Optional[str] = None
    running_control_id: Optional[str] = None
    segment_start_s: Optional[float] = None
    last_update_s: float = 0.0
    backlog_open_s: Optional[float] = None
    backlog_intervals: List[Tuple[float, float]] = field(default_factory=list)
    task_energy_j: float = 0.0
    control_energy_j: float = 0.0
    communication_energy_j: float = 0.0


@dataclass
class CapacityResult:
    start_s: float
    end_s: float
    productive_cycles: float
    wasted_cycles: float
    control_cycles: float
    idle_cycles: float
    provisioned_cycles: float
    backlog_exposure: float
    per_node: Dict[int, Dict[str, float]]

    def utilization(self) -> Dict[str, float]:
        den = self.provisioned_cycles
        return {
            "productive": self.productive_cycles / den,
            "wasted": self.wasted_cycles / den,
            "control": self.control_cycles / den,
            "idle": self.idle_cycles / den,
            "total_busy": (
                self.productive_cycles + self.wasted_cycles + self.control_cycles
            ) / den,
            "backlog_exposure": self.backlog_exposure,
        }


class ExecutionEngine:
    """Event-driven, one-CPU-per-UAV service engine."""

    def __init__(
        self,
        nodes: Iterable[Any],
        *,
        rng: Optional[random.Random] = None,
        task_power_w: float = 10.0,
        control_power_w: float = 10.0,
    ) -> None:
        self.rng = rng or random.Random(0)
        self.task_power_w = float(task_power_w)
        self.control_power_w = float(control_power_w)
        self.now_s = 0.0
        self._seq = itertools.count()
        self._events: List[Tuple[float, int, int, EventType, Dict[str, Any]]] = []
        self.nodes: Dict[int, NodeCpuState] = {}
        self.node_objects: Dict[int, Any] = {}
        for raw in nodes:
            if isinstance(raw, NodeCpuState):
                state = raw
                obj = None
            else:
                state = NodeCpuState(
                    node_id=int(raw.node_id),
                    frequency_hz=float(raw.fr_avg_hz()),
                    ram_total_gb=float(getattr(raw, "MP_tot_gb", math.inf)),
                    storage_total_gb=float(getattr(raw, "MS_tot_gb", math.inf)),
                    available=bool(getattr(raw, "available", True)),
                )
                obj = raw
                raw.cpu_state = state
            if state.frequency_hz <= 0:
                raise ValueError("node frequency must be positive")
            state._engine = self
            self.nodes[state.node_id] = state
            if obj is not None:
                self.node_objects[state.node_id] = obj
        if not self.nodes:
            raise ValueError("at least one provisioned node is required")
        self.attempts: Dict[str, TaskAttempt] = {}
        self.controls: Dict[str, ControlJob] = {}
        self.outcomes: Dict[int, TaskOutcome] = {}
        self.service_slices: List[ServiceSlice] = []
        self.event_audit: List[Dict[str, Any]] = []
        self.external_control_cycles: Dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Public submission API
    # ------------------------------------------------------------------
    def submit_task(
        self,
        *,
        task_id: int,
        arrival_s: float,
        deadline_s: float,
        cycles: float,
        size_kb: float,
        primary_node_id: int,
        primary_release_delay_s: float = 0.0,
        primary_upload_s: float = 0.0,
        primary_link_failure_s: Optional[float] = None,
        primary_compute_failure_service_s: Optional[float] = None,
        primary_rx_power_w: float = 0.0,
        primary_tx_node_id: Optional[int] = None,
        primary_tx_power_w: float = 0.0,
        backup: Optional[BackupPlan] = None,
    ) -> TaskOutcome:
        if task_id in self.outcomes:
            raise ValueError(f"duplicate task_id {task_id}")
        if deadline_s < arrival_s:
            raise ValueError("deadline precedes arrival")
        if cycles <= 0 or size_kb < 0:
            raise ValueError("invalid task demand")
        attempt_id = f"task-{task_id}:primary"
        attempt = self._new_attempt(
            attempt_id=attempt_id,
            task_id=task_id,
            node_id=primary_node_id,
            kind=AttemptKind.PRIMARY,
            arrival_s=arrival_s,
            ready_s=(
                arrival_s + max(0.0, primary_release_delay_s)
                + max(0.0, primary_upload_s)),
            deadline_s=deadline_s,
            cycles=cycles,
            size_kb=size_kb,
            compute_failure_service_s=primary_compute_failure_service_s,
            transmission_start_s=(
                arrival_s + max(0.0, primary_release_delay_s)),
            rx_power_w=primary_rx_power_w,
            tx_node_id=primary_tx_node_id,
            tx_power_w=primary_tx_power_w,
        )
        outcome = TaskOutcome(
            task_id=task_id,
            arrival_s=arrival_s,
            deadline_s=deadline_s,
            primary_attempt_id=attempt_id,
            backup_plan=backup,
        )
        self.outcomes[task_id] = outcome
        self._schedule(EventType.DEADLINE, deadline_s, task_id=task_id)
        self._start_transmission(attempt, primary_link_failure_s)
        return outcome

    def submit_control(
        self,
        *,
        control_id: str,
        node_id: Optional[int],
        release_s: float,
        cycles: float,
        algorithm: str,
        external: bool = False,
    ) -> ControlJob:
        if control_id in self.controls:
            raise ValueError(f"duplicate control_id {control_id}")
        if cycles < 0:
            raise ValueError("control cycles cannot be negative")
        if external:
            job = ControlJob(
                control_id, -1 if node_id is None else node_id, release_s,
                cycles, cycles, algorithm, external=True,
            )
            job.started_s = release_s
            job.completed_s = release_s
            self.controls[control_id] = job
            self.external_control_cycles[algorithm] += cycles
            self._audit("external_control", release_s, control_id=control_id,
                        algorithm=algorithm, cycles=cycles)
            return job
        if node_id is None or node_id not in self.nodes:
            raise ValueError("fog control job requires a valid node_id")
        job = ControlJob(
            control_id, node_id, release_s, cycles, cycles, algorithm,
        )
        self.controls[control_id] = job
        if release_s <= self.now_s + _EPS:
            self._control_arrival(max(release_s, self.now_s), control_id)
        else:
            self._schedule(EventType.CONTROL_ARRIVAL, release_s, control_id=control_id)
        return job

    def projected_control_completion_s(self, control_id: str) -> float:
        """Exact completion projection for a submitted FIFO control job.

        Controller jobs have strict priority over task work. Later arrivals join
        the FIFO behind this job, so the current running/queued prefix fully
        determines its completion time.
        """
        job = self.controls[control_id]
        if job.external or job.completed_s is not None:
            return float(job.completed_s if job.completed_s is not None else job.release_s)
        state = self.nodes[job.node_id]
        cycles = 0.0
        if state.running_control_id:
            cycles += self.controls[state.running_control_id].remaining_cycles
            if state.running_control_id == control_id:
                return self.now_s + cycles / state.frequency_hz
        for _seq, queued_id in sorted(state.control_queue):
            cycles += self.controls[queued_id].remaining_cycles
            if queued_id == control_id:
                return self.now_s + cycles / state.frequency_hz
        raise RuntimeError(f"control job {control_id} is not runnable")

    def projected_control_service_s(self, node_id: int, cycles: float) -> float:
        """Seconds until a hypothetical FIFO control job would complete now."""
        if node_id not in self.nodes or cycles < 0:
            raise ValueError("invalid hypothetical control job")
        state = self.nodes[node_id]
        pending = max(0.0, float(cycles))
        if state.running_control_id:
            pending += self.controls[state.running_control_id].remaining_cycles
        pending += sum(
            self.controls[control_id].remaining_cycles
            for _seq, control_id in state.control_queue)
        return pending / state.frequency_hz

    def control_backlog_cycles(self, node_id: int) -> float:
        """Running plus queued controller work remaining on one node."""
        if node_id not in self.nodes:
            raise ValueError("invalid control-backlog node")
        state = self.nodes[node_id]
        pending = (self.controls[state.running_control_id].remaining_cycles
                   if state.running_control_id else 0.0)
        return pending + sum(
            self.controls[control_id].remaining_cycles
            for _seq, control_id in state.control_queue)

    def set_node_available(
        self, node_id: int, available: bool, *, when_s: Optional[float] = None,
        reason: str = "external_state",
    ) -> None:
        """Change executor availability while preserving consumed service."""
        when = self.now_s if when_s is None else float(when_s)
        self.run(when)
        state = self.nodes[node_id]
        if state.available == bool(available):
            return
        self._sync_backlog(state, when)
        state.available = bool(available)
        obj = self.node_objects.get(node_id)
        if obj is not None:
            obj.available = bool(available)
        self._audit("node_availability", when, node_id=node_id,
                    available=bool(available), reason=reason)
        if not available and state.running_attempt_id is not None:
            attempt = self.attempts[state.running_attempt_id]
            self._stop_segment(state, when)
            state.running_attempt_id = None
            attempt.status = AttemptStatus.FAILED
            attempt.completed_s = when
            if attempt.kind == AttemptKind.PRIMARY:
                self._schedule(EventType.BACKUP_RELEASE, when,
                               task_id=attempt.task_id,
                               reason="primary_energy_depletion")
            else:
                self._maybe_terminal(attempt.task_id, "backup_energy_depletion")
        self._sync_backlog(state, when)

    def schedule_measurement_boundaries(self, start_s: float, end_s: float) -> None:
        if not 0 <= start_s < end_s:
            raise ValueError("invalid measurement window")
        self._schedule(EventType.MEASUREMENT_START, start_s)
        self._schedule(EventType.MEASUREMENT_END, end_s)

    def run(self, until_s: Optional[float] = None) -> None:
        """Process all events, or all events no later than ``until_s``."""
        if until_s is not None and until_s < self.now_s - _EPS:
            raise ValueError("cannot run backwards")
        while self._events and (until_s is None or self._events[0][0] <= until_s + _EPS):
            when, _priority, _seq, kind, payload = heapq.heappop(self._events)
            self.now_s = max(self.now_s, when)
            self._handle(kind, when, payload)
        if until_s is not None:
            self.now_s = until_s

    def drain_relevant(self, measurement_end_s: float) -> None:
        """Drain only events needed to classify tasks arriving by the window end."""
        relevant = {
            task_id for task_id, out in self.outcomes.items()
            if out.arrival_s <= measurement_end_s + _EPS
        }
        while relevant and any(not self.outcomes[t].terminal for t in relevant):
            if not self._events:
                self._finalize_stranded()
                break
            when, _priority, _seq, kind, payload = heapq.heappop(self._events)
            self.now_s = max(self.now_s, when)
            self._handle(kind, when, payload)
        self._close_all_segments(self.now_s, restart=False)
        self._close_backlogs(self.now_s)

    # ------------------------------------------------------------------
    # Shared live-state helpers
    # ------------------------------------------------------------------
    def remaining_workload(self, node_id: int) -> float:
        state = self.nodes[node_id]
        total = 0.0
        if state.running_attempt_id:
            total += self.attempts[state.running_attempt_id].remaining_cycles
        for queue in (state.backup_queue, state.primary_queue):
            total += sum(
                self.attempts[attempt_id].remaining_cycles
                for _deadline, _seq, attempt_id in queue
                if self.attempts[attempt_id].status
                in {AttemptStatus.QUEUED, AttemptStatus.PREEMPTED}
            )
        return total

    def queue_depth(self, node_id: int) -> int:
        state = self.nodes[node_id]
        ids = set()
        if state.running_attempt_id:
            ids.add(state.running_attempt_id)
        for queue in (state.backup_queue, state.primary_queue):
            ids.update(
                attempt_id for _deadline, _seq, attempt_id in queue
                if not self.attempts[attempt_id].terminal
            )
        return len(ids)

    def memory_occupancy(self, node_id: int) -> Tuple[float, float]:
        from config import (
            MEM_RAM_FLOOR_GB,
            MEM_RAM_GB_PER_KB,
            MEM_STORAGE_FLOOR_GB,
            MEM_STORAGE_GB_PER_KB,
        )
        state = self.nodes[node_id]
        ids = set()
        if state.running_attempt_id:
            ids.add(state.running_attempt_id)
        for queue in (state.backup_queue, state.primary_queue):
            ids.update(
                attempt_id for _d, _s, attempt_id in queue
                if not self.attempts[attempt_id].terminal
            )
        ram = sum(
            max(MEM_RAM_FLOOR_GB,
                self.attempts[i].size_kb * MEM_RAM_GB_PER_KB)
            for i in ids
        )
        storage = sum(
            max(MEM_STORAGE_FLOOR_GB,
                self.attempts[i].size_kb * MEM_STORAGE_GB_PER_KB)
            for i in ids
        )
        return min(ram, state.ram_total_gb), min(storage, state.storage_total_gb)

    def projected_edf_delay_s(
        self, node_id: int, *, deadline_s: float, backup: bool = False
    ) -> float:
        state = self.nodes[node_id]
        cycles = 0.0
        if state.running_control_id:
            cycles += self.controls[state.running_control_id].remaining_cycles
        cycles += sum(
            self.controls[cid].remaining_cycles
            for _seq, cid in state.control_queue
        )
        if state.running_attempt_id:
            cycles += self.attempts[state.running_attempt_id].remaining_cycles
        # Backup tasks always precede primary tasks. Within a class, EDF tasks
        # with an earlier/equal deadline precede the projected new attempt.
        cycles += sum(
            self.attempts[aid].remaining_cycles
            for d, _seq, aid in state.backup_queue
            if d <= deadline_s + _EPS and not self.attempts[aid].terminal
        )
        if not backup:
            cycles += sum(
                self.attempts[aid].remaining_cycles
                for d, _seq, aid in state.primary_queue
                if d <= deadline_s + _EPS and not self.attempts[aid].terminal
            )
        return cycles / state.frequency_hz

    # ------------------------------------------------------------------
    # Accounting and audit
    # ------------------------------------------------------------------
    def capacity_result(self, start_s: float, end_s: float) -> CapacityResult:
        if not start_s < end_s:
            raise ValueError("invalid capacity window")
        self._close_all_segments(self.now_s, restart=True)
        per_node: Dict[int, Dict[str, float]] = {}
        totals = defaultdict(float)
        duration = end_s - start_s
        for node_id, state in self.nodes.items():
            cap = state.frequency_hz * duration
            values = {"productive_cycles": 0.0, "wasted_cycles": 0.0,
                      "control_cycles": 0.0}
            for slc in self.service_slices:
                if slc.node_id != node_id:
                    continue
                overlap = max(0.0, min(end_s, slc.end_s) - max(start_s, slc.start_s))
                if overlap <= 0 or slc.end_s <= slc.start_s:
                    continue
                cycles = slc.cycles * overlap / (slc.end_s - slc.start_s)
                if slc.category == "control":
                    values["control_cycles"] += cycles
                else:
                    attempt = self.attempts[slc.work_id]
                    key = "productive_cycles" if attempt.useful else "wasted_cycles"
                    values[key] += cycles
            busy = sum(values.values())
            if busy > cap and busy - cap <= max(1.0, cap) * 1e-9:
                busy = cap
            if busy > cap + max(1.0, cap) * 1e-9:
                raise AssertionError(
                    f"node {node_id} service exceeds provisioned capacity")
            values["idle_cycles"] = max(0.0, cap - busy)
            values["provisioned_cycles"] = cap
            values["busy_utilization"] = busy / cap
            values["productive_utilization"] = values["productive_cycles"] / cap
            values["wasted_utilization"] = values["wasted_cycles"] / cap
            values["control_utilization"] = values["control_cycles"] / cap
            backlog_s = sum(
                max(0.0, min(end_s, b) - max(start_s, a))
                for a, b in state.backlog_intervals
            )
            if state.backlog_open_s is not None:
                backlog_s += max(
                    0.0, min(end_s, self.now_s) - max(start_s, state.backlog_open_s)
                )
            values["backlog_time_s"] = backlog_s
            per_node[node_id] = values
            for key, value in values.items():
                if key.endswith("_cycles"):
                    totals[key] += value
        provisioned = totals["provisioned_cycles"]
        weighted_backlog = sum(
            self.nodes[n].frequency_hz * per_node[n]["backlog_time_s"]
            for n in self.nodes
        ) / provisioned
        result = CapacityResult(
            start_s, end_s,
            totals["productive_cycles"], totals["wasted_cycles"],
            totals["control_cycles"], totals["idle_cycles"],
            provisioned, weighted_backlog, per_node,
        )
        util = result.utilization()
        if not math.isclose(
            util["productive"] + util["wasted"] + util["control"] + util["idle"],
            1.0, rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise AssertionError("capacity-fate categories do not conserve")
        return result

    def audit_rows(self) -> List[Dict[str, Any]]:
        rows = list(self.event_audit)
        for slc in self.service_slices:
            row = dataclasses.asdict(slc)
            row["record_type"] = "service_slice"
            if slc.category == "task":
                row["fate"] = (
                    "productive" if self.attempts[slc.work_id].useful else "wasted"
                )
            else:
                row["fate"] = "control"
            rows.append(row)
        return sorted(rows, key=lambda r: (
            float(r.get("time_s", r.get("start_s", 0.0))),
            r.get("record_type", ""),
        ))

    # ------------------------------------------------------------------
    # Internal event machinery
    # ------------------------------------------------------------------
    def _new_attempt(
        self, *, attempt_id: str, task_id: int, node_id: int, kind: AttemptKind,
        arrival_s: float, ready_s: float, deadline_s: float, cycles: float,
        size_kb: float, compute_failure_service_s: Optional[float],
        transmission_start_s: Optional[float] = None,
        rx_power_w: float = 0.0,
        tx_node_id: Optional[int] = None,
        tx_power_w: float = 0.0,
    ) -> TaskAttempt:
        if node_id not in self.nodes:
            raise ValueError(f"unknown node {node_id}")
        if attempt_id in self.attempts:
            raise ValueError(f"duplicate attempt {attempt_id}")
        failure_cycles = None
        if compute_failure_service_s is not None:
            failure_cycles = max(0.0, compute_failure_service_s) * self.nodes[node_id].frequency_hz
        attempt = TaskAttempt(
            attempt_id, task_id, node_id, kind, arrival_s, ready_s,
            deadline_s, cycles, cycles, size_kb,
            failure_after_cycles=failure_cycles,
            transmission_start_s=transmission_start_s,
            rx_power_w=max(0.0, float(rx_power_w)),
            tx_node_id=tx_node_id,
            tx_power_w=max(0.0, float(tx_power_w)),
        )
        self.attempts[attempt_id] = attempt
        return attempt

    def _start_transmission(
        self, attempt: TaskAttempt, link_failure_s: Optional[float]
    ) -> None:
        attempt.status = AttemptStatus.TRANSMITTING
        transmission_start = max(
            self.now_s,
            (attempt.transmission_start_s
             if attempt.transmission_start_s is not None
             else attempt.arrival_s))
        duration = max(0.0, attempt.ready_s - transmission_start)
        if link_failure_s is not None and link_failure_s < duration - _EPS:
            self._schedule(
                EventType.LINK_FAILURE,
                transmission_start + max(0.0, link_failure_s),
                attempt_id=attempt.attempt_id,
            )
        else:
            self._schedule(
                EventType.TASK_READY, attempt.ready_s,
                attempt_id=attempt.attempt_id,
            )

    def _charge_transmission(self, attempt: TaskAttempt, when: float) -> None:
        if attempt.transmission_energy_charged:
            return
        start = (attempt.transmission_start_s
                 if attempt.transmission_start_s is not None
                 else attempt.arrival_s)
        duration = max(0.0, min(when, attempt.ready_s) - start)
        receiver = self.node_objects.get(attempt.node_id)
        receiver_energy = attempt.rx_power_w * duration
        if receiver is not None and hasattr(receiver, "E_res_j"):
            receiver.E_res_j = max(
                0.0, float(receiver.E_res_j) - receiver_energy)
        self.nodes[attempt.node_id].communication_energy_j += receiver_energy
        if attempt.tx_node_id is not None and attempt.tx_node_id in self.nodes:
            transmitter = self.node_objects.get(attempt.tx_node_id)
            transmitter_energy = attempt.tx_power_w * duration
            if transmitter is not None and hasattr(transmitter, "E_res_j"):
                transmitter.E_res_j = max(
                    0.0, float(transmitter.E_res_j) - transmitter_energy)
            self.nodes[attempt.tx_node_id].communication_energy_j += transmitter_energy
        attempt.transmission_energy_charged = True
        attempt.transmission_end_s = when
        self._audit(
            "transmission_energy", when, attempt_id=attempt.attempt_id,
            task_id=attempt.task_id, duration_s=duration,
            receiver_energy_j=receiver_energy,
            transmitter_node_id=attempt.tx_node_id,
            transmitter_energy_j=(
                attempt.tx_power_w * duration
                if attempt.tx_node_id is not None else 0.0),
        )

    def _schedule(self, kind: EventType, when: float, **payload: Any) -> None:
        heapq.heappush(
            self._events,
            (float(when), EVENT_PRIORITY[kind], next(self._seq), kind, payload),
        )

    def _handle(self, kind: EventType, when: float, payload: Dict[str, Any]) -> None:
        if kind in {EventType.MEASUREMENT_START, EventType.MEASUREMENT_END}:
            self._close_all_segments(when, restart=True)
            self._audit(kind.value, when)
        elif kind == EventType.CONTROL_ARRIVAL:
            self._control_arrival(when, payload["control_id"])
        elif kind == EventType.TASK_READY:
            self._task_ready(when, payload["attempt_id"])
        elif kind == EventType.LINK_FAILURE:
            self._link_failure(when, payload["attempt_id"])
        elif kind == EventType.SERVICE_COMPLETION:
            self._service_completion(
                when, payload["node_id"], payload["work_id"],
                payload["generation"], payload["is_control"],
            )
        elif kind == EventType.COMPUTE_FAILURE:
            self._compute_failure(
                when, payload["node_id"], payload["attempt_id"],
                payload["generation"],
            )
        elif kind == EventType.DEADLINE:
            self._deadline(when, payload["task_id"])
        elif kind == EventType.BACKUP_RELEASE:
            self._release_backup(when, payload["task_id"], payload["reason"])

    def _control_arrival(self, when: float, control_id: str) -> None:
        job = self.controls[control_id]
        state = self.nodes[job.node_id]
        if state.running_control_id is not None and state.segment_start_s is not None:
            running_id = state.running_control_id
            self._stop_segment(state, when)
            running = self.controls[running_id]
            state.segment_start_s = when
            running.generation += 1
            self._schedule(
                EventType.SERVICE_COMPLETION,
                when + running.remaining_cycles / state.frequency_hz,
                node_id=state.node_id, work_id=running_id,
                generation=running.generation, is_control=True,
            )
        self._sync_backlog(state, when)
        if state.running_attempt_id is not None:
            self._stop_segment(state, when)
            attempt = self.attempts[state.running_attempt_id]
            attempt.status = AttemptStatus.PREEMPTED
            attempt.generation += 1
            queue = state.backup_queue if attempt.kind == AttemptKind.BACKUP else state.primary_queue
            heapq.heappush(queue, (attempt.deadline_s, next(self._seq), attempt.attempt_id))
            state.running_attempt_id = None
        heapq.heappush(state.control_queue, (next(self._seq), control_id))
        self._audit("control_arrival", when, node_id=state.node_id,
                    control_id=control_id, cycles=job.total_cycles)
        self._dispatch(state, when)

    def _task_ready(self, when: float, attempt_id: str) -> None:
        attempt = self.attempts[attempt_id]
        if attempt.status != AttemptStatus.TRANSMITTING:
            return
        self._charge_transmission(attempt, when)
        if when > attempt.deadline_s + _EPS:
            attempt.status = AttemptStatus.EXPIRED
            self._audit("pre_execution_expiry", when, attempt_id=attempt_id,
                        task_id=attempt.task_id, node_id=attempt.node_id)
            self._maybe_terminal(attempt.task_id, "ready_after_deadline")
            return
        state = self.nodes[attempt.node_id]
        self._sync_backlog(state, when)
        attempt.status = AttemptStatus.QUEUED
        queue = state.backup_queue if attempt.kind == AttemptKind.BACKUP else state.primary_queue
        heapq.heappush(queue, (attempt.deadline_s, next(self._seq), attempt_id))
        self._audit("task_ready", when, attempt_id=attempt_id,
                    task_id=attempt.task_id, node_id=attempt.node_id,
                    attempt_kind=attempt.kind.value)
        self._dispatch(state, when)

    def _link_failure(self, when: float, attempt_id: str) -> None:
        attempt = self.attempts[attempt_id]
        if attempt.status != AttemptStatus.TRANSMITTING:
            return
        self._charge_transmission(attempt, when)
        attempt.status = AttemptStatus.LINK_FAILED
        self._audit("link_failure", when, attempt_id=attempt_id,
                    task_id=attempt.task_id, node_id=attempt.node_id)
        if attempt.kind == AttemptKind.PRIMARY:
            self._schedule(EventType.BACKUP_RELEASE, when,
                           task_id=attempt.task_id, reason="primary_link_failure")
        else:
            self._maybe_terminal(attempt.task_id, "backup_link_failure")

    def _service_completion(
        self, when: float, node_id: int, work_id: str,
        generation: int, is_control: bool,
    ) -> None:
        state = self.nodes[node_id]
        current = state.running_control_id if is_control else state.running_attempt_id
        if current != work_id:
            return
        work = self.controls[work_id] if is_control else self.attempts[work_id]
        if work.generation != generation:
            return
        self._sync_backlog(state, when)
        self._stop_segment(state, when)
        if is_control:
            work.remaining_cycles = 0.0
            work.completed_s = when
            state.running_control_id = None
            self._audit("control_completion", when, node_id=node_id,
                        control_id=work_id, algorithm=work.algorithm)
        else:
            attempt = work
            attempt.remaining_cycles = 0.0
            attempt.completed_s = when
            state.running_attempt_id = None
            outcome = self.outcomes[attempt.task_id]
            if when <= attempt.deadline_s + _EPS and outcome.winner_attempt_id is None:
                attempt.status = AttemptStatus.SUCCEEDED
                attempt.useful = True
                outcome.winner_attempt_id = attempt.attempt_id
                outcome.delivered_s = when
                outcome.terminal = True
                self._supersede_other_attempts(outcome, when)
            elif outcome.winner_attempt_id is not None:
                attempt.status = AttemptStatus.SUPERSEDED
            else:
                attempt.status = AttemptStatus.LATE
                outcome.terminal = True
                outcome.drop_reason = "late_completion"
            self._audit("task_completion", when, node_id=node_id,
                        attempt_id=attempt.attempt_id, task_id=attempt.task_id,
                        status=attempt.status.value, useful=attempt.useful)
        self._dispatch(state, when)

    def _compute_failure(
        self, when: float, node_id: int, attempt_id: str, generation: int
    ) -> None:
        state = self.nodes[node_id]
        attempt = self.attempts[attempt_id]
        if state.running_attempt_id != attempt_id or attempt.generation != generation:
            return
        self._sync_backlog(state, when)
        self._stop_segment(state, when)
        state.running_attempt_id = None
        attempt.status = AttemptStatus.FAILED
        attempt.completed_s = when
        self._audit("compute_failure", when, node_id=node_id,
                    attempt_id=attempt_id, task_id=attempt.task_id,
                    consumed_cycles=attempt.service_cycles)
        if attempt.kind == AttemptKind.PRIMARY:
            self._schedule(EventType.BACKUP_RELEASE, when,
                           task_id=attempt.task_id, reason="primary_compute_failure")
        else:
            self._maybe_terminal(attempt.task_id, "backup_compute_failure")
        self._dispatch(state, when)

    def _deadline(self, when: float, task_id: int) -> None:
        outcome = self.outcomes[task_id]
        for attempt in self._task_attempts(task_id):
            if attempt.status == AttemptStatus.TRANSMITTING:
                self._charge_transmission(attempt, when)
            if attempt.status in {
                AttemptStatus.HELD, AttemptStatus.TRANSMITTING,
                AttemptStatus.QUEUED,
            }:
                attempt.status = AttemptStatus.EXPIRED
                self._audit("deadline_expiry", when, attempt_id=attempt.attempt_id,
                            task_id=task_id, node_id=attempt.node_id,
                            consumed_cycles=attempt.service_cycles)
                self._dispatch(self.nodes[attempt.node_id], when)
            elif (attempt.status == AttemptStatus.PREEMPTED
                  and attempt.started_s is None):
                attempt.status = AttemptStatus.EXPIRED
                self._audit("deadline_expiry", when, attempt_id=attempt.attempt_id,
                            task_id=task_id, node_id=attempt.node_id,
                            consumed_cycles=attempt.service_cycles)
        active_started = any(
            a.status == AttemptStatus.RUNNING
            or (a.status == AttemptStatus.PREEMPTED and a.started_s is not None)
            for a in self._task_attempts(task_id))
        if outcome.winner_attempt_id is None and not active_started:
            outcome.terminal = True
            outcome.drop_reason = outcome.drop_reason or "deadline_expiry"
        self._audit("deadline", when, task_id=task_id,
                    terminal=outcome.terminal, success=outcome.success)

    def _release_backup(self, when: float, task_id: int, reason: str) -> None:
        outcome = self.outcomes[task_id]
        if outcome.terminal or outcome.backup_attempt_id is not None:
            return
        plan = outcome.backup_plan
        if plan is None or when >= outcome.deadline_s - _EPS:
            self._maybe_terminal(task_id, "no_usable_backup")
            return
        is_link = reason == "primary_link_failure"
        delay = plan.fresh_upload_s if is_link else plan.forward_s
        link_failure = (
            plan.fresh_link_failure_s if is_link else plan.forward_link_failure_s
        )
        attempt_id = f"task-{task_id}:backup"
        attempt = self._new_attempt(
            attempt_id=attempt_id,
            task_id=task_id,
            node_id=plan.node_id,
            kind=AttemptKind.BACKUP,
            arrival_s=when,
            ready_s=when + max(0.0, delay),
            deadline_s=outcome.deadline_s,
            cycles=self.attempts[outcome.primary_attempt_id].total_cycles,
            size_kb=self.attempts[outcome.primary_attempt_id].size_kb,
            compute_failure_service_s=plan.compute_failure_service_s,
            transmission_start_s=when,
            rx_power_w=(
                plan.fresh_rx_power_w if is_link else plan.forward_rx_power_w),
            tx_node_id=(None if is_link else plan.forward_tx_node_id),
            tx_power_w=(0.0 if is_link else plan.forward_tx_power_w),
        )
        attempt.release_reason = reason
        outcome.backup_attempt_id = attempt_id
        self._audit("backup_release", when, task_id=task_id,
                    attempt_id=attempt_id, node_id=plan.node_id, reason=reason)
        self._start_transmission(attempt, link_failure)

    def _dispatch(self, state: NodeCpuState, when: float) -> None:
        self._sync_backlog(state, when)
        if not state.available:
            self._sync_backlog(state, when)
            return
        if state.running_control_id or state.running_attempt_id:
            self._sync_backlog(state, when)
            return
        while state.control_queue:
            _seq, cid = heapq.heappop(state.control_queue)
            job = self.controls[cid]
            if job.remaining_cycles > _EPS:
                state.running_control_id = cid
                if job.started_s is None:
                    job.started_s = when
                state.segment_start_s = when
                job.generation += 1
                self._schedule(
                    EventType.SERVICE_COMPLETION,
                    when + job.remaining_cycles / state.frequency_hz,
                    node_id=state.node_id, work_id=cid,
                    generation=job.generation, is_control=True,
                )
                self._sync_backlog(state, when)
                return
        for queue in (state.backup_queue, state.primary_queue):
            while queue:
                _deadline, _seq, aid = heapq.heappop(queue)
                attempt = self.attempts[aid]
                if attempt.status not in {
                    AttemptStatus.QUEUED, AttemptStatus.PREEMPTED
                }:
                    continue
                if (when > attempt.deadline_s + _EPS
                        and attempt.started_s is None):
                    attempt.status = AttemptStatus.EXPIRED
                    self._maybe_terminal(attempt.task_id, "queue_expiry")
                    continue
                state.running_attempt_id = aid
                attempt.status = AttemptStatus.RUNNING
                if attempt.started_s is None:
                    attempt.started_s = when
                state.segment_start_s = when
                attempt.generation += 1
                completion_s = when + attempt.remaining_cycles / state.frequency_hz
                failure_s = math.inf
                if attempt.failure_after_cycles is not None:
                    failure_left = attempt.failure_after_cycles - attempt.service_cycles
                    failure_s = when + max(0.0, failure_left) / state.frequency_hz
                if failure_s <= completion_s + _EPS:
                    self._schedule(
                        EventType.COMPUTE_FAILURE, failure_s,
                        node_id=state.node_id, attempt_id=aid,
                        generation=attempt.generation,
                    )
                else:
                    self._schedule(
                        EventType.SERVICE_COMPLETION, completion_s,
                        node_id=state.node_id, work_id=aid,
                        generation=attempt.generation, is_control=False,
                    )
                self._audit("task_start" if attempt.service_cycles <= _EPS else "task_resume",
                            when, node_id=state.node_id, attempt_id=aid,
                            task_id=attempt.task_id,
                            remaining_cycles=attempt.remaining_cycles)
                self._sync_backlog(state, when)
                return
        self._sync_backlog(state, when)

    def _stop_segment(self, state: NodeCpuState, when: float) -> None:
        if state.segment_start_s is None:
            return
        elapsed = max(0.0, when - state.segment_start_s)
        cycles = min(
            state.frequency_hz * elapsed,
            (self.controls[state.running_control_id].remaining_cycles
             if state.running_control_id
             else self.attempts[state.running_attempt_id].remaining_cycles),
        )
        if state.running_control_id:
            job = self.controls[state.running_control_id]
            job.remaining_cycles = max(0.0, job.remaining_cycles - cycles)
            state.control_energy_j += self.control_power_w * (
                cycles / state.frequency_hz
            )
            self.service_slices.append(ServiceSlice(
                state.node_id, state.segment_start_s, when, cycles,
                "control", job.control_id,
            ))
        elif state.running_attempt_id:
            attempt = self.attempts[state.running_attempt_id]
            attempt.remaining_cycles = max(0.0, attempt.remaining_cycles - cycles)
            attempt.service_cycles += cycles
            state.task_energy_j += self.task_power_w * (
                cycles / state.frequency_hz
            )
            self.service_slices.append(ServiceSlice(
                state.node_id, state.segment_start_s, when, cycles,
                "task", attempt.attempt_id, attempt.task_id, attempt.kind.value,
            ))
        self._charge_node_energy(state.node_id, cycles, state.running_control_id is not None)
        state.segment_start_s = None

    def _charge_node_energy(self, node_id: int, cycles: float, control: bool) -> None:
        obj = self.node_objects.get(node_id)
        if obj is None or not hasattr(obj, "E_res_j"):
            return
        power = self.control_power_w if control else self.task_power_w
        energy = power * cycles / self.nodes[node_id].frequency_hz
        obj.E_res_j = max(0.0, float(obj.E_res_j) - energy)

    def _close_all_segments(self, when: float, *, restart: bool) -> None:
        for state in self.nodes.values():
            if state.segment_start_s is None:
                continue
            running_control = state.running_control_id
            running_attempt = state.running_attempt_id
            self._stop_segment(state, when)
            if restart:
                state.segment_start_s = when
                if running_control:
                    self.controls[running_control].generation += 1
                    job = self.controls[running_control]
                    self._schedule(
                        EventType.SERVICE_COMPLETION,
                        when + job.remaining_cycles / state.frequency_hz,
                        node_id=state.node_id, work_id=running_control,
                        generation=job.generation, is_control=True,
                    )
                elif running_attempt:
                    attempt = self.attempts[running_attempt]
                    attempt.generation += 1
                    completion = when + attempt.remaining_cycles / state.frequency_hz
                    failure = math.inf
                    if attempt.failure_after_cycles is not None:
                        failure = when + max(
                            0.0, attempt.failure_after_cycles - attempt.service_cycles
                        ) / state.frequency_hz
                    event = (EventType.COMPUTE_FAILURE
                             if failure <= completion + _EPS
                             else EventType.SERVICE_COMPLETION)
                    if event == EventType.COMPUTE_FAILURE:
                        self._schedule(event, failure, node_id=state.node_id,
                                       attempt_id=running_attempt,
                                       generation=attempt.generation)
                    else:
                        self._schedule(event, completion, node_id=state.node_id,
                                       work_id=running_attempt,
                                       generation=attempt.generation,
                                       is_control=False)

    def _has_waiting_task(self, state: NodeCpuState) -> bool:
        return any(
            not self.attempts[aid].terminal
            for queue in (state.backup_queue, state.primary_queue)
            for _d, _s, aid in queue
        )

    def _sync_backlog(self, state: NodeCpuState, when: float) -> None:
        exposed = (
            (state.running_attempt_id is not None or state.running_control_id is not None)
            and self._has_waiting_task(state)
        )
        if exposed and state.backlog_open_s is None:
            state.backlog_open_s = when
        elif not exposed and state.backlog_open_s is not None:
            state.backlog_intervals.append((state.backlog_open_s, when))
            state.backlog_open_s = None

    def _close_backlogs(self, when: float) -> None:
        for state in self.nodes.values():
            if state.backlog_open_s is not None:
                state.backlog_intervals.append((state.backlog_open_s, when))
                state.backlog_open_s = None

    def _supersede_other_attempts(self, outcome: TaskOutcome, when: float) -> None:
        for attempt in self._task_attempts(outcome.task_id):
            if attempt.attempt_id == outcome.winner_attempt_id or attempt.terminal:
                continue
            state = self.nodes[attempt.node_id]
            if state.running_attempt_id == attempt.attempt_id:
                self._stop_segment(state, when)
                state.running_attempt_id = None
            attempt.status = AttemptStatus.SUPERSEDED
            attempt.completed_s = when
            self._dispatch(state, when)

    def _task_attempts(self, task_id: int) -> List[TaskAttempt]:
        outcome = self.outcomes[task_id]
        ids = [outcome.primary_attempt_id]
        if outcome.backup_attempt_id:
            ids.append(outcome.backup_attempt_id)
        return [self.attempts[i] for i in ids]

    def _maybe_terminal(self, task_id: int, reason: str) -> None:
        outcome = self.outcomes[task_id]
        if outcome.success:
            outcome.terminal = True
            return
        primary = self.attempts[outcome.primary_attempt_id]
        backup_pending = (
            outcome.backup_plan is not None
            and outcome.backup_attempt_id is None
            and primary.kind == AttemptKind.PRIMARY
            and primary.status in {
                AttemptStatus.LINK_FAILED, AttemptStatus.FAILED
            }
        )
        if backup_pending:
            return
        attempts = self._task_attempts(task_id)
        if all(a.terminal for a in attempts):
            outcome.terminal = True
            outcome.drop_reason = outcome.drop_reason or reason

    def _finalize_stranded(self) -> None:
        for outcome in self.outcomes.values():
            if not outcome.terminal:
                outcome.terminal = True
                outcome.drop_reason = "stranded_without_events"

    def _audit(self, event: str, when: float, **payload: Any) -> None:
        self.event_audit.append({
            "record_type": "event",
            "event": event,
            "time_s": float(when),
            **payload,
        })


def remaining_workload_cycles(node: Any) -> float:
    """Return exact remaining task cycles, with legacy queue fallback."""
    state = getattr(node, "cpu_state", None)
    engine = getattr(state, "_engine", None) if state is not None else None
    if engine is not None:
        return engine.remaining_workload(node.node_id)
    total = 0.0
    for entry in list(node.queue_backup) + list(node.queue_primary):
        if isinstance(entry, TaskAttempt):
            total += entry.remaining_cycles
        else:
            total += entry[0].cycles
    return total


def exact_queue_depth(node: Any) -> int:
    state = getattr(node, "cpu_state", None)
    engine = getattr(state, "_engine", None) if state is not None else None
    if engine is not None:
        return engine.queue_depth(node.node_id)
    return len(node.queue_primary) + len(node.queue_backup)


def exact_memory_occupancy(node: Any) -> Optional[Tuple[float, float]]:
    state = getattr(node, "cpu_state", None)
    engine = getattr(state, "_engine", None) if state is not None else None
    if engine is None:
        return None
    return engine.memory_occupancy(node.node_id)
