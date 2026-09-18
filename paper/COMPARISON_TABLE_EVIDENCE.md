# Comparison Table — Evidence Log

`✓` = the paper explicitly models, optimizes, evaluates, or implements the item.
`✗` = no such evidence found in the paper.
Every non-fixed entry below was decided by reading the abstract, system model,
problem formulation, algorithms, and evaluation sections of the actual paper
(PDFs in this repository). Entries for **F2E** and the **Proposed Method** are
fixed by specification and were not re-derived.

## 1. Final comparison table

| Existing Methods | Delay | Energy | Cost | Reliability | UAV | Fog Selection | Distribution | Fault Tolerance |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FU-Serve | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Con-Fog | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| 2DP-FHS | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| 3D-POS | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| FODAS | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| ReLIEF | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| An efficient master head selection for multi-EEG to multi-fog IoT network using 6G-driven FaaS | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| F2E: fog-enabled EEG architecture for healthcare services and computing | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Delay-Energy Aware Dynamic Master Head Selection in Intelligent Healthcare IoT Networks | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Computation Offloading Optimization for UAV-Based Cloud-Edge Collaborative Task Scheduling Strategy | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| **Proposed Method** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

## 2. Sources inspected

| # | Method | Paper / venue | File in repo |
|---|---|---|---|
| 1 | FU-Serve | I. Raju, A. Roy, "FU-Serve: Fog-enabled UAV-as-a-Service for IoT Applications," IEEE GLOBECOM 2023, DOI 10.1109/GLOBECOM54140.2023.10437547 | `FU-Serve_Fog-Enabled_UAV-as-a-Service_for_IoT_Applications.pdf` |
| 2 | Con-Fog | R. Imandi, A. Roy, K. Sethi, P. K. B. N., M. Guizani, "Con-Fog: Consensus-Driven Fog Node Selection in FU-Serve Platform for IoT Applications," IEEE IoT-J, vol. 12, no. 13, 2025, DOI 10.1109/JIOT.2025.3557858 | `Con-Fog_Consensus-Driven_Fog_Node_Selection_in_FU-Serve_Platform_for_IoT_Applications.pdf` |
| 3 | 2DP-FHS | S. H. Kurra, R. K. Rath, S. R. Sreeja, "2DP-FHS: 2D Pareto Optimized Fog Head Selection for Multiple EEG Healthcare Data Analysis and Computations," ICACDS 2024, CCIS 2194, pp. 58–68, DOI 10.1007/978-3-031-70906-7_6 | `2DP-FHS 2D Pareto Optimized Fog Head.pdf` |
| 4 | 3D-POS | E. S. S. Kaushal, R. K. Rath, S. R. Sreeja, A. Hazra, "3D-POS: 3D Pareto Optimized Head Selection for Fog-enabled Smart EEG Healthcare IoT" | `3D-POS 3D Pareto Optimized Head Selection.pdf` |
| 5 | FODAS | G. Nagabushnam, Y. Choi, K. H. Kim, "FODAS: A Novel Reinforcement Learning Approach for Efficient Task Scheduling in Fog Computing Network," IEEE FMEC 2024, DOI 10.1109/FMEC62297.2024.10710250 | `FODAS_A_Novel_Reinforcement_Learning_Approach...pdf` |
| 6 | ReLIEF | R. Siyadatzadeh et al., "ReLIEF: A Reinforcement-Learning-Based Real-Time Task Assignment Strategy in Emerging Fault-Tolerant Fog Computing," IEEE IoT-J, vol. 10, no. 12, 2023, DOI 10.1109/JIOT.2023.3240007 | `ReLIEF_A_Reinforcement-Learning-Based...pdf` |
| 7 | 6G-driven FaaS master head selection (E2F) | R. Nanda, Sakthivel P., R. K. Rath, A. Hazra, "An efficient master head selection for multi-EEG to multi-fog IoT network using 6G-driven FaaS," Computer Communications 248 (2026) 108429, DOI 10.1016/j.comcom.2026.108429 | `Faas 6G Comp Comm.pdf` |
| 8 | F2E | — (entry fixed by specification; not re-verified) | — |
| 9 | Delay–Energy Aware Dynamic Master Head Selection | G. K. Sameera, A. Patel, S. R. Sreeja, R. K. Rath, "Delay–Energy Aware Dynamic Master Head Selection in Intelligent Healthcare IoT Networks," IEEE CICN 2026, DOI 10.1109/CICN70047.2026.11594286 | `Delay-Energy_Aware_Dynamic_Master_Head_Selection...pdf` |
| 10 | UAV-based cloud-edge offloading | H. Chen, H. Cui, J. Wang, P. Cao, Y. He, M. Guizani, "Computation Offloading Optimization for UAV-Based Cloud-Edge Collaborative Task Scheduling Strategy," IEEE TCCN, vol. 11, no. 6, 2025 | `Computation_Offloading_Optimization_for_UAV-Based...pdf` |

## 3. Proof for every cell

Every cell of every researched row. `✗` is backed by an exhaustive keyword sweep of
the extracted paper text (case-insensitive, references separated from body), so the
absence is measured, not assumed. Line numbers refer to the extracted text.

### FU-Serve (GLOBECOM 2023)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Abstract: "address the issues of data transmission delay"; aim to "minimize the service delay while reducing the data transmission delay"; Figs. 2–3 plot transmission time vs. number of UAVs for static and dynamic fog nodes. | §I, §VI |
| Energy | ✓ | Residual Energy Ratio `RER = (E_init − E_con)/E_init` (Eq. 5) built from hovering (Eqs. 6–7) and flight (Eq. 8) energy; feeds `CCF` (Eq. 11) and the constraint `RER ≥ RER_th` (Eq. 12); Fig. 4 = average residual energy per UAV. | §V, §VI |
| Cost | ✗ | 8 cost-family hits, all descriptive: rent and monetary profit in the business view, CAPEX/OPEX when summarising others. The maximised fitness is `D = PCF + CCF` (Eq. 12) — no cost term, no cost metric. | §I, §III, §V |
| Reliability | ✗ | The only "reliab" string in the paper is the title of reference [3]. No reliability, failure-probability or availability quantity. | whole paper |
| UAV | ✓ | Architecture defines F-UAV / H-UAV / N-UAV classes; UAVs act as the dynamic fog nodes. | §III |
| Fog Selection | ✓ | §V "Optimal Dynamic Fog Node Selection": Genetic Algorithm maximises `D` subject to Eq. 12. | §V |
| Distribution | ✗ | No assignment or scheduling variable exists. Task allocation appears only when summarising Pathak [2] and Yao [9]. Eq. 2 maps physical to virtual UAVs by sensor type (sensing), not compute placement. | §II, §IV |
| Fault Tolerance | ✗ | 0 body occurrences of fault / failure / backup / redundant / replication / recovery. Re-election is triggered by energy depletion, not failure. | whole paper; §III |

### Con-Fog (IEEE IoT-J 2025)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Proximity constraint (Eq. 6) minimises distance "to reduce transmission latency"; a theorem on latency reduction; "Transmission Latency" is a primary evaluation metric. | §III-C, §IV-D, §V |
| Energy | ✓ | Effective Residual Energy `E_eff` (Eq. 11) is a term of the utility `U_i,j` (Eq. 12); energy constraint (Eq. 5); "Energy Consumption" evaluated. | §III, §IV, §V |
| Cost | ✓ | "The **cost function** minimised is described by (1)": `arg min Z = ΣΣ c(u_i,f_j)·x_ij + Σ p(1−Σ x_ij) − Σ E_j^{f,res}·x_ij`, "where `c(u_i,f_j)` denotes the **communication cost**, `p` is the penalty for unassigned UAVs"; §III-A: "minimises communication costs". | §III-A, §III-B |
| Reliability | ✗ | 9 "reliab" hits: one in related work (l. 273), one motivating sentence for the utility (l. 600), a closing adjective (l. 1412), rest references. The link-quality index is SNR (Eqs. 9–10) — channel quality, not failure probability or availability. | §IV-A |
| UAV | ✓ | UAVs are both the clients and the dynamic fog nodes of the FU-Serve platform. | §I, §III-A |
| Fog Selection | ✓ | Consensus/social-choice selection: utility values → per-UAV ranking → consensus ranking → selected fog node (Algorithm 1). | §IV-A |
| Distribution | ✗ | Algorithm 2 bin-packs **UAVs**, not tasks: each UAV is assigned to one fog node under `Σ R_i^{u,req}·x_ij ≤ T·R_j^{f,cur}` (Eq. 4). No UAV's tasks are ever split or scheduled across fog nodes, so the capacity-aware spread follows from device association. | §III-C, §IV-B |
| Fault Tolerance | ✗ | 0 body occurrences of fault / failure / backup / redundant / replication / recovery / failover. | whole paper |

### 2DP-FHS (ICACDS 2024, CCIS 2194)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Fog Delay Index `FDI = (PD + PrD + AQD)/D_net` (Eqs. 1–8) is one of the two Pareto objectives; Fig. 5 is a delay analysis over EEG and fog device counts. | §2.2.1, §3 |
| Energy | ✗ | 4 energy hits, none in the model: a related-work clause (l. 39), a motivating constraint (l. 86), "Bluetooth Low Energy" as the link (l. 359), and future work — "have aimed to include more parameters considering energy and storage to select optimal [FH]" (l. 563). `FPI` = idle physical memory + computational efficiency only (Eq. 12). | §2.2.1, §4 |
| Cost | ✗ | 2 hits, both summarising other papers ([5], [6]); the third match is the surname "Costa" in the reference list. | §1 |
| Reliability | ✗ | 0 occurrences of reliab / availability / failure probability anywhere. | whole paper |
| UAV | ✗ | 0 occurrences of UAV / drone / aerial / flying. EEG devices and fog devices only. | whole paper |
| Fog Selection | ✓ | 2D Pareto fog-head selection: non-dominated solution space, reference and utopia points, trade-off function (Eq. 13); Table 4 lists the selected FH per weight pair. | §2.2.2, §3 |
| Distribution | ✗ | "distribute the tasks among fog devices" is stated as the FH's role (§2.2) and "collaborative and distributed processing" as a layer function, but no policy, algorithm, or experiment implements it; `V(T_k)` is an input to Eqs. 3 and 9, not a decision. | §2.2, §3 |
| Fault Tolerance | ✓ | "an additional fog device, called as Alternate Fog Head (AFH), is also chosen; which takes control of the FH **in case of failure**" (ll. 323–324) — a backup head with stated failover. | §2.2.2 |

### 3D-POS

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Device Proximity Factor (Eqs. 1–4) is minimised in the objective (Eq. 11); "delay minimized" is an explicit solution case; average total delay is the evaluation metric (Figs. 5, 8). | §II-A, §III, §IV |
| Energy | ✓ | Device Energy Efficiency `DEE = (DTE − (DOE + DCE))/DTE` (Eqs. 5–7) is maximised in Eq. 11; constraint `DTE_j ≥ e_min`; Fig. 7 reports average energy consumption. | §II-A, §III, §IV |
| Cost | ✗ | 3 hits: two motivating clauses about cloud cost, and the conclusion — "we would implement a **cost efficient model** for the same work" as future work (l. 971). | §I, §V |
| Reliability | ✗ | 1 body hit: the abstract adjective "reliable data handling". No reliability quantity in the model, objective, or evaluation. | abstract |
| UAV | ✗ | 0 occurrences of UAV / drone / aerial / flying. | whole paper |
| Fog Selection | ✓ | 3D Pareto master-head selection over (DPF, DEE, DCC): Eq. 11, dominance sets Eqs. 15–16, reference/ideal points, Steps 1–5. | §III |
| Distribution | ✗ | Conclusion: "In the future work, we aim to develop an efficient **task distribution mechanism** for this work". | §V |
| Fault Tolerance | ✓ | Constraint C2: "`f*` assumes to have a **backup device** `f*_b` within a maximum threshold distance `D_max`"; Scenarios 1–2 of Step 5 select that backup fog device. | §III-A, §III-B |

### FODAS (IEEE FMEC 2024)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Objective minimises makespan `Z = max_j Σ_i x_ij e_ij` (Eq. 1) under deadline constraint `Σ_j x_ij e_ij ≤ d_i` (Eq. 3); makespan and deadline-meeting rate are headline metrics (−57.3%, +18%). | §III-A, §V |
| Energy | ✓ | Per-task energy `c_ij` with total-energy constraint `ΣΣ x_ij c_ij ≤ TEC` (Eq. 4); energy savings up to 80% reported per node count and task load. | §III-A, §V |
| Cost | ✗ | 1 hit, in related work ("reducing electricity costs and task rejection rates", Li et al.). The paper's own objective is makespan; energy is accounted separately as energy. | §II, §III-A |
| Reliability | ✗ | 0 occurrences of reliab / availability / failure probability. Deadline-meeting rate is a timeliness metric. | whole paper |
| UAV | ✗ | 0 occurrences of UAV / drone / aerial. Fog–cloud nodes only. | whole paper |
| Fog Selection | ✓ | "Node Selection: Nodes with sufficient computational capacity … are sorted based on the estimated completion time. The node with the lowest estimated completion time is selected for task scheduling." | §IV-A |
| Distribution | ✓ | Binary assignment `x_ij` with `Σ_j x_ij = 1` (Eq. 2); the EDF + multi-agent PPO/RNN scheduler dispatches aperiodic tasks from the global queue to heterogeneous fog nodes. | §III-A, §IV |
| Fault Tolerance | ✗ | 0 body occurrences of fault / failure / backup / redundant / replication (the single match is a reference title). | whole paper |

### ReLIEF (IEEE IoT-J 2023)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | The system model section defines the delay model; the RL agent works "by establishing a balance between **communication delay** and workload on each fog device"; tasks are real-time with deadlines. | abstract, §III, §IV |
| Energy | ✗ | 15 body hits, none in the method: 1 motivation (l. 19), 8 in §II related work, 1 generic RL illustration ("an equation with a combination of delay and energy **can be used** as a reward function", l. 374), 2 qualitative discussion (ll. 1041, 1110). Decisive: §IV-B lists "temperature, **power consumption**, bandwidth, and workload distribution" as candidate state metrics, then states "The **reliability and workload distribution** of the system are used" — power is explicitly excluded. No energy term in reward Eq. 21, no energy metric. | §IV-B, §V |
| Cost | ✗ | All 6 hits are in §II related work (describing Ghanavati [20] and others). No cost term in the model, reward, or evaluation. | §II |
| Reliability | ✓ | Explicit reliability model: per-node computation reliability and link reliability, per-task reliability as a sum of disjoint events, total system reliability as a product, and the constraint that it stay above a level `R_l`; reliability improved ~72%. | §III-C, §IV-B, §V |
| UAV | ✗ | 0 body occurrences of UAV / drone / aerial (1 match is a reference title). | whole paper |
| Fog Selection | ✓ | "The dynamic selection of appropriate fog nodes for the execution of the primary and backup tasks"; Algorithm 2 has the broker send the primary task to the selected node. | §I, §IV-B |
| Distribution | ✓ | Real-time task assignment across fog nodes with workload distribution as a reward term (Eq. 21); workload balancing improved 83.3%. | §IV-B, §V |
| Fault Tolerance | ✓ | "A novel **primary backup** task assignment strategy"; backup copies are dispatched for fault-tolerant execution and Fig. 8 sweeps different **failure rates**. | abstract, §IV, §V |

### An efficient master head selection … using 6G-driven FaaS (Computer Communications 2026)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Fog Service Delay `D` (network + computation delay) is one of the four MCDM criteria in Eq. 21; propagation, service and total delay evaluated (Figs. 6–8); 83.87% total-delay reduction vs. KCHE. | §3, §5 |
| Energy | ✓ | Fog Residual Energy `R` (Eqs. 15–16: processing + transmit + receive + idle energy) is a selection criterion; total energy and energy-per-task evaluated (Fig. 9, 4.52 J/task). | §3, §5 |
| Cost | ✗ | 5 hits: cloud-maintenance cost in the motivation, one related-work clause, and the qualitative "Cost" row of Table 1 comparing architectures. The objective (Eq. 21) contains only D, R, M, P. | Table 1, §3.3 |
| Reliability | ✗ | 4 hits, all adjectives about 6G links ("highly reliable … wireless transmission", "ultra-reliable and low-latency", "reliable device connectivity") plus a reference title. No reliability quantity is modelled or measured. | §1, §2 |
| UAV | ✗ | 0 occurrences of UAV / drone / aerial. EEG devices, fog devices, cloud. | whole paper |
| Fog Selection | ✓ | Correlation-weighted multi-criteria outranking master-head selection over (D, R, M, P): Eq. 21 with Conditions 1–4, Algorithm 1, complexity Theorem 2. | §3.3, §4.1 |
| Distribution | ✓ | The MH "receives the EEG data … and **manages task distribution among fog devices**", and it is measured: "The MH performs intelligent allocation based on residual energy, delay, and memory, resulting in near-uniform resource utilization … variance drops to 14–18%" (from 52% without MH selection), with throughput 64 tasks/s vs. 28 tasks/s for random distribution. | §4, §5.3 |
| Fault Tolerance | ✓ | "We select the highest value of the fog device as the master head and **the second highest value as the alternate master head**" (§4.1); Table 1 attributes the architecture's fault tolerance to it: "High (**alternate master head** + multi-fog task offloading)". | §4.1, Table 1 |

### Delay–Energy Aware Dynamic Master Head Selection (IEEE CICN 2026)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | Four-component delay model — transmission, propagation, processing, queuing (Eqs. 1–6) — jointly minimised in Eq. 10; queuing and total delay evaluated (Fig. 4); up to 35% reduction. | §II-1, §III-A, §IV |
| Energy | ✓ | Fog energy model `E_total = E_TR + E_Proc + E_Idle` (Eqs. 7–9) jointly minimised in Eq. 10; total energy consumption evaluated (Fig. 5); 60–70% improvement. | §II-2, §III-A, §IV |
| Cost | ✗ | 0 occurrences of cost / price / monetary / expenditure anywhere in the body. | whole paper |
| Reliability | ✗ | 0 occurrences of reliab / availability / failure probability in the body. | whole paper |
| UAV | ✗ | 0 occurrences of UAV / drone / aerial. | whole paper |
| Fog Selection | ✓ | Pareto-based dynamic master head selection: dominance classification, reference point, minimum-distance selection, FCE tie-break (Algorithm 1, Steps 1–6). | §III-B |
| Distribution | ✗ | Eq. 10 evaluates each candidate head over the **whole** task set `T`; there is no assignment variable, and the conclusion defers "adaptive task scheduling mechanisms" to future work. | §III-A, §V |
| Fault Tolerance | ✗ | 1 body hit: "we include a **fallback mechanism** for scenarios where `F+ = ∅`" (l. 524) — a fallback for an empty non-dominated set, selecting on Fog Computational Efficiency. Not node or link failure handling. | §III-B |

### Computation Offloading Optimization for UAV-Based Cloud-Edge … (IEEE TCCN 2025)

| Column | Mark | Proof | Where |
|---|:--:|---|---|
| Delay | ✓ | The objective minimises the maximum processing delay across all UEs (divided by the fairness index), over local, UAV and cloud execution times. | §II-C, §IV |
| Energy | ✓ | Local computing energy, UAV flight/propulsion energy, and the constraint that "the total energy consumption of each UE must not exceed its" budget, with `E_v = 500 kJ` for the UAV. | §II-B, §II-C, §IV |
| Cost | ✗ | 6 hits, all motivation or related work — e.g. "Integrating MEC servers with UAVs … can significantly decrease **system cost**, including computation delay and energy consumption [9]" describes reference [9], not this model. The paper's own objective is delay ÷ fairness. | §I |
| Reliability | ✗ | 4 hits, all adjectives: "5G ultra-reliable and low-latency communication (URLLC)" ×2, a related-work list, and "reliable MEC services to the regions". No reliability quantity. | §I |
| UAV | ✓ | "a single UAV outfitted with a nano MEC server"; its 3D flight trajectory `U = {u(i)}` is an optimisation variable, with LoS channel and propulsion-energy models. | §II |
| Fog Selection | ✗ | The system is "K UEs, **a single UAV** outfitted with a nano MEC server, and a cloud server" — there is no candidate set of fog/edge nodes to choose from; the binary variable `α_k(i)` schedules **users**, and `R` splits a task between UAV and cloud. | §II |
| Distribution | ✓ | Joint optimisation of user scheduling `α`, trajectory `U` and **task offloading ratio** `R = {R^uav_k, R^cloud_k}` by DDPG, splitting each task across local / UAV-edge / cloud execution. | §II-C, §III |
| Fault Tolerance | ✗ | 0 occurrences of fault / failure / backup / redundant / replication / recovery anywhere in the paper. | whole paper |

## 4. Classifications flagged as genuinely ambiguous

These are the calls where a defensible argument exists for the opposite mark. Each
was resolved conservatively (✗ unless the paper explicitly does the thing), per the
classification rules.

1. **2DP-FHS — Distribution (marked ✗).** The paper *states* that the Fog Head's
   purpose is to "distribute the tasks among fog devices" and lists "collaborative
   and distributed processing of the tasks among fog devices" as a fog-layer
   function, and `V(T_k)` (volume assigned to the k-th fog device) appears in Eqs.
   3 and 9. However, no distribution policy, algorithm, or distribution experiment
   is given — the assigned volume is an input assumption, not a decision made by
   the method. Marked ✗ on that basis.
2. **Delay–Energy Aware MH Selection — Distribution (marked ✗).** The MH is
   described as "distributing workloads to neighbouring fog nodes", and the results
   attribute delay reduction to "dynamically distributing workloads among
   heterogeneous fog nodes". But Eq. 10 evaluates each candidate head over the
   *whole* task set `T`, there is no assignment variable, and the conclusion defers
   "adaptive task scheduling mechanisms" to future work. Marked ✗.
3. **Con-Fog — Distribution (marked ✗).** Reversed from an earlier ✓ on review.
   Algorithm 2 is a genuine capacity-constrained best-fit allocation across
   multiple fog nodes, and "number of unassigned UAVs" is an evaluated metric,
   which is more machinery than any other selection-only paper here has. It is
   marked ✗ because the unit allocated is a UAV, not a task: the definition asks
   for tasks/workloads/jobs spread among computing nodes.
4. **Con-Fog — Reliability (marked ✗).** The Effective Link Quality Index (Eq. 9),
   defined as SNR (Eq. 10), is used in the selection utility and is described as
   measuring "the robustness of the communication link", and the paper claims it
   enhances "overall network reliability". This is a channel-quality metric, not a
   reliability/failure/availability model, so it does not meet the stated criterion.
5. **FU-Serve — Cost (marked ✗).** FU-Serve introduces a business model with a new
   "fog owner" actor, rent, and monetary profit. Nothing monetary is quantified or
   optimized (the GA maximizes `D = PCF + CCF`), so it fails the "distinct
   cost-related quantity or objective" test.
6. **3D-POS — Fault Tolerance (marked ✓).** The backup device is explicit
   (constraint C2 and Step 5), which satisfies the "backup or redundant nodes"
   criterion, but — unlike 2DP-FHS — the paper never states that the backup takes
   over on failure. The ✓ rests on the designation of a backup node alone.
7. **Computation Offloading (UAV-based) — Fog Selection (marked ✗).** If the
   local/UAV/cloud offloading target choice is read as "server selection", this
   would flip to ✓. It is marked ✗ because there is only one UAV edge server, so
   no node is chosen from a candidate set; the binary variable `α_k(i)` schedules
   *users*, not nodes.
8. **FU-Serve — Fault Tolerance (marked ✗).** Re-running the fog-node election
   after the elected node's energy depletes is a re-election trigger, not failure
   handling; no backup node, replication, or recovery is described.

## 5. Not re-verified (fixed by specification)

* **F2E: fog-enabled EEG architecture for healthcare services and computing** —
  Delay ✓, Energy ✓, Cost ✓, Reliability ✗, UAV ✗, Fog Selection ✓,
  Distribution ✗, Fault Tolerance ✗.
* **Proposed Method** — all eight columns ✓.
