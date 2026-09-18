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

## 3. Evidence for each ✓ (and the notable ✗)

| Method | Parameter / capability | Evidence from the paper | Source |
|---|---|---|---|
| FU-Serve | Delay ✓ | Stated goal is to "minimize the service delay while reducing the data transmission delay"; Figs. 2–3 evaluate transmission time vs. number of UAVs for static and dynamic fog nodes. | Abstract, §I-B, §VI |
| FU-Serve | Energy ✓ | Residual Energy Ratio `RER = (E_init − E_con)/E_init` (Eq. 5) with hovering/flying/transmission energy (Eqs. 6–8) enters the Current Condition Factor `CCF` (Eq. 11) and the constraint `RER ≥ RER_th` (Eq. 12); Fig. 4 evaluates average residual energy. | §V, §VI |
| FU-Serve | UAV ✓ | The whole architecture is Fog-enabled UAV-as-a-Service; F-UAV / H-UAV / N-UAV classes, UAVs act as dynamic fog nodes. | §III |
| FU-Serve | Fog Selection ✓ | "Optimal Dynamic Fog Node Selection": fitness `D = PCF + CCF` maximized by a Genetic Algorithm (Eq. 12) to elect the fog UAV. | §V |
| FU-Serve | Cost ✗ | Rent / monetary profit appear only in the descriptive "business view"; no cost term exists in the objective (Eq. 12) or in the evaluation. | §III, §V |
| FU-Serve | Fault Tolerance ✗ | Re-election is triggered by *energy depletion*, not by node or link failure; no backup, replication, or failover mechanism. | §III (fog layer) |
| Con-Fog | Delay ✓ | Proximity constraint (Eq. 6) minimizes distance "to reduce transmission latency"; Theorem on latency reduction; "Transmission Latency" is a primary evaluation metric. | §III-C, §IV-D, §V |
| Con-Fog | Energy ✓ | Effective Residual Energy `E_eff` (Eq. 11) is a term of the utility `U_i,j` (Eq. 12); energy constraint (Eq. 5); "Energy Consumption" evaluated. | §III, §IV, §V |
| Con-Fog | Cost ✓ | Explicit cost objective `arg min Z = ΣΣ c(u_i,f_j)·x_ij + Σ p(1−Σx_ij) − Σ E_j^{f,res}·x_ij` (Eq. 1) where `c(u_i,f_j)` is the communication cost and `p` the unassigned-UAV penalty. | §III-B |
| Con-Fog | UAV ✓ | UAVs are the clients and the dynamic fog nodes of the FU-Serve platform. | §I, §III-A |
| Con-Fog | Fog Selection ✓ | Consensus/social-choice fog node selection: utility values → per-UAV ranking → consensus ranking → selected fog node (Algorithm 1). | §IV-A |
| Con-Fog | Distribution ✗ | What Algorithm 2 bin-packs is **UAVs**, not tasks: each UAV is assigned to one fog node under `Σ R_i^{u,req}·x_ij ≤ T·R_j^{f,cur}` (Eq. 4). No task, job, or workload of a UAV is ever split or scheduled across fog nodes, so the capacity-aware spread is a consequence of device association, not task distribution. | §III-C, §IV-B |
| Con-Fog | Reliability ✗ | The link-quality index is SNR (Eqs. 9–10); "network reliability" appears only as a qualitative remark. No failure probability / availability / success-probability model. | §IV-A |
| 2DP-FHS | Delay ✓ | Fog Delay Index `FDI = (PD + PrD + AQD)/D_net` (Eqs. 1–8) is one of the two selection objectives; Fig. 5 is a delay analysis over EEG/fog device counts. | §2.2.1, §3 |
| 2DP-FHS | Fog Selection ✓ | 2D Pareto optimization over (FDI, FPI) with non-dominated solution space, utopia point, and trade-off function (Eq. 13) selects the Fog Head. | §2.2.2 |
| 2DP-FHS | Fault Tolerance ✓ | "An additional fog device, called as Alternate Fog Head (AFH), is also chosen; which takes control of the FH **in case of failure**" — explicit backup head with failover. | §2.2.2 |
| 2DP-FHS | Energy ✗ | Energy is named as *future* work: "have aimed to include more parameters considering energy and storage to select optimal [FH]". The FPI uses only idle physical memory and computational efficiency (Eq. 12). | §4 |
| 3D-POS | Delay ✓ | Device Proximity Factor (Eqs. 1–4) is the minimized objective in Eq. 11; "delay minimized" solution case; average total delay is the evaluation metric (Figs. 5, 8). | §II-A, §III, §IV |
| 3D-POS | Energy ✓ | Device Energy Efficiency `DEE = (DTE − (DOE + DCE))/DTE` (Eqs. 5–7) is maximized in Eq. 11; constraint `DTE_j ≥ e_min`; Fig. 7 reports average energy consumption. | §II-A, §III, §IV |
| 3D-POS | Fog Selection ✓ | 3D Pareto master head selection over (DPF, DEE, DCC), Eqs. 11–19. | §III |
| 3D-POS | Fault Tolerance ✓ | Constraint C2 requires the elected head to have a **backup device** `f*_b` within `D_max`, and Scenarios 1–2 explicitly select that backup fog device. | §III-A, §III-B (Step 5) |
| 3D-POS | Cost ✗, Distribution ✗ | Conclusion: "In the future work, we aim to develop an efficient **task distribution mechanism** … Further, we would implement a **cost efficient model**." Both are explicitly outside this paper. | §V |
| FODAS | Delay ✓ | Objective minimizes makespan `Z = max_j Σ_i x_ij e_ij` (Eq. 1) with deadline constraint `Σ_j x_ij e_ij ≤ d_i` (Eq. 3); makespan and deadline-meeting rate evaluated. | §III-A, §V |
| FODAS | Energy ✓ | Per-task energy `c_ij` and total energy constraint `ΣΣ x_ij c_ij ≤ TEC` (Eq. 4); "energy savings" is a headline metric (up to 80%). | §III-A, §V |
| FODAS | Fog Selection ✓ | "Node Selection: Nodes with sufficient computational capacity … are sorted based on the estimated completion time … the node with the lowest estimated completion time is selected." | §IV-A |
| FODAS | Distribution ✓ | Deadline-adaptive multi-agent RL scheduler assigns aperiodic tasks from the global queue to heterogeneous fog nodes (`x_ij`, Eq. 2; Algorithms/FODAScheduler). | §III-A, §IV |
| ReLIEF | Delay ✓ | Communication-delay model in the system model; the RL agent "establish[es] a balance between communication delay and workload on each fog device"; tasks are real-time with deadlines. | Abstract, §III, §IV |
| ReLIEF | Reliability ✓ | Explicit reliability model: computation reliability of each fog node and link reliability (Eqs. ~10–12), per-task reliability as a sum of disjoint events, total system reliability as a product, with constraint "total reliability … should be more than a specific reliability level `R_l`"; reliability improved by ~72%. | §III-C, §IV-B, §V |
| ReLIEF | Fog Selection ✓ | Q-learning selects the fog nodes that host the primary and the backup copy: "the dynamic selection of appropriate fog nodes"; broker sends the primary task to the selected node (Algorithm 2). | §I, §IV-B |
| ReLIEF | Distribution ✓ | Real-time task *assignment* across fog nodes with workload-distribution/load-balancing as a reward term (Eq. 21); workload balancing improved 83.3%. | §IV-B, §V |
| ReLIEF | Fault Tolerance ✓ | "A novel **primary backup** task assignment strategy"; backup copies are dispatched for fault-tolerant execution and the evaluation sweeps different **failure rates** (Fig. 8). | Abstract, §IV, §V |
| ReLIEF | Energy ✗ | No energy model, no energy term in the reward, no energy metric; energy appears only as motivation and as a qualitative side remark ("energy consumption may increase rather than other methods"). | §V |
| 6G-driven FaaS (E2F) | Delay ✓ | Fog Service Delay `D` (network + computation delay) is one of the four MCDM criteria in Eq. 21; propagation/service/total delay evaluated (Figs. 6–8), 83.87% total-delay reduction vs. KCHE. | §3, §5 |
| 6G-driven FaaS (E2F) | Energy ✓ | Fog Residual Energy `R` (Eqs. 15–16, processing + transmit + receive + idle energy) is a selection criterion; total energy consumption and energy-per-task evaluated (Fig. 9). | §3, §5 |
| 6G-driven FaaS (E2F) | Fog Selection ✓ | Correlation-weighted multi-criteria (D, R, M, P) outranking-flow master head selection, Eq. 21 + Algorithm 1. | §3.3, §4.1 |
| 6G-driven FaaS (E2F) | Distribution ✓ | The MH "manages task distribution among fog devices"; measured: "The MH performs intelligent allocation based on residual energy, delay, and memory, resulting in near-uniform resource utilization … variance drops to 14–18%", and throughput of 64 tasks/s vs. 28 tasks/s for random distribution. | §4, §5.3 |
| 6G-driven FaaS (E2F) | Fault Tolerance ✓ | "We select the highest value of the fog device as the master head and the **second highest value as the alternate master head**"; Table 1 lists the architecture's fault tolerance as "High (alternate master head + multi-fog task offloading)". | §4.1, Table 1 |
| 6G-driven FaaS (E2F) | Cost ✗ | "Cost" appears only as a qualitative row of the architecture comparison (Table 1) and in motivation; the objective (Eq. 21) has no cost term. | Table 1, §3.3 |
| Delay–Energy Aware MH Selection | Delay ✓ | Four-component delay model — transmission, propagation, processing, queuing (Eqs. 1–6) — jointly minimized in Eq. 10; queuing/total delay evaluated (Fig. 4), up to 35% reduction. | §II-1, §III-A, §IV |
| Delay–Energy Aware MH Selection | Energy ✓ | Fog energy model `E_total = E_TR + E_Proc + E_Idle` (Eqs. 7–9) jointly minimized in Eq. 10; total energy consumption evaluated (Fig. 5), 60–70% improvement. | §II-2, §III-A, §IV |
| Delay–Energy Aware MH Selection | Fog Selection ✓ | Pareto-based dynamic master head selection (Algorithm 1: dominance classification, reference point, minimum-distance selection, FCE fallback). | §III-B |
| Delay–Energy Aware MH Selection | Fault Tolerance ✗ | The only "fallback" is an algorithmic fallback for an empty non-dominated set (FCE-based), not node/link failure handling; no backup head, replication, or recovery. | §III-B |
| UAV-based cloud-edge offloading | Delay ✓ | Objective minimizes the maximum processing delay across UEs (subject to fairness), over local/UAV/cloud execution times. | §II-C, §IV |
| UAV-based cloud-edge offloading | Energy ✓ | Local computing energy, UAV propulsion/flight energy, and per-UE energy constraints ("the total energy consumption of each UE must not exceed its [budget]", `E_v = 500 kJ` for the UAV) bound the optimization. | §II-B/C, §IV |
| UAV-based cloud-edge offloading | UAV ✓ | A UAV carrying a nano MEC server is the edge node; its 3D flight trajectory is an optimization variable. | §II |
| UAV-based cloud-edge offloading | Distribution ✓ | Joint optimization of user scheduling `α`, trajectory `U`, and **task offloading ratio** `R = {R^uav_k, R^cloud_k}` splits each task across local / UAV-edge / cloud execution (DDPG). | §II-C, §III |
| UAV-based cloud-edge offloading | Fog Selection ✗ | The system has a **single** UAV edge server plus one cloud server; the decisions choose *users* (scheduling) and an offload *ratio*, not a node from a candidate set of fog/edge nodes. | §II |

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
