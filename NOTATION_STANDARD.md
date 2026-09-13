# Mathematical Notation Provenance Audit and Redesign

## Iteration 2

### Scope

This specification concerns **paper-facing mathematical notation only**. Python identifiers, simulation logic, parameter values, and numerical results are unchanged.

The rotary-wing propulsion model is the sole deliberate source-notation exception. Its notation remains exactly that of Zeng et al. The original source gives the model as Eq. (6).

A further constraint is imposed throughout this revision: symbols belonging to the protected propulsion equation are not reused for unrelated concepts wherever that reuse can reasonably be avoided.

---

## A. Global notation architecture

The common indexing system is:

| SymbolMeaning |                            |
| ------------- | -------------------------- |
| `\mathcal F`  | set of UAV fog nodes       |
| `\mathcal I`  | set of IoT devices         |
| `\mathcal K`  | set of computational tasks |
| `\mathcal C`  | set of Stage-1 criteria    |
| `i,j`         | UAV/fog-node indices, `i,j\in\mathcal F` |
| `m`           | IoT-device index, `m\in\mathcal I` |
| `k`           | task index                 |
| `q`           | Stage-1 criterion index    |
| `n`           | control-loop index         |

The main physical quantities are:

| SymbolMeaning        |                                |
| -------------------- | ------------------------------ |
| `b_k`                | payload of task `k`, bits      |
| `c_k`                | CPU-cycle demand of task `k`   |
| `T_k^{\mathrm{dl}}`  | relative deadline of task `k`  |
| `f_i`                | CPU clock frequency of UAV `i` |
| `E_i^{\mathrm{bat}}` | current battery energy         |
| `E_i^0`              | initial battery energy         |
| `\mathbf x_i[n]`     | 3-D position of UAV `i`        |
| `\mathbf y_m`        | 3-D position of IoT device `m` |
| `r_{mi}`             | IoT-to-UAV range               |
| `r_{ij}`             | inter-UAV range                |

Task payload is therefore **not** denoted by `s`, distance is **not** denoted by `d`, critic value is **not** denoted by `V`, and advantage is **not** denoted by `A`. Those symbols are reserved because they occur in the protected propulsion model.

---

# B. Geometry and wireless model

Instead of FU-Serve's coordinate-expanded `d_{ij}` notation, geometry is represented vectorially:

```math
r_{mi}[n] = \left\| \mathbf x_i[n]-\mathbf y_m \right\|_2,
```

```math
r_{ij}[n] = \left\| \mathbf x_i[n]-\mathbf x_j[n] \right\|_2.
```

FU-Serve explicitly uses `d_{ij}` and coordinate-wise `x_i,y_i,z_i` notation for its individual-distance model. The vector-and-range formulation therefore removes both the main symbol and the source's indexed-coordinate presentation.

The channel-gain model becomes

```math
G(r)=G_0r^{-\beta_{\mathrm{pl}}},
```

instead of the ReLIEF-derived

```math
g(d)=\upsilon_1d^{-\upsilon_2}.
```

The repository explicitly states that the current implementation uses the same algebraic form as ReLIEF Eq. (3).

Uplink data rate is

```math
\mathcal R_{mi}^{\uparrow} = B \log_2 \left( 1+ \frac{ G(r_{mi})P_m^{\mathrm{tx}} }{ N_0 } \right).
```

Hence the former `TR` notation is removed.

Transmission time becomes

```math
T_{kmi}^{\uparrow} = \frac{b_k} {\mathcal R_{mi}^{\uparrow}},
```

and execution time is

```math
T_{ki}^{\mathrm{cpu}} = \frac{c_k}{f_i}.
```

The end-to-end delay can consequently be written

```math
T_{ki}^{\mathrm{e2e}} = T_k^{\mathrm{ctl}} + T_{kmi}^{\uparrow} + T_{ki}^{\mathrm{wait}} + T_{ki}^{\mathrm{cpu}}.
```

This removes the ReLIEF-style `D,T_u,T_Q,T_e` family currently still present in the implementation comments.

---

# C. Workload and reliability

Pending CPU demand at UAV `i` is

```math
C_i^{\mathrm{q}} = \sum_{k\in\mathcal Q_i}c_k.
```

Mean queued demand is

```math
\bar C = \frac{1}{|\mathcal F|} \sum_{i\in\mathcal F}C_i^{\mathrm{q}},
```

and swarm-level workload imbalance is

```math
\Delta_C = \sum_{i\in\mathcal F} \left| C_i^{\mathrm{q}}-\bar C \right|.
```

This replaces the ReLIEF-derived family `W_i,\bar W,WL`. The implementation currently still describes the workload metric using `W_j,\bar W,WL`.

Let

```math
\zeta_i^{\mathrm{cpu}}
```

and

```math
\zeta_i^{\mathrm{link}}
```

denote the compute and communication failure hazard rates of UAV `i`. Single-node completion probability becomes

```math
p_{ki} = \exp \left[ -\zeta_i^{\mathrm{cpu}} T_{ki}^{\mathrm{cpu}} -\zeta_i^{\mathrm{link}} T_{kmi}^{\uparrow} \right].
```

Primary-backup reliability becomes

```math
p_k^{\mathrm{pair}} = 1- \left( 1-p_{k,i_k^{(p)}} \right) \left( 1-p_{k,i_k^{(b)}} \right).
```

Episode-wide joint reliability is

```math
p^{\mathrm{all}} = \prod_{k\in\mathcal K} p_k^{\mathrm{pair}},
```

or, numerically,

```math
\log_{10}p^{\mathrm{all}} = \sum_{k\in\mathcal K} \log_{10} p_k^{\mathrm{pair}}.
```

Thus `R_0,R_i,R_{\mathrm{sys}}`, and the old `R`-family are eliminated.

---

# D. Hardware tiering

The computational-tier metric is paper-facing only as

```math
\eta_i^{\mathrm{tier}} = \frac{ f_i }{ \vartheta_i \sigma_i^{\mathrm{MIPS}} 10^6 },
```

where `\vartheta_i` is the node-specific computation coefficient and `\sigma_i^{\mathrm{MIPS}}` its MIPS rating.

This deliberately avoids the current source-derived

```math
CE_{\mathrm{eff}} = \frac{f} {\varsigma C_{\mathrm{mips}}10^6}.
```

The repository states that this is the reciprocal of a computation-cost expression inherited from an original paper.

---

# E. Stage-1 criterion vector

The final implementation actually contains six criteria, not five:

```math
[ E_{\mathrm{res}}, ME, R, D_{\mathrm{ctrl}}, WL, ECT ].
```

This is explicit in `broker.py`.

The revised paper-facing vector for UAV `i` is

```math
\mathbf z_i = \begin{bmatrix} E_i^{\mathrm{bat}}, & ME_i, & \hat p_i, & T_i^{\mathrm{ctrl}}, & C_i^{\mathrm{q}}, & \tau_i^{\mathrm{cov}} \end{bmatrix},
```

where:

```math
\hat p_i
```

is the candidate's representative reliability under the nominal Stage-1 task, and

```math
\tau_i^{\mathrm{cov}}
```

is its estimated energy-coverage time under coordinator duty.

This resolves the previous ambiguity between task-specific `p_{ki}` and the representative reliability used by Stage 1.

---

# F. MEREC notation

The previous redesign did not go far enough because MEREC uses a whole distinctive family:

```math
n_{ij},\quad S_i,\quad S'_{ij},\quad E_j,\quad w_j.
```

The original method uses a logarithmic aggregate `S_i`, a leave-one-criterion-out quantity `S'_{ij}`, removal effect `E_j`, and normalized criterion weight `w_j`.

The revised family is:

```math
\nu_{iq}
```

for the normalized value supplied to the removal-effect computation,

```math
\kappa_i = \ln \left[ 1+ \frac{1}{|\mathcal C|} \sum_{q\in\mathcal C} \left| \ln\nu_{iq} \right| \right],
```

and, with criterion `q` excluded,

```math
\kappa_{i\setminus q} = \ln \left[ 1+ \frac{1}{|\mathcal C|} \sum_{\substack{h\in\mathcal C\\h\ne q}} \left| \ln\nu_{ih} \right| \right].
```

The influence caused by removing criterion `q` is

```math
\Gamma_q = \sum_{i\in\mathcal F} \left| \kappa_{i\setminus q} - \kappa_i \right|,
```

and the resulting criterion weight is

```math
w_q = \frac{\Gamma_q} {\sum_{h\in\mathcal C}\Gamma_h}.
```

Retaining `w_q` is acceptable because `w` for a normalized weight is conventional mathematical notation. The source-specific family `n,S,S',E` has been eliminated.

---

# G. SPOTIS notation

The source SPOTIS family consists of

```math
S_{ij}, \quad S_j^{\min}, \quad S_j^{\max}, \quad S_j^\star, \quad d_{ij}, \quad \tilde d_{ij}, \quad d(A_i,S^\star).
```

The source formulas explicitly use these quantities.

The revised notation uses the Stage-1 entry `z_{iq}`, with fixed bounds

```math
\underline z_q, \qquad \overline z_q,
```

and criterion target

```math
z_q^\circ.
```

The implementation-specific clipping operation is represented explicitly:

```math
\delta_{iq} = \frac{ \left| \operatorname{clip} \left( z_{iq}; \underline z_q, \overline z_q \right) - z_q^\circ \right| }{ \overline z_q-\underline z_q }.
```

The broker score is

```math
\Phi_i = \sum_{q\in\mathcal C} w_q\delta_{iq},
```

and the selected head is

```math
i^\star = \arg\min_{i\in\mathcal F} \Phi_i.
```

This is an independent notation family. Neither the `S`-family nor the SPOTIS `d(A_i,S^\star)` construction remains.

---

# H. FU-Serve baseline

The previous notation

```math
F_i^{\mathrm{past}}, \qquad F_i^{\mathrm{now}}
```

was not sufficiently independent because FU-Serve explicitly constructs a Past Condition Factor and Current Condition Factor and adds them.

The revised notation does **not** create corresponding historical/current factor variables.

Instead, the complete baseline utility is written directly as

```math
\Psi_i^{\mathrm{FU}} = n_i^{\mathrm{svc}} + \bar\ell_i^{\mathrm{hist}} + \frac{ \left(E_i^{\mathrm{bat}}/E_i^0\right) n_i^{\mathrm{nbr}} }{ \bar r_i^{\mathrm{nbr}} / r_{\max}^{\mathrm{nbr}} }.
```

Here:

```math
n_i^{\mathrm{svc}}
```

is the number of previous fog-head services,

```math
\bar\ell_i^{\mathrm{hist}}
```

is historical mean service load,

```math
n_i^{\mathrm{nbr}}
```

is the number of eligible neighboring UAVs,

```math
\bar r_i^{\mathrm{nbr}}
```

is their average range, and

```math
r_{\max}^{\mathrm{nbr}}
```

is the sensing-range parameter.

Thus the source-specific families

```math
PCF,\ CCF,\ RER,\ DC_i,\ AD_i,\ R,\ \mathcal D
```

are all absent.

The normalized distance in the denominator is retained because it is a documented project adaptation in the implementation, not a notation inherited from FU-Serve.

---

# I. 2DP-FHS baseline

The paper implementation explicitly identifies the source scalarization as

```math
\alpha FDI-\beta FPI.
```

The revised delay quantity is denoted

```math
L_i^{(2)},
```

and the performance quantity is

```math
Q_i^{(2)}.
```

Let

```math
\tau_i^{(2)}
```

denote the raw delay quantity used by the baseline. Then

```math
L_i^{(2)} = \frac{\tau_i^{(2)}} {\sum_j\tau_j^{(2)}}.
```

The implementation's performance expression is represented as

```math
Q_i^{(2)} = \frac{1}{2} \frac{ M_i^{\mathrm{ram}}-B_i^{\mathrm{q}} }{ M_i^{\mathrm{ram}} } + \frac{1}{2} \frac{ \sigma_i^{\mathrm{MIPS}}10^6 }{ f_i }.
```

The trade-off score becomes

```math
\Omega_i^{(2)} = \omega_L^{(2)} \frac{L_i^{(2)}}{\sum_jL_j^{(2)}} - \omega_Q^{(2)} \frac{Q_i^{(2)}}{\sum_jQ_j^{(2)}}.
```

Thus

```math
FDI,\quad FPI,\quad IPM,\quad CE,\quad \alpha,\quad \beta
```

do not survive in the paper-facing notation.

The dimensional concern in the source-derived processing-delay expression remains a modelling issue and is **not** silently corrected by this notation exercise.

---

# J. 3D-POS baseline

The implementation currently follows a DPF/DEE/DCC objective family.

The revised proximity quantity is

```math
J_i^{\mathrm{prox}} = \frac{ \bar r_i^{\mathrm{IoT}} + \bar r_i^{\mathrm{fog}} }{ \displaystyle \sum_m r_{mi} + \sum_{j\ne i}r_{ij} }.
```

Estimated offloading and computation energies are

```math
E_i^{\uparrow} = P^{\mathrm{comm}} \frac{ 8B_i^{\mathrm{q}} }{ \bar{\mathcal R}_i^{\uparrow} },
```

```math
E_i^{\mathrm{cpu}} = P^{\mathrm{cpu}} \frac{ C_i^{\mathrm{q}} }{ f_i }.
```

The 3D-POS energy index is

```math
J_i^{\mathrm{en}} = \frac{ E_i^0- \left( E_i^{\uparrow}+E_i^{\mathrm{cpu}} \right) }{ E_i^0 }.
```

Its capacity index is

```math
J_i^{\mathrm{cap}} = \frac{1}{2} \frac{ n_i^{\mathrm{core}}f_i }{ \sigma_i^{\mathrm{MIPS}}10^6 } + \frac{1}{2} \frac{ M_i^{\mathrm{ram}}-M_i^{\mathrm{occ}} }{ M_i^{\mathrm{ram}} }.
```

The final scalarization becomes

```math
\Omega_i^{(3)} = \omega_r^{(3)} \widetilde J_i^{\mathrm{prox}} - \omega_E^{(3)} \widetilde J_i^{\mathrm{en}} - \omega_C^{(3)} \widetilde J_i^{\mathrm{cap}}.
```

This eliminates

```math
DPF,\quad DOE,\quad DCE,\quad DEE,\quad CPI,\quad MUF,\quad DCC
```

and the implementation's `tau_weights` notation.

The source's exact glyphs cannot be independently extracted from the repository PDF through the GitHub connector, so the 3D-POS provenance judgment remains slightly less certain than the FU-Serve, MEREC, SPOTIS, and Zeng judgments.

---

# K. ReLIEF baseline

The repository explicitly states that its ReLIEF implementation follows source Eq. (1) for Q-learning, Eq. (2) for action selection, and Eq. (21) for reward.

The revised state is

```math
\boldsymbol\xi_k^{\mathrm R},
```

and the dispatch choice is

```math
\boldsymbol\chi_k^{\mathrm R} = \left( i_k^{(p)},i_k^{(b)} \right).
```

The action-value table is

```math
\mathcal H^{\mathrm R} \left( \boldsymbol\xi, \boldsymbol\chi \right),
```

rather than `Q(s,a)`.

The ReLIEF reward is

```math
u_k^{\mathrm R} = \omega_C^{\mathrm R} \frac{ C_\star-\Delta_C }{ C_\star } + \omega_T^{\mathrm R} \frac{ T_k^{\mathrm{dl}}-T_k^{\mathrm{e2e}} }{ T_k^{\mathrm{dl}} } + \omega_p^{\mathrm R} \frac{ p_k^{\mathrm{pair}}-p_\star }{ p_\star }.
```

The update becomes

```math
\mathcal H_{n+1}^{\mathrm R} (\boldsymbol\xi,\boldsymbol\chi) = \mathcal H_n^{\mathrm R} (\boldsymbol\xi,\boldsymbol\chi) + \eta_{\mathrm R} \left[ u_n^{\mathrm R} + \gamma_{\mathrm R} \max_{\boldsymbol\chi'} \mathcal H_n^{\mathrm R} (\boldsymbol\xi',\boldsymbol\chi') - \mathcal H_n^{\mathrm R} (\boldsymbol\xi,\boldsymbol\chi) \right].
```

The greedy policy is

```math
\boldsymbol\chi^\star (\boldsymbol\xi) = \arg\max_{\boldsymbol\chi} \mathcal H^{\mathrm R} (\boldsymbol\xi,\boldsymbol\chi).
```

This removes the source-specific

```math
Q,\ s,\ a,\ r,\ \rho_1,\rho_2,\rho_3,\ WL,\ D,\ R,\ R_K
```

family.

Keeping `\gamma_{\mathrm R}` for a discount factor is acceptable because `\gamma` is conventional reinforcement-learning notation.

---

# L. PPO notation

The Stage-2 learned policy uses the following separate namespace:

```math
\boldsymbol\xi_k
```

for the observation/state representation,

```math
\mathbf e_i
```

for the embedding of UAV `i`,

```math
\mathbf g
```

for the global attention context,

and

```math
\boldsymbol\chi_k = \left( i_k^{(p)},i_k^{(b)} \right)
```

for the dispatch decision.

This fixes the previous conflicts in which `c` denoted both CPU demand and neural context and `h` denoted both embeddings and actions.

The critic is

```math
\mathcal J_\phi (\boldsymbol\xi_k),
```

not `V_\phi(s_k)`.

The TD residual is

```math
\delta_k^{\mathrm{td}} = u_k + \gamma \mathcal J_\phi (\boldsymbol\xi_{k+1}) - \mathcal J_\phi (\boldsymbol\xi_k).
```

Let

```math
\iota_k \in \{0,1\}
```

denote the terminal indicator. GAE is written

```math
\psi_k = \delta_k^{\mathrm{td}} + \gamma\lambda_{\mathrm{GAE}} (1-\iota_k) \psi_{k+1}.
```

The critic target is

```math
\widehat{\mathcal J}_k = \psi_k + \mathcal J_\phi (\boldsymbol\xi_k).
```

The PPO probability ratio is

```math
\varphi_k(\theta) = \frac{ \pi_\theta ( \boldsymbol\chi_k \mid \boldsymbol\xi_k ) }{ \pi_{\theta_{\mathrm{old}}} ( \boldsymbol\chi_k \mid \boldsymbol\xi_k ) }.
```

Thus `V`, `A`, `s`, and `r` are not used for critic value, advantage, state, or reward.

---

# M. Protected propulsion notation

The following equation is intentionally unchanged:

```math
P(V) = P_0 \left( 1+\frac{3V^2}{U_{\mathrm{tip}}^2} \right) + P_i \left( \sqrt{ 1+\frac{V^4}{4v_0^4} } - \frac{V^2}{2v_0^2} \right)^{1/2} + \frac{1}{2} d_0\rho s A V^3.
```

Its symbols

```math
P(V),P_0,P_i,V,U_{\mathrm{tip}},v_0,d_0,\rho,s,A
```

are intentionally preserved. The source gives this expression as Eq. (6).

---

# N. Fresh notation audit

## 1. Remaining notation collisions

| My notationReference notationReference paperType of overlapMust change?Reason |                                        |                         |              |    |                                                                                                                   |
| ----------------------------------------------------------------------------- | -------------------------------------- | ----------------------- | ------------ | -- | ----------------------------------------------------------------------------------------------------------------- |
| `P(V),P_0,P_i,V,U_{\mathrm{tip}},v_0,d_0,\rho,s,A`                            | identical                              | Zeng et al.             | Exact        | No | Explicit protected exception                                                                                      |
| `w_q`                                                                         | `w_j`                                  | MEREC / SPOTIS          | Conventional | No | `w` for a weight is generic mathematical notation and no longer belongs to the surrounding source notation family |
| `B,N_0,P^{\mathrm{tx}}`                                                       | similar standard communication symbols | several wireless papers | Conventional | No | Standard communications notation                                                                                  |
| `\pi_\theta,\gamma,\lambda_{\mathrm{GAE}}`                                    | standard RL notation                   | PPO/GAE literature      | Conventional | No | Changing these would reduce clarity without increasing genuine independence                                       |
| `E_i^0,E_i^{\mathrm{bat}}`                                                    | various `E`-based energy quantities    | several UAV papers      | Conventional | No | `E` for energy is not a distinctive source convention                                                             |
| `\arg\min,\arg\max,\sum,\prod,\\|\cdot\\|_2`                                  | same                                   | many sources            | Conventional | No | Universal mathematical language                                                                                   |

**No unprotected paper-specific notation collision remains in the revised specification that I can substantiate from the accessible sources.**

The previously problematic FU-Serve, MEREC, SPOTIS, ReLIEF, 2DP-FHS, and 3D-POS families have all been broken at the family level rather than through index-only changes.

---

## 2. Internal notation problems

No major internal symbol collision remains in the revised specification.

The former duplicate uses of `I_i^{\mathrm{cpu}}` and `m_i^{\mathrm{free}}` have been removed. The 2DP and 3DP quantities now occupy separate namespaces.

Task CPU demand `c_k` no longer conflicts with the neural global context, which is now `\mathbf g`.

The dispatch variable no longer conflicts with the neural node embedding: dispatch is `\boldsymbol\chi_k`, while node embeddings are `\mathbf e_i`.

The protected `A,V,s,d_0,\rho` symbols are no longer reused for advantage, critic value, task size, terminal indicator, distance, or reward.

Stage 1 now has an explicit notation for all six implemented criteria, so representative candidate reliability is no longer conflated with task-specific reliability.

One implementation-level consistency problem nevertheless remains: the current `arch_diagram.py` still shows the old reader-facing notation, including `s_i,l_i,d,g(d),TR,T_u,R,WL,S_{\min},S_{\max},V_\phi`, and `r` as reward.

It also still labels the Stage-1 matrix as `20\times5`, whereas the final implementation contains six criteria.

Therefore the **notation specification is internally consistent, but the existing reader-facing architecture figure has not yet been migrated to it**.

---

## 3. Verdict

### Notation independence: 9.6/10

The new system no longer reproduces any accessible reference paper's distinctive symbol family except the explicitly protected Zeng propulsion equation.

The strongest improvements are:

FU-Serve's PCF/CCF architecture is no longer mirrored through renamed “past/current” variables.

MEREC's complete `n,S,S',E` family has been replaced, rather than only renaming the removal effect.

SPOTIS's `S,d,S^\star` family has been replaced by `z,\delta,\Phi`.

ReLIEF's `Q,s,a,r,R,WL,D` family has been replaced as a coherent system.

The 2DP and 3DP acronym families and their source-specific coefficient notation have been removed.

The remaining uncertainty is mostly evidentiary: exact equation glyphs in the binary 2DP-FHS and 3D-POS PDFs cannot be extracted through the GitHub connector.

### Notation quality: 9.3/10

The notation is now concise enough for IEEE presentation while retaining semantic separation between physical quantities, Stage-1 quantities, baseline-only quantities, and RL quantities.

The principal remaining problem is **application**, not architecture: current reader-facing figures still contain the superseded notation.

### Verdict

**PASS for the revised notation specification.**

**The repository/manuscript as currently rendered is not yet paper-wide PASS**, because `arch_diagram.py` and potentially the compiled manuscript still contain the old symbols. A final paper-wide certification requires those reader-facing occurrences to be migrated and the manuscript source or searchable manuscript text to be checked.