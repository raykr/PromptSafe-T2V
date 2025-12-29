下面给你一版**“仿真这一节的完整正文内容”**，已经帮你把 **setup + 结果描述 + 图引用** 全部自然地串在一起，**不繁琐、不像技术报告**，但**信息是完备的**。
这是我按 **JAS / Automatica / TAC** 的常见风格，给你压到一个“一气呵成”的版本，你可以**直接整体粘贴作为 Numerical Example / Simulation Results 一节**。

---

## Numerical Example

In this section, a nonlinear numerical example is presented to illustrate the performance of the proposed second-order OCP-based algorithm for computing open-loop Nash equilibria in nonlinear systems.

### Problem setup

Consider the following discrete-time nonlinear system:
[
x_{k+1}
= a x_k

* c,\frac{x_k^3}{1+x_k^2}
* b_1 u_k
* b_2 w_k,
  \qquad k=0,1,\dots,N-1,
  ]
  where (x_k\in\mathbb{R}) denotes the system state, and (u_k\in\mathbb{R}) and (w_k\in\mathbb{R}) are the control inputs of Player~1 and Player~2, respectively.
  The nonlinear term (\frac{x_k^3}{1+x_k^2}) corresponds to a saturated Duffing-type nonlinearity, which preserves nonlinear behavior while preventing numerical instability for large state magnitudes.

The system parameters are chosen as
[
a=0.90,\quad b_1=0.80,\quad b_2=0.60,\quad c=0.12.
]

Each player minimizes a quadratic performance index with Nash coupling:
[
\begin{aligned}
J_1 &= \sum_{k=0}^{N-1}
\big(q_1 x_k^2 + r_1 u_k^2 + s_1 u_k w_k\big)

* p_1 x_N^2,\
  J_2 &= \sum_{k=0}^{N-1}
  \big(q_2 x_k^2 + r_2 w_k^2 + s_2 u_k w_k\big)
* p_2 x_N^2.
  \end{aligned}
  ]
  The weighting parameters are set to
  [
  \begin{aligned}
  &q_1=1.0,\ r_1=1.2,\ s_1=0.5,\ p_1=2.0,\
  &q_2=0.9,\ r_2=1.1,\ s_2=0.4,\ p_2=1.5.
  \end{aligned}
  ]
  The cross terms (u_k w_k) introduce strategic interaction between the two players and ensure a genuine Nash-coupled optimization problem.

The prediction horizon is selected as (N=25). To examine robustness with respect to the initial state, three different initial conditions are considered:
[
x_0 \in {1,;2,;3}.
]

### Numerical results

The open-loop Nash equilibrium is computed by solving the coupled stationarity conditions using the proposed structured second-order algorithm, where exact first- and second-order information is obtained via forward–backward recursions. For comparison, a high-accuracy reference solution is computed using the same second-order scheme with a tighter stopping tolerance and additional iterations, and is used solely for validation.

The simulation results are shown in Fig.~X. Fig.~X(a) illustrates the optimal state trajectories for the three initial conditions. In all cases, the state converges rapidly to the origin, indicating that the computed Nash equilibrium effectively stabilizes the nonlinear system. The trajectories produced by the proposed algorithm closely match the reference solutions, demonstrating high numerical accuracy.

Figs.~X(b) and X(c) depict the corresponding control sequences of Player~1 and Player~2, respectively. The control inputs (u_k) and (w_k) exhibit smooth transient behavior and converge toward zero as the state approaches equilibrium. Moreover, the algorithmic solutions are in excellent agreement with the reference results across all initial conditions, confirming that the proposed method accurately captures the Nash-coupled structure and converges to the open-loop Nash equilibrium rather than a centralized optimum.

Overall, this example demonstrates that the proposed algorithm can reliably compute Nash-coupled optimal control strategies for nonlinear discrete-time systems, yielding stable state trajectories and well-behaved control profiles under nonlinear dynamics and strategic interactions.

---

### 图注（与你当前三子图严格一致）

> **Fig. X.** Nonlinear Nash-coupled optimal trajectories under three different initial conditions (x_0=1,2,3):
> (a) optimal state;
> (b) Player-1 control (u_k);
> (c) Player-2 control (w_k).

---

## 现在这一节在“审稿人眼里”的状态

* ✔ **结构完整**：模型 → 参数 → 算法 → 结果
* ✔ **非线性突出**，但不过度解释
* ✔ **Nash-coupled 主线清晰**，没有被“合成控制”带偏
* ✔ 篇幅和信息密度 **正好是期刊仿真节的黄金区间**

如果你愿意，下一步我可以帮你做三件**“投稿前最后一刀”**的事情之一：

1. 🔹 按 **JAS 页数习惯**再压缩 10–15%
2. 🔹 帮你写一段 **Remark**，点出“与 LQR / centralized OC 的本质区别”
3. 🔹 帮你逐句对照 **Algorithm 部分符号**，做一次“全文一致性体检”

你选一个，我继续陪你把这篇打磨到**稳过审**的程度。
