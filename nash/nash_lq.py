import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Nash-coupled LQ toy example (two controllers)
# System:
#   x_{k+1} = a x_k + b1 u_k + b2 v_k
# Costs:
#   J1 = Σ ( q1 x_k^2 + r1 u_k^2 + s1 u_k v_k ) + p1 x_N^2
#   J2 = Σ ( q2 x_k^2 + r2 v_k^2 + s2 u_k v_k ) + p2 x_N^2
#
# Nash conditions:
#   dJ1/du_k = 0,  dJ2/dv_k = 0   for k=0..N-1
#
# We'll compare:
#   - "Algorithm": Newton on Nash mapping F(z)=0 using exact Jacobian
#   - "Exact": one-shot linear solve z* = -J^{-1}F(0) (since LQ => F is affine)
# ============================================================

def rollout(a, b1, b2, x0, u, v):
    N = len(u)
    x = np.zeros(N + 1, dtype=float)
    x[0] = float(x0)
    for k in range(N):
        x[k + 1] = a * x[k] + b1 * u[k] + b2 * v[k]
    return x

def nash_residual(a, b1, b2, x0, u, v, q1, r1, s1, p1, q2, r2, s2, p2):
    """
    Costates:
      lam1_N = 2 p1 x_N,  lam1_k = 2 q1 x_k + a lam1_{k+1}
      lam2_N = 2 p2 x_N,  lam2_k = 2 q2 x_k + a lam2_{k+1}
    Stationarity:
      g1_k = 2 r1 u_k + s1 v_k + b1 lam1_{k+1}
      g2_k = 2 r2 v_k + s2 u_k + b2 lam2_{k+1}
    """
    N = len(u)
    x = rollout(a, b1, b2, x0, u, v)

    lam1 = np.zeros(N + 1, dtype=float)
    lam2 = np.zeros(N + 1, dtype=float)
    lam1[N] = 2.0 * p1 * x[N]
    lam2[N] = 2.0 * p2 * x[N]
    for k in range(N - 1, -1, -1):
        lam1[k] = 2.0 * q1 * x[k] + a * lam1[k + 1]
        lam2[k] = 2.0 * q2 * x[k] + a * lam2[k + 1]

    g1 = np.zeros(N, dtype=float)
    g2 = np.zeros(N, dtype=float)
    for k in range(N):
        g1[k] = 2.0 * r1 * u[k] + s1 * v[k] + b1 * lam1[k + 1]
        g2[k] = 2.0 * r2 * v[k] + s2 * u[k] + b2 * lam2[k + 1]

    F = np.concatenate([g1, g2], axis=0)
    return F, x

def compute_state_sensitivities(a, b1, b2, N):
    DxDu = np.zeros((N + 1, N), dtype=float)
    DxDv = np.zeros((N + 1, N), dtype=float)
    for j in range(N):
        DxDu[j + 1, j] += b1
        DxDv[j + 1, j] += b2
        for t in range(j + 2, N + 1):
            DxDu[t, j] = a * DxDu[t - 1, j]
            DxDv[t, j] = a * DxDv[t - 1, j]
    return DxDu, DxDv

def compute_costate_sensitivities(a, q, p, DxDctrl):
    N = DxDctrl.shape[0] - 1
    Dlam = np.zeros_like(DxDctrl)
    Dlam[N, :] = 2.0 * p * DxDctrl[N, :]
    for k in range(N - 1, -1, -1):
        Dlam[k, :] = 2.0 * q * DxDctrl[k, :] + a * Dlam[k + 1, :]
    return Dlam

def nash_jacobian(a, b1, b2, N, q1, r1, s1, p1, q2, r2, s2, p2):
    DxDu, DxDv = compute_state_sensitivities(a, b1, b2, N)
    Dlam1Du = compute_costate_sensitivities(a, q1, p1, DxDu)
    Dlam1Dv = compute_costate_sensitivities(a, q1, p1, DxDv)
    Dlam2Du = compute_costate_sensitivities(a, q2, p2, DxDu)
    Dlam2Dv = compute_costate_sensitivities(a, q2, p2, DxDv)

    Juu = np.zeros((N, N), dtype=float)
    Juv = np.zeros((N, N), dtype=float)
    Jvu = np.zeros((N, N), dtype=float)
    Jvv = np.zeros((N, N), dtype=float)

    for k in range(N):
        Juu[k, :] = b1 * Dlam1Du[k + 1, :]
        Juv[k, :] = b1 * Dlam1Dv[k + 1, :]
        Jvu[k, :] = b2 * Dlam2Du[k + 1, :]
        Jvv[k, :] = b2 * Dlam2Dv[k + 1, :]

        Juu[k, k] += 2.0 * r1
        Juv[k, k] += s1
        Jvu[k, k] += s2
        Jvv[k, k] += 2.0 * r2

    top = np.concatenate([Juu, Juv], axis=1)
    bot = np.concatenate([Jvu, Jvv], axis=1)
    return np.concatenate([top, bot], axis=0)

def exact_open_loop_nash(a, b1, b2, x0, N, q1, r1, s1, p1, q2, r2, s2, p2):
    J = nash_jacobian(a, b1, b2, N, q1, r1, s1, p1, q2, r2, s2, p2)
    z0 = np.zeros(2 * N, dtype=float)
    F0, _ = nash_residual(a, b1, b2, x0, z0[:N], z0[N:], q1, r1, s1, p1, q2, r2, s2, p2)
    z_star = np.linalg.solve(J, -F0)
    u_star = z_star[:N]
    v_star = z_star[N:]
    x_star = rollout(a, b1, b2, x0, u_star, v_star)
    return u_star, v_star, x_star

def newton_solve_nash(a, b1, b2, x0, N, q1, r1, s1, p1, q2, r2, s2, p2,
                      max_iter=30, tol=1e-12):
    J = nash_jacobian(a, b1, b2, N, q1, r1, s1, p1, q2, r2, s2, p2)
    J = J + 1e-14 * np.eye(J.shape[0])

    z = np.zeros(2 * N, dtype=float)
    for _ in range(max_iter):
        F, _ = nash_residual(a, b1, b2, x0, z[:N], z[N:], q1, r1, s1, p1, q2, r2, s2, p2)
        if np.linalg.norm(F, np.inf) < tol:
            break
        dz = np.linalg.solve(J, -F)

        # very light backtracking to avoid overshoot (usually 1.0 works for LQ)
        alpha = 1.0
        z_new = z + alpha * dz
        F_new, _ = nash_residual(a, b1, b2, x0, z_new[:N], z_new[N:], q1, r1, s1, p1, q2, r2, s2, p2)
        while np.linalg.norm(F_new, np.inf) > 0.9 * np.linalg.norm(F, np.inf) and alpha > 1e-6:
            alpha *= 0.5
            z_new = z + alpha * dz
            F_new, _ = nash_residual(a, b1, b2, x0, z_new[:N], z_new[N:], q1, r1, s1, p1, q2, r2, s2, p2)

        z = z_new

    u = z[:N]
    v = z[N:]
    x = rollout(a, b1, b2, x0, u, v)
    return u, v, x

def make_fig2_like_nash_two_panel():
    # -------- parameters: you can tune to match your manuscript style --------
    a = 1.8
    b1 = 0.9
    b2 = 0.7
    N = 15

    q1, r1, s1, p1 = 1.0, 3.0, 0.6, 3.0
    q2, r2, s2, p2 = 0.8, 2.5, 0.5, 2.0

    x0_list = [1.0, 2.0, 3.0]

    alg_x, alg_utot = [], []
    ex_x,  ex_utot  = [], []

    for x0 in x0_list:
        u_alg, v_alg, x_alg = newton_solve_nash(a, b1, b2, x0, N, q1, r1, s1, p1, q2, r2, s2, p2)
        u_ex,  v_ex,  x_ex  = exact_open_loop_nash(a, b1, b2, x0, N, q1, r1, s1, p1, q2, r2, s2, p2)

        # 合成控制：真实进入系统的控制作用
        utot_alg = b1 * u_alg + b2 * v_alg
        utot_ex  = b1 * u_ex  + b2 * v_ex

        alg_x.append(x_alg);  alg_utot.append(utot_alg)
        ex_x.append(x_ex);    ex_utot.append(utot_ex)

    k_state = np.arange(N + 1)
    k_ctrl  = np.arange(N)

    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    plt.figure(figsize=(10, 4))

    # (a) state
    plt.subplot(1, 2, 1)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, alg_x[i], color=colors[i], linewidth=2, label=f"Test {i+1}: Algorithm")
        plt.plot(k_state, ex_x[i], linestyle="None", marker=markers[i], color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact")
    plt.xlabel("k"); plt.ylabel("State")
    plt.title("(a) Optimal state (Nash-coupled)")
    plt.grid(True); plt.legend(fontsize=8)

    # (b) "control": use aggregated control action
    plt.subplot(1, 2, 2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_utot[i], color=colors[i], linewidth=2, label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, ex_utot[i], linestyle="None", marker=markers[i], color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact")
    plt.xlabel("k"); plt.ylabel(r"Aggregated control $b_1u_k+b_2v_k$")
    plt.title("(b) Optimal control action")
    plt.grid(True); plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nash_fig2_like.png", dpi=150, bbox_inches="tight")
    plt.close()

def make_fig2_like_nash_three_panel():
    # -------- parameters: tune to match your manuscript if needed --------
    a = 1.8
    b1 = 0.9
    b2 = 0.7
    N = 15

    q1, r1, s1, p1 = 1.0, 3.0, 0.6, 3.0
    q2, r2, s2, p2 = 0.8, 2.5, 0.5, 2.0

    x0_list = [1.0, 2.0, 3.0]

    alg_x, alg_u, alg_v = [], [], []
    ex_x,  ex_u,  ex_v  = [], [], []

    for x0 in x0_list:
        # Algorithm (Newton on Nash mapping with exact Jacobian)
        u_alg, v_alg, x_alg = newton_solve_nash(
            a, b1, b2, x0, N, q1, r1, s1, p1, q2, r2, s2, p2,
            max_iter=30, tol=1e-12
        )

        # Exact open-loop Nash (one linear solve)
        u_ex, v_ex, x_ex = exact_open_loop_nash(
            a, b1, b2, x0, N, q1, r1, s1, p1, q2, r2, s2, p2
        )

        alg_x.append(x_alg); alg_u.append(u_alg); alg_v.append(v_alg)
        ex_x.append(x_ex);   ex_u.append(u_ex);   ex_v.append(v_ex)

    k_state = np.arange(N + 1)
    k_ctrl  = np.arange(N)

    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    plt.figure(figsize=(12, 4))

    # (a) state
    plt.subplot(1, 3, 1)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, alg_x[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_state, ex_x[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact")
    plt.xlabel("k")
    plt.ylabel("State")
    plt.title("(a) Optimal state (Nash-coupled)")
    plt.grid(True)
    plt.legend(fontsize=8)

    # (b) player-1 control u_k
    plt.subplot(1, 3, 2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_u[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, ex_u[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact")
    plt.xlabel("k")
    plt.ylabel(r"Control $u_k$")
    plt.title("(b) Player-1 control")
    plt.grid(True)
    plt.legend(fontsize=8)

    # (c) player-2 control v_k
    plt.subplot(1, 3, 3)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_v[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, ex_v[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact")
    plt.xlabel("k")
    plt.ylabel(r"Control $v_k$")
    plt.title("(c) Player-2 control")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nash_fig2_like_nash_two_panel.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_fig2_like_nash_two_panel()
    make_fig2_like_nash_three_panel()
