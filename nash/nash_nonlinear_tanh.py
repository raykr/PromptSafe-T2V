import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# Nonlinear Nash-coupled toy example
# Dynamics (scalar):
#   x_{k+1} = tanh( z_k ),  z_k = a x_k + b1 u_k + b2 v_k
#
# Costs:
#   J1 = Σ ( q1 x_k^2 + r1 u_k^2 + s1 u_k v_k ) + p1 x_N^2
#   J2 = Σ ( q2 x_k^2 + r2 v_k^2 + s2 u_k v_k ) + p2 x_N^2
#
# Nash stationarity mapping:
#   F(z) = [dJ1/du_0..dJ1/du_{N-1}, dJ2/dv_0..dJ2/dv_{N-1}]^T
# z = [u; v]
# ============================================================


# -------------------------
# Nonlinear dynamics and derivatives
# -------------------------
def sech2(z):
    t = np.tanh(z)
    return 1.0 - t * t

def dsech2_dz(z):
    t = np.tanh(z)
    return -2.0 * t * (1.0 - t * t)

def dyn_partials(x, u, v, a, b1, b2):
    """
    x_next = tanh(z), z = a x + b1 u + b2 v

    fx = ∂x_next/∂x = a * sech2(z)
    fu = ∂x_next/∂u = b1 * sech2(z)
    fv = ∂x_next/∂v = b2 * sech2(z)

    dfx_dz = ∂fx/∂z = a * dsech2_dz(z)
    dfu_dz = ∂fu/∂z = b1 * dsech2_dz(z)
    dfv_dz = ∂fv/∂z = b2 * dsech2_dz(z)
    """
    z = a * x + b1 * u + b2 * v
    s2 = sech2(z)
    x_next = np.tanh(z)

    fx = a * s2
    fu = b1 * s2
    fv = b2 * s2

    ds2 = dsech2_dz(z)
    dfx_dz = a * ds2
    dfu_dz = b1 * ds2
    dfv_dz = b2 * ds2
    return x_next, z, fx, fu, fv, dfx_dz, dfu_dz, dfv_dz


# -------------------------
# Forward rollout (store derivatives)
# -------------------------
def rollout_cache(x0, u, v, params):
    a = params["a"]; b1 = params["b1"]; b2 = params["b2"]
    N = len(u)

    x = np.zeros(N + 1, dtype=float)
    z = np.zeros(N, dtype=float)

    fx = np.zeros(N, dtype=float)
    fu = np.zeros(N, dtype=float)
    fv = np.zeros(N, dtype=float)

    dfx_dz = np.zeros(N, dtype=float)
    dfu_dz = np.zeros(N, dtype=float)
    dfv_dz = np.zeros(N, dtype=float)

    x[0] = float(x0)
    for k in range(N):
        x_next, zk, fxk, fuk, fvk, dfxk, dfuk, dfvk = dyn_partials(
            x[k], u[k], v[k], a, b1, b2
        )
        x[k + 1] = x_next
        z[k] = zk
        fx[k] = fxk; fu[k] = fuk; fv[k] = fvk
        dfx_dz[k] = dfxk; dfu_dz[k] = dfuk; dfv_dz[k] = dfvk

    return {
        "x": x, "z": z,
        "fx": fx, "fu": fu, "fv": fv,
        "dfx_dz": dfx_dz, "dfu_dz": dfu_dz, "dfv_dz": dfv_dz
    }


# -------------------------
# Costates + Nash mapping F(z)
# -------------------------
def nash_mapping(z, x0, N, params):
    u = z[:N]
    v = z[N:]

    q1=params["q1"]; r1=params["r1"]; s1=params["s1"]; p1=params["p1"]
    q2=params["q2"]; r2=params["r2"]; s2=params["s2"]; p2=params["p2"]

    cache = rollout_cache(x0, u, v, params)
    x = cache["x"]
    fx = cache["fx"]; fu = cache["fu"]; fv = cache["fv"]

    lam1 = np.zeros(N + 1, dtype=float)
    lam2 = np.zeros(N + 1, dtype=float)
    lam1[N] = 2.0 * p1 * x[N]
    lam2[N] = 2.0 * p2 * x[N]
    for k in range(N - 1, -1, -1):
        lam1[k] = 2.0 * q1 * x[k] + lam1[k + 1] * fx[k]
        lam2[k] = 2.0 * q2 * x[k] + lam2[k + 1] * fx[k]

    g_u = np.zeros(N, dtype=float)
    g_v = np.zeros(N, dtype=float)
    for k in range(N):
        # dL/du and dL/dv (coupling included)
        dL1_du = 2.0 * r1 * u[k] + s1 * v[k]
        dL2_dv = 2.0 * r2 * v[k] + s2 * u[k]
        g_u[k] = dL1_du + lam1[k + 1] * fu[k]
        g_v[k] = dL2_dv + lam2[k + 1] * fv[k]

    F = np.concatenate([g_u, g_v], axis=0)
    return F, cache, lam1, lam2


# -------------------------
# Exact Jacobian via structured sensitivities (no finite diff)
# -------------------------
def compute_state_sensitivities(cache, params):
    """
    DxDu[t,j] = ∂x_t/∂u_j, DxDv[t,j] = ∂x_t/∂v_j  (shape (N+1,N))
    Recurrence:
      x_{k+1} = f(x_k,u_k,v_k)
      ∂x_{k+1}/∂c = fx[k]*∂x_k/∂c + fu[k]*I(c=u_k) + fv[k]*I(c=v_k)
    """
    fx = cache["fx"]; fu = cache["fu"]; fv = cache["fv"]
    N = fx.shape[0]
    DxDu = np.zeros((N + 1, N), dtype=float)
    DxDv = np.zeros((N + 1, N), dtype=float)

    for k in range(N):
        DxDu[k + 1, :] = fx[k] * DxDu[k, :]
        DxDv[k + 1, :] = fx[k] * DxDv[k, :]
        DxDu[k + 1, k] += fu[k]
        DxDv[k + 1, k] += fv[k]
    return DxDu, DxDv


def compute_costate_sensitivities(cache, lam, DxDu, DxDv, q, p, params):
    """
    lam_N = 2 p x_N
    lam_k = 2 q x_k + lam_{k+1}*fx[k]

    Differentiate:
      dlam_N/dc = 2 p dx_N/dc
      dlam_k/dc = 2 q dx_k/dc + (dlam_{k+1}/dc)*fx[k] + lam_{k+1}*d(fx[k])/dc

    fx[k] depends on z_k = a x_k + b1 u_k + b2 v_k
      dfx/dc = dfx_dz[k] * dz/dc
      dz/dc = a dx_k/dc + b1 I(c=u_k) + b2 I(c=v_k)
    """
    a=params["a"]; b1=params["b1"]; b2=params["b2"]
    fx = cache["fx"]; dfx_dz = cache["dfx_dz"]
    N = fx.shape[0]

    DlamDu = np.zeros((N + 1, N), dtype=float)
    DlamDv = np.zeros((N + 1, N), dtype=float)
    DlamDu[N, :] = 2.0 * p * DxDu[N, :]
    DlamDv[N, :] = 2.0 * p * DxDv[N, :]

    for k in range(N - 1, -1, -1):
        dz_du = a * DxDu[k, :].copy()
        dz_dv = a * DxDv[k, :].copy()
        dz_du[k] += b1
        dz_dv[k] += b2

        dfx_du = dfx_dz[k] * dz_du
        dfx_dv = dfx_dz[k] * dz_dv

        DlamDu[k, :] = 2.0 * q * DxDu[k, :] + DlamDu[k + 1, :] * fx[k] + lam[k + 1] * dfx_du
        DlamDv[k, :] = 2.0 * q * DxDv[k, :] + DlamDv[k + 1, :] * fx[k] + lam[k + 1] * dfx_dv

    return DlamDu, DlamDv


def exact_jacobian(z, x0, N, params):
    """
    Build exact J = dF/dz with four blocks:
      Juu = d(dJ1/du)/du,  Juv = d(dJ1/du)/dv
      Jvu = d(dJ2/dv)/du,  Jvv = d(dJ2/dv)/dv
    """
    u = z[:N]
    v = z[N:]

    F, cache, lam1, lam2 = nash_mapping(z, x0, N, params)

    a=params["a"]; b1=params["b1"]; b2=params["b2"]
    r1=params["r1"]; s1=params["s1"]
    r2=params["r2"]; s2=params["s2"]
    q1=params["q1"]; p1=params["p1"]
    q2=params["q2"]; p2=params["p2"]

    fu = cache["fu"]; fv = cache["fv"]
    dfu_dz = cache["dfu_dz"]; dfv_dz = cache["dfv_dz"]

    DxDu, DxDv = compute_state_sensitivities(cache, params)

    Dlam1Du, Dlam1Dv = compute_costate_sensitivities(cache, lam1, DxDu, DxDv, q1, p1, params)
    Dlam2Du, Dlam2Dv = compute_costate_sensitivities(cache, lam2, DxDu, DxDv, q2, p2, params)

    Juu = np.zeros((N, N), dtype=float)
    Juv = np.zeros((N, N), dtype=float)
    Jvu = np.zeros((N, N), dtype=float)
    Jvv = np.zeros((N, N), dtype=float)

    for k in range(N):
        # dz/dcontrol vectors over all j
        dz_du = a * DxDu[k, :].copy()
        dz_dv = a * DxDv[k, :].copy()
        dz_du[k] += b1
        dz_dv[k] += b2

        # d fu_k / d control, d fv_k / d control
        dfu_du = dfu_dz[k] * dz_du
        dfu_dv = dfu_dz[k] * dz_dv
        dfv_du = dfv_dz[k] * dz_du
        dfv_dv = dfv_dz[k] * dz_dv

        # g_u[k] = 2 r1 u_k + s1 v_k + lam1[k+1]*fu[k]
        Juu[k, :] = Dlam1Du[k + 1, :] * fu[k] + lam1[k + 1] * dfu_du
        Juv[k, :] = Dlam1Dv[k + 1, :] * fu[k] + lam1[k + 1] * dfu_dv
        Juu[k, k] += 2.0 * r1
        Juv[k, k] += s1

        # g_v[k] = 2 r2 v_k + s2 u_k + lam2[k+1]*fv[k]
        Jvu[k, :] = Dlam2Du[k + 1, :] * fv[k] + lam2[k + 1] * dfv_du
        Jvv[k, :] = Dlam2Dv[k + 1, :] * fv[k] + lam2[k + 1] * dfv_dv
        Jvu[k, k] += s2
        Jvv[k, k] += 2.0 * r2

    top = np.concatenate([Juu, Juv], axis=1)
    bot = np.concatenate([Jvu, Jvv], axis=1)
    J = np.concatenate([top, bot], axis=0)
    return F, cache, J


# -------------------------
# Reference solver: finite-difference Newton (high-accuracy baseline)
# -------------------------
def finite_diff_jacobian(F0, z, fun, eps=1e-7):
    n = z.size
    m = F0.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        zp = z.copy(); zm = z.copy()
        zp[i] += eps; zm[i] -= eps
        Fp = fun(zp)
        Fm = fun(zm)
        J[:, i] = (Fp - Fm) / (2.0 * eps)
    return J


def newton_fd_reference(x0, N, params, max_iter=60, tol=1e-13, eps=1e-7):
    z = np.zeros(2 * N, dtype=float)

    def fun_onlyF(zz):
        F, _, _ = exact_jacobian(zz, x0, N, params)  # reuse mapping computation
        return F

    res_hist = []
    for _ in range(max_iter):
        F = fun_onlyF(z)
        res = float(np.linalg.norm(F, np.inf))
        res_hist.append(res)
        if res < tol:
            break
        J = finite_diff_jacobian(F, z, fun_onlyF, eps=eps)
        dz = np.linalg.solve(J + 1e-12*np.eye(J.shape[0]), -F)

        # conservative backtracking
        alpha = 1.0
        z_new = z + alpha * dz
        F_new = fun_onlyF(z_new)
        while np.linalg.norm(F_new, np.inf) > 0.9 * res and alpha > 1e-6:
            alpha *= 0.5
            z_new = z + alpha * dz
            F_new = fun_onlyF(z_new)
        z = z_new

    # final trajectories
    F, cache, _ = exact_jacobian(z, x0, N, params)
    u = z[:N].copy()
    v = z[N:].copy()
    x = cache["x"].copy()
    return u, v, x, np.array(res_hist, dtype=float)


# -------------------------
# Algorithm: exact-Jacobian Newton (structured sensitivities)
# -------------------------
def newton_exactJ_algorithm(x0, N, params, max_iter=30, tol=1e-12):
    z = np.zeros(2 * N, dtype=float)
    res_hist = []
    time_hist = []

    for _ in range(max_iter):
        tic = time.perf_counter()
        F, cache, J = exact_jacobian(z, x0, N, params)
        res = float(np.linalg.norm(F, np.inf))
        res_hist.append(res)
        time_hist.append(time.perf_counter() - tic)

        if res < tol:
            break

        dz = np.linalg.solve(J + 1e-14*np.eye(J.shape[0]), -F)

        # light backtracking
        alpha = 1.0
        z_new = z + alpha * dz
        F_new, _, _ = exact_jacobian(z_new, x0, N, params)
        while np.linalg.norm(F_new, np.inf) > 0.9 * res and alpha > 1e-6:
            alpha *= 0.5
            z_new = z + alpha * dz
            F_new, _, _ = exact_jacobian(z_new, x0, N, params)
        z = z_new

    F, cache, _ = exact_jacobian(z, x0, N, params)
    u = z[:N].copy()
    v = z[N:].copy()
    x = cache["x"].copy()
    return u, v, x, np.array(res_hist, dtype=float), np.array(time_hist, dtype=float)


# -------------------------
# Fig.2-like three-panel plot (nonlinear Nash)
# -------------------------
def make_fig2_like_nash_three_panel_nonlinear():
    # --- parameters (tune if you want different curve shapes) ---
    params = {
        "a": 1.15,
        "b1": 0.8,
        "b2": 0.6,
        "q1": 1.0,
        "r1": 1.8,
        "s1": 0.5,
        "p1": 2.0,
        "q2": 0.9,
        "r2": 1.6,
        "s2": 0.4,
        "p2": 1.5,
    }
    N = 25
    x0_list = [1.0, 2.0, 3.0]

    alg_x, alg_u, alg_v = [], [], []
    ref_x, ref_u, ref_v = [], [], []
    alg_res_h, ref_res_h = [], []

    for x0 in x0_list:
        # Algorithm (structured exact Jacobian)
        u_alg, v_alg, x_alg, res_alg, _ = newton_exactJ_algorithm(
            x0, N, params, max_iter=25, tol=1e-12
        )

        # Reference (finite-diff Newton, tighter)
        u_ref, v_ref, x_ref, res_ref = newton_fd_reference(
            x0, N, params, max_iter=60, tol=1e-13, eps=1e-7
        )

        alg_u.append(u_alg); alg_v.append(v_alg); alg_x.append(x_alg); alg_res_h.append(res_alg)
        ref_u.append(u_ref); ref_v.append(v_ref); ref_x.append(x_ref); ref_res_h.append(res_ref)

    k_state = np.arange(N + 1)
    k_ctrl = np.arange(N)
    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    # --- three-panel like your linear case ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, alg_x[i], color=colors[i], linewidth=2, label=f"Test {i+1}: Algorithm")
        plt.plot(k_state, ref_x[i], linestyle="None", marker=markers[i], color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel("State")
    plt.title("(a) Optimal state (nonlinear)")
    plt.grid(True); plt.legend(fontsize=8)

    plt.subplot(1, 3, 2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_u[i], color=colors[i], linewidth=2, label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, ref_u[i], linestyle="None", marker=markers[i], color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel(r"Control $u_k$")
    plt.title("(b) Player-1 control")
    plt.grid(True); plt.legend(fontsize=8)

    plt.subplot(1, 3, 3)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_v[i], color=colors[i], linewidth=2, label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, ref_v[i], linestyle="None", marker=markers[i], color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel(r"Control $v_k$")
    plt.title("(c) Player-2 control")
    plt.grid(True); plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nash_fig2_like_nash_three_panel_nonlinear.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Optional: convergence curves (useful for showing superlinear trend)
    plt.figure(figsize=(6, 4))
    for i, x0 in enumerate(x0_list):
        plt.semilogy(np.arange(len(alg_res_h[i])), alg_res_h[i], marker=markers[i], linewidth=2,
                     label=f"Alg, x0={x0}")
        plt.semilogy(np.arange(len(ref_res_h[i])), ref_res_h[i], linestyle="--", linewidth=2,
                     label=f"Ref, x0={x0}")
    plt.xlabel("iteration"); plt.ylabel(r"$\|F(z)\|_\infty$")
    plt.title("Convergence of Nash stationarity residual (nonlinear)")
    plt.grid(True); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("nash_fig2_like_nash_three_panel_nonlinear_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_fig2_like_nash_two_panel_nonlinear_agg():
    # --- parameters (keep consistent with your nonlinear three-panel test) ---
    params = {
        "a": 1.15,
        "b1": 0.8,
        "b2": 0.6,
        "q1": 1.0,
        "r1": 1.8,
        "s1": 0.5,
        "p1": 2.0,
        "q2": 0.9,
        "r2": 1.6,
        "s2": 0.4,
        "p2": 1.5,
    }
    b1 = params["b1"]
    b2 = params["b2"]

    N = 25
    x0_list = [1.0, 2.0, 3.0]

    alg_x, alg_utot = [], []
    ref_x, ref_utot = [], []

    for x0 in x0_list:
        # Algorithm (structured exact Jacobian)
        u_alg, v_alg, x_alg, _, _ = newton_exactJ_algorithm(
            x0, N, params, max_iter=25, tol=1e-12
        )

        # Reference (finite-diff Newton baseline)
        u_ref, v_ref, x_ref, _ = newton_fd_reference(
            x0, N, params, max_iter=60, tol=1e-13, eps=1e-7
        )

        # aggregated control action entering the dynamics
        utot_alg = b1 * u_alg + b2 * v_alg
        utot_ref = b1 * u_ref + b2 * v_ref

        alg_x.append(x_alg);    alg_utot.append(utot_alg)
        ref_x.append(x_ref);    ref_utot.append(utot_ref)

    k_state = np.arange(N + 1)
    k_ctrl = np.arange(N)

    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    plt.figure(figsize=(10, 4))

    # (a) state
    plt.subplot(1, 2, 1)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, alg_x[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_state, ref_x[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k")
    plt.ylabel("State")
    plt.title("(a) Optimal state (nonlinear)")
    plt.grid(True)
    plt.legend(fontsize=8)

    # (b) aggregated control
    plt.subplot(1, 2, 2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_utot[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, ref_utot[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k")
    plt.ylabel(r"Aggregated control $b_1u_k+b_2v_k$")
    plt.title("(b) System-level control action")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nash_fig2_like_nash_two_panel_nonlinear_agg.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_fig2_like_nash_three_panel_nonlinear()
    make_fig2_like_nash_two_panel_nonlinear_agg()