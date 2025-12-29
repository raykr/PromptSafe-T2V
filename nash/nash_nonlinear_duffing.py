import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Nonlinear Nash-coupled toy example (Duffing-type)
#
# Dynamics (scalar):
#   x_{k+1} = a x_k + c x_k^3 + b1 u_k + b2 v_k
#
# Costs:
#   J1 = Σ ( q1 x_k^2 + r1 u_k^2 + s1 u_k v_k ) + p1 x_N^2
#   J2 = Σ ( q2 x_k^2 + r2 v_k^2 + s2 u_k v_k ) + p2 x_N^2
#
# Nash stationarity mapping:
#   F(z) = [∂J1/∂u_0..∂J1/∂u_{N-1},  ∂J2/∂v_0..∂J2/∂v_{N-1}]^T
# z = [u; v]
#
# We solve F(z)=0 by Newton with an exact (structured) Jacobian.
# Reference solution is produced by stricter tolerance + more iterations.
# ============================================================


# -------------------------
# Dynamics + derivatives
# -------------------------
def dyn_partials(x, u, v, a, b1, b2, c):
    """
    x_next = a x + c x^3 + b1 u + b2 v
    fx = ∂x_next/∂x = a + 3c x^2
    fu = ∂x_next/∂u = b1
    fv = ∂x_next/∂v = b2
    fxx = ∂fx/∂x = 6c x   (used in costate sensitivity)
    """
    x_next = a * x + c * (x ** 3) + b1 * u + b2 * v
    fx = a + 3.0 * c * (x ** 2)
    fu = b1
    fv = b2
    fxx = 6.0 * c * x
    return x_next, fx, fu, fv, fxx


def rollout_cache(x0, u, v, params):
    a = params["a"]; b1 = params["b1"]; b2 = params["b2"]; c = params["c"]
    N = len(u)

    x = np.zeros(N + 1, dtype=float)
    fx = np.zeros(N, dtype=float)
    fu = np.zeros(N, dtype=float)
    fv = np.zeros(N, dtype=float)
    fxx = np.zeros(N, dtype=float)

    x[0] = float(x0)
    for k in range(N):
        x_next, fxk, fuk, fvk, fxxk = dyn_partials(x[k], u[k], v[k], a, b1, b2, c)
        x[k + 1] = x_next
        fx[k] = fxk
        fu[k] = fuk
        fv[k] = fvk
        fxx[k] = fxxk

    return {"x": x, "fx": fx, "fu": fu, "fv": fv, "fxx": fxx}


# -------------------------
# Sensitivities
# -------------------------
def compute_state_sensitivities(cache):
    """
    DxDu[k+1,:] = fx[k]*DxDu[k,:] + fu[k]*e_k
    DxDv[k+1,:] = fx[k]*DxDv[k,:] + fv[k]*e_k
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


def compute_costate_sensitivities(cache, lam, DxDu, DxDv, q, p):
    """
    lam_N = 2 p x_N
    lam_k = 2 q x_k + lam_{k+1} * fx[k]

    dlam_N/dc = 2 p dx_N/dc
    dlam_k/dc = 2 q dx_k/dc + (dlam_{k+1}/dc)*fx[k] + lam_{k+1}*(dfx/dc)

    Here:
      fx[k] = a + 3c x_k^2
      dfx/dc = (dfx/dx)*dx/dc, with dfx/dx = fxx[k] = 6c x_k
    """
    fx = cache["fx"]; fxx = cache["fxx"]
    N = fx.shape[0]

    DlamDu = np.zeros((N + 1, N), dtype=float)
    DlamDv = np.zeros((N + 1, N), dtype=float)

    DlamDu[N, :] = 2.0 * p * DxDu[N, :]
    DlamDv[N, :] = 2.0 * p * DxDv[N, :]

    for k in range(N - 1, -1, -1):
        dfx_du = fxx[k] * DxDu[k, :]
        dfx_dv = fxx[k] * DxDv[k, :]

        DlamDu[k, :] = 2.0 * q * DxDu[k, :] + DlamDu[k + 1, :] * fx[k] + lam[k + 1] * dfx_du
        DlamDv[k, :] = 2.0 * q * DxDv[k, :] + DlamDv[k + 1, :] * fx[k] + lam[k + 1] * dfx_dv

    return DlamDu, DlamDv


# -------------------------
# Nash mapping + exact Jacobian
# -------------------------
def exact_jacobian(z, x0, N, params):
    u = z[:N]
    v = z[N:]

    q1=params["q1"]; r1=params["r1"]; s1=params["s1"]; p1=params["p1"]
    q2=params["q2"]; r2=params["r2"]; s2=params["s2"]; p2=params["p2"]

    cache = rollout_cache(x0, u, v, params)
    x = cache["x"]
    fx = cache["fx"]; fu = cache["fu"]; fv = cache["fv"]

    # costates (PMP)
    lam1 = np.zeros(N + 1, dtype=float)
    lam2 = np.zeros(N + 1, dtype=float)
    lam1[N] = 2.0 * p1 * x[N]
    lam2[N] = 2.0 * p2 * x[N]
    for k in range(N - 1, -1, -1):
        lam1[k] = 2.0 * q1 * x[k] + lam1[k + 1] * fx[k]
        lam2[k] = 2.0 * q2 * x[k] + lam2[k + 1] * fx[k]

    # Nash stationarity F(z)
    g_u = np.zeros(N, dtype=float)
    g_v = np.zeros(N, dtype=float)
    for k in range(N):
        g_u[k] = (2.0 * r1 * u[k] + s1 * v[k]) + lam1[k + 1] * fu[k]
        g_v[k] = (2.0 * r2 * v[k] + s2 * u[k]) + lam2[k + 1] * fv[k]
    F = np.concatenate([g_u, g_v], axis=0)

    # structured sensitivities for Jacobian
    DxDu, DxDv = compute_state_sensitivities(cache)
    Dlam1Du, Dlam1Dv = compute_costate_sensitivities(cache, lam1, DxDu, DxDv, q1, p1)
    Dlam2Du, Dlam2Dv = compute_costate_sensitivities(cache, lam2, DxDu, DxDv, q2, p2)

    # Jacobian blocks
    Juu = np.zeros((N, N), dtype=float)
    Juv = np.zeros((N, N), dtype=float)
    Jvu = np.zeros((N, N), dtype=float)
    Jvv = np.zeros((N, N), dtype=float)

    for k in range(N):
        # g_u[k] = 2r1 u_k + s1 v_k + lam1[k+1]*fu[k]
        Juu[k, :] = Dlam1Du[k + 1, :] * fu[k]
        Juv[k, :] = Dlam1Dv[k + 1, :] * fu[k]
        Juu[k, k] += 2.0 * r1
        Juv[k, k] += s1

        # g_v[k] = 2r2 v_k + s2 u_k + lam2[k+1]*fv[k]
        Jvu[k, :] = Dlam2Du[k + 1, :] * fv[k]
        Jvv[k, :] = Dlam2Dv[k + 1, :] * fv[k]
        Jvu[k, k] += s2
        Jvv[k, k] += 2.0 * r2

    top = np.concatenate([Juu, Juv], axis=1)
    bot = np.concatenate([Jvu, Jvv], axis=1)
    J = np.concatenate([top, bot], axis=0)

    return F, cache, J


# -------------------------
# Newton solver with exact Jacobian + backtracking
# -------------------------
def newton_exactJ(x0, N, params, z_init=None, max_iter=30, tol=1e-12, verbose=False):
    if z_init is None:
        z = np.zeros(2 * N, dtype=float)
    else:
        z = z_init.astype(float).copy()

    res_hist = []

    for it in range(max_iter):
        F, cache, J = exact_jacobian(z, x0, N, params)
        res = float(np.linalg.norm(F, np.inf))
        res_hist.append(res)

        if verbose:
            print(f"[iter {it:02d}] ||F||_inf = {res:.3e}")

        if res < tol:
            break

        # Newton step
        dz = np.linalg.solve(J + 1e-12 * np.eye(J.shape[0]), -F)

        # Backtracking on residual
        alpha = 1.0
        z_new = z + alpha * dz
        F_new, _, _ = exact_jacobian(z_new, x0, N, params)
        res_new = float(np.linalg.norm(F_new, np.inf))

        bt = 0
        while res_new > 0.9 * res and alpha > 1e-6 and bt < 30:
            alpha *= 0.5
            z_new = z + alpha * dz
            F_new, _, _ = exact_jacobian(z_new, x0, N, params)
            res_new = float(np.linalg.norm(F_new, np.inf))
            bt += 1

        z = z_new

    # final trajectories
    F, cache, _ = exact_jacobian(z, x0, N, params)
    u = z[:N].copy()
    v = z[N:].copy()
    x = cache["x"].copy()
    return u, v, x, np.array(res_hist, dtype=float), z


# -------------------------
# Fig.2-like two-panel (state + aggregated control)
# -------------------------
def make_fig2_like_duffing_two_panel():
    # --- Recommended "Fig.2-like" parameters (weak nonlinearity) ---
    params = {
        "a": 0.85,
        "c": 0.05,     # nonlinearity strength: try 0.02~0.08
        "b1": 0.8,
        "b2": 0.6,
        "q1": 1.0, "r1": 2.0, "s1": 0.4, "p1": 2.0,
        "q2": 0.9, "r2": 1.8, "s2": 0.3, "p2": 1.5,
    }
    b1 = params["b1"]
    b2 = params["b2"]

    N = 25
    x0_list = [1.0, 2.0, 3.0]

    alg_x, alg_utot = [], []
    ref_x, ref_utot = [], []

    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    for x0 in x0_list:
        # ---------------- Algorithm ----------------
        # (moderate tolerance/iters)
        u_alg, v_alg, x_alg, res_alg, z_alg = newton_exactJ(
            x0, N, params, z_init=None, max_iter=25, tol=1e-12, verbose=False
        )

        # ---------------- Reference ----------------
        # High-accuracy solution using stricter tolerance + more iterations.
        # Warm-start from algorithm solution for robustness.
        u_ref, v_ref, x_ref, res_ref, z_ref = newton_exactJ(
            x0, N, params, z_init=z_alg, max_iter=80, tol=1e-14, verbose=False
        )

        # Aggregated control entering the dynamics
        utot_alg = b1 * u_alg + b2 * v_alg
        utot_ref = b1 * u_ref + b2 * v_ref

        alg_x.append(x_alg);    alg_utot.append(utot_alg)
        ref_x.append(x_ref);    ref_utot.append(utot_ref)

        # Optional: print final residuals (sanity check)
        # print(f"x0={x0}: ALG res={res_alg[-1]:.2e}, REF res={res_ref[-1]:.2e}")

    k_state = np.arange(N + 1)
    k_ctrl  = np.arange(N)

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
    plt.title("(a) Optimal state (nonlinear Duffing)")
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
    plt.savefig("nash_nonlinear_duffing.png", dpi=150, bbox_inches="tight")
    plt.close()


def _auto_ylim_from_series(series_list, pad_ratio=0.08, min_span=1e-3):
    """
    series_list: list of 1D arrays (concatenated all tests if you want)
    Create a reasonable ylim so that small signals are visible.
    """
    y = np.concatenate([np.asarray(s).ravel() for s in series_list])
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = y_max - y_min
    if span < min_span:
        mid = 0.5 * (y_min + y_max)
        span = min_span
        y_min = mid - 0.5 * span
        y_max = mid + 0.5 * span
    pad = pad_ratio * span
    return y_min - pad, y_max + pad


def solve_three_tests_duffing(params, N, x0_list):
    """
    Solve for each x0:
      Algorithm: moderate tol/iters
      Reference: high-accuracy (warm-start from Algorithm)
    Return dict of trajectories.
    """
    b1 = params["b1"]
    b2 = params["b2"]

    out = {
        "alg": {"x": [], "u": [], "v": [], "utot": []},
        "ref": {"x": [], "u": [], "v": [], "utot": []},
    }

    for x0 in x0_list:
        # Algorithm
        u_alg, v_alg, x_alg, _, z_alg = newton_exactJ(
            x0, N, params, z_init=None, max_iter=25, tol=1e-12, verbose=False
        )
        # Reference (high precision)
        u_ref, v_ref, x_ref, _, _ = newton_exactJ(
            x0, N, params, z_init=z_alg, max_iter=80, tol=1e-14, verbose=False
        )

        out["alg"]["x"].append(x_alg)
        out["alg"]["u"].append(u_alg)
        out["alg"]["v"].append(v_alg)
        out["alg"]["utot"].append(b1 * u_alg + b2 * v_alg)

        out["ref"]["x"].append(x_ref)
        out["ref"]["u"].append(u_ref)
        out["ref"]["v"].append(v_ref)
        out["ref"]["utot"].append(b1 * u_ref + b2 * v_ref)

    return out


def make_fig2_like_duffing_two_panel_agg(params=None, N=25, x0_list=None):
    if params is None:
        params = {
            "a": 0.85, "c": 0.05, "b1": 0.8, "b2": 0.6,
            "q1": 1.0, "r1": 2.0, "s1": 0.4, "p1": 2.0,
            "q2": 0.9, "r2": 1.8, "s2": 0.3, "p2": 1.5,
        }
    if x0_list is None:
        x0_list = [1.0, 2.0, 3.0]

    sol = solve_three_tests_duffing(params, N, x0_list)

    k_state = np.arange(N + 1)
    k_ctrl = np.arange(N)

    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    plt.figure(figsize=(10, 4))

    # (a) state
    plt.subplot(1, 2, 1)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, sol["alg"]["x"][i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_state, sol["ref"]["x"][i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel("State")
    plt.title("(a) Optimal state (nonlinear Duffing)")
    plt.grid(True); plt.legend(fontsize=8)

    # (b) aggregated control
    plt.subplot(1, 2, 2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, sol["alg"]["utot"][i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, sol["ref"]["utot"][i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k")
    plt.ylabel(r"Aggregated control $b_1u_k+b_2v_k$")
    plt.title("(b) System-level control action")
    plt.grid(True)
    # auto y-lim so tiny curves are visible (fix "only red line" illusion)
    ylo, yhi = _auto_ylim_from_series(sol["alg"]["utot"] + sol["ref"]["utot"], min_span=1e-3)
    plt.ylim([ylo, yhi])
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nash_fig2_like_nash_two_panel_nonlinear_duffing.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_fig2_like_duffing_three_panel_uv(params=None, N=25, x0_list=None):
    if params is None:
        params = {
            "a": 0.85, "c": 0.05, "b1": 0.8, "b2": 0.6,
            "q1": 1.0, "r1": 2.0, "s1": 0.4, "p1": 2.0,
            "q2": 0.9, "r2": 1.8, "s2": 0.3, "p2": 1.5,
        }
    if x0_list is None:
        x0_list = [1.0, 2.0, 3.0]

    sol = solve_three_tests_duffing(params, N, x0_list)

    k_state = np.arange(N + 1)
    k_ctrl = np.arange(N)

    colors  = ["r", "g", "b"]
    markers = ["o", "s", "x"]

    plt.figure(figsize=(12, 4))

    # (a) state
    plt.subplot(1, 3, 1)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, sol["alg"]["x"][i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_state, sol["ref"]["x"][i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel("State")
    plt.title("(a) Optimal state (nonlinear Duffing)")
    plt.grid(True); plt.legend(fontsize=8)

    # (b) Player-1 control u_k
    plt.subplot(1, 3, 2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, sol["alg"]["u"][i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, sol["ref"]["u"][i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel(r"Control $u_k$")
    plt.title("(b) Player-1 control")
    plt.grid(True)
    ylo, yhi = _auto_ylim_from_series(sol["alg"]["u"] + sol["ref"]["u"], min_span=1e-3)
    plt.ylim([ylo, yhi])
    plt.legend(fontsize=8)

    # (c) Player-2 control v_k
    plt.subplot(1, 3, 3)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, sol["alg"]["v"][i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm")
        plt.plot(k_ctrl, sol["ref"]["v"][i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=5,
                 label=f"Test {i+1}: Reference")
    plt.xlabel("k"); plt.ylabel(r"Control $v_k$")
    plt.title("(c) Player-2 control")
    plt.grid(True)
    ylo, yhi = _auto_ylim_from_series(sol["alg"]["v"] + sol["ref"]["v"], min_span=1e-3)
    plt.ylim([ylo, yhi])
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("nash_fig2_like_nash_three_panel_nonlinear_duffing_uv.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    make_fig2_like_duffing_two_panel()
    make_fig2_like_duffing_two_panel_agg()
    make_fig2_like_duffing_three_panel_uv()
