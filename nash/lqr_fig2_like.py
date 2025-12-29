import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Problem: scalar discrete-time LQR
#    x_{k+1} = a x_k + b u_k
#    J = sum_{k=0}^{N-1} (q x_k^2 + r u_k^2) + p x_N^2
# ============================================================

def rollout_lqr(a, b, x0, u):
    """Forward rollout states x[0..N] for given control u[0..N-1]."""
    N = len(u)
    x = np.zeros(N + 1)
    x[0] = x0
    for k in range(N):
        x[k+1] = a * x[k] + b * u[k]
    return x

def gradient_lqr(a, b, q, r, p, x0, u):
    """
    Gradient via PMP recursion (costate lambda):
      lambda_N = 2 p x_N
      lambda_k = 2 q x_k + a lambda_{k+1}
      grad_k = dH/du_k = 2 r u_k + b lambda_{k+1}
    """
    N = len(u)
    x = rollout_lqr(a, b, x0, u)

    lam = np.zeros(N + 1)
    lam[N] = 2.0 * p * x[N]
    for k in range(N-1, -1, -1):
        lam[k] = 2.0 * q * x[k] + a * lam[k+1]

    grad = np.zeros(N)
    for k in range(N):
        grad[k] = 2.0 * r * u[k] + b * lam[k+1]
    return x, grad

def hessian_row_lqr(a, b, q, r, p, N, row_i):
    """
    Row-wise Hessian construction for LQR (matches the "row-wise" FBDE idea,
    but here it has a closed recursion form in (alpha, beta).

    Forward recursion for beta:
      beta_0 = 0
      beta_{k+1} = a beta_k,                      if k != i
                = b + a beta_k,                  if k == i

    Backward recursion for alpha:
      alpha_N = 2 p beta_N
      alpha_k = a alpha_{k+1} + 2 q beta_k

    Assemble ith row H[i,k]:
      if k == i:  2 r + b * alpha_{k+1}
      else:       b * alpha_{k+1}
    """
    beta = np.zeros(N + 1)
    beta[0] = 0.0
    for k in range(N):
        if k == row_i:
            beta[k+1] = b + a * beta[k]
        else:
            beta[k+1] = a * beta[k]

    alpha = np.zeros(N + 1)
    alpha[N] = 2.0 * p * beta[N]
    for k in range(N-1, -1, -1):
        alpha[k] = a * alpha[k+1] + 2.0 * q * beta[k]

    row = np.zeros(N)
    for k in range(N):
        if k == row_i:
            row[k] = 2.0 * r + b * alpha[k+1]
        else:
            row[k] = b * alpha[k+1]
    return row

def hessian_lqr(a, b, q, r, p, N):
    """Assemble full Hessian by rows (row-wise construction)."""
    H = np.zeros((N, N))
    for i in range(N):
        H[i, :] = hessian_row_lqr(a, b, q, r, p, N, i)
    return H

# ============================================================
# 2) Algorithm 1 (explicit second-order PMP recursion version)
#    Practical realization used in the TIE-style LQR demo:
#      g0 = (R + H)^(-1) grad
#      gj = (R + H)^(-1) (grad + R * g_{j-1}),  j=1..iter
#      u <- u - g_iter
# ============================================================

def algorithm1_solve_lqr(a, b, q, r, p, x0, N,
                         R_coeff=0.1,
                         max_outer=50,
                         tol=1e-10,
                         u_init=None):
    """
    Return:
      u_star, x_star, history dict
    """
    if u_init is None:
        u = np.zeros(N)
    else:
        u = u_init.astype(float).copy()

    R = R_coeff * np.eye(N)
    H = hessian_lqr(a, b, q, r, p, N)  # constant for LQR

    # Pre-factor (R+H) once (SPD for typical settings)
    A = R + H
    # small reg for numeric safety
    reg = 1e-12 * np.eye(N)
    A_reg = A + reg

    hist_grad_inf = []
    hist_step_inf = []

    for outer in range(max_outer):
        x, grad = gradient_lqr(a, b, q, r, p, x0, u)
        grad_inf = np.linalg.norm(grad, ord=np.inf)
        hist_grad_inf.append(grad_inf)

        if grad_inf < tol:
            break

        # inner recursion g_j
        g_prev = np.linalg.solve(A_reg, grad)  # g0
        g = g_prev
        if outer >= 1:
            for _ in range(outer):  # j=1..outer
                g = np.linalg.solve(A_reg, grad + R @ g_prev)
                g_prev = g

        step_inf = np.linalg.norm(g, ord=np.inf)
        hist_step_inf.append(step_inf)

        u = u - g

    x_final = rollout_lqr(a, b, x0, u)
    history = {
        "grad_inf": np.array(hist_grad_inf),
        "step_inf": np.array(hist_step_inf)
    }
    return u, x_final, history

# ============================================================
# 3) "Exact" solution by discrete Riccati (finite-horizon)
#    Backward:
#      P_N = p
#      P_k = q + a^2 P_{k+1} - (a b P_{k+1})^2 / (r + b^2 P_{k+1})
#    Control:
#      K_k = (r + b^2 P_{k+1})^{-1} b a P_{k+1}
#      u_k = - K_k x_k
# ============================================================

def riccati_exact_lqr(a, b, q, r, p, x0, N):
    P = np.zeros(N + 1)
    P[N] = p
    for k in range(N-1, -1, -1):
        denom = r + (b*b) * P[k+1]
        P[k] = q + (a*a) * P[k+1] - ((a*b*P[k+1])**2) / denom

    x = np.zeros(N + 1)
    u = np.zeros(N)
    x[0] = x0
    for k in range(N):
        denom = r + (b*b) * P[k+1]
        K = (b * a * P[k+1]) / denom
        u[k] = -K * x[k]
        x[k+1] = a * x[k] + b * u[k]
    return u, x

# ============================================================
# 4) Produce Fig.2-like plots: 3 initial conditions
# ============================================================

def make_fig2_like():
    # Parameters (you can match your paper’s LQR demo)
    a = 1.8
    b = 0.9
    q = 1.0
    r = 3.0
    p = 3.0
    N = 15
    R_coeff = 0.1

    x0_list = [1.0, 2.0, 3.0]

    # Store for plotting
    alg_states = []
    alg_controls = []
    ex_states = []
    ex_controls = []

    for x0 in x0_list:
        # Algorithm 1
        u_alg, x_alg, hist = algorithm1_solve_lqr(
            a, b, q, r, p, x0, N,
            R_coeff=R_coeff,
            max_outer=60,
            tol=1e-12
        )

        # "Exact" Riccati
        u_ex, x_ex = riccati_exact_lqr(a, b, q, r, p, x0, N)

        alg_states.append(x_alg)
        alg_controls.append(u_alg)
        ex_states.append(x_ex)
        ex_controls.append(u_ex)

    k_state = np.arange(N + 1)
    k_ctrl = np.arange(N)

    # (a) state
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    colors = ["r", "g", "b"]
    markers = ["o", "s", "x"]
    for i, x0 in enumerate(x0_list):
        plt.plot(k_state, alg_states[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm 1")
        plt.plot(k_state, ex_states[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact solution")
    plt.xlabel("k")
    plt.ylabel("State")
    plt.title("(a) Optimal state")
    plt.grid(True)
    plt.legend(fontsize=8)

    # (b) control
    plt.subplot(1,2,2)
    for i, x0 in enumerate(x0_list):
        plt.plot(k_ctrl, alg_controls[i], color=colors[i], linewidth=2,
                 label=f"Test {i+1}: Algorithm 1")
        plt.plot(k_ctrl, ex_controls[i], linestyle="None", marker=markers[i],
                 color=colors[i], markersize=6,
                 label=f"Test {i+1}: Exact solution")
    plt.xlabel("k")
    plt.ylabel("Control")
    plt.title("(b) Optimal control")
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("lqr_fig2_like.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    make_fig2_like()
