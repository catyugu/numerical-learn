'''
In this program, we solve the heat equation using finite difference method.
The heat equation is:
    u_t = u_xx
where u is the temperature, x is the position, t is time.
We then show that the explicit forward Euler method (or other explicit schemes) 
converges if and only if the time step size is small enough.
In this case, to say, dt < 1/2 * dx^2.
Even a slightly larger time step size will cause the solution to diverge.
Here we denote dx^2 / dt as mu.
'''

import numpy as np
import matplotlib.pyplot as plt
import tqdm

N_POINTS = 51
LENGTH = 1.0
T = 1.0
dx = LENGTH / (N_POINTS - 1)

def exact_solution(x, t):
    return np.exp(- np.pi * np.pi * t) * np.sin(np.pi * x)
    

# Forward Euler:This is conditionally stable/convergent.
def solve_forward_euler(mu = 0.5):
    dt = mu * dx**2
    n_steps = int(T / dt)
    u = np.zeros(N_POINTS)
    # Initial Condition: u(x) = sin(pi * x)
    u[1:-1] = np.sin(np.pi * np.linspace(0, LENGTH, N_POINTS - 2))
    u[0] = u[1] = 0
    for _ in tqdm.tqdm(range(n_steps)):
        u_next = np.zeros(N_POINTS)
        u_next[1:-1] = mu * (u[0:-2] + u[2:] - 2 * u[1:-1]) + u[1:-1]
        u = u_next
    return u

# Backward Euler: This is unconditionally stable.
def solve_backward_euler(mu = 0.5):
    dt = mu * dx**2
    n_steps = int(T / dt)
    u = np.zeros(N_POINTS)
    # Initial Condition: u(x) = sin(pi * x)
    u[1:-1] = np.sin(np.pi * np.linspace(0, LENGTH, N_POINTS - 2))
    u[0] = u[-1] = 0
    A = np.zeros((N_POINTS - 2, N_POINTS - 2))
    for i in range(N_POINTS - 2):
        A[i, i] = 1 + 2 * mu
        if i > 0:
            A[i, i - 1] = -mu
        if i < N_POINTS - 3:
            A[i, i + 1] = -mu
    A_inv = np.linalg.inv(A)
    for _ in tqdm.tqdm(range(n_steps)):
        b = u[1:-1]
        u_inner = A_inv @ b
        u[1:-1] = u_inner
    return u

# Crank-Nicolson: This is unconditionally stable. (semi-implicit)
def solve_crank_nicolson(mu = 0.5):
    dt = mu * dx**2
    n_steps = int(T / dt)
    u = np.zeros(N_POINTS)
    # Initial Condition: u(x) = sin(pi * x)
    u[1:-1] = np.sin(np.pi * np.linspace(0, LENGTH, N_POINTS - 2))
    u[0] = u[-1] = 0
    A = np.zeros((N_POINTS - 2, N_POINTS - 2))
    B = np.zeros((N_POINTS - 2, N_POINTS - 2))
    for i in range(N_POINTS - 2):
        A[i, i] = 1 + mu
        B[i, i] = 1 - mu
        if i > 0:
            A[i, i - 1] = -mu / 2
            B[i, i - 1] = mu / 2
        if i < N_POINTS - 3:
            A[i, i + 1] = -mu / 2
            B[i, i + 1] = mu / 2
    A_inv = np.linalg.inv(A)
    for _ in tqdm.tqdm(range(n_steps)):
        b = B @ u[1:-1]
        u_inner = A_inv @ b
        u[1:-1] = u_inner
    return u


if __name__ == "__main__":
    u_fe_converge1 = solve_forward_euler(mu=0.25)
    u_fe_divergent = solve_forward_euler(mu=0.51)
    u_be_1 = solve_backward_euler(mu=0.25)
    u_be_2 = solve_backward_euler(mu=1.0)
    u_cn_1 = solve_crank_nicolson(mu=0.25)
    u_cn_2 = solve_crank_nicolson(mu=1.0)
    x = np.linspace(0, LENGTH, N_POINTS)
    u_exact = exact_solution(x, T)

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    plt.plot(x, u_exact, 'r-', label="Exact Solution", linewidth=2)
    plt.plot(x, u_fe_converge1, 'g--', label=f"FE Convergent (mu = 0.25)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x, u_fe_divergent, ':', label=f"FE Divergent (mu = 0.51)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(x, u_exact, 'r-', label="Exact Solution", linewidth=2)
    plt.plot(x, u_be_1, 'g--', label=f"BDF Convergent (mu = 0.25)", linewidth=2)
    plt.plot(x, u_be_2, 'b:', label=f"BDF Convergent (mu = 1.0)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(x, u_exact, 'r-', label="Exact Solution", linewidth=2)
    plt.plot(x, u_cn_1, 'g--', label=f"CN Convergent (mu = 0.25)", linewidth=2)
    plt.plot(x, u_cn_2, 'b:', label=f"CN Convergent (mu = 1.0)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    
    plt.show()