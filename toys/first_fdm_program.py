'''
In this program, we solve the heat equation using finite difference method.
The heat equation is:
    u_t = u_xx
where u is the temperature, x is the position, t is time.
We then show that the solution converges if and only if the time step size is small enough.
In this case, to say, dt < 1/2 * dx^2.
Even a slightly larger time step size will cause the solution to diverge.
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
    
def solve_fdm(miu = 0.5):
    dt = miu * dx**2
    n_steps = int(T / dt)
    u = np.zeros(N_POINTS)
    # Initial Condition: u(x) = sin(pi * x)
    u[1:-1] = np.sin(np.pi * np.linspace(0, LENGTH, N_POINTS - 2))
    u[0] = u[1] = 0
    for _ in tqdm.tqdm(range(n_steps)):
        u_next = np.zeros(N_POINTS)
        u_next[1:-1] = miu * (u[0:-2] + u[2:] - 2 * u[1:-1]) + u[1:-1]
        u = u_next
    return u

if __name__ == "__main__":
    u_converge1= solve_fdm(miu=0.25)
    u_converge2 = solve_fdm(miu=0.5)
    u_divergent = solve_fdm(miu=0.51)
    x = np.linspace(0, LENGTH, N_POINTS)
    u_exact = exact_solution(x, T)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, u_exact, 'r-', label="Exact Solution", linewidth=2)
    plt.plot(x, u_converge1, 'g--', label=f"FDM Convergent (miu = 0.25)", linewidth=2)
    plt.plot(x, u_converge2, 'y.', label=f"FDM Convergent (miu = 0.5)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, u_divergent, ':', label=f"FDM Divergent (miu = 0.501)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    plt.show()