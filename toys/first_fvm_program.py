import numpy as np
import matplotlib.pyplot as plt
import tqdm

N_POINTS = 41
DOMAIN_SIZE = 1.0
N_ITERATIONS = 500
TIME_STEP_LENGTH = 0.001
KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    
    X, Y = np.meshgrid(x, y)
    
    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)
    
    maximum_timestep_length = 0.25 * element_length**2 / KINEMATIC_VISCOSITY
    if TIME_STEP_LENGTH > maximum_timestep_length:
        raise RuntimeError(f"TIME_STEP_LENGTH is too large, The maximum time step length is {maximum_timestep_length}")
    
    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * element_length)
        return diff
    
    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * element_length)
        return diff
    
    def laplacian(f):
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1] - 4 * f[1:-1, 1:-1]) / (element_length ** 2)
        return lap
    
    for _ in tqdm.trange(N_ITERATIONS):
        du_prev_dx = central_difference_x(u_prev)
        du_prev_dy = central_difference_y(u_prev)
        dv_prev_dx = central_difference_x(v_prev)
        dv_prev_dy = central_difference_y(v_prev)
        laplacian_u_prev = laplacian(u_prev)
        laplacian_v_prev = laplacian(v_prev)
        
        u_tent = u_prev + TIME_STEP_LENGTH * (
            - (u_prev * du_prev_dx + v_prev * du_prev_dy)
            + KINEMATIC_VISCOSITY * laplacian_u_prev
        )
        
        v_tent = v_prev + TIME_STEP_LENGTH * (
            - (u_prev * dv_prev_dx + v_prev * dv_prev_dy)
            + KINEMATIC_VISCOSITY * laplacian_v_prev
        )
        
        # BC: Homogeneous Dirichlet for u and v except for the top, where we have a horizontal velocity prescribed
        u_tent[0, :] = 0.0
        u_tent[:, 0] = 0.0
        u_tent[:, -1] = 0.0
        u_tent[-1, :] = HORIZONTAL_VELOCITY_TOP
        
        v_tent[0, :] = 0.0
        v_tent[-1, :] = 0.0
        v_tent[:, 0] = 0.0
        v_tent[:, -1] = 0.0
        
        du_tent_dx = central_difference_x(u_tent)
        dv_tent_dy = central_difference_y(v_tent)
        
        # Compute a pressure correction by solving the Poisson equation
        rhs = DENSITY * (du_tent_dx + dv_tent_dy) / TIME_STEP_LENGTH
        
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (
                p_prev[1:-1, 2:] + p_prev[1:-1, :-2] + p_prev[2:, 1:-1] + p_prev[:-2, 1:-1] - element_length**2 * rhs[1:-1, 1:-1]
            )
        
            # Pressure BC: Homogeneous Neumann (zero-gradient) on all sides
            p_next[:, -1] = p_next[:, -2]
            p_next[:, 0] = p_next[:, 1]
            p_next[0, :] = p_next[1, :]
            p_next[-1, :] = 0 # For the top, we have Dirichlet BC
            
            p_prev = p_next
        
        dp_next_dx = central_difference_x(p_next)
        dp_next_dy = central_difference_y(p_next)
            
        # Correct the velocities so that the fluid is incompressible
        u_next = u_tent - TIME_STEP_LENGTH * dp_next_dx / DENSITY
        v_next = v_tent - TIME_STEP_LENGTH * dp_next_dy / DENSITY
        
        # BC: Homogeneous Dirichlet for u and v except for the top, where we have a horizontal velocity prescribed
        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = HORIZONTAL_VELOCITY_TOP
    
        v_next[0, :] = 0.0
        v_next[-1, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        
        # Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next
        
    plt.figure()
    plt.contourf(X, Y, p_next)
    plt.colorbar()
    
    plt.streamplot(X, Y, u_next, v_next, color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Velocity field')
    plt.show()
        
if __name__ == "__main__":
    main()