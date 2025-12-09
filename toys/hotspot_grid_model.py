import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ==========================================
# 1. Physical Constants (Silicon-like block)
# ==========================================
L = 0.001       # Thickness (m)
A = 1e-4        # Area (m^2)
k = 100         # Thermal Conductivity (W/mK)
rho = 2330      # Density (kg/m^3)
cp = 800        # Specific Heat (J/kgK)
Q_in = 10       # Heat Input (Watts)

# Derived Parameters
R_total = L / (k * A)           # Total Thermal Resistance
C_total = L * A * rho * cp      # Total Thermal Capacitance

# ==========================================
# 2. The "Ground Truth" Model (High-Res FDM)
# ==========================================
# We split the block into N small slices to simulate 
# the continuous "Distributed" nature of reality.
N = 50
dx = L / N
C_node = C_total / N
R_node = R_total / N

def high_res_fdm(T, t):
    # T represents temperature at N nodes.
    # Node 0 is the Source side.
    # Node N-1 is connected to the Heat Sink (Fixed at 0).
    
    dTdt = np.zeros(N)
    
    # Node 0 (Source Interface)
    # Heat comes in (Q_in), Heat leaves to Node 1
    # Resistance between centers is R_node
    flux_out = (T[0] - T[1]) / R_node
    dTdt[0] = (Q_in - flux_out) / C_node
    
    # Internal Nodes (1 to N-2)
    for i in range(1, N-1):
        flux_in = (T[i-1] - T[i]) / R_node
        flux_out = (T[i] - T[i+1]) / R_node
        dTdt[i] = (flux_in - flux_out) / C_node
        
    # Node N-1 (Boundary near Sink)
    # Connected to Sink (Temp=0) via half-node resistance
    flux_in = (T[N-2] - T[N-1]) / R_node
    flux_to_sink = (T[N-1] - 0) / (R_node / 2) # Closer to boundary
    dTdt[N-1] = (flux_in - flux_to_sink) / C_node
    
    return dTdt

# ==========================================
# 3. The "Lumped" Models (HotSpot Style)
# ==========================================
# Equation: C_eff * dT/dt = Q_in - (T - T_sink)/R_total

def lumped_model(T, t, scaling_factor):
    C_eff = C_total * scaling_factor
    dTdt = (Q_in - (T - 0)/R_total) / C_eff
    return dTdt

# ==========================================
# 4. Run Simulation
# ==========================================
t = np.linspace(0, 0.05, 1000) # Simulate for 0.5 seconds
T_init_fdm = np.zeros(N)
T_init_lumped = 0.0

# Solve High Res
sol_fdm = odeint(high_res_fdm, T_init_fdm, t)
T_surface_truth = sol_fdm[:, 0] # We only care about the Source temperature

# Solve Lumped Models
sol_lump_half = odeint(lumped_model, T_init_lumped, t, args=(1.0/2.0,))
sol_lump_third = odeint(lumped_model, T_init_lumped, t, args=(1.0/3.0,))

# ==========================================
# 5. Plotting
# ==========================================
plt.figure(figsize=(10, 6))

plt.plot(t, T_surface_truth, 'k-', linewidth=3, label='Ground Truth (FDM N=50)')
plt.plot(t, sol_lump_half, 'b--', label='Lumped (Factor 1/2) - Old HotSpot')
plt.plot(t, sol_lump_third, 'r--', linewidth=2, label='Lumped (Factor 1/3) - New HotSpot')

plt.title("Step Response: Distributed vs Lumped Capacitance")
plt.xlabel("Time (s)")
plt.ylabel("Temperature Rise (K)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()