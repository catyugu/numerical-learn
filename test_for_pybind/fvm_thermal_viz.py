import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure the build directory is in path (modify if needed)
import build.cxx_simlib as cxx_simlib

# --- Simulation Parameters ---
NX, NY = 100, 100    # Grid size
H = 0.01             # Spacing (meters)
ALPHA = 0.02          # Thermal diffusivity
DT_LIMIT = (H*H) / (4.0 * ALPHA)  # Stability limit for explicit scheme
DT = DT_LIMIT * 0.8  # Safe time step

print(f"Grid: {NX}x{NY} | DT: {DT:.6f}s")

# --- Initialize Solver ---
solver = cxx_simlib.Heat2D(NX, NY, H, ALPHA)
solver.set_uniform(25.0) # Initial Ambient Temp 25C

# --- Configure Advanced BCs ---
# 1. LEFT: Dirichlet (Fixed Hot Wall)
solver.set_bc("left", cxx_simlib.BC_DIRICHLET, 60.0)

# 2. RIGHT: Robin (Convective Cooling)
solver.set_bc("right", cxx_simlib.BC_ROBIN, 0.0, h_conv=50.0, t_amb=20.0)

# 3. TOP: Neumann (Insulated / Zero Flux)
solver.set_bc("top", cxx_simlib.BC_NEUMANN, 0.0) 

# 4. BOTTOM: Neumann (Heat Flux In)
solver.set_bc("bottom", cxx_simlib.BC_NEUMANN, 50.0) 

# --- Source Term Utility ---
# Create meshgrid for Gaussian calculation
x_coords = np.linspace(0, NX*H, NX)
y_coords = np.linspace(0, NY*H, NY)
X, Y = np.meshgrid(x_coords, y_coords)

def get_gaussian_source(center_x, center_y, power, width=0.1):
    """
    Returns a 2D array representing a Gaussian heat source.
    power: Intensity (K/s)
    width: Spread in meters
    """
    dist_sq = (X - center_x)**2 + (Y - center_y)**2
    source = power * np.exp(-dist_sq / (2 * width**2))
    return source

# --- Visualization Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(9, 8))

data_view = solver.get_view()
img = ax.imshow(data_view, cmap='magma', origin='lower', 
                vmin=20.0, vmax=100.0, extent=[0, NX*H, 0, NY*H])

plt.colorbar(img, label="Temperature (C)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")

# --- Text Annotations for BCs ---
ax.text(0.02, 0.5, "DIRICHLET\n(60 C)", transform=ax.transAxes, color='white', 
        ha='left', va='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
ax.text(0.98, 0.5, "ROBIN\n(Conv -> 20C)", transform=ax.transAxes, color='white', 
        ha='right', va='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
ax.text(0.5, 0.98, "NEUMANN (Insulated)", transform=ax.transAxes, color='white', 
        ha='center', va='top', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
ax.text(0.5, 0.02, "NEUMANN (Flux In)", transform=ax.transAxes, color='white', 
        ha='center', va='bottom', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

# --- Main Loop ---
print("Running... Close plot window to stop.")
steps_per_frame = 10
sim_time = 0.0
frame_count = 0

try:
    start_time = time.time()

    while plt.fignum_exists(fig.number):
        
        # 1. Update Physics
        for _ in range(steps_per_frame):
            sim_time += DT
            
            # --- Dynamic Heat Source ---
            # Move source in a circle

            cx = (NX*H)/2
            cy = (NY*H)/2   
            power_t = 200.0 * np.exp(-((sim_time - 0.5)**2) / 2.0)  # Peak at t=0.5s
            
            # Generate Gaussian field (Power=500 K/s peak)
            source_field = get_gaussian_source(cx, cy, power=power_t, width=0.2)
            
            # Apply Source -> Step Diffusion
            solver.add_heat_source(source_field, DT)
            solver.step(DT)

        # 2. Update Viz
        img.set_data(data_view)
    
        
        ax.set_title(f"Time: {sim_time:.3f}s | BCs & Time Varying Gaussian Source")
        plt.pause(0.001)
        
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"FPS: {frame_count / elapsed:.1f}")

except KeyboardInterrupt:
    print("Stopped.")

plt.ioff()
plt.show()