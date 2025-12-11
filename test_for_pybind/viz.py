import build.hpc_sim as hpc_sim
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Initialize C++ Backend
num_particles = 1000
sim = hpc_sim.Simulation(num_particles)

# 2. Access Custom Structures
p0 = sim.get_particle(0)
print(f"Initial particle 0 status: ID={p0.id}, X={p0.x}")

# 3. Setup Real-time Visualization
plt.ion() # Interactive mode
fig, ax = plt.subplots()
scatter = ax.scatter([], [])
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

print("Starting Simulation...")
start_time = time.time()

for i in range(1000):
    # RUN COMPUTE IN C++
    sim.step(0.1) 

    # FAST DATA ACCESS
    # This calls 'get_positions_view'. 
    # It does NOT copy the array. 'data' is a view into C++ memory.
    data = sim.get_positions() 
    
    # Update Plot
    scatter.set_offsets(data)
    fig.canvas.draw()
    fig.canvas.flush_events()

print(f"Simulation finished. Final particle 0 X: {p0.x}")