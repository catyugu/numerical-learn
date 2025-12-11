#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // Essential for HPC data exchange
#include <pybind11/stl.h>   // For std::vector conversion
#include <vector>
#include <cmath>
#include <iostream>

namespace py = pybind11;

// --- 1. Custom Structure (The basic unit) ---
struct Particle {
    double x, y;
    double vx, vy;
    int id;

    Particle(double x, double y, int id) : x(x), y(y), vx(0), vy(0), id(id) {}
};

// --- 2. The HPC Engine ---
class Simulation {
private:
    std::vector<Particle> particles;
    size_t num_particles;

    // We keep a separate raw buffer for positions to show "Zero-Copy" access for plotting
    // (In real HPC, you often separate data structures like "Struct of Arrays" vs "Array of Structs")
    std::vector<double> pos_buffer; 

public:
    Simulation(size_t n) : num_particles(n) {
        pos_buffer.resize(n * 2); // x and y for each particle
        for (size_t i = 0; i < n; ++i) {
            particles.emplace_back(i * 0.1, i * 0.1, i);
            // Initialize randomness or logic here
        }
        sync_buffer();
    }

    // A computational intensive step
    void step(double dt) {
        // CALL_GUARD: Release Python GIL so this runs in true parallel C++ threads if needed
        py::gil_scoped_release release; 
        
        for (size_t i = 0; i < num_particles; ++i) {
            // Fake physics: Simple harmonic motion + drift
            particles[i].x += std::cos(particles[i].y) * dt;
            particles[i].y += std::sin(particles[i].x) * dt;
            
            // Sync to the raw buffer (for visualization)
            pos_buffer[2*i] = particles[i].x;
            pos_buffer[2*i+1] = particles[i].y;
        }
    }

    void sync_buffer() {
        for (size_t i = 0; i < num_particles; ++i) {
            pos_buffer[2*i] = particles[i].x;
            pos_buffer[2*i+1] = particles[i].y;
        }
    }

    // --- 3. The "HPC" Bridge Method ---
    // Returns a NumPy array that VIEWS the C++ memory (no copy!)
    py::array_t<double> get_positions_view() {
        // Create a Python object that points to our C++ vector data
        // Shape: [num_particles, 2]
        return py::array_t<double>(
            { (long)num_particles, 2l }, // Shape
            { 2 * sizeof(double), sizeof(double) }, // Strides (bytes to step)
            pos_buffer.data(), // Pointer to data
            py::cast(this) // Tie lifetime to this Simulation instance
        );
    }
    
    // Allow Python to inspect specific particles (Custom Type handling)
    Particle& get_particle(size_t index) {
        if (index >= num_particles) throw std::out_of_range("Index out of bounds");
        return particles[index];
    }
};

// --- 4. Binding Code ---
PYBIND11_MODULE(hpc_sim, m) {
    // Bind the Custom Struct
    py::class_<Particle>(m, "Particle")
        .def(py::init<double, double, int>())
        .def_readwrite("x", &Particle::x)
        .def_readwrite("y", &Particle::y)
        .def_readwrite("id", &Particle::id);

    // Bind the Engine
    py::class_<Simulation>(m, "Simulation")
        .def(py::init<size_t>())
        .def("step", &Simulation::step, "Run one simulation step (releases GIL)")
        .def("get_particle", &Simulation::get_particle, py::return_value_policy::reference) // Return reference, don't copy!
        .def("get_positions", &Simulation::get_positions_view, "Get zero-copy numpy view");
}