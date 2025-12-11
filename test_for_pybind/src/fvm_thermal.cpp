#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Enum for Boundary Condition Types
enum BCType {
    BC_DIRICHLET = 0, // Fixed Temperature
    BC_NEUMANN = 1,   // Fixed Flux (dT/dn = val)
    BC_ROBIN = 2      // Convection (-k dT/dn = h(T - T_inf))
};

struct BCConfig {
    BCType type;
    double value;  // For Dirichlet: Temp; For Neumann: Flux gradient
    double h_conv; // For Robin: Convection coefficient (h/k approx)
    double t_amb;  // For Robin: Ambient Temperature
};

class Heat2D {
private:
    int nx, ny;
    int total_cells;
    double h;            // Grid spacing
    double alpha;        // Thermal diffusivity

    // Boundary Configurations (Left, Right, Bottom, Top)
    BCConfig bc_left, bc_right, bc_bottom, bc_top;

    std::vector<double> T_curr;
    std::vector<double> T_next;

    inline int idx(int x, int y) const { return y * nx + x; }

public:
    Heat2D(int nx_in, int ny_in, double h_in, double alpha_in) 
        : nx(nx_in), ny(ny_in), h(h_in), alpha(alpha_in) {
        total_cells = nx * ny;
        T_curr.resize(total_cells, 0.0);
        T_next.resize(total_cells, 0.0);
        
        // Default to Dirichlet 0.0 everywhere
        BCConfig default_bc = {BC_DIRICHLET, 0.0, 0.0, 0.0};
        bc_left = bc_right = bc_bottom = bc_top = default_bc;
    }

    void set_uniform_field(double T_val) {
        std::fill(T_curr.begin(), T_curr.end(), T_val);
        std::fill(T_next.begin(), T_next.end(), T_val);
    }

    // --- Boundary Condition Setup ---
    void set_bc(std::string side, int type, double val, double h_c = 0.0, double t_a = 0.0) {
        BCConfig cfg = {(BCType)type, val, h_c, t_a};
        if (side == "left") bc_left = cfg;
        else if (side == "right") bc_right = cfg;
        else if (side == "bottom") bc_bottom = cfg;
        else if (side == "top") bc_top = cfg;
    }

    // --- External Heat Source (Operator Splitting) ---
    // Adds Source(x,y) * dt to the current temperature field.
    // Python calculates the shape (e.g. Gaussian), C++ applies it efficiently.
    void add_heat_source(py::array_t<double> source_array, double dt) {
        auto r = source_array.unchecked<2>(); // 2D access without bounds check
        
        // Safety check on shape
        if (r.shape(0) != ny || r.shape(1) != nx) {
            throw std::runtime_error("Source array shape must match grid dimensions");
        }

        // We add energy directly to T_curr before the diffusion step
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                T_curr[idx(x, y)] += r(y, x) * dt;
            }
        }
    }

    // --- Main Solver Step ---
    void step(double dt) {
        double coeff = (alpha * dt) / (h * h);
        
        py::gil_scoped_release release; // Release Python GIL

        // 1. Interior Update (Diffusion)
        for (int y = 1; y < ny - 1; ++y) {
            for (int x = 1; x < nx - 1; ++x) {
                int c = idx(x, y);
                double T_sum = T_curr[idx(x+1, y)] + T_curr[idx(x-1, y)] +
                               T_curr[idx(x, y+1)] + T_curr[idx(x, y-1)];
                T_next[c] = T_curr[c] + coeff * (T_sum - 4.0 * T_curr[c]);
            }
        }

        // 2. Apply Boundary Conditions
        // Note: Corners are handled implicitly by the order of operations or simple averaging.
        // For this demo, we apply Left/Right then Top/Bottom.
        
        auto apply_side = [&](BCConfig& bc, int x, int y, int x_in, int y_in) {
            int c = idx(x, y);
            int c_in = idx(x_in, y_in); // The internal neighbor

            if (bc.type == BC_DIRICHLET) {
                T_next[c] = bc.value;
            } 
            else if (bc.type == BC_NEUMANN) {
                // T_boundary = T_inner + dx * Gradient
                // Gradient is positive if pointing INWARD direction of axis check
                // Simple first-order approx
                T_next[c] = T_next[c_in] + h * bc.value;
            }
            else if (bc.type == BC_ROBIN) {
                // T_b = (T_inner + Bi * T_amb) / (1 + Bi)
                // Bi = h_conv * dx / k (simplified here as just h_conv * dx)
                double Bi = bc.h_conv * h; 
                T_next[c] = (T_next[c_in] + Bi * bc.t_amb) / (1.0 + Bi);
            }
        };

        // Left Edge (x=0, neighbor x=1)
        for (int y = 0; y < ny; ++y) apply_side(bc_left, 0, y, 1, y);
        // Right Edge (x=nx-1, neighbor x=nx-2)
        for (int y = 0; y < ny; ++y) apply_side(bc_right, nx-1, y, nx-2, y);
        // Bottom Edge (y=0, neighbor y=1)
        for (int x = 0; x < nx; ++x) apply_side(bc_bottom, x, 0, x, 1);
        // Top Edge (y=ny-1, neighbor y=ny-2)
        for (int x = 0; x < nx; ++x) apply_side(bc_top, x, ny-1, x, ny-2);

        std::swap(T_curr, T_next);
    }

    py::array_t<double> get_view() {
        return py::array_t<double>(
            { (long)ny, (long)nx },
            { (long)nx*sizeof(double), sizeof(double) },
            T_curr.data(),
            py::cast(this)
        );
    }
};

PYBIND11_MODULE(hpc_sim, m) {
    // Export BC Type Enums
    m.attr("BC_DIRICHLET") = (int)BC_DIRICHLET;
    m.attr("BC_NEUMANN") = (int)BC_NEUMANN;
    m.attr("BC_ROBIN") = (int)BC_ROBIN;

    py::class_<Heat2D>(m, "Heat2D")
        .def(py::init<int, int, double, double>(), 
             py::arg("nx"), py::arg("ny"), py::arg("h"), py::arg("alpha"))
        .def("set_uniform", &Heat2D::set_uniform_field)
        .def("set_bc", &Heat2D::set_bc, 
             py::arg("side"), py::arg("type"), py::arg("val"), py::arg("h_conv")=0.0, py::arg("t_amb")=0.0)
        .def("add_heat_source", &Heat2D::add_heat_source, "Add external source field * dt to T")
        .def("step", &Heat2D::step)
        .def("get_view", &Heat2D::get_view);
}