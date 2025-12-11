#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

enum BCType { BC_DIRICHLET = 0, BC_NEUMANN = 1, BC_ROBIN = 2 };

struct BCConfig {
    BCType type;
    double value;  // Dirichlet: Temp; Neumann: Flux Gradient
    double h_conv; // Robin: Convection Coeff
    double t_amb;  // Robin: Ambient Temp
};

class Heat2D {
private:
    int nx, ny;
    double h_grid;      
    double alpha;
    double coeff; // Precomputed: alpha * dt / h^2

    // Flat buffers
    std::vector<double> T_curr;
    std::vector<double> T_next;

    // Boundary Configurations
    BCConfig bc_left, bc_right, bc_bottom, bc_top;

public:
    Heat2D(int nx_in, int ny_in, double h_in, double alpha_in) 
        : nx(nx_in), ny(ny_in), h_grid(h_in), alpha(alpha_in) {
        T_curr.resize(nx * ny, 0.0);
        T_next.resize(nx * ny, 0.0);
        
        // Default safe BCs
        BCConfig def = {BC_DIRICHLET, 0.0, 0.0, 0.0};
        bc_left = bc_right = bc_bottom = bc_top = def;
    }

    void set_uniform_field(double T_val) {
        std::fill(T_curr.begin(), T_curr.end(), T_val);
        std::fill(T_next.begin(), T_next.end(), T_val);
    }

    void set_bc(std::string side, int type, double val, double h_c = 0.0, double t_a = 0.0) {
        BCConfig cfg = {(BCType)type, val, h_c, t_a};
        if (side == "left") bc_left = cfg;
        else if (side == "right") bc_right = cfg;
        else if (side == "bottom") bc_bottom = cfg;
        else if (side == "top") bc_top = cfg;
    }

    void add_heat_source(py::array_t<double> source_array, double dt) {
        auto r = source_array.unchecked<2>(); 
        double* ptr = T_curr.data();
        for (int i = 0; i < nx * ny; ++i) {
             // Flat iteration is faster for source addition
             // Note: We assume source_array is Row-Major and contiguous matches T_curr
             ptr[i] += r.data(0,0)[i] * dt;
        }
    }

    // --- Helper: Calculate 1D Flux Term from a Boundary ---
    // Returns the contribution to the temperature change: (alpha * dt / h) * Flux_Gradient
    inline double get_bc_flux_term(const BCConfig& bc, double T_cell) {
        if (bc.type == BC_DIRICHLET) {
            // Flux = k * (T_wall - T_cell) / (h/2)
            // Term = (alpha*dt/h) * [ (T_w - T_c) / (h/2) ]
            //      = (alpha*dt/h^2) * 2 * (T_w - T_c)
            //      = coeff * 2 * (T_w - T_c)
            return 2.0 * coeff * (bc.value - T_cell);
        }
        else if (bc.type == BC_NEUMANN) {
            // Flux Gradient is given directly as bc.value
            // Term = coeff * h * value
            return coeff * h_grid * bc.value;
        }
        else { // ROBIN
            // q = h_conv * (T_amb - T_wall) = k * (T_wall - T_cell) / (h/2)
            // Solve for T_wall, then compute flux.
            // Simplified approximation for explicit stability:
            // Flux ~ h_conv * (T_amb - T_cell)
            // This is a first-order approximation sufficient for viz.
            // Term = (alpha*dt/h) * (h_conv/k) * (T_amb - T_cell)
            // Note: bc.h_conv in python was passed as h_conv/k approx
            return coeff * h_grid * bc.h_conv * (bc.t_amb - T_cell); 
        }
    }

    void step(double dt) {
        coeff = (alpha * dt) / (h_grid * h_grid);
        
        // Raw pointers for speed (replaces std::vector access)
        const double* __restrict__ p_curr = T_curr.data();
        double* __restrict__ p_next = T_next.data();

        py::gil_scoped_release release;

        // ---------------------------------------------------------
        // 1. CORE INTERIOR LOOP (Optimized for Vectorization)
        // Range: x=[1, nx-2], y=[1, ny-2]
        // ---------------------------------------------------------
        for (int y = 1; y < ny - 1; ++y) {
            // Pre-calculate row offsets to avoid multiplication in inner loop
            int row_offset = y * nx;
            int prev_row = (y - 1) * nx;
            int next_row = (y + 1) * nx;

            for (int x = 1; x < nx - 1; ++x) {
                int c = row_offset + x;
                
                // Standard 5-point stencil
                // T_new = T + coeff * (Left + Right + Up + Down - 4*T)
                double term = p_curr[c - 1] +      // Left
                              p_curr[c + 1] +      // Right
                              p_curr[prev_row + x] + // Bottom
                              p_curr[next_row + x] - // Top
                              4.0 * p_curr[c];
                
                p_next[c] = p_curr[c] + coeff * term;
            }
        }

        // ---------------------------------------------------------
        // 2. BOUNDARY HANDLING (Exact Flux Application)
        // We process edges separately to avoid 'if' statements in the core loop.
        // ---------------------------------------------------------

        // A. Left Edge (x=0) & Right Edge (x=nx-1) for interior Y
        for (int y = 1; y < ny - 1; ++y) {
            int c_left = y * nx;
            int c_right = y * nx + (nx - 1);

            // Left Node: Has Right Neighbor (internal) + Left BC
            double flux_E = coeff * (p_curr[c_left + 1] - p_curr[c_left]);
            double flux_W = get_bc_flux_term(bc_left, p_curr[c_left]); 
            double flux_N = coeff * (p_curr[c_left + nx] - p_curr[c_left]);
            double flux_S = coeff * (p_curr[c_left - nx] - p_curr[c_left]);
            p_next[c_left] = p_curr[c_left] + flux_E + flux_W + flux_N + flux_S;

            // Right Node: Has Left Neighbor (internal) + Right BC
            double flux_W_r = coeff * (p_curr[c_right - 1] - p_curr[c_right]);
            double flux_E_r = get_bc_flux_term(bc_right, p_curr[c_right]);
            double flux_N_r = coeff * (p_curr[c_right + nx] - p_curr[c_right]);
            double flux_S_r = coeff * (p_curr[c_right - nx] - p_curr[c_right]);
            p_next[c_right] = p_curr[c_right] + flux_W_r + flux_E_r + flux_N_r + flux_S_r;
        }

        // B. Bottom Edge (y=0) & Top Edge (y=ny-1) for interior X
        for (int x = 1; x < nx - 1; ++x) {
            int c_bot = x;
            int c_top = (ny - 1) * nx + x;

            // Bottom Node: Has Top Neighbor (internal) + Bottom BC
            double flux_N = coeff * (p_curr[c_bot + nx] - p_curr[c_bot]);
            double flux_S = get_bc_flux_term(bc_bottom, p_curr[c_bot]);
            double flux_E = coeff * (p_curr[c_bot + 1] - p_curr[c_bot]);
            double flux_W = coeff * (p_curr[c_bot - 1] - p_curr[c_bot]);
            p_next[c_bot] = p_curr[c_bot] + flux_N + flux_S + flux_E + flux_W;

            // Top Node: Has Bottom Neighbor (internal) + Top BC
            double flux_S_t = coeff * (p_curr[c_top - nx] - p_curr[c_top]);
            double flux_N_t = get_bc_flux_term(bc_top, p_curr[c_top]);
            double flux_E_t = coeff * (p_curr[c_top + 1] - p_curr[c_top]);
            double flux_W_t = coeff * (p_curr[c_top - 1] - p_curr[c_top]);
            p_next[c_top] = p_curr[c_top] + flux_S_t + flux_N_t + flux_E_t + flux_W_t;
        }

        // C. Corners (Simplify: Average of adjacent BC fluxes)
        // Example: Bottom-Left (0,0)
        {
            int c = 0;
            double f_E = coeff * (p_curr[c+1] - p_curr[c]);
            double f_N = coeff * (p_curr[c+nx] - p_curr[c]);
            double f_W = get_bc_flux_term(bc_left, p_curr[c]);
            double f_S = get_bc_flux_term(bc_bottom, p_curr[c]);
            p_next[c] = p_curr[c] + f_E + f_N + f_W + f_S;
        }
        // Bottom-Right
        {
            int c = nx - 1;
            double f_W = coeff * (p_curr[c-1] - p_curr[c]);
            double f_N = coeff * (p_curr[c+nx] - p_curr[c]);
            double f_E = get_bc_flux_term(bc_right, p_curr[c]);
            double f_S = get_bc_flux_term(bc_bottom, p_curr[c]);
            p_next[c] = p_curr[c] + f_W + f_N + f_E + f_S;
        }
        // Top-Left
        {
            int c = (ny - 1) * nx;
            double f_E = coeff * (p_curr[c+1] - p_curr[c]);
            double f_S = coeff * (p_curr[c-nx] - p_curr[c]);
            double f_W = get_bc_flux_term(bc_left, p_curr[c]);
            double f_N = get_bc_flux_term(bc_top, p_curr[c]);
            p_next[c] = p_curr[c] + f_E + f_S + f_W + f_N;
        }
        // Top-Right
        {
            int c = (ny - 1) * nx + nx - 1;
            double f_W = coeff * (p_curr[c-1] - p_curr[c]);
            double f_S = coeff * (p_curr[c-nx] - p_curr[c]);
            double f_E = get_bc_flux_term(bc_right, p_curr[c]);
            double f_N = get_bc_flux_term(bc_top, p_curr[c]);
            p_next[c] = p_curr[c] + f_W + f_S + f_E + f_N;
        }

        // Swap buffers
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
    m.attr("BC_DIRICHLET") = (int)BC_DIRICHLET;
    m.attr("BC_NEUMANN") = (int)BC_NEUMANN;
    m.attr("BC_ROBIN") = (int)BC_ROBIN;

    py::class_<Heat2D>(m, "Heat2D")
        .def(py::init<int, int, double, double>(), 
             py::arg("nx"), py::arg("ny"), py::arg("h"), py::arg("alpha"))
        .def("set_uniform", &Heat2D::set_uniform_field)
        .def("set_bc", &Heat2D::set_bc, 
             py::arg("side"), py::arg("type"), py::arg("val"), py::arg("h_conv")=0.0, py::arg("t_amb")=0.0)
        .def("add_heat_source", &Heat2D::add_heat_source)
        .def("step", &Heat2D::step)
        .def("get_view", &Heat2D::get_view);
}