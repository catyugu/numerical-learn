#include <pybind11/pybind11.h>

namespace py = pybind11;
int add(int i, int j) {
    return i + j;
}

struct Pet {
    Pet(const std::string &name) : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
};

PYBIND11_MODULE(pybind_test, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function that adds two numbers");
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;

    // If you like OOP!
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}

