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
    py::class_<Pet>(m, "Pet", py::dynamic_attr()) 
    // (Only with dynamic_attr will it be possible to add new attributes in python code)
        .def(py::init<const std::string &>())
        .def_readwrite("name", &Pet::name) // To allow direct access to the attribute
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName)
        .def("__repr__",
        [](const Pet &a) {
            return "<example.Pet named '" + a.name + "'>";
        }   // Only with this will the print(pet) display some info
    );
}

