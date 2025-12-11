cmake_minimum_required(VERSION 3.10)
project(cxx_simlib)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.11.1  # Pin a specific version for stability
)
# This downloads and adds pybind11 to your project only
FetchContent_MakeAvailable(pybind11)

# Now you can use it normally
pybind11_add_module(cxx_simlib 
    src/fvm_thermal.cpp   
)

pybind11_add_module(pybind_test
    src/test.cpp
)

# Optimization flags (Critical for HPC)
if(MSVC)
    target_compile_options(cxx_simlib PRIVATE /O2)
    target_compile_options(pybind_test PRIVATE /O2)
else()
    target_compile_options(cxx_simlib PRIVATE -O3 -march=native -ffast-math -flto)
    target_compile_options(pybind_test PRIVATE -O3 -march=native -ffast-math -flto)
endif()