cmake_minimum_required(VERSION 3.10)
project(hpc_sim)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.11.1  # Pin a specific version for stability
)
# This downloads and adds pybind11 to your project only
FetchContent_MakeAvailable(pybind11)

# Now you can use it normally
pybind11_add_module(hpc_sim 
    src/fvm_thermal.cpp   
)

# Optimization flags (Critical for HPC)
if(MSVC)
    target_compile_options(hpc_sim PRIVATE /O2)
else()
    target_compile_options(hpc_sim PRIVATE -O3 -march=native -ffast-math -flto)
endif()