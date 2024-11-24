cmake_minimum_required(VERSION 3.14)
project(ex1-with-cmake C)

find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
pkg_check_modules(PETSC_PKG REQUIRED IMPORTED_TARGET petsc)
pkg_check_modules(PETSC_PKG REQUIRED IMPORTED_TARGET slepc)
pkg_check_modules(PETSC_PKG REQUIRED IMPORTED_TARGET PETSc)
pkg_check_modules(SLEPC_PKG REQUIRED IMPORTED_TARGET SLEPc)

add_executable(slepc_example slepc_example.c)

if(MPI_COMPILE_FLAGS)
  set_target_properties(slepc_example PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(slepc_example PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(slepc_example PUBLIC
        ${MPI_LIBRARIES}
        PkgConfig::PETSC_PKG
        PkgConfig::SLEPC_PKG)
