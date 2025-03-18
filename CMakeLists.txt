cmake_minimum_required(VERSION 3.16)
project(svd VERSION 1.0.0 DESCRIPTION "Implementations of SVD" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")

# Поиск зависимостей
find_package(Eigen3 NO_MODULE REQUIRED)
find_package(Qt6 COMPONENTS Widgets Charts REQUIRED)
find_package(LAPACK REQUIRED)

# Попытка найти LAPACKE явно
find_library(LAPACKE_LIBRARY NAMES lapacke)
if(NOT LAPACKE_LIBRARY)
    message(FATAL_ERROR "LAPACKE library not found. Please install liblapacke-dev or equivalent.")
endif()

set(SOURCE_FILES
    src/GolubKahanSVD.cpp
    src/iterative_refinement.cpp
    src/jacobi.cpp
    src/generate_svd.cpp
    src/testing.cpp
)
set(HEADER_FILES
    src/iterative_refinement.h
    src/generate_svd.h
    src/jacobi.h
    src/dqds.h
    src/testing.h
)

add_library(svd ${SOURCE_FILES} ${HEADER_FILES})
target_compile_features(svd PUBLIC cxx_std_20)
target_include_directories(svd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/svd>
)
target_link_libraries(svd PUBLIC Eigen3::Eigen LAPACK::LAPACK ${LAPACKE_LIBRARY})

install(TARGETS svd
    EXPORT svdTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${HEADER_FILES} DESTINATION include/svd)

add_executable(svd_test src/test_main.cpp)
target_compile_features(svd_test PRIVATE cxx_std_20)
target_link_libraries(svd_test PRIVATE svd)
install(TARGETS svd_test RUNTIME DESTINATION bin)

set(GUI_SOURCE_FILES
    src/gui.cpp
    src/main.cpp
)
set(GUI_HEADER_FILES
    src/gui.h
)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(svd_gui ${GUI_SOURCE_FILES} ${GUI_HEADER_FILES})
target_compile_features(svd_gui PRIVATE cxx_std_20)
target_link_libraries(svd_gui PRIVATE svd Qt6::Widgets Qt6::Charts)
target_include_directories(svd_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
install(TARGETS svd_gui RUNTIME DESTINATION bin)

install(EXPORT svdTargets
    FILE svdTargets.cmake
    NAMESPACE svd::
    DESTINATION lib/cmake/svd
)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/svdConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/svd"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/svdConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/svdConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/svdConfigVersion.cmake"
    DESTINATION lib/cmake/svd
)
