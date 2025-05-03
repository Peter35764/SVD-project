# TODO добавить создание папки генерации сборки и папки для бинарников + переписать инструкцию

cmake_minimum_required(VERSION 3.10)

project(svd VERSION 1.0.0 DESCRIPTION "Implementations of experimental SVD algorithms")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")


# ========== Конфигурация сборки ===========

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THREADS 4 CACHE STRING "количество потоков") # "количество потоков при запуске тестов (один тест в один поток, сами тесты (пока) не параллельны)

configure_file(src/config.h.in config.h)
include_directories(${CMAKE_BINARY_DIR})

# ==========================================


# ============== Зависимости: ==============

find_package(Eigen3 REQUIRED)
find_package(lapacke REQUIRED)
find_package(Boost REQUIRED)

# При возникновении ошибок сборки на unix-like системах можно попробовать установку через pkg-config:

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(EIGEN3 REQUIRED eigen3)
# pkg_check_modules(LAPACKE REQUIRED lapacke)

# При этом нужно внести изменение в сборку целевых объектов:

# add_library(svd ${SOURCE_FILES} ${HEADER_FILES}
#   src/givens_refinement.h)
# target_compile_features(svd PRIVATE cxx_std_20)
# target_include_directories(svd PUBLIC ${EIGEN3_INCLUDE_DIRS})
# target_link_libraries(svd ${LAPACKE_LIBRARIES})

# ==========================================


# =========== Файлы фреймворка: ============

SET(HEADER_FILES
  # Tests
  src/lib/tests/generate_svd.h
  src/lib/tests/SVD_Test.h
  src/lib/tests/SVD_Test.hpp
  src/lib/tests/SVD_Test_config.h

  # Parf&Drozdov
  src/lib/idea1_givref/givens_refinement.h
  src/lib/idea1_givref/givens_refinement.hpp
  src/lib/idea2_revjac/reverse_jacobi.h
  src/lib/idea2_revjac/reverse_jacobi.hpp
  src/lib/idea3_naive/naive_mrrr.h
  src/lib/idea3_naive/naive_mrrr.hpp

  # Legacy
  src/lib/idea1_givref/legacy/v0_givens_refinement.h
  src/lib/idea1_givref/legacy/v0_givens_refinement.hpp
  src/lib/idea2_revjac/legacy/v0_reverse_jacobi.h
  src/lib/idea2_revjac/legacy/v0_reverse_jacobi.hpp
  src/lib/idea3_naive/legacy/v0_naive_mrrr.h
  src/lib/idea3_naive/legacy/v0_naive_mrrr.hpp

  # Umbrella header
  src/lib/SVD_project.h
)

SET(SOURCE_FILES
  # Tests run
  src/testing.cpp

  # Parf&Drozdov
  src/givens_refinement.cpp
  src/reverse_jacobi.cpp
  src/naive_mrrr.cpp
)

# ==========================================


# ============= Цели сборки: ===============

# Библиотека
add_library(svd ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(svd Eigen3::Eigen lapacke Boost::boost)

# Основные тесты  (testing.cpp)
add_executable(svd_test src/testing.cpp)
target_link_libraries(svd_test PRIVATE svd Eigen3::Eigen lapacke)

# Сценарий тестирования, описанный в givens_refinement.cpp  (idea 1)
add_executable(svd_test_givref src/givens_refinement.cpp)
target_link_libraries(svd_test_givref PRIVATE svd Eigen3::Eigen lapacke)

# Сценарий тестирования, описанный в reverse_jacobi.cpp  (idea 2)
add_executable(svd_test_revjac src/reverse_jacobi.cpp)
target_link_libraries(svd_test_revjac PRIVATE svd Eigen3::Eigen lapacke)

 # Сценарий тестирования, описанный в mrrr.cpp  (idea 3)
add_executable(svd_test_mrrr src/naive_mrrr.cpp)
target_link_libraries(svd_test_mrrr PRIVATE svd Eigen3::Eigen lapacke)

# ==========================================


# ======== Цели установки (TODO): ==========
# добавить установку фреймворка как библиотеки и поддержку очистки того, что устанавливает фреймворк
