cmake_minimum_required(VERSION 3.10)

project(svd VERSION 1.0.0 DESCRIPTION "Implementations of experimental SVD algorithms")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")

# TODO добавить создание папки генерации сборки и папки для бинарников + переписать инструкцию

# ========== Зависимости: ==========

find_package(Eigen3 REQUIRED)
find_package(lapacke REQUIRED)

# При возникновении ошибок сборки на unix-like системах можно попробовать установку через pkg-config:

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(EIGEN3 REQUIRED eigen3)
# pkg_check_modules(LAPACKE REQUIRED lapacke)

# При этом нужно внести изменение в сборку целевых объектов (строки с подключением библиотек):

# С помощью find-package:
# add_library(svd ${SOURCE_FILES} ${HEADER_FILES}
#   src/givens_refinement.h)
# target_compile_features(svd PRIVATE cxx_std_20)
# target_link_libraries(svd Eigen3::Eigen)
# target_link_libraries(svd lapacke)

# С помощью pkg-config:
# add_library(svd ${SOURCE_FILES} ${HEADER_FILES}
#   src/givens_refinement.h)
# target_compile_features(svd PRIVATE cxx_std_20)
# target_include_directories(svd PUBLIC ${EIGEN3_INCLUDE_DIRS})
# target_link_libraries(svd ${LAPACKE_LIBRARIES})

# ========= Генерация конфиг файла =========

set(THREADS_NUM 1 CACHE STRING "количество потоков")
configure_file(src/config.h.in config.h)
include_directories(${CMAKE_BINARY_DIR})

# ==========================================


# ======= Файлы фреймворка: ========

SET(HEADER_FILES
  src/generate_svd.h
  src/dqds.h
  src/iterative_refinement.h
  src/jacobi.h
  # # Parf&Drozdov
  src/mrrr.h
  src/mrrr.hpp
  # src/givens_refinement.h
  # src/givens_refinement.hpp
)

SET(SOURCE_FILES
  src/generate_svd.cpp
  src/GolubKahanSVD.cpp
  src/iterative_refinement.cpp
  src/jacobi.cpp
  src/testing.cpp
  # # Parf&Drozdov
  src/mrrr.cpp
  # src/givens_refinement.cpp
)

# ==================================


# ========= Цели сборки: ==========

# Библиотека
add_library(svd ${SOURCE_FILES} ${HEADER_FILES})
target_compile_features(svd PRIVATE cxx_std_20)
target_link_libraries(svd Eigen3::Eigen lapacke)

# Сценарий тестирования, описанный в testing.cpp
add_executable(svd_test src/testing.cpp)
target_compile_features(svd_test PRIVATE cxx_std_20)
target_link_libraries(svd_test PRIVATE svd Eigen3::Eigen lapacke)

 # Сценарий тестирования, описанный в givens_refinement.cpp
add_executable(svd_test_mrrr src/mrrr.cpp)
target_compile_features(svd_test_mrrr PRIVATE cxx_std_20)
target_link_libraries(svd_test_mrrr PRIVATE svd Eigen3::Eigen lapacke)

# ==================================

# ========= Цели установки (TODO): ==========
# добавить установку фреймворка как полноценной библиотеки и поддержку вменяемой очистки того, что устанавливает фреймворк
