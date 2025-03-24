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

# ==================================


# ======= Файлы фреймворка: ========

SET(HEADER_FILES
  src/generate_svd.h
  src/dqds.h
  src/iterative_refinement.h
  src/jacobi.h
  # Parf&Drozdov
  src/reverse_jacobi.h
  src/reverse_jacobi.hpp
)

SET(SOURCE_FILES
  src/generate_svd.cpp
  src/GolubKahanSVD.cpp
  src/iterative_refinement.cpp
  src/jacobi.cpp
  src/testing.cpp
  # Parf&Drozdov
  src/reverse_jacobi.cpp
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
add_executable(svd_test_revjac src/reverse_jacobi.cpp)
target_compile_features(svd_test_revjac PRIVATE cxx_std_20)
target_link_libraries(svd_test_revjac PRIVATE svd Eigen3::Eigen lapacke)

# ==================================


# ========= Цели установки (TODO): ==========
# добавить поддержку вменяемой очистки того, что устанавливает фреймворк

install(
  TARGETS svd
  LIBRARY DESTINATION lib
)
install(
  FILES ${HEADER_FILES}
  DESTINATION include/library_name
)

# ==================================
