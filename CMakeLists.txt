cmake_minimum_required(VERSION 3.12)

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
  # # Parf&Drozdov
  # src/givens_refinement.h
  # src/givens_refinement.hpp
)

SET(SOURCE_FILES
  src/generate_svd.cpp
  src/GolubKahanSVD.cpp
  src/iterative_refinement.cpp
  src/jacobi.cpp


install(
  TARGETS svd
  LIBRARY DESTINATION lib
)
install(

