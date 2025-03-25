cmake_minimum_required(VERSION 3.12)
# Установка количества потоков по умолчанию
if(NOT CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL 4)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")

find_package(Eigen3 NO_MODULE)

project(svd VERSION 1.0.0 DESCRIPTION "Implementations of SVD")

SET(SOURCE_FILES
  src/GolubKahanSVD.cpp
  src/iterative_refinement.cpp
  src/jacobi.cpp
  src/generate_svd.cpp
  src/mrrr.cpp
)
SET(HEADER_FILES
  src/iterative_refinement.h
  src/generate_svd.h
  src/jacobi.h
  src/dqds.h
  src/mrrr.h
  src/mrrr.hpp
)

# Основная библиотека
add_library(svd ${SOURCE_FILES} ${HEADER_FILES}) 
target_compile_features(svd PRIVATE cxx_std_20)
target_include_directories(svd PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(svd Eigen3::Eigen lapacke)

install(
  TARGETS svd
  LIBRARY DESTINATION lib
)
install(
  FILES ${HEADER_FILES} 
  DESTINATION include/svd
)

# Основной тестовый исполняемый файл
add_executable(svd_test src/testing.cpp)
target_compile_features(svd_test PRIVATE cxx_std_20)
target_link_libraries(svd_test svd Eigen3::Eigen lapacke)

# Отдельный тест для MRRR
add_executable(mrrr_test src/mrrr.cpp)
target_compile_features(mrrr_test PRIVATE cxx_std_20)
target_link_libraries(mrrr_test svd Eigen3::Eigen lapacke)

# Опционально: добавить тестирование через CTest
enable_testing()
add_test(NAME svd_test COMMAND svd_test)
add_test(NAME mrrr_test COMMAND mrrr_test)
