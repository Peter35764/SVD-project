cmake_minimum_required(VERSION 3.10)

project(svd_iterative_refinement VERSION 1.0.0 DESCRIPTION "SVD algorithm: Iterative Refinement")

# ========== Пользовательские настройки ==========
set(THREADS 20 CACHE STRING "Number of threads to use for testing")
configure_file(${CMAKE_SOURCE_DIR}/src/config.h.in ${CMAKE_BINARY_DIR}/config.h)

# ========== Компилятор и системные флаги ========
find_package(OpenMP COMPONENTS CXX REQUIRED) # Ищем OpenMP только для CXX

# ========== Внешние зависимости ==========
find_package(Eigen3 REQUIRED)
find_package(LAPACKE REQUIRED)
find_package(Boost REQUIRED)
find_package(BLAS REQUIRED) # Убедись, что BLAS находится

if(NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS library not found. Check installation and CMake output.")
endif()

# ========== Пути поиска (Windows/MSYS) ==========
set(CMAKE_PREFIX_PATH "C:/msys64/mingw64")
set(CMAKE_IGNORE_PATH "C:/Users/schuc/anaconda3") # Проверь актуальность

# ========== Определение INTERFACE библиотеки "svd" ==========
add_library(svd INTERFACE)
target_include_directories(svd INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_link_libraries(svd INTERFACE
    Eigen3::Eigen
    lapacke # LAPACKE обычно линкуется с BLAS, но ${BLAS_LIBRARIES} для исполняемого файла надежнее
    Boost::boost
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
target_compile_features(svd INTERFACE cxx_std_20)

# ========== Тесты ==========
# Собираем исполняемый файл из всех частей
add_executable(svd_test_all
    src/testing_all.cpp          # Главный файл с main() и run_precision_study
    src/test_common.cpp          # Файл с определением std::mutex cout_mutex
    src/test_cases_20_24.cpp

    src/test_cases_25_29.cpp
    src/test_cases_30_35.cpp  
    src/test_cases_36_40.cpp
    src/test_cases_41_45.cpp
    src/test_cases_46_50.cpp
    src/test_cases_51_55.cpp
    src/test_cases_56_60.cpp
    src/test_cases_61_65.cpp
    src/test_cases_66_70.cpp
    src/test_cases_71_75.cpp
    src/test_cases_76_80.cpp
    src/test_cases_81_85.cpp
    src/test_cases_86_90.cpp
    src/test_cases_91_95.cpp
    src/test_cases_96_100.cpp  # Или до 101, если нужно
)

target_compile_features(svd_test_all PRIVATE cxx_std_20)

if(OpenMP_CXX_FOUND)
    target_compile_options(svd_test_all PRIVATE ${OpenMP_CXX_FLAGS})
endif()

target_compile_options(svd_test_all PRIVATE
    -Wa,-mbig-obj
    -g0 # Для релизных сборок можно использовать -O2 или -O3
)

target_link_libraries(svd_test_all PRIVATE
    svd             # Твоя интерфейсная библиотека
    ${BLAS_LIBRARIES} # Явная линковка с BLAS
)

# ПРЕДКОМПИЛИРОВАННЫЕ ЗАГОЛОВКИ (PCH) - ОЧЕНЬ РЕКОМЕНДУЕТСЯ!
# PCH будет применяться к компиляции КАЖДОГО из .cpp файлов цели svd_test_all.
target_precompile_headers(svd_test_all PRIVATE
    # Главный общий заголовок, который включает всё остальное:
    "src/test_common.h"

    # Можно также явно перечислить самые "тяжелые" из test_common.h,
    # но включения test_common.h должно быть достаточно, если он хорошо структурирован.
    # <Eigen/Dense>
    # <boost/multiprecision/cpp_dec_float.hpp>
    # <boost/multiprecision/eigen.hpp>
    # <vector>
    # <string>
    # <iostream>
)

# Вывод информации о найденном OpenMP (для отладки)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP CXX Libraries (может быть пустым для IMPORTED целей): ${OpenMP_CXX_LIBRARIES}")
else()
    message(WARNING "OpenMP for CXX not found, parallel features might be disabled.")
endif()
