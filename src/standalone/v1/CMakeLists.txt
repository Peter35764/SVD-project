cmake_minimum_required(VERSION 3.10)

project(svd_iterative_refinement VERSION 1.0.0 DESCRIPTION "SVD algorithm: Iterative Refinement")

# ========== Пользовательские настройки ==========
set(THREADS 20 CACHE STRING "Number of threads to use for testing")
configure_file(${CMAKE_SOURCE_DIR}/src/config.h.in ${CMAKE_BINARY_DIR}/config.h)
include_directories(${CMAKE_BINARY_DIR})

# ========== Компилятор и системные флаги ========
find_package(OpenMP REQUIRED)

# ========== Внешние зависимости ==========
find_package(Eigen3 REQUIRED)
find_package(LAPACKE REQUIRED) # LAPACKE часто зависит от BLAS
find_package(Boost REQUIRED) # Обратите внимание на предупреждение CMP0167, если используете старую версию CMake/Boost

# --- Возвращаемся к стандартному поиску BLAS ---
set(CMAKE_FIND_DEBUG_MODE TRUE) # Оставляем отладку
find_package(BLAS)
set(CMAKE_FIND_DEBUG_MODE FALSE) # Выключаем отладку после

if(BLAS_FOUND)
    message(STATUS "Found BLAS library (using FindBLAS): ${BLAS_LIBRARIES}")
else()
    # Если стандартный поиск тоже не сработал, выдаем ошибку
    message(FATAL_ERROR "BLAS library not found using standard FindBLAS module. Check installation in C:/msys64/mingw64 and CMake output.")
endif()
# --- Конец блока поиска BLAS ---

# ========== Пути поиска (Windows/MSYS) ==========
set(CMAKE_PREFIX_PATH "C:/msys64/mingw64")
set(CMAKE_IGNORE_PATH "C:/Users/schuc/anaconda3")

# ========== Заголовки и исходники ==========
set(HEADER_FILES
    src/generate_svd.h
    src/iterative_refinement_1.h
    src/iterative_refinement_1.hpp
    src/iterative_refinement_4.h    
    src/iterative_refinement_4.hpp  
    src/iterative_refinement_5.h
    src/iterative_refinement_5.hpp
    src/iterative_refinement_6.h
    src/iterative_refinement_6.hpp
    src/iterative_refinement_8.h
    src/iterative_refinement_8.hpp
)

set(SVD_LIB_SOURCES
    src/generate_svd.cpp
)

# Создание статической библиотеки без main
add_library(svd STATIC ${SVD_LIB_SOURCES} ${HEADER_FILES})
target_compile_features(svd PRIVATE cxx_std_20)
target_compile_options(svd PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd PUBLIC Eigen3::Eigen lapacke Boost::boost ${OpenMP_CXX_LIBRARIES})

# ========== Тесты ==========

# Тест для Алгоритма 1
add_executable(svd_test_algo1 src/testing_algo1.cpp)
target_compile_features(svd_test_algo1 PRIVATE cxx_std_20)
target_compile_options(svd_test_algo1 PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd_test_algo1 PRIVATE svd Eigen3::Eigen lapacke ${OpenMP_CXX_LIBRARIES})



# Тест для Алгоритма 4
add_executable(svd_test_algo4 src/testing_algo4.cpp)
target_compile_features(svd_test_algo4 PRIVATE cxx_std_20)
target_compile_options(svd_test_algo4 PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd_test_algo4 PRIVATE svd Eigen3::Eigen lapacke ${OpenMP_CXX_LIBRARIES} ${BLAS_LIBRARIES})

# Тест для Алгоритма 5
add_executable(svd_test_algo5 src/testing_algo5.cpp)
target_compile_features(svd_test_algo5 PRIVATE cxx_std_20)
target_compile_options(svd_test_algo5 PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd_test_algo5 PRIVATE svd Eigen3::Eigen lapacke ${OpenMP_CXX_LIBRARIES} ${BLAS_LIBRARIES})

# Тест для Алгоритма 6
add_executable(svd_test_algo6 src/testing_algo6.cpp)
target_compile_features(svd_test_algo6 PRIVATE cxx_std_20)
target_compile_options(svd_test_algo6 PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd_test_algo6 PRIVATE svd Eigen3::Eigen lapacke ${OpenMP_CXX_LIBRARIES} ${BLAS_LIBRARIES})

# Тест для Алгоритма 8
add_executable(svd_test_algo8 src/testing_algo8.cpp)
target_compile_features(svd_test_algo8 PRIVATE cxx_std_20)
target_compile_options(svd_test_algo8 PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd_test_algo8 PRIVATE svd Eigen3::Eigen lapacke ${OpenMP_CXX_LIBRARIES} ${BLAS_LIBRARIES})

# ОБЪЕДИНЕННЫЙ ТЕСТ 
add_executable(svd_test_all src/testing_all.cpp)
target_compile_features(svd_test_all PRIVATE cxx_std_20)
target_compile_options(svd_test_all PRIVATE -Wa,-mbig-obj -g0 ${OpenMP_CXX_FLAGS})
target_link_libraries(svd_test_all PRIVATE svd Eigen3::Eigen lapacke ${OpenMP_CXX_LIBRARIES} ${BLAS_LIBRARIES})