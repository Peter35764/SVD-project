cmake_minimum_required(VERSION 3.10)

project(svd_iterative_refinement VERSION 1.0.0 DESCRIPTION "SVD algorithm: Iterative Refinement")

# ========== Пользовательские настройки ==========
set(THREADS 20 CACHE STRING "Number of threads to use for testing")
configure_file(${CMAKE_SOURCE_DIR}/src/config.h.in ${CMAKE_BINARY_DIR}/config.h)
# include_directories(${CMAKE_BINARY_DIR}) # Глобальное добавление CMAKE_BINARY_DIR не обязательно, если оно добавляется к целям

# ========== Компилятор и системные флаги ========
find_package(OpenMP REQUIRED)

# ========== Внешние зависимости ==========
find_package(Eigen3 REQUIRED)
find_package(LAPACKE REQUIRED) # LAPACKE часто зависит от BLAS
find_package(Boost REQUIRED) # Обратите внимание на предупреждение CMP0167, если используете старую версию CMake/Boost

# --- Возвращаемся к стандартному поиску BLAS ---
# set(CMAKE_FIND_DEBUG_MODE TRUE) # Можно закомментировать или убрать, если BLAS находится стабильно
find_package(BLAS) 
# set(CMAKE_FIND_DEBUG_MODE FALSE) # Можно закомментировать или убрать

if(BLAS_FOUND)
    message(STATUS "Found BLAS library (using FindBLAS): ${BLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS library not found using standard FindBLAS module. Check installation in C:/msys64/mingw64 and CMake output.")
endif()
# --- Конец блока поиска BLAS ---

# ========== Пути поиска (Windows/MSYS) ==========
set(CMAKE_PREFIX_PATH "C:/msys64/mingw64")
set(CMAKE_IGNORE_PATH "C:/Users/schuc/anaconda3")

# ========== Определение INTERFACE библиотеки "svd" ==========
add_library(svd INTERFACE)

target_include_directories(svd INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src 
    ${CMAKE_BINARY_DIR}             
)

target_link_libraries(svd INTERFACE 
    Eigen3::Eigen 
    lapacke 
    
    Boost::boost 
    OpenMP::OpenMP_CXX # Это правильный современный способ линковки с OpenMP, если find_package(OpenMP) его предоставляет
)

target_compile_features(svd INTERFACE cxx_std_20)

# ========== Тесты ==========
# ОБЪЕДИНЕННЫЙ ТЕСТ 
add_executable(svd_test_all src/testing_all.cpp)
target_compile_features(svd_test_all PRIVATE cxx_std_20)

target_compile_options(svd_test_all PRIVATE 
    -Wa,-mbig-obj 
    -g0 
    ${OpenMP_CXX_FLAGS} # Флаги OpenMP добавляются здесь к конечной цели
)

target_link_libraries(svd_test_all PRIVATE 
    svd # svd принесет Eigen, LAPACKE, Boost и OpenMP (если OpenMP::OpenMP_CXX работает)
    ${BLAS_LIBRARIES}
    # Если OpenMP::OpenMP_CXX в target_link_libraries(svd INTERFACE ...) не передает библиотеки для линковки,
    # то можно добавить ${OpenMP_CXX_LIBRARIES} здесь, или OpenMP::OpenMP_CXX еще раз, если это необходимо.
    # Но обычно, если OpenMP::OpenMP_CXX это INTERFACE цель, она передаст все нужное.
)