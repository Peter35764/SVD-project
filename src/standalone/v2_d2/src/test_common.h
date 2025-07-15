// Файл: src/test_common.h
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

// Стандартные библиотеки
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Eigen
#include <Eigen/Dense> // Включает Core и SVD

// Boost.Multiprecision
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/eigen.hpp> // Для совместимости Eigen и Boost.Multiprecision
#include <boost/math/special_functions/fpclassify.hpp> // Для boost::math::isnan

// Заголовки проекта
#include "config.h"    // Для THREADS
#include "svd_types.h" // <--- ВАЖНО: здесь определяется SVD_Project::PrecisionType

// Подключаем все варианты iterative_refinement и generate_svd,
// так как perform_single_precision_run будет их использовать через шаблоны.
#include "iterative_refinement_1.h" // Убедись, что эти файлы существуют в src/
#include "iterative_refinement_4.h"
#include "iterative_refinement_5.h"
#include "iterative_refinement_6.h"
#include "iterative_refinement_8.h"
#include "generate_svd.h"         // Убедись, что этот файл существует в src/

// Объявление глобального мьютекса (определение будет в test_common.cpp)
extern std::mutex cout_mutex;

// Структура для хранения результатов одного запуска теста
struct PrecisionStudyResultRow {
    int algorithm_id;
    int digits;
    std::string epsilon_str;
    int matrix_rows;
    int matrix_cols;
    float sigma_ratio_for_generator_float;
    int iterations_count;
    float time_taken_seconds;
    std::string achieved_sigma_relative_error_str;
    std::string achieved_u_ortho_error_str;
    std::string achieved_v_ortho_error_str;
};

// Шаблонная функция для преобразования значения в строку с научной нотацией
template<typename T>
std::string value_to_string_scientific(T val, int precision = -1) {
    std::ostringstream oss;
    if (precision == -1) {
        if (std::numeric_limits<T>::is_specialized && std::numeric_limits<T>::max_digits10 > 0) {
            precision = std::numeric_limits<T>::max_digits10;
        } else if (std::numeric_limits<T>::is_specialized && std::numeric_limits<T>::digits10 > 0) {
            precision = std::numeric_limits<T>::digits10 + 3;
        } else {
            precision = 20;
        }
    }
    oss << std::scientific << std::setprecision(precision) << val;
    return oss.str();
}

// ПОЛНОЕ ОПРЕДЕЛЕНИЕ шаблонной функции perform_single_precision_run
template<unsigned int NumDigits,
         int AlgoID,
         template<typename> class SVDAlgoClass,
         template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow perform_single_precision_run(
    int actual_digits_for_log,
    const std::pair<int, int>& mat_size_pair,
    float sigma_ratio_for_gen_float,
    int num_runs_for_averaging)
{
    // Теперь SVD_Project::PrecisionType должен быть известен из svd_types.h
    using CurrentPrecisionType = SVD_Project::PrecisionType<NumDigits>;

    CurrentPrecisionType current_epsilon_T = std::numeric_limits<CurrentPrecisionType>::epsilon();
    std::string epsilon_as_string = value_to_string_scientific(current_epsilon_T);

    int num_rows = mat_size_pair.first;
    int num_cols = mat_size_pair.second;

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "      Matrix: " << num_rows << "x" << num_cols
                  << ", SigmaRatioGen (float): " << sigma_ratio_for_gen_float
                  << ", Digits (template): " << NumDigits << " (log as " << actual_digits_for_log <<")..." << std::flush;
    }

    long long cumulative_iterations_total = 0;
    double cumulative_time_seconds_total = 0.0;
    CurrentPrecisionType cumulative_achieved_sigma_error = CurrentPrecisionType(0);
    CurrentPrecisionType cumulative_achieved_u_ortho_error = CurrentPrecisionType(0);
    CurrentPrecisionType cumulative_achieved_v_ortho_error = CurrentPrecisionType(0);
    int successful_runs_count = 0;

    for (int run_idx = 0; run_idx < num_runs_for_averaging; ++run_idx) {
        try {
            CurrentPrecisionType sigma_min_val = CurrentPrecisionType(1.0);
            CurrentPrecisionType sigma_ratio_val = CurrentPrecisionType(sigma_ratio_for_gen_float);
            CurrentPrecisionType sigma_max_val = sigma_min_val * sigma_ratio_val;

            SVDGeneratorEngine<CurrentPrecisionType> svd_problem_generator(
                num_rows, num_cols,
                sigma_min_val,
                sigma_max_val);
            svd_problem_generator.generate();

            Eigen::Matrix<CurrentPrecisionType, Eigen::Dynamic, Eigen::Dynamic> A_test_matrix = svd_problem_generator.MatrixA();
            Eigen::Matrix<CurrentPrecisionType, Eigen::Dynamic, 1> True_Singular_Values_vector = svd_problem_generator.SingularValues();

            Eigen::Index k_min_mn_eigen_idx = static_cast<Eigen::Index>(std::min(num_rows, num_cols));
            if (True_Singular_Values_vector.size() != k_min_mn_eigen_idx) {
                Eigen::Matrix<CurrentPrecisionType, Eigen::Dynamic, 1> temp_sv =
                    Eigen::Matrix<CurrentPrecisionType, Eigen::Dynamic, 1>::Zero(k_min_mn_eigen_idx);
                if (True_Singular_Values_vector.size() > 0) {
                    Eigen::Index current_true_sv_size = True_Singular_Values_vector.size();
                    Eigen::Index head_elements_to_copy = std::min(current_true_sv_size, k_min_mn_eigen_idx);
                    temp_sv.head(head_elements_to_copy) = True_Singular_Values_vector.head(head_elements_to_copy);
                }
                True_Singular_Values_vector = temp_sv;
            }

            std::string history_file_base_name = "log_A" + std::to_string(AlgoID) +
                                                 "_d" + std::to_string(actual_digits_for_log) +
                                                 "_" + std::to_string(num_rows) + "x" + std::to_string(num_cols) +
                                                 "_sr" + std::to_string(static_cast<int>(std::round(sigma_ratio_for_gen_float))) +
                                                 "_run" + std::to_string(run_idx);

            SVDAlgoClass<CurrentPrecisionType> svd_solver(
                A_test_matrix,
                True_Singular_Values_vector,
                history_file_base_name);

            cumulative_iterations_total += svd_solver.iterations_taken();
            cumulative_time_seconds_total += svd_solver.time_taken_s();

            // Используем boost::math::isnan для типов Boost.Multiprecision
            if (!boost::math::isnan(svd_solver.achieved_sigma_relative_error())) {
                cumulative_achieved_sigma_error += svd_solver.achieved_sigma_relative_error();
            }
            if (!boost::math::isnan(svd_solver.achieved_U_ortho_error())) {
                cumulative_achieved_u_ortho_error += svd_solver.achieved_U_ortho_error();
            }
            if (!boost::math::isnan(svd_solver.achieved_V_ortho_error())) {
                cumulative_achieved_v_ortho_error += svd_solver.achieved_V_ortho_error();
            }
            successful_runs_count++;

        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "\nEXCEPTION: AlgoID=" << AlgoID << ", Digits=" << actual_digits_for_log
                      << ", Matrix=" << num_rows << "x" << num_cols << "\n  Error: " << e.what() << std::endl;
        } catch (...) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "\nUNKNOWN EXCEPTION: AlgoID=" << AlgoID << ", Digits=" << actual_digits_for_log
                      << ", Matrix=" << num_rows << "x" << num_cols << std::endl;
        }
    }

    PrecisionStudyResultRow result_row_data;
    result_row_data.algorithm_id = AlgoID;
    result_row_data.digits = actual_digits_for_log;
    result_row_data.epsilon_str = epsilon_as_string;
    result_row_data.matrix_rows = num_rows;
    result_row_data.matrix_cols = num_cols;
    result_row_data.sigma_ratio_for_generator_float = sigma_ratio_for_gen_float;

    if (successful_runs_count > 0) {
        result_row_data.iterations_count = static_cast<int>(std::round(static_cast<double>(cumulative_iterations_total) / successful_runs_count));
        result_row_data.time_taken_seconds = static_cast<float>(cumulative_time_seconds_total / successful_runs_count);
        result_row_data.achieved_sigma_relative_error_str = value_to_string_scientific(cumulative_achieved_sigma_error / successful_runs_count);
        result_row_data.achieved_u_ortho_error_str = value_to_string_scientific(cumulative_achieved_u_ortho_error / successful_runs_count);
        result_row_data.achieved_v_ortho_error_str = value_to_string_scientific(cumulative_achieved_v_ortho_error / successful_runs_count);
    } else {
        result_row_data.iterations_count = -1;
        result_row_data.time_taken_seconds = -1.0f;
        result_row_data.achieved_sigma_relative_error_str = "RunError";
        result_row_data.achieved_u_ortho_error_str = "RunError";
        result_row_data.achieved_v_ortho_error_str = "RunError";
    }

    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << " Done. AvgIter: " << result_row_data.iterations_count
                  << ", AvgTime: " << std::fixed << std::setprecision(4) << result_row_data.time_taken_seconds << "s"
                  << ", AvgSigmaRelErr: " << result_row_data.achieved_sigma_relative_error_str << std::endl;
    }
    return result_row_data;
}

#endif // TEST_COMMON_H
