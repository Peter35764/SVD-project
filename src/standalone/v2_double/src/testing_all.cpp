#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <numeric>

#include <boost/multiprecision/cpp_dec_float.hpp>
// #include <boost/multiprecision/cmath.hpp> // Убедились, что это не нужно

#include "config.h" 
#include "iterative_refinement_1.h" 
#include "iterative_refinement_4.h" 
#include "iterative_refinement_5.h" 
#include "iterative_refinement_6.h" 
#include "iterative_refinement_8.h" 
#include "generate_svd.h"         

std::mutex cout_mutex;

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

struct PrecisionStudyResultRow {
    int algorithm_id;
    int digits; 
    std::string epsilon_str; 
    int matrix_rows;
    int matrix_cols;
    double sigma_ratio_for_generator_double; 
    int iterations_count;
    double time_taken_seconds;
    std::string achieved_sigma_relative_error_str;
    std::string achieved_u_ortho_error_str; 
    std::string achieved_v_ortho_error_str; 
};

template<unsigned int NumDigits,
         int AlgoID, 
         template<typename> class SVDAlgoClass,
         template<typename> class SVDGeneratorEngine >
PrecisionStudyResultRow perform_single_precision_run(
    int actual_digits_for_log,
    const std::pair<int, int>& mat_size_pair,
    double sigma_ratio_for_gen_double,
    int num_runs_for_averaging
) {
    using CurrentPrecisionType = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<NumDigits>>;
    
    CurrentPrecisionType current_epsilon_T = std::numeric_limits<CurrentPrecisionType>::epsilon();
    std::string epsilon_as_string = value_to_string_scientific(current_epsilon_T);

    int num_rows = mat_size_pair.first;
    int num_cols = mat_size_pair.second;

    { 
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "    Matrix: " << num_rows << "x" << num_cols 
                  << ", SigmaRatioGen (double): " << sigma_ratio_for_gen_double 
                  << ", Digits: " << actual_digits_for_log << "..." << std::flush;
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
            CurrentPrecisionType sigma_ratio_val = CurrentPrecisionType(sigma_ratio_for_gen_double);
            CurrentPrecisionType sigma_max_val = sigma_min_val * sigma_ratio_val;

            SVDGeneratorEngine<CurrentPrecisionType> svd_problem_generator(
                num_rows, num_cols, 
                sigma_min_val,    
                sigma_max_val     
            );
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
                                                 "_sr" + std::to_string(static_cast<int>(std::round(sigma_ratio_for_gen_double))) +
                                                 "_run" + std::to_string(run_idx);
            
            SVDAlgoClass<CurrentPrecisionType> svd_solver(
                A_test_matrix, 
                True_Singular_Values_vector, 
                history_file_base_name
            );
            
            cumulative_iterations_total += svd_solver.iterations_taken();
            cumulative_time_seconds_total += svd_solver.time_taken_s();
            
            if (!boost::math::isnan(svd_solver.achieved_sigma_relative_error()))
                 cumulative_achieved_sigma_error += svd_solver.achieved_sigma_relative_error();
            if (!boost::math::isnan(svd_solver.achieved_U_ortho_error()))
                 cumulative_achieved_u_ortho_error += svd_solver.achieved_U_ortho_error();
            if (!boost::math::isnan(svd_solver.achieved_V_ortho_error()))
                 cumulative_achieved_v_ortho_error += svd_solver.achieved_V_ortho_error();
            
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
    result_row_data.sigma_ratio_for_generator_double = sigma_ratio_for_gen_double;
    
    if (successful_runs_count > 0) {
        result_row_data.iterations_count = static_cast<int>(cumulative_iterations_total / successful_runs_count);
        result_row_data.time_taken_seconds = cumulative_time_seconds_total / successful_runs_count;
        result_row_data.achieved_sigma_relative_error_str = value_to_string_scientific(cumulative_achieved_sigma_error / successful_runs_count);
        result_row_data.achieved_u_ortho_error_str = value_to_string_scientific(cumulative_achieved_u_ortho_error / successful_runs_count);
        result_row_data.achieved_v_ortho_error_str = value_to_string_scientific(cumulative_achieved_v_ortho_error / successful_runs_count);
    } else { 
        result_row_data.iterations_count = -1; 
        result_row_data.time_taken_seconds = -1.0;
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

template<int AlgoID, 
         template<typename> class SVDAlgoClass,
         template<typename> class SVDGeneratorEngine >
void run_precision_study(
    const std::string& output_csv_filename_base,    
    const std::vector<int>& digit_levels_config,    
    const std::vector<std::pair<int, int>>& matrix_sizes_config, 
    const std::vector<double>& sigma_ratios_config, 
    int num_runs_for_averaging                       
) {
    std::string full_output_csv_filename = output_csv_filename_base + "_algo" + std::to_string(AlgoID) + ".csv";
    std::ofstream csv_file(full_output_csv_filename);

    if (!csv_file.is_open()) {
        std::lock_guard<std::mutex> lock(cout_mutex); 
        std::cerr << "Error: Could not open output CSV file: " << full_output_csv_filename << std::endl;
        return;
    }

    csv_file << "AlgorithmID,Digits,Epsilon,MatrixRows,MatrixCols,SigmaRatioGen,"
             << "Iterations,Time_s,AchievedSigmaRelError,AchievedUOrthoError,AchievedVOrthoError\n";

    for (int current_digits_val : digit_levels_config) { 
        { 
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\n---> Testing Algorithm ID: " << AlgoID << " with Digits: " << current_digits_val << std::endl;
        }

        for (const auto& mat_size_pair : matrix_sizes_config) {
            if (AlgoID == 4 && mat_size_pair.first < mat_size_pair.second) {
                 std::lock_guard<std::mutex> lock(cout_mutex);
                 std::cout << "  Skipping Matrix: " << mat_size_pair.first << "x" << mat_size_pair.second << " for Algo " << AlgoID 
                           << " (constraint m>=n)" << std::endl;
                 continue;
            }
            
            for (double sigma_ratio_for_gen_double : sigma_ratios_config) {
                PrecisionStudyResultRow result_row_data;

                switch (current_digits_val) {
                    case 21: result_row_data = perform_single_precision_run<21, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 22: result_row_data = perform_single_precision_run<22, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;   
                    case 23: result_row_data = perform_single_precision_run<23, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 24: result_row_data = perform_single_precision_run<24, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 25: result_row_data = perform_single_precision_run<25, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 26: result_row_data = perform_single_precision_run<26, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
        
                    case 27: result_row_data = perform_single_precision_run<27, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 28: result_row_data = perform_single_precision_run<28, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 29: result_row_data = perform_single_precision_run<29, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 30: result_row_data = perform_single_precision_run<30, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 31: result_row_data = perform_single_precision_run<31, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 32: result_row_data = perform_single_precision_run<32, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 33: result_row_data = perform_single_precision_run<33, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 34: result_row_data = perform_single_precision_run<34, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 35: result_row_data = perform_single_precision_run<35, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 36: result_row_data = perform_single_precision_run<36, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 37: result_row_data = perform_single_precision_run<37, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 38: result_row_data = perform_single_precision_run<38, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 39: result_row_data = perform_single_precision_run<39, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 40: result_row_data = perform_single_precision_run<40, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 41: result_row_data = perform_single_precision_run<41, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 42: result_row_data = perform_single_precision_run<42, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 43: result_row_data = perform_single_precision_run<43, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 44: result_row_data = perform_single_precision_run<44, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 45: result_row_data = perform_single_precision_run<45, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 46: result_row_data = perform_single_precision_run<46, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 47: result_row_data = perform_single_precision_run<47, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 48: result_row_data = perform_single_precision_run<48, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 49: result_row_data = perform_single_precision_run<49, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 50: result_row_data = perform_single_precision_run<50, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 51: result_row_data = perform_single_precision_run<51, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 52: result_row_data = perform_single_precision_run<52, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 53: result_row_data = perform_single_precision_run<53, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 54: result_row_data = perform_single_precision_run<54, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 55: result_row_data = perform_single_precision_run<55, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 56: result_row_data = perform_single_precision_run<56, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 57: result_row_data = perform_single_precision_run<57, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 58: result_row_data = perform_single_precision_run<58, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 59: result_row_data = perform_single_precision_run<59, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 60: result_row_data = perform_single_precision_run<60, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 61: result_row_data = perform_single_precision_run<61, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 62: result_row_data = perform_single_precision_run<62, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 63: result_row_data = perform_single_precision_run<63, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 64: result_row_data = perform_single_precision_run<64, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 65: result_row_data = perform_single_precision_run<65, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 66: result_row_data = perform_single_precision_run<66, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 67: result_row_data = perform_single_precision_run<67, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 68: result_row_data = perform_single_precision_run<68, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 69: result_row_data = perform_single_precision_run<69, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 70: result_row_data = perform_single_precision_run<70, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 71: result_row_data = perform_single_precision_run<71, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 72: result_row_data = perform_single_precision_run<72, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 73: result_row_data = perform_single_precision_run<73, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 74: result_row_data = perform_single_precision_run<74, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 75: result_row_data = perform_single_precision_run<75, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 76: result_row_data = perform_single_precision_run<76, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 77: result_row_data = perform_single_precision_run<77, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 78: result_row_data = perform_single_precision_run<78, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 79: result_row_data = perform_single_precision_run<79, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 80: result_row_data = perform_single_precision_run<80, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 81: result_row_data = perform_single_precision_run<81, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 82: result_row_data = perform_single_precision_run<82, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 83: result_row_data = perform_single_precision_run<83, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 84: result_row_data = perform_single_precision_run<84, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 85: result_row_data = perform_single_precision_run<85, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 86: result_row_data = perform_single_precision_run<86, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 87: result_row_data = perform_single_precision_run<87, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 88: result_row_data = perform_single_precision_run<88, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 89: result_row_data = perform_single_precision_run<89, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 90: result_row_data = perform_single_precision_run<90, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 91: result_row_data = perform_single_precision_run<91, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 92: result_row_data = perform_single_precision_run<92, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 93: result_row_data = perform_single_precision_run<93, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 94: result_row_data = perform_single_precision_run<94, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 95: result_row_data = perform_single_precision_run<95, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 96: result_row_data = perform_single_precision_run<96, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 97: result_row_data = perform_single_precision_run<97, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 98: result_row_data = perform_single_precision_run<98, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 99: result_row_data = perform_single_precision_run<99, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 100: result_row_data = perform_single_precision_run<100, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    case 101: result_row_data = perform_single_precision_run<101, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_double, num_runs_for_averaging); break;
                    default:
                        std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cerr << "    Skipping Digits: " << current_digits_val << " - not explicitly handled in switch-case." << std::endl;
                        result_row_data.algorithm_id = AlgoID; result_row_data.digits = current_digits_val; result_row_data.iterations_count = -2;
                        result_row_data.achieved_sigma_relative_error_str = "UnhandledDigits";
                        result_row_data.matrix_rows = mat_size_pair.first; result_row_data.matrix_cols = mat_size_pair.second;
                        result_row_data.sigma_ratio_for_generator_double = sigma_ratio_for_gen_double;
                        result_row_data.time_taken_seconds = 0.0;
                        result_row_data.epsilon_str = "N/A";
                        result_row_data.achieved_u_ortho_error_str = "UnhandledDigits";
                        result_row_data.achieved_v_ortho_error_str = "UnhandledDigits";
                        break;
                }
                
                if (result_row_data.iterations_count != -2) {
                     csv_file << result_row_data.algorithm_id << ","
                         << result_row_data.digits << ","
                         << "\"" << result_row_data.epsilon_str << "\"" << "," 
                         << result_row_data.matrix_rows << ","
                         << result_row_data.matrix_cols << ","
                         << value_to_string_scientific(result_row_data.sigma_ratio_for_generator_double, 6) << ","
                         << result_row_data.iterations_count << ","
                         << std::fixed << std::setprecision(6) << result_row_data.time_taken_seconds << ","
                         << result_row_data.achieved_sigma_relative_error_str << ","
                         << result_row_data.achieved_u_ortho_error_str << ","
                         << result_row_data.achieved_v_ortho_error_str << "\n";
                    csv_file.flush(); 
                }
            } 
        } 
    } 

    csv_file.close();
    { 
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\n---> Precision study for Algorithm ID " << AlgoID << " finished." 
                  << " Results saved to: " << full_output_csv_filename << std::endl;
    }
} 


int main() {
    std::cout << "SVD Algorithm Precision Study Test Suite" << std::endl;
    std::cout << "Compilation Date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<int> digit_levels_to_study_config;

    int min_exponent_for_epsilon = 20;  
    int max_exponent_for_epsilon = 100; 

    std::cout << "Configuring digit levels for target epsilons from approximately 1e-" 
              << min_exponent_for_epsilon << " to 1e-" << max_exponent_for_epsilon << "." << std::endl;
    std::cout << "This means Digits will range from " << (min_exponent_for_epsilon + 1) 
              << " to " << (max_exponent_for_epsilon + 1) << "." << std::endl;

    for (int exponent = min_exponent_for_epsilon; exponent <= max_exponent_for_epsilon; ++exponent) {
        digit_levels_to_study_config.push_back(exponent + 1);
    }
    
    std::vector<std::pair<int, int>> matrix_sizes_config = {
        {50, 50}
    };

    std::vector<double> sigma_ratios_config = {100}; 

    int num_runs_for_averaging_config = 1; 

    std::cout << "Study Configuration:" << std::endl;
    std::cout << "  Digit levels to test (" << digit_levels_to_study_config.size() << " levels): "; 
    if (!digit_levels_to_study_config.empty()) {
        std::cout << digit_levels_to_study_config.front() << "..." << digit_levels_to_study_config.back();
    }
    std::cout << std::endl;
    std::cout << "  Matrix sizes (rows x cols): "; 
    for(size_t i=0; i < matrix_sizes_config.size(); ++i) {
        std::cout << matrix_sizes_config[i].first << "x" << matrix_sizes_config[i].second << (i == matrix_sizes_config.size()-1 ? "" : "; ");
    }
    std::cout << std::endl;
    std::cout << "  Sigma ratios for generator (sigma_max/sigma_min): "; 
    for(size_t i=0; i < sigma_ratios_config.size(); ++i) {
        std::cout << sigma_ratios_config[i] << (i == sigma_ratios_config.size()-1 ? "" : ", ");
    }
    std::cout << std::endl;
    std::cout << "  Number of runs for averaging: " << num_runs_for_averaging_config << std::endl;
    std::cout << "============================================================" << std::endl;

    run_precision_study<1, 
                        SVD_Project::AOIR_SVD_1, 
                        SVD_Project::SVDGenerator>(
        "precision_study_results",      
        digit_levels_to_study_config,   
        matrix_sizes_config,
        sigma_ratios_config,
        num_runs_for_averaging_config
    );

    run_precision_study<4, 
                        SVD_Project::AOIR_SVD_4, 
                        SVD_Project::SVDGenerator>(
        "precision_study_results",      
        digit_levels_to_study_config,   
        matrix_sizes_config,
        sigma_ratios_config,
        num_runs_for_averaging_config
    );
    run_precision_study<5, 
                        SVD_Project::AOIR_SVD_5, 
                        SVD_Project::SVDGenerator>(
        "precision_study_results",      
        digit_levels_to_study_config,   
        matrix_sizes_config,
        sigma_ratios_config,
        num_runs_for_averaging_config
    );
    run_precision_study<6, 
                        SVD_Project::AOIR_SVD_6, 
                        SVD_Project::SVDGenerator>(
        "precision_study_results",      
        digit_levels_to_study_config,   
        matrix_sizes_config,
        sigma_ratios_config,
        num_runs_for_averaging_config
    );
    run_precision_study<8, 
                        SVD_Project::AOIR_SVD_8, 
                        SVD_Project::SVDGenerator>(
        "precision_study_results",      
        digit_levels_to_study_config,   
        matrix_sizes_config,
        sigma_ratios_config,
        num_runs_for_averaging_config
    );
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "All configured studies finished." << std::endl;
    std::cout << "Check CSV files starting with 'precision_study_results_algoX.csv'." << std::endl;

    return 0;
}