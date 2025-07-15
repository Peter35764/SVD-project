#ifndef ITERATIVE_REFINEMENT_1_HPP
#define ITERATIVE_REFINEMENT_1_HPP

#include "iterative_refinement_1.h" 
#include <Eigen/Dense>               
#include <limits>                    
#include <cmath>                     

#include <cassert>   
#include <vector>    
#include <iostream>  
#include <fstream>   
#include <string>    
#include <iomanip>   
#include <stdexcept> 
#include <chrono>    

#include <boost/multiprecision/cpp_dec_float.hpp> 
#include <boost/math/special_functions/fpclassify.hpp> // Для boost::math::isnan

// OpenBLAS здесь не используется для Алгоритма 1 (по предположению)
// extern "C" {
// #include <openblas/cblas.h>
// }

namespace SVD_Project {

// Псевдонимы для типов LowPrec и MatrixLowPrec здесь не нужны, если нет CBLAS
// using LowPrec = double;
// using MatrixLowPrec = Eigen::Matrix<LowPrec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; 

template<typename T>
AOIR_SVD_1<T>::AOIR_SVD_1() {} 

template<typename T>
AOIR_SVD_1<T>::AOIR_SVD_1(
    const MatrixDyn& A,                    
    const VectorDyn& true_singular_values, 
    const std::string& history_filename_base 
) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("Input matrix A is empty in AOIR_SVD_1 constructor.");
    }

    const int m = A.rows();
    const int n = A.cols();

    Eigen::MatrixXd A_double = A.template cast<double>();
    Eigen::BDCSVD<Eigen::MatrixXd> svd_double_initial(A_double, Eigen::ComputeThinU | Eigen::ComputeThinV); 

    MatrixDyn initial_U_thin = svd_double_initial.matrixU().template cast<T>(); 
    MatrixDyn initial_V_thin = svd_double_initial.matrixV().template cast<T>(); 
    
    MatrixDyn U_full = MatrixDyn::Identity(m, m);
    if (initial_U_thin.cols() > 0 && initial_U_thin.rows() == m) {
        if (initial_U_thin.cols() <= m) {
             U_full.leftCols(initial_U_thin.cols()) = initial_U_thin;
        } else { 
            U_full.leftCols(m) = initial_U_thin.leftCols(m); 
            // std::cerr << "Warning (AOIR_SVD_1): Initial U from BDCSVD had more columns ("
            //           << initial_U_thin.cols() << ") than rows (" << m << "). Truncated." << std::endl;
        }
    } else if (initial_U_thin.cols() > 0) { 
        // std::cerr << "Warning (AOIR_SVD_1): Initial U from BDCSVD has " << initial_U_thin.rows() 
        //           << " rows, expected " << m << ". Using Identity for U_full." << std::endl;
    }
   
    MatrixDyn V_full = MatrixDyn::Identity(n, n);
    if (initial_V_thin.cols() > 0 && initial_V_thin.rows() == n) {
        if (initial_V_thin.cols() <= n) {
            V_full.leftCols(initial_V_thin.cols()) = initial_V_thin;
        } else {
            V_full.leftCols(n) = initial_V_thin.leftCols(n);
            // std::cerr << "Warning (AOIR_SVD_1): Initial V from BDCSVD had more columns ("
            //           << initial_V_thin.cols() << ") than rows (" << n << "). Truncated." << std::endl;
        }
    } else if (initial_V_thin.cols() > 0) {
        // std::cerr << "Warning (AOIR_SVD_1): Initial V from BDCSVD has " << initial_V_thin.rows() 
        //           << " rows, expected " << n << ". Using Identity for V_full." << std::endl;
    }
    
    const T noise_level = T("1e-4"); 
    if (boost::multiprecision::abs(noise_level) > std::numeric_limits<T>::epsilon()) {
        MatrixDyn noise_U_matrix = MatrixDyn::Random(m, m); 
        MatrixDyn noise_V_matrix = MatrixDyn::Random(n, n);
        U_full += noise_level * noise_U_matrix;
        V_full += noise_level * noise_V_matrix;
    }
    
    SVDAlgorithmResult<T> result_data = MSVD_SVD_Refined(A, U_full, V_full, history_filename_base, true_singular_values);

    this->U_computed = result_data.U;
    this->S_computed_diag_matrix = result_data.S_diag_matrix;
    this->V_computed = result_data.V;
    this->iterations_taken_ = result_data.iterations_taken;
    this->achieved_sigma_relative_error_ = result_data.achieved_sigma_relative_error;
    this->achieved_U_ortho_error_ = result_data.achieved_U_ortho_error;
    this->achieved_V_ortho_error_ = result_data.achieved_V_ortho_error;
    this->time_taken_s_ = result_data.time_taken_s;
}

template<typename T>
SVDAlgorithmResult<T> AOIR_SVD_1<T>::MSVD_SVD_Refined(
    const MatrixDyn& A,        
    const MatrixDyn& U_initial, 
    const MatrixDyn& V_initial, 
    const std::string& history_filename_base, 
    const VectorDyn& true_singular_values     
) {
    auto overall_start_time = std::chrono::high_resolution_clock::now(); 
    SVDAlgorithmResult<T> algo_result_output; 

    const int m = A.rows();
    const int n = A.cols();
    const int k_min_mn = std::min(m, n); 

    if (m < n) { 
        throw std::runtime_error("MSVD_SVD_Refined (Algo 1) currently requires m >= n.");
    }

    const T machine_epsilon = std::numeric_limits<T>::epsilon();
    const T sqrt_machine_epsilon = boost::multiprecision::sqrt(machine_epsilon); 

    const int max_iterations = 10000; 
    const int min_iterations_for_stagnation_check = 5; 
    const T stagnation_improvement_threshold_factor = T("1e-1"); 
    const T direct_convergence_threshold_factor_ortho = T("10.0"); 
    const T direct_convergence_threshold_factor_sigma = T("1e3");  

    MatrixDyn U = U_initial;
    MatrixDyn V = V_initial;
    
    VectorDyn current_sigma_vector = VectorDyn::Zero(n);
    MatrixDyn Sigma_n_matrix = MatrixDyn::Zero(n, n); 

    T sigma_relative_error_prev_iter = std::numeric_limits<T>::infinity();
    T internal_R_norm_prev_iter = std::numeric_limits<T>::infinity(); 
    T internal_S_norm_prev_iter = std::numeric_limits<T>::infinity(); 

    std::ofstream history_log_file;
    if (!history_filename_base.empty()) {
        std::string full_history_filename = history_filename_base + "_conv1_details.csv"; // Изменено на conv1
        history_log_file.open(full_history_filename);
        if (history_log_file) {
            history_log_file << "Iteration,SigmaRelError,Reported_U_OrthoError,Reported_V_OrthoError,"
                             << "Internal_R_norm,Internal_S_norm,StepTime_us,"
                             << "DeltaSigmaRelError,Delta_Internal_R_norm,Delta_Internal_S_norm\n";
        } else {
            // std::cerr << "Warning (AOIR_SVD_1): Could not open history log file: " << full_history_filename << std::endl;
        }
    }

    for (int iter_count = 0; iter_count < max_iterations; ++iter_count) {
        auto iter_step_start_time = std::chrono::high_resolution_clock::now();

        MatrixDyn R_full = MatrixDyn::Identity(m, m) - U.transpose() * U; 
        MatrixDyn S_full = MatrixDyn::Identity(n, n) - V.transpose() * V; 
        MatrixDyn T_full = U.transpose() * A * V; 
        
        MatrixDyn R11 = R_full.topLeftCorner(n, n);
        MatrixDyn T1 = T_full.topLeftCorner(n, n);

        for (int i = 0; i < n; ++i) { 
            T r_ii = R11(i, i); 
            T s_ii = S_full(i, i); 
            T t_ii = T1(i, i);

            T denom_sigma = T("1.0") - T("0.5") * (r_ii + s_ii);
            Sigma_n_matrix(i, i) = (boost::multiprecision::abs(denom_sigma) < machine_epsilon * T("10.0")) ? t_ii : (t_ii / denom_sigma);
            current_sigma_vector(i) = Sigma_n_matrix(i, i);
        }
        
        T current_sigma_relative_error = std::numeric_limits<T>::quiet_NaN(); 
        if (true_singular_values.size() >= k_min_mn && k_min_mn > 0) { 
            T diff_norm_squared = T("0.0");
            T true_norm_squared = T("0.0");
            for(int i = 0; i < k_min_mn; ++i) {
                T diff = current_sigma_vector(i) - true_singular_values(i); 
                diff_norm_squared += diff * diff;
                true_norm_squared += true_singular_values(i) * true_singular_values(i);
            }
            if (true_norm_squared > machine_epsilon * machine_epsilon) { 
                 current_sigma_relative_error = boost::multiprecision::sqrt(diff_norm_squared / true_norm_squared);
            } else if (diff_norm_squared > machine_epsilon * machine_epsilon) { 
                current_sigma_relative_error = boost::multiprecision::sqrt(diff_norm_squared);
            } else { 
                current_sigma_relative_error = T("0.0");
            }
        }

        MatrixDyn Eye_m_for_report = MatrixDyn::Identity(m,m);
        MatrixDyn Eye_n_for_report = MatrixDyn::Identity(n,n);
        T current_U_ortho_error_reported = (Eye_m_for_report - U * U.transpose()).norm();
        T current_V_ortho_error_reported = (Eye_n_for_report - V * V.transpose()).norm();
        
        T current_internal_R_norm = R_full.norm(); 
        T current_internal_S_norm = S_full.norm(); 

        T delta_sigma_relative_error_abs = boost::multiprecision::abs(sigma_relative_error_prev_iter - current_sigma_relative_error);
        T delta_internal_R_norm_abs = boost::multiprecision::abs(internal_R_norm_prev_iter - current_internal_R_norm);
        T delta_internal_S_norm_abs = boost::multiprecision::abs(internal_S_norm_prev_iter - current_internal_S_norm);

        algo_result_output.iterations_taken = iter_count + 1;
        algo_result_output.achieved_sigma_relative_error = current_sigma_relative_error;
        algo_result_output.achieved_U_ortho_error = current_U_ortho_error_reported; 
        algo_result_output.achieved_V_ortho_error = current_V_ortho_error_reported; 

        if (history_log_file.is_open()) {
             const int log_prec = std::numeric_limits<T>::digits10 > 0 ? std::numeric_limits<T>::digits10 + 2 : 20;
             auto iter_step_end_time_for_log = std::chrono::high_resolution_clock::now();
             auto iter_duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(iter_step_end_time_for_log - iter_step_start_time).count();
             history_log_file << iter_count << ","
                              << std::scientific << std::setprecision(log_prec) << current_sigma_relative_error << ","
                              << std::scientific << std::setprecision(log_prec) << current_U_ortho_error_reported << ","
                              << std::scientific << std::setprecision(log_prec) << current_V_ortho_error_reported << ","
                              << std::scientific << std::setprecision(log_prec) << current_internal_R_norm << ","
                              << std::scientific << std::setprecision(log_prec) << current_internal_S_norm << ","
                              << iter_duration_microseconds << ","
                              << std::scientific << std::setprecision(log_prec) << delta_sigma_relative_error_abs << ","
                              << std::scientific << std::setprecision(log_prec) << delta_internal_R_norm_abs << ","
                              << std::scientific << std::setprecision(log_prec) << delta_internal_S_norm_abs << "\n";
        }
        
        bool sigma_converged = (!boost::math::isnan(current_sigma_relative_error) && current_sigma_relative_error < direct_convergence_threshold_factor_sigma * machine_epsilon) ||
                               (delta_sigma_relative_error_abs < stagnation_improvement_threshold_factor * machine_epsilon * (sigma_relative_error_prev_iter > T(1) ? sigma_relative_error_prev_iter : T(1) ) );
        bool U_internal_ortho_converged = (current_internal_R_norm < direct_convergence_threshold_factor_ortho * machine_epsilon) ||
                                          (delta_internal_R_norm_abs < stagnation_improvement_threshold_factor * machine_epsilon);
        bool V_internal_ortho_converged = (current_internal_S_norm < direct_convergence_threshold_factor_ortho * machine_epsilon) ||
                                          (delta_internal_S_norm_abs < stagnation_improvement_threshold_factor * machine_epsilon);

        if (iter_count >= min_iterations_for_stagnation_check) { 
            if (sigma_converged && U_internal_ortho_converged && V_internal_ortho_converged) {
                break; 
            }
        }
        
        if (iter_count == max_iterations - 1 && history_log_file.is_open()) { 
            history_log_file << "Warning: Reached max_iterations (" << max_iterations << "). SigmaRelErr=" 
                             << current_sigma_relative_error << ", UOrthoInternal=" << current_internal_R_norm 
                             << ", VOrthoInternal=" << current_internal_S_norm << "\n";
        }

        sigma_relative_error_prev_iter = current_sigma_relative_error;
        internal_R_norm_prev_iter = current_internal_R_norm;
        internal_S_norm_prev_iter = current_internal_S_norm;

        MatrixDyn F_correction = MatrixDyn::Zero(m, m); 
        MatrixDyn G_correction = MatrixDyn::Zero(n, n); 

        for (int i = 0; i < n; ++i) { 
            F_correction(i, i) = R11(i, i) / T("2.0");
            G_correction(i, i) = S_full(i, i) / T("2.0");
        }
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue; 
                
                T alpha_term = T1(i, j) + current_sigma_vector(j) * R11(i, j);
                T beta_term  = T1(j, i) + current_sigma_vector(j) * S_full(i, j); 
                
                T sigma_i_val = current_sigma_vector(i);
                T sigma_j_val = current_sigma_vector(j);
                T denominator = sigma_j_val * sigma_j_val - sigma_i_val * sigma_i_val;

                if (boost::multiprecision::abs(denominator) > sqrt_machine_epsilon * boost::multiprecision::abs(sigma_j_val * sigma_j_val) ) {
                    F_correction(i, j) = (alpha_term * sigma_j_val + beta_term * sigma_i_val) / denominator;
                    G_correction(i, j) = (alpha_term * sigma_i_val + beta_term * sigma_j_val) / denominator;
                } else {
                    F_correction(i, j) = T("0.0"); 
                    G_correction(i, j) = T("0.0");
                }
            }
        }

        const int m_minus_n_rows = m - n; 
        if (m > n && m_minus_n_rows > 0) {
            MatrixDyn Sigma_n_inverse = MatrixDyn::Zero(n,n); 
            bool is_Sigma_n_invertible = true;
            for(int i=0; i<n; ++i) { 
                if (boost::multiprecision::abs(Sigma_n_matrix(i,i)) < sqrt_machine_epsilon) {
                    is_Sigma_n_invertible = false; break;
                }
                Sigma_n_inverse(i,i) = T("1.0") / Sigma_n_matrix(i,i);
            }

            MatrixDyn F12_block, F21_block; 
            if (is_Sigma_n_invertible) {
                MatrixDyn U_right_part = U.rightCols(m_minus_n_rows); 
                MatrixDyn T2_matrix_term = U_right_part.transpose() * A * V; 
                MatrixDyn T2_transpose_term = T2_matrix_term.transpose();    
                
                F12_block.noalias() = -Sigma_n_inverse * T2_transpose_term; 
                
                MatrixDyn R21_block_term = R_full.block(n, 0, m_minus_n_rows, n); 
                F21_block = R21_block_term - F12_block.transpose(); 
            } else { 
                F12_block = MatrixDyn::Zero(n, m_minus_n_rows);
                F21_block = MatrixDyn::Zero(m_minus_n_rows, n);
            }
            MatrixDyn R22_block_term = R_full.block(n, n, m_minus_n_rows, m_minus_n_rows);
            MatrixDyn F22_block = T("0.5") * R22_block_term; 

            F_correction.block(0, n, n, m_minus_n_rows) = F12_block;
            F_correction.block(n, 0, m_minus_n_rows, n) = F21_block;
            F_correction.block(n, n, m_minus_n_rows, m_minus_n_rows) = F22_block;
        }
        
        // ОТЛИЧИЕ АЛГОРИТМА 1 от АЛГОРИТМА 4: Обновление U и V без CBLAS, в текущей точности T
        U.noalias() += U * F_correction;
        V.noalias() += V * G_correction;

    } 

    if (history_log_file.is_open()) {
        history_log_file.close();
    }

    algo_result_output.U = U;
    algo_result_output.V = V;
    algo_result_output.S_diag_matrix = MatrixDyn::Zero(m,n);
    for(int i=0; i < std::min(n, k_min_mn) ; ++i) { 
         algo_result_output.S_diag_matrix(i,i) = current_sigma_vector(i); 
    }
    
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    algo_result_output.time_taken_s = std::chrono::duration<double>(overall_end_time - overall_start_time).count();
    
    return algo_result_output;
}

} 

#endif // ITERATIVE_REFINEMENT_1_HPP