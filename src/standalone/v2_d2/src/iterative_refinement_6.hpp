// Файл: src/iterative_refinement_6.hpp
#ifndef ITERATIVE_REFINEMENT_6_HPP
#define ITERATIVE_REFINEMENT_6_HPP

#include "iterative_refinement_6.h" // Объявление класса AOIR_SVD_6

#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <vector>
#include <iostream> // Для отладочных сообщений (можно убрать)
#include <fstream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <chrono>

// #include <boost/multiprecision/cpp_dec_float.hpp> // Уже включен через svd_types.h -> test_common.h
#include <boost/math/special_functions/fpclassify.hpp> // Для boost::math::isnan

// extern "C" { // УДАЛЯЕМ, так как cblas_sgemm больше не используется
// #include <openblas/cblas.h>
// }

namespace SVD_Project {

// Вспомогательная структура для определения типов с пониженной точностью
template<typename T_working_precision>
struct LowPrecisionTypes_Algo6 { // Уникальное имя для этого алгоритма
    static constexpr int working_digits = std::numeric_limits<T_working_precision>::digits10;
    static constexpr int calculated_low_prec_digits = working_digits / 2;
    static constexpr int low_prec_target_digits = (calculated_low_prec_digits > 0) ? calculated_low_prec_digits : 1;

    using LowPrecBackend = boost::multiprecision::backends::cpp_dec_float<low_prec_target_digits>;
    using Scalar = boost::multiprecision::number<LowPrecBackend, boost::multiprecision::et_on>; // Предполагаем et_on
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
};

// using LowPrec = float; // УДАЛЕНО
// using MatrixLowPrec = Eigen::Matrix<LowPrec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // УДАЛЕНО

template<typename T>
AOIR_SVD_6<T>::AOIR_SVD_6() {}

template<typename T>
AOIR_SVD_6<T>::AOIR_SVD_6(
    const MatrixDyn& A,
    const VectorDyn& true_singular_values,
    const std::string& history_filename_base)
{
    // Определяем типы пониженной точности
    using LowPrecScalar = typename LowPrecisionTypes_Algo6<T>::Scalar;
    using MatrixLowPrec = typename LowPrecisionTypes_Algo6<T>::Matrix;

    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("Input matrix A is empty in AOIR_SVD_6 constructor.");
    }

    const int m = A.rows();
    const int n = A.cols();

    // Начальное SVD в пониженной точности
    MatrixLowPrec A_low_prec = A.template cast<LowPrecScalar>();
    Eigen::BDCSVD<MatrixLowPrec> svd_initial_low_prec(A_low_prec, Eigen::ComputeThinU | Eigen::ComputeThinV);

    MatrixDyn initial_U_thin = svd_initial_low_prec.matrixU().template cast<T>();
    MatrixDyn initial_V_thin = svd_initial_low_prec.matrixV().template cast<T>();

    // Формирование U_full, V_full
    MatrixDyn U_full = MatrixDyn::Identity(m, m);
    if (initial_U_thin.cols() > 0 && initial_U_thin.rows() == m) {
        if (initial_U_thin.cols() <= m) U_full.leftCols(initial_U_thin.cols()) = initial_U_thin;
        else U_full.leftCols(m) = initial_U_thin.leftCols(m);
    } // (обработка предупреждений опущена для краткости, но должна быть как в оригинале)

    MatrixDyn V_full = MatrixDyn::Identity(n, n);
    if (initial_V_thin.cols() > 0 && initial_V_thin.rows() == n) {
        if (initial_V_thin.cols() <= n) V_full.leftCols(initial_V_thin.cols()) = initial_V_thin;
        else V_full.leftCols(n) = initial_V_thin.leftCols(n);
    } // (обработка предупреждений опущена)

    // Добавление шума
    const T noise_level = T("1e-4");
    if (boost::multiprecision::abs(noise_level) > std::numeric_limits<T>::epsilon()) {
        U_full += noise_level * MatrixDyn::Random(m, m);
        V_full += noise_level * MatrixDyn::Random(n, n);
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
SVDAlgorithmResult<T> AOIR_SVD_6<T>::MSVD_SVD_Refined(
    const MatrixDyn& A,
    const MatrixDyn& U_initial,
    const MatrixDyn& V_initial,
    const std::string& history_filename_base,
    const VectorDyn& true_singular_values)
{
    // Определяем типы пониженной точности
    using LowPrecScalar = typename LowPrecisionTypes_Algo6<T>::Scalar;
    using MatrixLowPrec = typename LowPrecisionTypes_Algo6<T>::Matrix;

    auto overall_start_time = std::chrono::high_resolution_clock::now();
    SVDAlgorithmResult<T> algo_result_output;

    const int m = A.rows();
    const int n = A.cols();
    const int k_min_mn = std::min(m, n);
    const int m_minus_n = m - n;

    if (m < n) {
        throw std::runtime_error("MSVD_SVD_Refined (Algo 6) currently requires m >= n.");
    }

    const T machine_epsilon = std::numeric_limits<T>::epsilon();
    const T sqrt_machine_epsilon = boost::multiprecision::sqrt(machine_epsilon);

    // ... (параметры сходимости) ...
    const int max_iterations = 15; // Пример
    const int min_iterations_for_stagnation_check = 5;
    const T stagnation_improvement_threshold_factor = T("1e-1");
    const T direct_convergence_threshold_factor_ortho = T("10.0");
    const T direct_convergence_threshold_factor_sigma = T("1e3");

    MatrixDyn U = U_initial;
    MatrixDyn V = V_initial;
    VectorDyn current_sigma_vector = VectorDyn::Zero(n);
    MatrixDyn Sigma_n_matrix = MatrixDyn::Zero(n, n);

    // ... (переменные для отслеживания сходимости) ...
    T sigma_relative_error_prev_iter = std::numeric_limits<T>::infinity();
    T internal_R_norm_prev_iter = std::numeric_limits<T>::infinity();
    T internal_S_norm_prev_iter = std::numeric_limits<T>::infinity();


    std::ofstream history_log_file;
    if (!history_filename_base.empty()) {
        std::string full_history_filename = history_filename_base + "_conv6_details.csv"; // conv6
        history_log_file.open(full_history_filename);
        if (history_log_file) {
             history_log_file << "Iteration,SigmaRelError,Reported_U_OrthoError,Reported_V_OrthoError,"
                             << "Internal_R_norm,Internal_S_norm,StepTime_us,"
                             << "DeltaSigmaRelError,Delta_Internal_R_norm,Delta_Internal_S_norm\n";
        }
    }

    for (int iter_count = 0; iter_count < max_iterations; ++iter_count) {
        auto iter_step_start_time = std::chrono::high_resolution_clock::now();

        MatrixDyn R_full = MatrixDyn::Identity(m, m) - U.transpose() * U;
        MatrixDyn S_full_ortho = MatrixDyn::Identity(n, n) - V.transpose() * V; // S_{ortho}

        MatrixDyn U1 = U.leftCols(n);
        MatrixDyn U2;
        if (m > n && m_minus_n > 0) {
            U2 = U.rightCols(m_minus_n);
        }

        MatrixDyn P; P.noalias() = A * V;
        MatrixDyn R_block = R_full.topLeftCorner(n, n); // R_{11} из R_full
        MatrixDyn S_algo_block = S_full_ortho;          // S_{algo} = S_{ortho}

        VectorDyn t_diag(n);
        for (int i = 0; i < n; ++i) {
            T r_ii = R_block(i, i);
            T s_ii = S_algo_block(i, i);
            t_diag(i) = U1.col(i).dot(P.col(i)); // t_ii = u1_i^T * (A*v_i)
            T denom_sigma = T("1.0") - T("0.5") * (r_ii + s_ii);
            Sigma_n_matrix(i, i) = (boost::multiprecision::abs(denom_sigma) < machine_epsilon * T("10.0")) ? t_diag(i) : (t_diag(i) / denom_sigma);
            current_sigma_vector(i) = Sigma_n_matrix(i, i);
        }

        for (int i = 0; i < n; ++i) { 
            T r_ii = R_block(i, i);
            T s_ii = S_algo_block(i, i);
            t_diag(i) = U1.col(i).dot(P.col(i)); 
            T denom_sigma = T("1.0") - T("0.5") * (r_ii + s_ii);
            Sigma_n_matrix(i, i) = (boost::multiprecision::abs(denom_sigma) < machine_epsilon * T("10.0")) ? t_diag(i) : (t_diag(i) / denom_sigma);
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
        T current_internal_S_norm = S_full_ortho.norm(); 

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

        MatrixDyn Cy; Cy.noalias() = P - U1 * Sigma_n_matrix; // Cy = A*V - U1*Sigma_n
        MatrixDyn Ca, Cb;
        // Ca = U1^T * Cy  (в пониженной точности)
        {
            MatrixLowPrec U1_low = U1.template cast<LowPrecScalar>();
            MatrixLowPrec Cy_low = Cy.template cast<LowPrecScalar>();
            MatrixLowPrec Ca_low(n, n);
            Ca_low.noalias() = U1_low.transpose() * Cy_low;
            Ca = Ca_low.template cast<T>();
        }
        // Cb = Ca^T - Sigma_n * R_block + S_algo_block * Sigma_n
        Cb.noalias() = Ca.transpose() - Sigma_n_matrix * R_block + S_algo_block * Sigma_n_matrix;

        MatrixDyn D_mat = Sigma_n_matrix * Ca + Cb * Sigma_n_matrix;
        MatrixDyn E_mat = Ca * Sigma_n_matrix + Sigma_n_matrix * Cb;

        MatrixDyn G_correction = MatrixDyn::Zero(n, n);
        MatrixDyn F11_correction = MatrixDyn::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            G_correction(i, i) = S_algo_block(i, i) / T("2.0"); // Используем S_algo_block
            F11_correction(i, i) = R_block(i, i) / T("2.0");    // Используем R_block
            T sigma2_i = current_sigma_vector(i) * current_sigma_vector(i);
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    T sigma2_j = current_sigma_vector(j) * current_sigma_vector(j);
                    T denominator = sigma2_j - sigma2_i;
                    if (boost::multiprecision::abs(denominator) > sqrt_machine_epsilon * boost::multiprecision::abs(sigma2_j)) {
                        G_correction(i, j) = D_mat(i, j) / denominator;
                        F11_correction(i, j) = E_mat(i, j) / denominator;
                    } else {
                        G_correction(i, j) = T("0.0");
                        F11_correction(i, j) = T("0.0");
                    }
                }
            }
        }

        MatrixDyn F_correction = MatrixDyn::Zero(m, m);
        F_correction.topLeftCorner(n, n) = F11_correction;

        if (m > n && m_minus_n > 0) {
            MatrixDyn Sigma_n_inv_for_F = MatrixDyn::Zero(n,n);
            bool invertible_for_F = true;
            for(int i=0; i<n; ++i) {
                if (boost::multiprecision::abs(current_sigma_vector(i)) < sqrt_machine_epsilon) {
                    invertible_for_F = false; break;
                }
                Sigma_n_inv_for_F(i,i) = T("1.0") / current_sigma_vector(i);
            }
            MatrixDyn F12_block, F21_block, F22_block;
            if (invertible_for_F) {
                MatrixDyn PTU2; PTU2.noalias() = P.transpose() * U2;
                F12_block.noalias() = -Sigma_n_inv_for_F * PTU2.transpose(); // Проверь эту формулу по источнику Алго6

                MatrixDyn U2Cy_high; // U2^T * Cy (в пониженной точности)
                {
                    MatrixLowPrec U2_low = U2.template cast<LowPrecScalar>();
                    MatrixLowPrec Cy_low = Cy.template cast<LowPrecScalar>();
                    MatrixLowPrec U2TCy_low(m_minus_n, n);
                    U2TCy_low.noalias() = U2_low.transpose() * Cy_low;
                    U2Cy_high = U2TCy_low.template cast<T>();
                }
                F21_block.noalias() = U2Cy_high * Sigma_n_inv_for_F;
            } else {
                F12_block = MatrixDyn::Zero(n, m_minus_n);
                F21_block = MatrixDyn::Zero(m_minus_n, n);
            }
            MatrixDyn R_full_bottom_right_block = R_full.bottomRightCorner(m_minus_n, m_minus_n);
            F22_block.noalias() = T("0.5") * R_full_bottom_right_block;

            F_correction.block(0, n, n, m_minus_n) = F12_block;
            F_correction.block(n, 0, m_minus_n, n) = F21_block;
            F_correction.block(n, n, m_minus_n, m_minus_n) = F22_block;
        }

        // Обновление U и V с использованием пониженной точности для умножения
        {
            MatrixLowPrec U_curr_low = U.template cast<LowPrecScalar>();
            MatrixLowPrec F_corr_low = F_correction.template cast<LowPrecScalar>();
            MatrixLowPrec DeltaU_low_res(m, m);
            DeltaU_low_res.noalias() = U_curr_low * F_corr_low;
            U += DeltaU_low_res.template cast<T>();
        }
        {
            MatrixLowPrec V_curr_low = V.template cast<LowPrecScalar>();
            MatrixLowPrec G_corr_low = G_correction.template cast<LowPrecScalar>();
            MatrixLowPrec DeltaV_low_res(n, n);
            DeltaV_low_res.noalias() = V_curr_low * G_corr_low;
            V += DeltaV_low_res.template cast<T>();
        }
        // ... (обновление sigma_relative_error_prev_iter и т.д.) ...
    } // Конец for iter_count

    if (history_log_file.is_open()) {
        history_log_file.close();
    }

    algo_result_output.U = U;
    algo_result_output.V = V;
    algo_result_output.S_diag_matrix = MatrixDyn::Zero(m,n);
    for(int i=0; i < std::min(n, k_min_mn) ; ++i) {
        if (i < current_sigma_vector.size()) {
            algo_result_output.S_diag_matrix(i,i) = current_sigma_vector(i);
        }
    }

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    algo_result_output.time_taken_s = std::chrono::duration<double>(overall_end_time - overall_start_time).count(); // Изменено на double

    return algo_result_output;
}

} // namespace SVD_Project

#endif // ITERATIVE_REFINEMENT_6_HPP
