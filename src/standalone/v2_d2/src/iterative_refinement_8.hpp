// Файл: src/iterative_refinement_8.hpp
#ifndef ITERATIVE_REFINEMENT_8_HPP
#define ITERATIVE_REFINEMENT_8_HPP

#include "iterative_refinement_8.h" // Объявление класса AOIR_SVD_8

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
#include <type_traits> // Для std::is_same, если понадобится

// #include <boost/multiprecision/cpp_dec_float.hpp> // Уже включен через svd_types.h -> test_common.h
#include <boost/math/special_functions/fpclassify.hpp> // Для boost::math::isnan

// extern "C" { // УДАЛЯЕМ, так как cblas_sgemm больше не используется
// #include <openblas/cblas.h>
// }

namespace SVD_Project {

// Вспомогательная структура для определения типов с пониженной точностью
template<typename T_working_precision>
struct LowPrecisionTypes_Algo8 { // Уникальное имя для этого алгоритма
    static constexpr int working_digits = std::numeric_limits<T_working_precision>::digits10;
    static constexpr int calculated_low_prec_digits = working_digits / 2;
    static constexpr int low_prec_target_digits = (calculated_low_prec_digits > 0) ? calculated_low_prec_digits : 1;

    using LowPrecBackend = boost::multiprecision::backends::cpp_dec_float<low_prec_target_digits>;
    using Scalar = boost::multiprecision::number<LowPrecBackend, boost::multiprecision::et_on>; // Предполагаем et_on
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
};

// using LowPrec = float; // УДАЛЕНО
// using MatrixLP = Eigen::Matrix<LowPrec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // УДАЛЕНО

template<typename T>
AOIR_SVD_8<T>::AOIR_SVD_8() {}

template<typename T>
AOIR_SVD_8<T>::AOIR_SVD_8(
    const MatrixDyn& A,
    const VectorDyn& true_singular_values,
    const std::string& history_filename_base)
{
    // Определяем типы пониженной точности
    using LowPrecScalar = typename LowPrecisionTypes_Algo8<T>::Scalar;
    using MatrixLowPrec = typename LowPrecisionTypes_Algo8<T>::Matrix;

    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("Input matrix A is empty in AOIR_SVD_8 constructor.");
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
    } // (обработка предупреждений опущена для краткости)

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
SVDAlgorithmResult<T> AOIR_SVD_8<T>::MSVD_SVD_Refined(
    const MatrixDyn& A,
    const MatrixDyn& U_initial,
    const MatrixDyn& V_initial,
    const std::string& history_filename_base,
    const VectorDyn& true_singular_values)
{
    // Определяем типы пониженной точности
    using LowPrecScalar = typename LowPrecisionTypes_Algo8<T>::Scalar;
    using MatrixLowPrec = typename LowPrecisionTypes_Algo8<T>::Matrix; // Переименовал из MatrixLP

    auto overall_start_time = std::chrono::high_resolution_clock::now();
    SVDAlgorithmResult<T> algo_result_output;

    const int m = A.rows();
    const int n = A.cols();
    const int k_min_mn = std::min(m, n);
    const int m_minus_n = m - n;

    if (m < n) {
        throw std::runtime_error("MSVD_SVD_Refined (Algo 8) currently requires m >= n.");
    }

    const T machine_epsilon = std::numeric_limits<T>::epsilon();
    const T sqrt_machine_epsilon = boost::multiprecision::sqrt(machine_epsilon);

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

    const T sqrt2 = boost::multiprecision::sqrt(T("2.0"));
    const T inv_sqrt2 = T("1.0") / sqrt2;
    const T one = T("1.0");
    const T two = T("2.0");
    const T half = T("0.5");
    const T ten = T("10.0");

    std::ofstream history_log_file;
    if (!history_filename_base.empty()) {
        std::string full_history_filename = history_filename_base + "_conv8_details.csv"; // conv8
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
        MatrixDyn S_full_ortho = MatrixDyn::Identity(n, n) - V.transpose() * V;

        MatrixDyn U1 = U.leftCols(n);
        MatrixDyn P; P.noalias() = A * V;
        MatrixDyn Q; Q.noalias() = A.transpose() * U1;

        VectorDyn r_diag(n), t_diag(n);
        for (int i = 0; i < n; ++i) {
            T u_norm_sq = U.col(i).squaredNorm();
            T v_norm_sq = V.col(i).squaredNorm();
            r_diag(i) = one - (u_norm_sq + v_norm_sq) / two;
            t_diag(i) = U1.col(i).dot(P.col(i));
            T denom_sigma = one - r_diag(i);
            Sigma_n_matrix(i, i) = (boost::multiprecision::abs(denom_sigma) < machine_epsilon * ten) ? t_diag(i) : (t_diag(i) / denom_sigma);
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

        MatrixDyn U2;
        if (m > n && m_minus_n > 0) {
            U2 = U.rightCols(m_minus_n);
        } else {
            U2.resize(m, 0); // Убедимся, что U2 имеет правильные размеры, даже если пустая
        }

        MatrixDyn P1_mat; P1_mat.noalias() = Q - V * Sigma_n_matrix.transpose();
        MatrixDyn P2_mat; P2_mat.noalias() = P - U1 * Sigma_n_matrix;

        MatrixDyn P3_mat, P4_mat;
        // P3_mat = V^T * P1_mat (в пониженной точности)
        {
            MatrixLowPrec V_low = V.template cast<LowPrecScalar>();
            MatrixLowPrec P1_low = P1_mat.template cast<LowPrecScalar>();
            MatrixLowPrec P3_low(n, n); // V(n,n), P1_mat(n,n) -> P3_mat(n,n)
            P3_low.noalias() = V_low.transpose() * P1_low;
            P3_mat = P3_low.template cast<T>();
        }
        // P4_mat = U1^T * P2_mat (в пониженной точности)
        {
            MatrixLowPrec U1_low = U1.template cast<LowPrecScalar>();
            MatrixLowPrec P2_low = P2_mat.template cast<LowPrecScalar>();
            MatrixLowPrec P4_low(n, n); // U1(m,n), P2_mat(m,n) -> P4_mat(n,n)
            P4_low.noalias() = U1_low.transpose() * P2_low;
            P4_mat = P4_low.template cast<T>();
        }

        MatrixDyn Q1_mat = (half * (P3_mat + P4_mat));
        MatrixDyn Q2_mat = (half * (P3_mat - P4_mat));
        MatrixDyn Q3_mat, Q4_mat;

        if (m > n && m_minus_n > 0) {
            MatrixDyn PTU2; PTU2.noalias() = P.transpose() * U2; // P(m,n), U2(m,m-n) -> P^T(n,m) * U2(m,m-n) -> (n,m-n)
            Q3_mat = inv_sqrt2 * PTU2;

            MatrixDyn U2P2_high; // U2^T * P2_mat (в пониженной точности)
            {
                MatrixLowPrec U2_low = U2.template cast<LowPrecScalar>();
                MatrixLowPrec P2_low = P2_mat.template cast<LowPrecScalar>();
                MatrixLowPrec U2P2_low(m_minus_n, n); // U2(m,m-n), P2_mat(m,n) -> U2^T(m-n,m) * P2_mat(m,n) -> (m-n,n)
                U2P2_low.noalias() = U2_low.transpose() * P2_low;
                U2P2_high = U2P2_low.template cast<T>();
            }
            Q4_mat = inv_sqrt2 * U2P2_high;
        } else {
            Q3_mat.resize(n, 0); Q3_mat.setZero();
            Q4_mat.resize(0, n); Q4_mat.setZero(); // Размеры (m-n, n), т.е. (0, n)
        }

        MatrixDyn E1_mat = MatrixDyn::Zero(n, n); MatrixDyn E2_mat = MatrixDyn::Zero(n, n);
        MatrixDyn E3_mat = MatrixDyn::Zero(n, m_minus_n); MatrixDyn E4_mat = MatrixDyn::Zero(m_minus_n, n);
        MatrixDyn E5_mat = MatrixDyn::Zero(m_minus_n, m_minus_n);

        for(int i=0; i<n; ++i) {
            E1_mat(i,i) = r_diag(i) / two;
            for(int j=0; j<n; ++j) {
                if (i == j) {
                    E2_mat(i,j) = T("0.0"); continue;
                }
                T sigma_i = current_sigma_vector(i); T sigma_j = current_sigma_vector(j);
                T denom1 = sigma_j - sigma_i;
                T denom2 = sigma_i + sigma_j;
                if (boost::multiprecision::abs(denom1) > machine_epsilon * (boost::multiprecision::abs(sigma_i) + boost::multiprecision::abs(sigma_j))) {
                    E1_mat(i,j) = Q1_mat(i,j) / denom1;
                }
                if (denom2 > machine_epsilon) { // Условие из оригинального кода
                    E2_mat(i,j) = Q2_mat(i,j) / denom2;
                }
            }
        }

        if (m > n && m_minus_n > 0) {
            MatrixDyn Sigma_n_inv_for_E = MatrixDyn::Zero(n,n);
            bool invertible_for_E = true;
            for(int i=0; i<n; ++i) {
                if (boost::multiprecision::abs(current_sigma_vector(i)) < sqrt_machine_epsilon) { invertible_for_E = false; break; }
                Sigma_n_inv_for_E(i,i) = one / current_sigma_vector(i);
            }
            if (invertible_for_E) {
                // Q3_mat (n, m-n)
                E3_mat.noalias() = Sigma_n_inv_for_E * Q3_mat; // (n,n) * (n,m-n) -> (n,m-n)
                // Q4_mat (m-n, n)
                E4_mat.noalias() = Q4_mat * Sigma_n_inv_for_E; // (m-n,n) * (n,n) -> (m-n,n)
            }
            MatrixDyn R_full_bottom_right_block_E5 = R_full.bottomRightCorner(m_minus_n, m_minus_n);
            E5_mat.noalias() = half * R_full_bottom_right_block_E5;
        }

        MatrixDyn F_correction_final = MatrixDyn::Zero(m, m);
        MatrixDyn G_correction_final = E1_mat + E2_mat;
        F_correction_final.topLeftCorner(n,n) = E1_mat - E2_mat;
        if (m > n && m_minus_n > 0) {
            F_correction_final.block(0, n, n, m_minus_n) = inv_sqrt2 * E3_mat;
            F_correction_final.block(n, 0, m_minus_n, n) = inv_sqrt2 * E4_mat;
            F_correction_final.block(n, n, m_minus_n, m_minus_n) = E5_mat;
        }

        // Обновление U и V с использованием пониженной точности для умножения
        {
            MatrixLowPrec U_curr_low = U.template cast<LowPrecScalar>();
            MatrixLowPrec F_corr_low = F_correction_final.template cast<LowPrecScalar>();
            MatrixLowPrec DeltaU_low_res(m, m);
            DeltaU_low_res.noalias() = U_curr_low * F_corr_low;
            U += DeltaU_low_res.template cast<T>();
        }
        {
            MatrixLowPrec V_curr_low = V.template cast<LowPrecScalar>();
            MatrixLowPrec G_corr_low = G_correction_final.template cast<LowPrecScalar>();
            MatrixLowPrec DeltaV_low_res(n, n);
            DeltaV_low_res.noalias() = V_curr_low * G_corr_low;
            V += DeltaV_low_res.template cast<T>();
        }
        // ... (обновление sigma_relative_error_prev_iter и т.д.) ...
    } // Конец for iter_count

    // ... (закрытие лог файла, заполнение algo_result_output) ...
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

#endif // ITERATIVE_REFINEMENT_8_HPP
