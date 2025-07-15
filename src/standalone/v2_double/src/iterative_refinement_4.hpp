#ifndef ITERATIVE_REFINEMENT_4_HPP
#define ITERATIVE_REFINEMENT_4_HPP

#include "iterative_refinement_4.h" // Содержит SVDAlgorithmResult и объявление AOIR_SVD_4
#include <Eigen/Dense>               // Основные возможности Eigen
#include <limits>                    // Для std::numeric_limits
#include <cmath>                     // Для std::round (если используется где-то еще)

#include <cassert>   // Для assert
#include <vector>    // Для std::vector
#include <iostream>  // Для std::cout, std::cerr
#include <fstream>   // Для std::ofstream
#include <string>    // Для std::string
#include <iomanip>   // Для std::setprecision, std::scientific, std::fixed
#include <stdexcept> // Для std::runtime_error
#include <chrono>    // Для замера времени

// Заголовки Boost Multiprecision
#include <boost/multiprecision/cpp_dec_float.hpp> // Основной тип для высокой точности

// Подключение OpenBLAS для смешанной точности
extern "C" {
#include <openblas/cblas.h>
}

namespace SVD_Project {

// Псевдонимы для типов, используемых в низкоточных вычислениях с BLAS
using LowPrec = double;
using MatrixLowPrec = Eigen::Matrix<LowPrec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>; // ColMajor важен для BLAS

// --- Реализация Конструкторов ---

// Конструктор по умолчанию
template<typename T>
AOIR_SVD_4<T>::AOIR_SVD_4() {} // Пустой, если не используется

// Основной конструктор, выполняющий SVD разложение
template<typename T>
AOIR_SVD_4<T>::AOIR_SVD_4(
    const MatrixDyn& A,                    // Входная матрица m x n
    const VectorDyn& true_singular_values, // Вектор "истинных" сингулярных чисел для расчета ошибки
    const std::string& history_filename_base // Базовое имя для файла лога истории сходимости (опционально)
) {
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::runtime_error("Input matrix A is empty in AOIR_SVD_4 constructor.");
    }

    const int m = A.rows();
    const int n = A.cols();

    // 1. Получаем начальное приближение U_thin (m x k_rank), V_thin (n x k_rank)
    //    с помощью стандартного SVD (Eigen::BDCSVD) в double точности.
    Eigen::MatrixXd A_double = A.template cast<double>();
    Eigen::BDCSVD<Eigen::MatrixXd> svd_double_initial(A_double, Eigen::ComputeThinU | Eigen::ComputeThinV); // Используем ComputeThinU/V

    MatrixDyn initial_U_thin = svd_double_initial.matrixU().template cast<T>(); // m x k_rank
    MatrixDyn initial_V_thin = svd_double_initial.matrixV().template cast<T>(); // n x k_rank
    // int k_rank = svd_double_initial.singularValues().size(); // Фактический ранг от BDCSVD

    // 2. Расширяем/формируем начальные U_full (m x m) и V_full (n x n)
    //    Они должны быть (близки к) ортогональным матрицам.
    //    Если BDCSVD вернул что-то осмысленное, используем это как основу.
    MatrixDyn U_full = MatrixDyn::Identity(m, m);
    if (initial_U_thin.cols() > 0 && initial_U_thin.rows() == m) {
        if (initial_U_thin.cols() <= m) {
             U_full.leftCols(initial_U_thin.cols()) = initial_U_thin;
        } else { // Очень редкий случай, если BDCSVD вернет больше столбцов, чем m
            U_full.leftCols(m) = initial_U_thin.leftCols(m); // Обрезаем
            std::cerr << "Warning (AOIR_SVD_4): Initial U from BDCSVD had more columns ("
                      << initial_U_thin.cols() << ") than rows (" << m << "). Truncated." << std::endl;
        }
    } else if (initial_U_thin.cols() > 0) { // Если количество строк не совпадает
        std::cerr << "Warning (AOIR_SVD_4): Initial U from BDCSVD has " << initial_U_thin.rows() 
                  << " rows, expected " << m << ". Using Identity for U_full." << std::endl;
    }
    // Если initial_U_thin пустой, U_full останется Identity.

    MatrixDyn V_full = MatrixDyn::Identity(n, n);
    if (initial_V_thin.cols() > 0 && initial_V_thin.rows() == n) {
        if (initial_V_thin.cols() <= n) {
            V_full.leftCols(initial_V_thin.cols()) = initial_V_thin;
        } else {
            V_full.leftCols(n) = initial_V_thin.leftCols(n);
            std::cerr << "Warning (AOIR_SVD_4): Initial V from BDCSVD had more columns ("
                      << initial_V_thin.cols() << ") than rows (" << n << "). Truncated." << std::endl;
        }
    } else if (initial_V_thin.cols() > 0) {
        std::cerr << "Warning (AOIR_SVD_4): Initial V from BDCSVD has " << initial_V_thin.rows() 
                  << " rows, expected " << n << ". Using Identity for V_full." << std::endl;
    }
    // Если initial_V_thin пустой, V_full останется Identity.

    // --- НАЧАЛО БЛОКА ДОБАВЛЕНИЯ ШУМА ---
    const T noise_level = T("1e-4"); 
    
    // Условие добавления шума, как в вашем оригинальном коде:
    // шум добавляется, если он значим по сравнению с машинным эпсилон текущего типа T.
    if (boost::multiprecision::abs(noise_level) > std::numeric_limits<T>::epsilon()) {
        // Генерируем случайные матрицы шума того же размера, что и U_full, V_full
        // Eigen::Matrix::Random() генерирует элементы в диапазоне [-1, 1]
        MatrixDyn noise_U_matrix = MatrixDyn::Random(m, m); 
        MatrixDyn noise_V_matrix = MatrixDyn::Random(n, n);

        U_full += noise_level * noise_U_matrix;
        V_full += noise_level * noise_V_matrix;

        // ОПЦИОНАЛЬНО: Ре-ортогонализация U_full и V_full после добавления шума.
        // Ваш оригинальный алгоритм этого не делал, и итерационное уточнение
        // само по себе должно восстанавливать ортогональность.
        // Если вы обнаружите, что это необходимо, можно раскомментировать:
        /*
        bool perform_reorthogonalization_after_noise = false; // Установите в true, если нужно
        if (perform_reorthogonalization_after_noise) {
            Eigen::HouseholderQR<MatrixDyn> qr_decomposition_U(U_full);
            U_full = qr_decomposition_U.householderQ(); 
            // Убедимся, что U_full осталась m x m (для некоторых версий Eigen householderQ() может вернуть m x min(m,rank))
            if (U_full.cols() < m) { 
                 MatrixDyn temp_U_ortho = MatrixDyn::Identity(m,m);
                 temp_U_ortho.leftCols(U_full.cols()) = U_full;
                 U_full = temp_U_ortho;
            }
        
            Eigen::HouseholderQR<MatrixDyn> qr_decomposition_V(V_full);
            V_full = qr_decomposition_V.householderQ();
            if (V_full.cols() < n) {
                 MatrixDyn temp_V_ortho = MatrixDyn::Identity(n,n);
                 temp_V_ortho.leftCols(V_full.cols()) = V_full;
                 V_full = temp_V_ortho;
            }
        }
        */
    }
    // --- КОНЕЦ БЛОКА ДОБАВЛЕНИЯ ШУМА ---

    // 3. Запускаем основной уточняющий алгоритм с (возможно, зашумленными) U_full и V_full
    SVDAlgorithmResult<T> result_data = MSVD_SVD_Refined(A, U_full, V_full, history_filename_base, true_singular_values);

    // 4. Сохраняем результаты работы алгоритма в членах класса
    this->U_computed = result_data.U;
    this->S_computed_diag_matrix = result_data.S_diag_matrix;
    this->V_computed = result_data.V;
    this->iterations_taken_ = result_data.iterations_taken;
    this->achieved_sigma_relative_error_ = result_data.achieved_sigma_relative_error;
    this->achieved_U_ortho_error_ = result_data.achieved_U_ortho_error;
    this->achieved_V_ortho_error_ = result_data.achieved_V_ortho_error;
    this->time_taken_s_ = result_data.time_taken_s;
}


// --- Основной Метод Итерационного Уточнения SVD ---
// Эта функция содержит ядро вашего Алгоритма 4.
// Я постарался максимально сохранить вашу оригинальную логику вычислений внутри цикла.
template<typename T>
SVDAlgorithmResult<T> AOIR_SVD_4<T>::MSVD_SVD_Refined(
    const MatrixDyn& A,        // Входная матрица m x n
    const MatrixDyn& U_initial, // Начальное приближение для U, (m x m), ортогональная
    const MatrixDyn& V_initial, // Начальное приближение для V, (n x n), ортогональная
    const std::string& history_filename_base, // Базовое имя для файла лога
    const VectorDyn& true_singular_values     // "Истинные" сингулярные числа для расчета ошибки
) {
    auto overall_start_time = std::chrono::high_resolution_clock::now(); // Замер общего времени работы
    SVDAlgorithmResult<T> algo_result_output; // Структура для возврата результатов

    const int m = A.rows();
    const int n = A.cols();
    const int k_min_mn = std::min(m, n); // Количество сингулярных чисел, которые мы ожидаем найти

    if (m < n) { 
        // Ваш оригинальный алгоритм был ориентирован на m >= n.
        // Если требуется поддержка m < n, нужно либо транспонировать A и поменять U/V,
        // либо адаптировать сам алгоритм. Пока оставим это ограничение.
        throw std::runtime_error("MSVD_SVD_Refined (Algo 4 original logic) requires m >= n.");
    }

    // Параметры для работы с текущей точностью типа T
    const T machine_epsilon = std::numeric_limits<T>::epsilon();
    const T sqrt_machine_epsilon = boost::multiprecision::sqrt(machine_epsilon); // Порог для знаменателей, инверсий и т.д.

    // Параметры для адаптивного критерия остановки цикла
    const int max_iterations = 10000; // Максимальное количество итераций во избежание бесконечного цикла
    const int min_iterations_for_stagnation_check = 5; // Минимальное количество итераций перед проверкой стагнации
    // Факторы для порогов сходимости (можно настраивать)
    const T stagnation_improvement_threshold_factor = T("1e-1"); // abs(err_prev - err_curr) < factor * eps * (масштаб ошибки)
    const T direct_convergence_threshold_factor_ortho = T("10.0"); // err_curr < factor * eps (для ошибок ортогональности)
    const T direct_convergence_threshold_factor_sigma = T("1e3");  // err_curr < factor * eps (для относительной ошибки сигм, может быть больше)

    // Рабочие матрицы U (m x m) и V (n x n)
    MatrixDyn U = U_initial;
    MatrixDyn V = V_initial;
    
    // Вектор для текущих сингулярных чисел (n элементов, так как Sigma_n была n x n)
    VectorDyn current_sigma_vector = VectorDyn::Zero(n);
    // Матрица Sigma_n (n x n) с current_sigma_vector на диагонали (как в вашем оригинале)
    MatrixDyn Sigma_n_matrix = MatrixDyn::Zero(n, n); 

    // Переменные для отслеживания сходимости ошибок между итерациями
    T sigma_relative_error_prev_iter = std::numeric_limits<T>::infinity();
    T internal_R_norm_prev_iter = std::numeric_limits<T>::infinity(); // Для ||I - U^T U||
    T internal_S_norm_prev_iter = std::numeric_limits<T>::infinity(); // Для ||I - V^T V||

    // Логирование истории сходимости (если указано имя файла)
    std::ofstream history_log_file;
    if (!history_filename_base.empty()) {
        std::string full_history_filename = history_filename_base + "_conv4_details.csv";
        history_log_file.open(full_history_filename);
        if (history_log_file) {
            history_log_file << "Iteration,SigmaRelError,Reported_U_OrthoError,Reported_V_OrthoError,"
                             << "Internal_R_norm,Internal_S_norm,StepTime_us,"
                             << "DeltaSigmaRelError,Delta_Internal_R_norm,Delta_Internal_S_norm\n";
        } else {
            std::cerr << "Warning (AOIR_SVD_4): Could not open history log file: " << full_history_filename << std::endl;
        }
    }

    // Основной итерационный цикл
    for (int iter_count = 0; iter_count < max_iterations; ++iter_count) {
        auto iter_step_start_time = std::chrono::high_resolution_clock::now();

        // === Начало блока вычислений, СТРОГО соответствующего вашей ОРИГИНАЛЬНОЙ логике итерации ===
        // Матрицы R_full, S_full, T_full, R11, T1 определяются и используются так, как это было у вас.
        // U предполагается (m x m), V предполагается (n x n).
        MatrixDyn R_full = MatrixDyn::Identity(m, m) - U.transpose() * U; // Ошибка U^T*U от I_m
        MatrixDyn S_full = MatrixDyn::Identity(n, n) - V.transpose() * V; // Ошибка V^T*V от I_n
        MatrixDyn T_full = U.transpose() * A * V; // Проекция A: (m x m)^T * (m x n) * (n x n) -> (m x n)
        
        // R11 и T1 берутся как n x n верхние левые подматрицы (n = A.cols())
        MatrixDyn R11 = R_full.topLeftCorner(n, n);
        MatrixDyn T1 = T_full.topLeftCorner(n, n);

        // Обновление диагональных элементов Sigma_n_matrix и вектора current_sigma_vector
        // (формула и порог machine_epsilon * T("10.0") взяты из вашего оригинального кода)
        for (int i = 0; i < n; ++i) { // Цикл до n (A.cols())
            T r_ii = R11(i, i); 
            T s_ii = S_full(i, i); 
            T t_ii = T1(i, i);

            T denom_sigma = T("1.0") - T("0.5") * (r_ii + s_ii);
            Sigma_n_matrix(i, i) = (boost::multiprecision::abs(denom_sigma) < machine_epsilon * T("10.0")) ? t_ii : (t_ii / denom_sigma);
            current_sigma_vector(i) = Sigma_n_matrix(i, i);
        }
        // === Конец блока расчета Sigma_n ===

        // --- Расчет ошибок для отчета и критерия адаптивной остановки ---
        // 1. Относительная ошибка сингулярных чисел (с использованием true_singular_values)
        T current_sigma_relative_error = std::numeric_limits<T>::quiet_NaN(); // Инициализация NaN
        if (true_singular_values.size() >= k_min_mn && k_min_mn > 0) { // Сравниваем k_min_mn сингулярных чисел
            T diff_norm_squared = T("0.0");
            T true_norm_squared = T("0.0");
            for(int i = 0; i < k_min_mn; ++i) {
                // current_sigma_vector имеет n элементов, true_singular_values - k_min_mn
                T diff = current_sigma_vector(i) - true_singular_values(i); 
                diff_norm_squared += diff * diff;
                true_norm_squared += true_singular_values(i) * true_singular_values(i);
            }
            if (true_norm_squared > machine_epsilon * machine_epsilon) { // Избегаем деления на ноль
                 current_sigma_relative_error = boost::multiprecision::sqrt(diff_norm_squared / true_norm_squared);
            } else if (diff_norm_squared > machine_epsilon * machine_epsilon) { // Если истинные малы, а вычисленные нет
                current_sigma_relative_error = boost::multiprecision::sqrt(diff_norm_squared);
            } else { // Оба малы
                current_sigma_relative_error = T("0.0");
            }
        }

        // 2. Ошибки ортогональности для отчета: ||I - U*U^T|| и ||I - V*V^T||
        //    (U и V здесь полные m x m и n x n матрицы)
        MatrixDyn Eye_m_for_report = MatrixDyn::Identity(m,m);
        MatrixDyn Eye_n_for_report = MatrixDyn::Identity(n,n);
        T current_U_ortho_error_reported = (Eye_m_for_report - U * U.transpose()).norm();
        T current_V_ortho_error_reported = (Eye_n_for_report - V * V.transpose()).norm();
        
        // 3. Внутренние нормы ошибок ортогональности (те, что использовались в вашем оригинальном коде для проверки сходимости)
        T current_internal_R_norm = R_full.norm(); // Норма ||I - U^T U||
        T current_internal_S_norm = S_full.norm(); // Норма ||I - V^T V||

        // --- Проверка сходимости (адаптивная) ---
        T delta_sigma_relative_error_abs = boost::multiprecision::abs(sigma_relative_error_prev_iter - current_sigma_relative_error);
        T delta_internal_R_norm_abs = boost::multiprecision::abs(internal_R_norm_prev_iter - current_internal_R_norm);
        T delta_internal_S_norm_abs = boost::multiprecision::abs(internal_S_norm_prev_iter - current_internal_S_norm);

        // Обновляем результаты, которые будут возвращены, на каждой итерации
        algo_result_output.iterations_taken = iter_count + 1;
        algo_result_output.achieved_sigma_relative_error = current_sigma_relative_error;
        algo_result_output.achieved_U_ortho_error = current_U_ortho_error_reported; // Отчетная ошибка U
        algo_result_output.achieved_V_ortho_error = current_V_ortho_error_reported; // Отчетная ошибка V

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
        
        // Критерии остановки:
        // 1. Относительная ошибка сингулярных чисел достаточно мала ИЛИ ее изменение незначительно.
        bool sigma_converged = (!boost::math::isnan(current_sigma_relative_error) && current_sigma_relative_error < direct_convergence_threshold_factor_sigma * machine_epsilon) ||
                               (delta_sigma_relative_error_abs < stagnation_improvement_threshold_factor * machine_epsilon * (sigma_relative_error_prev_iter > T(1) ? sigma_relative_error_prev_iter : T(1) ) );
        // 2. Внутренние ошибки ортогональности (как в вашем оригинале) достаточно малы ИЛИ их изменение незначительно.
        bool U_internal_ortho_converged = (current_internal_R_norm < direct_convergence_threshold_factor_ortho * machine_epsilon) ||
                                          (delta_internal_R_norm_abs < stagnation_improvement_threshold_factor * machine_epsilon);
        bool V_internal_ortho_converged = (current_internal_S_norm < direct_convergence_threshold_factor_ortho * machine_epsilon) ||
                                          (delta_internal_S_norm_abs < stagnation_improvement_threshold_factor * machine_epsilon);

        if (iter_count >= min_iterations_for_stagnation_check) { // Начинаем проверку только после нескольких итераций
            if (sigma_converged && U_internal_ortho_converged && V_internal_ortho_converged) {
                break; // Выход из цикла, если все условия сходимости выполнены
            }
        }

        // Сохраняем текущие значения ошибок для сравнения на следующей итерации
        sigma_relative_error_prev_iter = current_sigma_relative_error;
        internal_R_norm_prev_iter = current_internal_R_norm;
        internal_S_norm_prev_iter = current_internal_S_norm;


        // === Начало блока вычисления поправочных матриц F и G, СТРОГО соответствующего вашей ОРИГИНАЛЬНОЙ логике ===
        // `current_sigma_vector` соответствует вашему `sigma(n)`
        // `R11`, `S_full`, `T1` уже определены выше и соответствуют вашим именам и содержимому.

        MatrixDyn F_correction = MatrixDyn::Zero(m, m); // Поправочная матрица для U (m x m)
        MatrixDyn G_correction = MatrixDyn::Zero(n, n); // Поправочная матрица для V (n x n)

        // Диагональные части F_correction и G_correction (для первых n элементов, как в вашем коде)
        // F_ii = R11_ii / 2, G_ii = S_full_ii / 2
        for (int i = 0; i < n; ++i) { // Цикл до n (A.cols())
            F_correction(i, i) = R11(i, i) / T("2.0");
            G_correction(i, i) = S_full(i, i) / T("2.0");
        }
        
        // Внедиагональные части F_correction и G_correction (для первых n x n блоков)
        // F_ij = (alpha*sj + beta*si) / (sj^2 - si^2)
        // G_ij = (alpha*si + beta*sj) / (sj^2 - si^2)
        // alpha = T1_ij + sigma_j * R11_ij
        // beta  = T1_ji + sigma_j * S_full_ij (!!! S_full_ij, а не S_full_ji !!! - это из вашего оригинального кода)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue; // Пропускаем диагональные элементы
                
                T alpha_term = T1(i, j) + current_sigma_vector(j) * R11(i, j);
                T beta_term  = T1(j, i) + current_sigma_vector(j) * S_full(i, j); // Используем S_full(i,j) как в вашем оригинале
                
                T sigma_i_val = current_sigma_vector(i);
                T sigma_j_val = current_sigma_vector(j);
                T denominator = sigma_j_val * sigma_j_val - sigma_i_val * sigma_i_val;

                // Порог для знаменателя (sqrt_machine_epsilon * |sj^2|) взят из вашего оригинального кода (threshold * abs(sj*sj))
                if (boost::multiprecision::abs(denominator) > sqrt_machine_epsilon * boost::multiprecision::abs(sigma_j_val * sigma_j_val) ) {
                    F_correction(i, j) = (alpha_term * sigma_j_val + beta_term * sigma_i_val) / denominator;
                    G_correction(i, j) = (alpha_term * sigma_i_val + beta_term * sigma_j_val) / denominator;
                } else {
                    F_correction(i, j) = T("0.0"); // Если знаменатель слишком мал
                    G_correction(i, j) = T("0.0");
                }
            }
        }

        // Обработка случая m > n для матрицы F_correction (m x m)
        // (Логика взята из вашего оригинального кода)
        const int m_minus_n_rows = m - n; // Количество "дополнительных" строк в U
        if (m > n && m_minus_n_rows > 0) {
            // Инверсия Sigma_n_matrix (n x n) для использования в формулах
            MatrixDyn Sigma_n_inverse = MatrixDyn::Zero(n,n); 
            bool is_Sigma_n_invertible = true;
            for(int i=0; i<n; ++i) { // Инвертируем диагональные элементы Sigma_n_matrix
                // Порог для инверсии (sqrt_machine_epsilon) взят из вашего оригинального кода (threshold)
                if (boost::multiprecision::abs(Sigma_n_matrix(i,i)) < sqrt_machine_epsilon) {
                    is_Sigma_n_invertible = false; break;
                }
                Sigma_n_inverse(i,i) = T("1.0") / Sigma_n_matrix(i,i);
            }

            MatrixDyn F12_block, F21_block; // F12 (n x m-n), F21 (m-n x n)
            if (is_Sigma_n_invertible) {
                // T2_transpose = (U_right_part^T * A * V)^T
                // U_right_part это U.block(0, n, m, m_minus_n_rows) в вашем оригинале не было, было U.rightCols(m_minus_n)
                // Если U - это m x m, то U.rightCols(m_minus_n_rows) это блок m x (m-n)
                MatrixDyn U_right_part = U.rightCols(m_minus_n_rows); // m x (m-n)
                MatrixDyn T2_matrix_term = U_right_part.transpose() * A * V; // (m-n) x n
                MatrixDyn T2_transpose_term = T2_matrix_term.transpose();    // n x (m-n)
                
                F12_block.noalias() = -Sigma_n_inverse * T2_transpose_term; // (n x n) * (n x (m-n)) -> (n x (m-n))
                
                // R_full (m x m). R21_block = R_full.block(n, 0, m_minus_n_rows, n); (нижний левый блок R_full)
                MatrixDyn R21_block_term = R_full.block(n, 0, m_minus_n_rows, n); 
                F21_block = R21_block_term - F12_block.transpose(); // (m-n x n)
            } else { // Если Sigma_n не удалось инвертировать
                F12_block = MatrixDyn::Zero(n, m_minus_n_rows);
                F21_block = MatrixDyn::Zero(m_minus_n_rows, n);
            }
            // R22_block = R_full.block(n, n, m_minus_n_rows, m_minus_n_rows); (нижний правый блок R_full)
            MatrixDyn R22_block_term = R_full.block(n, n, m_minus_n_rows, m_minus_n_rows);
            MatrixDyn F22_block = T("0.5") * R22_block_term; // (m-n x m-n)

            // Заполняем соответствующие блоки в F_correction
            F_correction.block(0, n, n, m_minus_n_rows) = F12_block;
            F_correction.block(n, 0, m_minus_n_rows, n) = F21_block;
            F_correction.block(n, n, m_minus_n_rows, m_minus_n_rows) = F22_block;
        }
        // === Конец блока вычисления F и G ===


        // --- Обновление матриц U и V с использованием смешанной точности (cblas_dgemm) ---
        // U_new = U_old + U_old * F_correction
        // V_new = V_old + V_old * G_correction
        // (Эта логика обновления соответствует вашему оригинальному коду)
        { // Обновление U
            MatrixLowPrec U_current_low_prec = U.template cast<LowPrec>();           // m x m
            MatrixLowPrec F_correction_low_prec = F_correction.template cast<LowPrec>(); // m x m
            MatrixLowPrec DeltaU_low_prec_blas_result(m, m); // Результат U*F

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, m, m, // M, N, K for U*F
                        1.0,      // alpha
                        U_current_low_prec.data(), U_current_low_prec.outerStride(),
                        F_correction_low_prec.data(), F_correction_low_prec.outerStride(),
                        0.0,      // beta
                        DeltaU_low_prec_blas_result.data(), DeltaU_low_prec_blas_result.outerStride());
            
            U += DeltaU_low_prec_blas_result.template cast<T>(); // U_new = U_old + DeltaU
        }
        { // Обновление V
            MatrixLowPrec V_current_low_prec = V.template cast<LowPrec>();           // n x n
            MatrixLowPrec G_correction_low_prec = G_correction.template cast<LowPrec>(); // n x n
            MatrixLowPrec DeltaV_low_prec_blas_result(n, n); // Результат V*G

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, // M, N, K for V*G
                        1.0,      // alpha
                        V_current_low_prec.data(), V_current_low_prec.outerStride(),
                        G_correction_low_prec.data(), G_correction_low_prec.outerStride(),
                        0.0,      // beta
                        DeltaV_low_prec_blas_result.data(), DeltaV_low_prec_blas_result.outerStride());

            V += DeltaV_low_prec_blas_result.template cast<T>(); // V_new = V_old + DeltaV
        }
        // --- Конец блока обновления U и V ---

    } // Конец основного итерационного цикла (iter_count)

    if (history_log_file.is_open()) {
        history_log_file.close();
    }

    // Заполняем структуру с результатами
    algo_result_output.U = U;
    algo_result_output.V = V;
    // Формируем S_diag_matrix (m x n) из последнего вычисленного вектора current_sigma_vector (который имеет n элементов)
    algo_result_output.S_diag_matrix = MatrixDyn::Zero(m,n);
    for(int i=0; i < std::min(n, k_min_mn) ; ++i) { 
         algo_result_output.S_diag_matrix(i,i) = current_sigma_vector(i); // Используем значения из current_sigma_vector
    }
    // Значения iterations_taken и achieved_..._error уже установлены на последней итерации цикла.

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    algo_result_output.time_taken_s = std::chrono::duration<double>(overall_end_time - overall_start_time).count();
    
    return algo_result_output;
}

} // namespace SVD_Project

#endif // ITERATIVE_REFINEMENT_4_HPP