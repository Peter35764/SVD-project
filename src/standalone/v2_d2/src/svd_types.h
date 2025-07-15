#ifndef SVD_TYPES_H
#define SVD_TYPES_H

#include <Eigen/Core> // Для Eigen::Dynamic и Eigen::Matrix
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/number.hpp>
#include <string>   // Для std::string в SVDAlgorithmResult (если используется для имен файлов и т.д.)
#include <limits>   // Для std::numeric_limits

namespace SVD_Project {

// Основной тип для высокоточных вычислений
template<unsigned int NumDigits>
using PrecisionType = boost::multiprecision::number<
    boost::multiprecision::backends::cpp_dec_float<NumDigits>,
    boost::multiprecision::et_on
>;

// Определение структуры для результатов работы алгоритмов SVD
// Это определение должно быть здесь, чтобы все iterative_refinement_X.h могли его видеть.
template<typename T>
struct SVDAlgorithmResult {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> U;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S_diag_matrix; // Диагональная матрица S (m x n)
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V;
    int iterations_taken = 0;
    T achieved_sigma_relative_error = std::numeric_limits<T>::quiet_NaN();
    T achieved_U_ortho_error = std::numeric_limits<T>::quiet_NaN(); // Ошибка ортогональности для U (например, ||I - U*U^T|| или ||I - U^T*U||)
    T achieved_V_ortho_error = std::numeric_limits<T>::quiet_NaN(); // Ошибка ортогональности для V (например, ||I - V*V^T|| или ||I - V^T*V||)
    double time_taken_s = 0.0; // Время выполнения в секундах
    // Можно добавить другие поля, если необходимо
};

} // namespace SVD_Project

#endif // SVD_TYPES_H
