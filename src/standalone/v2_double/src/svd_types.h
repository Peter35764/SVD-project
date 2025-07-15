// src/svd_types.h
#ifndef SVD_TYPES_H
#define SVD_TYPES_H

#include <Eigen/Core> // Нужен для Eigen::Matrix
#include <string>     // Нужен для std::string
#include <limits>     // Нужен для std::numeric_limits

// Не нужно #include <boost/multiprecision/cpp_dec_float.hpp> здесь,
// так как T_Scalar будет конкретным типом multiprecision, передаваемым как шаблонный параметр.

namespace SVD_Project {

template<typename T_Scalar>
struct SVDAlgorithmResult {
    using MatrixDyn = Eigen::Matrix<T_Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    MatrixDyn U;
    MatrixDyn S_diag_matrix; 
    MatrixDyn V;

    int iterations_taken = 0;
    T_Scalar achieved_sigma_relative_error = std::numeric_limits<T_Scalar>::quiet_NaN();
    T_Scalar achieved_U_ortho_error = std::numeric_limits<T_Scalar>::quiet_NaN(); 
    T_Scalar achieved_V_ortho_error = std::numeric_limits<T_Scalar>::quiet_NaN(); 
    double time_taken_s = 0.0;
};

// Сюда же можно вынести и другие общие типы или структуры, если они появятся.
// template<typename T_Scalar> 
// using Matrix = Eigen::Matrix<T_Scalar, Eigen::Dynamic, Eigen::Dynamic>; // Это уже есть как MatrixDyn
// template<typename T_Scalar> 
// using Vector = Eigen::Matrix<T_Scalar, Eigen::Dynamic, 1>; // Это уже есть как VectorDyn

} // namespace SVD_Project

#endif // SVD_TYPES_H