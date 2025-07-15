#ifndef ITERATIVE_REFINEMENT_6_H
#define ITERATIVE_REFINEMENT_6_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <string>
#include <vector>
#include <limits>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include "svd_types.h" 

namespace SVD_Project {

template<typename T>
class AOIR_SVD_6 {
public:
    using MatrixDyn = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorDyn = Eigen::Matrix<T, Eigen::Dynamic, 1>;

private:
    MatrixDyn U_computed;
    MatrixDyn S_computed_diag_matrix;
    MatrixDyn V_computed;

    int iterations_taken_ = 0;
    T achieved_sigma_relative_error_ = std::numeric_limits<T>::quiet_NaN();
    T achieved_U_ortho_error_ = std::numeric_limits<T>::quiet_NaN();
    T achieved_V_ortho_error_ = std::numeric_limits<T>::quiet_NaN();
    double time_taken_s_ = 0.0;

    SVDAlgorithmResult<T> MSVD_SVD_Refined(
        const MatrixDyn& A,
        const MatrixDyn& U_initial, 
        const MatrixDyn& V_initial, 
        const std::string& history_filename_base,
        const VectorDyn& true_singular_values
    );

public:
    AOIR_SVD_6();

    explicit AOIR_SVD_6(
        const MatrixDyn& A,
        const VectorDyn& true_singular_values,
        const std::string& history_filename_base = ""
    );

    MatrixDyn matrixU() const { return U_computed; }
    MatrixDyn matrixV() const { return V_computed; }
    MatrixDyn singularValues() const { return S_computed_diag_matrix; }

    int iterations_taken() const { return iterations_taken_; }
    T achieved_sigma_relative_error() const { return achieved_sigma_relative_error_; }
    T achieved_U_ortho_error() const { return achieved_U_ortho_error_; }
    T achieved_V_ortho_error() const { return achieved_V_ortho_error_; }
    double time_taken_s() const { return time_taken_s_; }
};

} 

#include "iterative_refinement_6.hpp"

#endif // ITERATIVE_REFINEMENT_6_H