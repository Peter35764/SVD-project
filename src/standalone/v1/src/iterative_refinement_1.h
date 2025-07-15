#ifndef ITERATIVE_REFINEMENT_1_H
#define ITERATIVE_REFINEMENT_1_H

#include <Eigen/Core>
#include <Eigen/SVD>

namespace SVD_Project {

template<typename T>
class AOIR_SVD_1 {
private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> U;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> S;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V;

    void Set_U(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A);
    void Set_V(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A);
    void Set_S(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A);

    AOIR_SVD_1 MSVD_SVD(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Ui,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Vi,
        const std::string& history_filename_base = ""); 

public:
    AOIR_SVD_1();
    explicit AOIR_SVD_1(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A,
        const std::string& history_filename_base = "");

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrixU() const;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrixV() const;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> singularValues() const;
};

} 

#include "iterative_refinement_1.hpp"

#endif 
