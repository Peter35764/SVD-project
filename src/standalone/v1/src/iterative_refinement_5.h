#ifndef ITERATIVE_REFINEMENT_5_H
#define ITERATIVE_REFINEMENT_5_H

#include <Eigen/Core>
#include <Eigen/SVD>
#include <string>

namespace SVD_Project {

template<typename T>
class AOIR_SVD_5 {
private:

    using MatrixDyn = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    MatrixDyn U;
    MatrixDyn S;
    MatrixDyn V;


    void Set_U(const MatrixDyn& A);
    void Set_V(const MatrixDyn& A);

    void Set_S(const MatrixDyn& A);


    AOIR_SVD_5 MSVD_SVD(const MatrixDyn& A,
                        const MatrixDyn& Ui,
                        const MatrixDyn& Vi,
                        const std::string& history_filename_base);

public:

    AOIR_SVD_5();

    explicit AOIR_SVD_5(const MatrixDyn& A,
                        const std::string& history_filename_base = "");


    MatrixDyn matrixU() const;
    MatrixDyn matrixV() const;
    MatrixDyn singularValues() const;
};

}


#include "iterative_refinement_5.hpp"

#endif