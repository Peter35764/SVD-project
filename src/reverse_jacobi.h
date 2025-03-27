#ifndef REVERSE_JACOBI_H
#define REVERSE_JACOBI_H

#include <Eigen/Core>
#include <Eigen/Jacobi>
#include <Eigen/SVD>
#include <Eigen/src/Core/util/ForwardDeclarations.h>
#include <Eigen/src/SVD/SVDBase.h>
#include <boost/math/tools/minima.hpp>

namespace SVD_Project {

// Allow only Sub_SVD implementations derived from Eigen::SVDBase
template<typename Sub_SVD>
requires std::derived_from<Sub_SVD, Eigen::SVDBase<Sub_SVD>> class RevJac_SVD : public Sub_SVD
{
private:
    typedef Eigen::internal::traits<Sub_SVD>::MatrixType MatrixType;
    typedef Eigen::internal::traits<MatrixType>::Scalar Scalar;

public:
    RevJac_SVD(const MatrixType &initial, unsigned int computationOptions);
};

} // namespace SVD_Project

template<typename Sub_SVD>
struct Eigen::internal::traits<SVD_Project::RevJac_SVD<Sub_SVD>>
    : Eigen::internal::traits<typename Sub_SVD::MatrixType>
{
    typedef Eigen::internal::traits<Sub_SVD>::MatrixType MatrixType;
    typedef Eigen::internal::traits<MatrixType>::Scalar Scalar;
};

#include "reverse_jacobi.hpp"

#endif // REVERSE_JACOBI_H