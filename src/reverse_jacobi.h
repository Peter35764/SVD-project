#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/Core>
#include <Eigen/SVD>

namespace SVD_Project {

template<typename _MatrixType>
class RevJac_SVD : public Eigen::SVDBase<RevJac_SVD<_MatrixType> >
{
    typedef Eigen::SVDBase<RevJac_SVD> Base;

public:
    RevJac_SVD();
};

} // namespace SVD_Project

#include "reverse_jacobi.hpp"

#endif // GIVENS_REFINEMENT_H
