#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/Core>
#include <Eigen/SVD>

namespace SVD_Project {

template<typename _MatrixType>
class GivRef_SVD : public Eigen::SVDBase<GivRef_SVD<_MatrixType> >
{
    typedef Eigen::SVDBase<GivRef_SVD> Base;

public:
};

} // namespace SVD_Project

#include "givens_refinement.hpp"

#endif // GIVENS_REFINEMENT_H
