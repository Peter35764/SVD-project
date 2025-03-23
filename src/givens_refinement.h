#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/Core>
#include <Eigen/SVD>

namespace SVD_Project {
/*
 * Класс SVD разложения наследуется от класса Eigen::SVDBase, причем должен существовать 
 * конструктор класса, который вторым параметром принимает настройку вычислений 
 * матриц U и V, т.е. thin или full.
 */

template<typename _MatrixType>
class GivRef_SVD : public Eigen::SVDBase<GivRef_SVD<_MatrixType> >
{
    typedef Eigen::SVDBase<GivRef_SVD> Base;

public:
    GivRef_SVD();
};

} // namespace SVD_Project

#include "givens_refinement.hpp"

#endif // GIVENS_REFINEMENT_H
