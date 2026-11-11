#pragma once

#include <lapacke.h>
#include <Eigen/SVD>
#include <stdexcept>
#include <string>

template <typename _MatrixType>
class DGESDD_SVD : public Eigen::SVDBase<DGESDD_SVD<_MatrixType>> {
 private:
  typedef Eigen::SVDBase<DGESDD_SVD<_MatrixType>> Base;

 public:
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index Index;

  DGESDD_SVD() {}
  DGESDD_SVD(const _MatrixType& matrix, unsigned int computationOptions = 0);

  DGESDD_SVD<_MatrixType>& compute(const _MatrixType& matrix,
                                   unsigned int computationOptions);
  DGESDD_SVD<_MatrixType>& compute(const _MatrixType& matrix) {
    return compute(matrix, 0);
  }
};

template <typename _MatrixType>
struct Eigen::internal::traits<DGESDD_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "dgesdd.cpp"
