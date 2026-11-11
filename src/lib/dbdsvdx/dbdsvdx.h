#pragma once

#include <lapacke.h>
#include <Eigen/SVD>

/**
 * @brief SVD implementation wrapping LAPACK's DGESVDX / DBDSVDX
 * (bisection and inverse iteration for bidiagonal SVD / dqds-based values)
 */
template <typename _MatrixType>
class DBDSVDX_SVD : public Eigen::SVDBase<DBDSVDX_SVD<_MatrixType>> {
 private:
  typedef Eigen::SVDBase<DBDSVDX_SVD<_MatrixType>> Base;

 public:
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using Index = typename MatrixType::Index;

  DBDSVDX_SVD() = default;
  DBDSVDX_SVD(const MatrixType& matrix, unsigned int computationOptions);

  DBDSVDX_SVD& compute(const MatrixType& matrix, unsigned int computationOptions);
};

template <typename _MatrixType>
using DGESVDX_SVD = DBDSVDX_SVD<_MatrixType>;

template <typename _MatrixType>
struct Eigen::internal::traits<DBDSVDX_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "dbdsvdx.cpp"
