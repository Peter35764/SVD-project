#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace SVD_Project {

template <typename _MatrixType>
class Newton_Polar {
 public:
  typedef _MatrixType                             MatrixType;
  typedef typename MatrixType::Scalar             Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index                            Index;

  Newton_Polar() {}

  MatrixType compute(
      const MatrixType& A,
      Index max_iter = 100,
      RealScalar tol = std::numeric_limits<RealScalar>::epsilon());

 private:
  static RealScalar norm1(const MatrixType& M) {
    return M.cwiseAbs().colwise().sum().maxCoeff();
  }
};

template <typename _MatrixType>
class Newton_Polar_SVD
    : public Eigen::SVDBase<Newton_Polar_SVD<_MatrixType>> {
 private:
  typedef Eigen::SVDBase<Newton_Polar_SVD<_MatrixType>> Base;

 public:
  typedef _MatrixType                             MatrixType;
  typedef typename MatrixType::Scalar             Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index                            Index;

  Newton_Polar_SVD() {}
  Newton_Polar_SVD(const _MatrixType& matrix,
                   unsigned int computationOptions = 0);

  Newton_Polar_SVD<_MatrixType>& compute(const _MatrixType& matrix,
                                         unsigned int computationOptions);
  Newton_Polar_SVD<_MatrixType>& compute(const _MatrixType& matrix) {
    return compute(matrix, 0);
  }
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::Newton_Polar_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "newton_polar.hpp"