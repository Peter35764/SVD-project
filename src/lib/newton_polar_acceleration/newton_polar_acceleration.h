#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace SVD_Project {
template <typename _MatrixType>
class Newton_Polar_Acceleration {
 public:
  typedef _MatrixType                             MatrixType;
  typedef typename MatrixType::Scalar             Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index                            Index;

  Newton_Polar_Acceleration() = default;

  void       setScalingThreshold(RealScalar tau) { m_tau = tau; }
  RealScalar scalingThreshold() const            { return m_tau; }

  Index steps()       const { return m_steps; }
  Index scaledSteps() const { return m_scaled_steps; }

  MatrixType compute(
      const MatrixType& A,
      Index max_iter = 100,
      RealScalar tol = std::numeric_limits<RealScalar>::epsilon());

 private:
  static RealScalar norm1  (const MatrixType& M);
  static RealScalar normInf(const MatrixType& M);

  RealScalar m_tau          = RealScalar(1e-2);
  Index      m_steps        = 0;
  Index      m_scaled_steps = 0;
};

template <typename _MatrixType>
class Newton_Polar_SVD_Acceleration
    : public Eigen::SVDBase<Newton_Polar_SVD_Acceleration<_MatrixType>> {
 private:
  typedef Eigen::SVDBase<Newton_Polar_SVD_Acceleration<_MatrixType>> Base;

 public:
  typedef _MatrixType                             MatrixType;
  typedef typename MatrixType::Scalar             Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index                            Index;

  Newton_Polar_SVD_Acceleration() {}
  Newton_Polar_SVD_Acceleration(const _MatrixType& matrix,
                                unsigned int computationOptions = 0);

  Newton_Polar_SVD_Acceleration<_MatrixType>& compute(
      const _MatrixType& matrix, unsigned int computationOptions);
  Newton_Polar_SVD_Acceleration<_MatrixType>& compute(const _MatrixType& matrix) {
    return compute(matrix, 0);
  }

  void  setScalingThreshold(RealScalar tau) { m_tau = tau; }
  Index lastSteps()       const { return m_steps; }
  Index lastScaledSteps() const { return m_scaled_steps; }

 private:
  RealScalar m_tau          = RealScalar(1e-2);
  Index      m_steps        = 0;
  Index      m_scaled_steps = 0;
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<
    SVD_Project::Newton_Polar_SVD_Acceleration<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "newton_polar_acceleration.hpp"