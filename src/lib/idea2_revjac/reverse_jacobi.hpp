#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>
#include <boost/math/tools/minima.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

#include "reverse_jacobi.h"

namespace SVD_Project {

template <typename _MatrixType>
RevJac_SVD<_MatrixType>& RevJac_SVD<_MatrixType>::Compute(
    const MatrixType& initial, unsigned int computationOptions) {
  const size_t MAX_ITERATIONS = 1000;
  const double TOLERANCE = 1e-10;

  assert(this->m_singularValues.rows() ==
             std::min(initial.rows(), initial.cols()) &&
         "Singular value vector size and matrix size do not match");

  // This algorithm is intended to calculate full U and V matrices, thus
  // other values do not make sense here
  this->m_computeFullU = true;
  this->m_computeFullV = true;

  // Initialize matrix for accumulating right rotations in order not to
  // transpose it 2 times each iteration
  auto transposedMatrixV = this->m_matrixV.transpose();

  // Initialize the current approximation
  MatrixType currentApproximation =
      this->m_matrixU * this->m_singularValues.asDiagonal() * transposedMatrixV;

  // DEBUG: Output divergence before iterating
  Scalar divergence = (currentApproximation - initial).norm();
  if (this->m_divOstream) {
    *this->m_divOstream << "Divergence: " << std::to_string(divergence) << "\n";
  }

  Scalar w, x, y, z, c1, s1;
  // Main iteration loop
  for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter) {
    for (Index i = 0; i < initial.rows() - 1; ++i) {
      for (Index j = i + 1; j < initial.cols(); ++j) {
        w = currentApproximation(i, i);
        x = currentApproximation(i, j);
        y = currentApproximation(j, i);
        z = currentApproximation(j, j);
        this->USVD(w, x, y, z, c1, s1);

        auto leftRotation = Eigen::JacobiRotation<Scalar>(c1, s1);

        this->m_matrixU.applyOnTheLeft(i, j, leftRotation.adjoint());
        currentApproximation.applyOnTheLeft(i, j, leftRotation.adjoint());

        auto minimizedFunction = [currentApproximation, initial, i,
                                  j](Scalar phi) {
          MatrixType tempApproximation = currentApproximation;
          Scalar c = std::cos(phi);
          Scalar s = std::sin(phi);
          Eigen::JacobiRotation<Scalar> rotation =
              Eigen::JacobiRotation<Scalar>(c, s);
          tempApproximation.applyOnTheRight(i, j, rotation);
          return (tempApproximation - initial).norm();
        };

        auto result = boost::math::tools::brent_find_minima(
            minimizedFunction, -std::numbers::pi / 4, std::numbers::pi / 4,
            std::numeric_limits<Scalar>::digits);

        Scalar phi = result.first;
        Scalar c2 = std::cos(phi);
        Scalar s2 = std::sin(phi);
        auto rightRotation = Eigen::JacobiRotation<Scalar>(c2, s2);

        transposedMatrixV.applyOnTheRight(i, j, rightRotation);
        currentApproximation.applyOnTheRight(i, j, rightRotation);
      }

      // DEBUG: Recalculate divergence for debugging purposes and to check if
      // convergence has been reached
      Scalar divergence = (currentApproximation - initial).norm();

      if (this->m_divOstream) {
        *this->m_divOstream << "Divergence: " << std::to_string(divergence)
                            << "\n";
      }

      // Establish if convergence has been reached
      if (divergence < TOLERANCE) {
        break;
      }
    }
  }

  this->m_matrixV = transposedMatrixV.transpose();
  this->m_isInitialized = true;
  return *this;
}

// USVD algorithm implementation used to calculate cos and sin for left jacobi
// rotation symmertizing the 2x2 matrix
// Article: R. P. Brent, F. T. Luk, and C. Van Loan, Computation of the
// singular value decomposition using mesh-connected processors
template <typename _MatrixType>
void RevJac_SVD<_MatrixType>::USVD(Scalar w, Scalar x, Scalar y, Scalar z,
                                   Scalar& c1, Scalar& s1) {
  const double EPS = 1e-30;
  bool flag = false;
  if (y == 0 && z == 0) {
    y = x;
    x = 0;
    flag = true;
  }

  Scalar mu1 = w + z;
  Scalar mu2 = x - y;

  Scalar c, s;
  if (std::abs(mu2) <= EPS * std::abs(mu1)) {
    c = 1;
    s = 0;
  } else {
    Scalar rho = mu1 / mu2;
    Scalar sgnRho = (rho > 0) - (rho < 0);
    s = sgnRho / std::sqrt(1 + rho * rho);
    c = s * rho;
  }

  mu1 = s * (x + y) + c * (z - w);
  mu2 = 2 * (c * x - s * z);

  Scalar c2, s2;
  if (std::abs(mu2) <= EPS * std::abs(mu1)) {
    c2 = 1;
    s2 = 0;
  } else {
    Scalar rho2 = mu1 / mu2;
    Scalar sgnRho2 = (rho2 > 0) - (rho2 < 0);
    Scalar t2 = sgnRho2 / (std::abs(rho2) + std::sqrt(1 + rho2 * rho2));
    c2 = 1 / std::sqrt(1 + t2 * t2);
    s2 = c2 * t2;
  }

  c1 = c2 * c - s2 * s;
  s1 = s2 * c + c2 * s;
  Scalar d1 = c1 * (w * c1 - x * s2) - s1 * (y * c2 - z * s2);
  Scalar d2 = s1 * (w * s2 + x * c2) + c1 * (y * s2 + z * c2);

  if (flag) {
    // Used in original implementation but not here because we don't need
    // these values c2 = c1; s2 = s1;
    c1 = 1;
    s1 = 0;
  }
}

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP
