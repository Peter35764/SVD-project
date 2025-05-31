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
RevJac_SVD<_MatrixType>::RevJac_SVD(const MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    unsigned int computationOptions) {
  Index m = initial.rows();
  Index n = initial.cols();
  this->m_matrixU = MatrixDynamic::Identity(m, m);
  this->m_matrixV = MatrixDynamic::Identity(n, n);
  this->m_singularValues = singularValues;
  Compute(initial, computationOptions);
}

template <typename _MatrixType>
RevJac_SVD<_MatrixType>::RevJac_SVD(const MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    std::ostream* os,
                                    unsigned int computationOptions) {
  Index m = initial.rows();
  Index n = initial.cols();
  this->m_matrixU = MatrixDynamic::Identity(m, m);
  this->m_matrixV = MatrixDynamic::Identity(n, n);
  this->m_singularValues = singularValues;
  this->m_divOstream = os;
  Compute(initial, computationOptions);
};

template <typename _MatrixType>
RevJac_SVD<_MatrixType>::RevJac_SVD(const MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    const MatrixDynamic& matrixU,
                                    const MatrixDynamic& matrixV,
                                    unsigned int computationOptions) {
  this->m_singularValues = singularValues;
  this->m_matrixU = matrixU;
  this->m_matrixV = matrixV;
  Compute(initial, computationOptions);
};

template <typename _MatrixType>
RevJac_SVD<_MatrixType>::RevJac_SVD(const MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    const MatrixDynamic& matrixU,
                                    const MatrixDynamic& matrixV,
                                    std::ostream* os,
                                    unsigned int computationOptions) {
  this->m_singularValues = singularValues;
  this->m_matrixU = matrixU;
  this->m_matrixV = matrixV;
  this->m_divOstream = os;
  Compute(initial, computationOptions);
};

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

        // Calculate such sin and cos values that J^T * A simmetrizes the
        // desired 2 by 2
        this->calculateSymmetrizingRotation(w, x, y, z, c1, s1);

        auto leftRotation = Eigen::JacobiRotation<Scalar>(c1, s1);

        this->m_matrixU.applyOnTheLeft(i, j, leftRotation.adjoint());
        currentApproximation.applyOnTheLeft(i, j, leftRotation.adjoint());

        auto minimizedFunction = [currentApproximation, initial, i,
                                  j](Scalar phi) {
          MatrixType tempApproximation = currentApproximation;
          Scalar c = std::cos(phi);
          Scalar s = std::sin(phi);
          Eigen::JacobiRotation<Scalar> rightRotation =
              Eigen::JacobiRotation<Scalar>(c, s);
          tempApproximation.applyOnTheRight(i, j, rightRotation);
          return (tempApproximation - initial).norm();
        };

        // Minimize the function and get the resulting angle value
        auto result = boost::math::tools::brent_find_minima(
            minimizedFunction, -std::numbers::pi / 4, std::numbers::pi / 4,
            std::numeric_limits<Scalar>::digits);

        Scalar phi = result.first;
        Scalar c2 = std::cos(phi);
        Scalar s2 = std::sin(phi);
        auto rightRotation = Eigen::JacobiRotation<Scalar>(c2, s2);

        // Apply the corresponding rotation on the right
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

// The original USVD algorithm implementation is used here to calculate cos and
// sin for left jacobi rotation symmertizing the 2x2 matrix
// Article: R. P. Brent, F. T. Luk, and C. Van Loan, Computation of the singular
// value decomposition using mesh-connected processors
template <typename _MatrixType>
void RevJac_SVD<_MatrixType>::calculateSymmetrizingRotation(Scalar w, Scalar x,
                                                            Scalar y, Scalar z,
                                                            Scalar& c,
                                                            Scalar& s) {
  const double EPS = 1e-30;

  Scalar mu1 = w + z;
  Scalar mu2 = x - y;

  if (std::abs(mu2) <= EPS * std::abs(mu1)) {
    c = 1;
    s = 0;
  } else {
    Scalar rho = mu1 / mu2;
    Scalar sgnRho = (rho > 0) - (rho < 0);
    s = sgnRho / std::sqrt(1 + rho * rho);
    c = s * rho;
  }
}

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP
