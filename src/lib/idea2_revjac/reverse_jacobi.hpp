#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>
#include <boost/math/tools/minima.hpp>
#include <cassert>
#include <cmath>

#include "reverse_jacobi.h"

namespace SVD_Project {

const size_t MAX_ITERATIONS = 1000;
const double TOLERANCE = 1e-10;

template <typename _MatrixType>
RevJac_SVD<_MatrixType>& RevJac_SVD<_MatrixType>::Compute(
    const MatrixType& initial, unsigned int computationOptions) {
  assert(this->m_singularValues.rows() ==
             std::min(initial.rows(), initial.cols()) &&
         "Singular value vector size and matrix size do not match");

  // This algorithm is intended to calculate full U and V matrices, thus other
  // values do not make sense here
  this->m_computeFullU = true;
  this->m_computeFullV = true;

  // Initialize the current approximation
  MatrixType currentApproximation = this->m_matrixU *
                                    this->m_singularValues.asDiagonal() *
                                    this->m_matrixV.adjoint();

  // Output divergence before iterating (debug)
  Scalar divergence = (currentApproximation - initial).norm();
  if (m_divOstream) {
    *m_divOstream << "Divergence: " << std::to_string(divergence) << "\n";
  }

  // This vector represents the order in which the algorithm traverses elements
  // of reconstructed matrix
  std::vector<std::pair<std::pair<Index, Index>, Scalar>> traversalOrder;
  traversalOrder.reserve(initial.rows() * initial.cols());

  // Populate the vector with current per-element-divergence (^A_ij - A_ij)
  // and their corresponding indices
  for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter) {
    for (Index i = 0; i < initial.rows(); ++i) {
      for (Index j = 0; j < initial.cols(); ++j) {
        if(i == j) continue;
        traversalOrder.push_back(
            {{i, j}, std::abs(currentApproximation(i, j) - initial(i, j))});
      }
    }

    // Sort the vector from biggest divergence to smallest
    std::sort(traversalOrder.begin(), traversalOrder.end(),
              [](auto a, auto b) { return a.second > b.second; });

    // Perform the main loop
    for (auto [indices, _] : traversalOrder) {
      Index i = indices.first;
      Index j = indices.second;

      // Calculate first the left and then the right rotation.
      for (auto rotType : {RotationType::Left, RotationType::Right}) {
        // Here we create a lambda representing the functions being minimized,
        // parametrized by cosign value:
        // ||^A * J_ij(c) - A|| -> min and ||J_ij^T(C) * ^A - A|| -> min
        auto minimizedFunction = [currentApproximation, initial, i, j,
                                  rotType](Scalar angle) {
          MatrixType tempApproximation = currentApproximation;
          Scalar c = cos(angle);
          Scalar s = sin(angle);
          Eigen::JacobiRotation<Scalar> rotation =
              Eigen::JacobiRotation<Scalar>(c, s);

          if (rotType == RotationType::Left) {
            tempApproximation.applyOnTheLeft(i, j, rotation.adjoint());
          } else {
            tempApproximation.applyOnTheRight(i, j, rotation);
          }

          return (tempApproximation - initial).norm();
        };

        // Minimize the function and get the result cosign value
        auto result = boost::math::tools::brent_find_minima(
            minimizedFunction, -M_PI, M_PI, std::numeric_limits<Scalar>::digits);

        Scalar angle = result.first;
        Scalar c = cos(angle);
        Scalar s = sin(angle);
        auto rotation = Eigen::JacobiRotation<Scalar>(c, s);

        // Apply the corresponding rotation on the left/right
        if (rotType == RotationType::Left) {
          this->m_matrixU.applyOnTheLeft(i, j, rotation.adjoint());
          currentApproximation.applyOnTheLeft(i, j, rotation.adjoint());
        } else {
          this->m_matrixV.adjointInPlace();
          this->m_matrixV.applyOnTheRight(i, j, rotation);
          this->m_matrixV.adjointInPlace();
          currentApproximation.applyOnTheRight(i, j, rotation);
        }
      }
    }

    // Clear the ordering vector, preserving the capacity
    traversalOrder.clear();

    // Recalculate divergence for debugging purposes and to check if convergence
    // has been reached
    Scalar divergence = (currentApproximation - initial).norm();

    if (m_divOstream) {
      *m_divOstream << "Divergence: " << std::to_string(divergence) << "\n";
    }

    // Establish if convergence has been reached
    if (divergence < TOLERANCE) {
      break;
    }
  }

  this->m_isInitialized = true;
  return *this;
};

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP
