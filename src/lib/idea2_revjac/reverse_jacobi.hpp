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


  for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter) {
    for (Index i = 0; i < initial.rows(); ++i) {
        for (Index j = 0; j < initial.cols(); ++j) {
            // Make left and right rotation
            for (const auto& rotType : {RotationType::Left, RotationType::Right}) {
                Scalar angle = 0.0;
                Scalar A = 0.0;
                Scalar B = 0.0;

                // Consider Jacobi rotation on plane (i, j)
                // and find optimal angle to minimize divergence
                for (const auto& t_I : {i, j})
                    for (Index t_J = 0; t_J < initial.cols(); ++t_J)
                        A += currentApproximation(t_I, t_J) * initial(t_I, t_J);

                // Formula for finding stationary points
                // look at ReverseJacobi in ideas.tex

                if (rotType == RotationType::Left) {
                    for (Index t_j = 0; t_j < initial.cols(); ++t_j) {
                        B += currentApproximation(i, t_j) * initial(j, t_j);
                    }
                    for (Index t_j = 0; t_j < initial.cols(); ++t_j) {
                        B -= currentApproximation(j, t_j) * initial(i, t_j);
                    }
                    // in case of B / A = inf atan(inf) will return PI/2
                    angle = atan(B / A);

                    Eigen::JacobiRotation<Scalar> rotation
                        = Eigen::JacobiRotation<Scalar>(cos(angle), sin(angle));
                    this->m_matrixU.applyOnTheLeft(i, j, rotation.adjoint());
                    currentApproximation.applyOnTheLeft(i, j, rotation.adjoint());
                }
                else {
                    for (Index t_i = 0; t_i < initial.rows(); ++t_i) {
                        B += currentApproximation(t_i, i) * initial(t_i, j);
                    }
                    for (Index t_i = 0; t_i < initial.rows(); ++t_i) {
                        B -= currentApproximation(t_i, j) * initial(t_i, i);
                    }
                    angle = atan(B / A);

                    Eigen::JacobiRotation<Scalar> rotation
                        = Eigen::JacobiRotation<Scalar>(cos(angle), sin(angle));
                    this->m_matrixV.adjointInPlace();
                    this->m_matrixV.applyOnTheRight(i, j, rotation);
                    this->m_matrixV.adjointInPlace();
                    currentApproximation.applyOnTheRight(i, j, rotation);
                }
          }
      }
    }
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
