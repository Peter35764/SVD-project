#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>
#include <boost/math/tools/minima.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

#include "reverse_jacobi.h"

namespace SVD_Project {

const size_t MAX_ITERATIONS = 10;
const double TOLERANCE = 1e-10;

template <typename _MatrixType>
RevJac_SVD<_MatrixType>& RevJac_SVD<_MatrixType>::Compute(
    const MatrixType& initial, unsigned int computationOptions) {
  assert(this->m_singularValues.rows() ==
             std::min(initial.rows(), initial.cols()) &&
         "Singular value vector size and matrix size do not match");

  this->m_computeFullU = true;
  this->m_computeFullV = true;

  MatrixType currentApproximation = this->m_matrixU *
                                    this->m_singularValues.asDiagonal() *
                                    this->m_matrixV.adjoint();

  for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter) {
    for (Index i = 0; i < initial.rows(); ++i) {
      for (Index j = 0; j < initial.cols(); ++j) {
        // Calculate first the left and then the right rotation.
        for (auto rotType : {RotationType::Left, RotationType::Right}) {
          auto minimizedFunction = [currentApproximation, initial, i, j,
                                    rotType](Scalar c) {
            MatrixType tempApproximation = currentApproximation;
            Scalar s = std::sqrt(1 - c * c);
            Eigen::JacobiRotation<Scalar> rotation =
                Eigen::JacobiRotation<Scalar>(c, s);

            if (rotType == RotationType::Left) {
              tempApproximation.applyOnTheLeft(i, j, rotation.adjoint());
            } else {
              tempApproximation.applyOnTheRight(i, j, rotation);
            }

            return (tempApproximation - initial).norm();
          };

          auto result = boost::math::tools::brent_find_minima(
              minimizedFunction, 0.0, 1.0, std::numeric_limits<Scalar>::digits);

          Scalar c = result.first;
          Scalar s = std::sqrt(1 - c * c);
          auto rotation = Eigen::JacobiRotation<Scalar>(c, s);

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
    }

    Scalar divergence = (currentApproximation - initial).norm();

    if (m_divOstream) {
      *m_divOstream << "Divergence: " << std::to_string(divergence)
                    << std::endl;
    }

    if (divergence < TOLERANCE) {
      break;
    }
  }

  this->m_isInitialized = true;
  return *this;
};

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP
