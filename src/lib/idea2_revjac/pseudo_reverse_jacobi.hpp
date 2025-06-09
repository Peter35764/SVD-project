#ifndef PSEUDO_REVERSE_JACOBI_HPP
#define PSEUDO_REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>
#include <boost/math/tools/minima.hpp>
#include <cassert>
#include <cmath>

#include "pseudo_reverse_jacobi.h"

namespace SVD_Project {

template <typename _MatrixType>
PseudoRevJac_SVD<_MatrixType>::PseudoRevJac_SVD(
    const MatrixType& initial, const VectorDynamic& singularValues,
    unsigned int computationOptions) {
  Index m = initial.rows();
  Index n = initial.cols();
  this->m_matrixU = MatrixDynamic::Identity(m, m);
  this->m_matrixV = MatrixDynamic::Identity(n, n);
  this->m_singularValues = singularValues;
  Compute(initial, computationOptions);
}

template <typename _MatrixType>
PseudoRevJac_SVD<_MatrixType>::PseudoRevJac_SVD(
    const MatrixType& initial, const VectorDynamic& singularValues,
    std::ostream* os, unsigned int computationOptions) {
  Index m = initial.rows();
  Index n = initial.cols();
  this->m_matrixU = MatrixDynamic::Identity(m, m);
  this->m_matrixV = MatrixDynamic::Identity(n, n);
  this->m_singularValues = singularValues;
  this->m_divOstream = os;
  Compute(initial, computationOptions);
};

template <typename _MatrixType>
PseudoRevJac_SVD<_MatrixType>::PseudoRevJac_SVD(
    const MatrixType& initial, const VectorDynamic& singularValues,
    const MatrixDynamic& matrixU, const MatrixDynamic& matrixV,
    unsigned int computationOptions) {
  this->m_singularValues = singularValues;
  this->m_matrixU = matrixU;
  this->m_matrixV = matrixV;
  Compute(initial, computationOptions);
};

template <typename _MatrixType>
PseudoRevJac_SVD<_MatrixType>::PseudoRevJac_SVD(
    const MatrixType& initial, const VectorDynamic& singularValues,
    const MatrixDynamic& matrixU, const MatrixDynamic& matrixV,
    std::ostream* os, unsigned int computationOptions) {
  this->m_singularValues = singularValues;
  this->m_matrixU = matrixU;
  this->m_matrixV = matrixV;
  this->m_divOstream = os;
  Compute(initial, computationOptions);
};

template <typename _MatrixType>
PseudoRevJac_SVD<_MatrixType>& PseudoRevJac_SVD<_MatrixType>::Compute(
    const MatrixType& initial, unsigned int computationOptions) {
  const size_t MAX_ITERATIONS = 5000;
  const double TOLERANCE = 1e-10;

  assert(this->m_singularValues.rows() ==
             std::min(initial.rows(), initial.cols()) &&
         "Singular value vector size and matrix size do not match");

  // This algorithm is intended to calculate full U and V matrices, thus
  // other values do not make sense here
  this->m_computeFullU = true;
  this->m_computeFullV = true;

  // Initialize the current approximation
  MatrixType currentApproximation = this->m_matrixU *
                                    this->m_singularValues.asDiagonal() *
                                    this->m_matrixV.adjoint();

  // Output divergence before iterating (debug)
  Scalar divergence = (currentApproximation - initial).norm();
  if (this->m_divOstream) {
    *this->m_divOstream << "Divergence: " << std::to_string(divergence) << "\n";
  }

  // This vector represents the order in which the algorithm traverses elements
  // of reconstructed matrix.
  // max count of this vector is MAX_ITERATIONS * n(n-1)/2
  // with each iteration the size increases by n(n-1)/2 elements
  // its working but there are no explainations =)
  std::vector<std::pair<std::pair<Index, Index>, Scalar>> traversalOrder;
  traversalOrder.reserve(MAX_ITERATIONS * initial.rows() *
                         (initial.cols() - 1) / 2);

  // Populate the vector with current per-element-divergence (^A_ij - A_ij)
  // and their corresponding indices
  for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter) {
    for (Index i = 0; i < initial.rows(); ++i) {
      for (Index j = i + 1; j < initial.cols(); ++j) {
        traversalOrder.push_back(
            {{i, j}, std::abs(currentApproximation(i, j) - initial(i, j))});
      }
    }

    // Sort the vector from biggest divergence to smallest
    std::sort(traversalOrder.begin(), traversalOrder.end(),
              [](auto a, auto b) { return a.second > b.second; });

    // Perform the main loop
    for (const auto& [indices, _] : traversalOrder) {
      Index i = indices.first;
      Index j = indices.second;
      // Make left and right rotation
      for (const auto& rotType : {RotationType::Left, RotationType::Right}) {
        Scalar angle = 0.0;
        Scalar A = 0.0;
        Scalar B = 0.0;

        // Consider Jacobi rotation on rows or columns (i, j)
        // rotation type accordingly
        // and find optimal angle to minimize divergence

        // Pay attention to this implementation: an algorithm  doesn't find
        // global minimum of divergence
        // in a fact, it's looking for a stationary point
        // regardless of the type of point: local maximum, global minimum and
        // etc.

        // Formula for finding stationary points
        // look at ReverseJacobi in ideas.tex page 18-19
        if (rotType == RotationType::Left) {
          for (const auto& t_I : {i, j})
            for (Index t_J = 0; t_J < initial.cols(); ++t_J)
              A += currentApproximation(t_I, t_J) * initial(t_I, t_J);

          for (Index t_j = 0; t_j < initial.cols(); ++t_j) {
            B += currentApproximation(i, t_j) * initial(j, t_j);
          }
          for (Index t_j = 0; t_j < initial.cols(); ++t_j) {
            B -= currentApproximation(j, t_j) * initial(i, t_j);
          }
          // in case of B / A = inf atan(inf) will return PI/2
          angle = atan(B / A);

          Eigen::JacobiRotation<Scalar> rotation =
              Eigen::JacobiRotation<Scalar>(cos(angle), sin(angle));
          this->m_matrixU.applyOnTheLeft(i, j, rotation.adjoint());
          currentApproximation.applyOnTheLeft(i, j, rotation.adjoint());
        } else {
          for (const auto& t_I : {i, j})
            for (Index t_J = 0; t_J < initial.cols(); ++t_J)
              A += currentApproximation(t_J, t_I) * initial(t_J, t_I);

          for (Index t_i = 0; t_i < initial.rows(); ++t_i) {
            B += currentApproximation(t_i, i) * initial(t_i, j);
          }
          for (Index t_i = 0; t_i < initial.rows(); ++t_i) {
            B -= currentApproximation(t_i, j) * initial(t_i, i);
          }
          angle = atan(B / A);

          Eigen::JacobiRotation<Scalar> rotation =
              Eigen::JacobiRotation<Scalar>(cos(angle), sin(angle));
          this->m_matrixV.adjointInPlace();
          this->m_matrixV.applyOnTheRight(i, j, rotation);
          this->m_matrixV.adjointInPlace();
          currentApproximation.applyOnTheRight(i, j, rotation);
        }
      }
    }
    // Recalculate divergence for debugging purposes and to check if convergence
    // has been reached
    // traversalOrder.clear(); otherwise convergence rate is slower
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

#endif  // PSEUDO_REVERSE_JACOBI_HPP
