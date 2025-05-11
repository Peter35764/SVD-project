#ifndef GIVENS_REFINEMENT_HPP
#define GIVENS_REFINEMENT_HPP

#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <cmath>
#include <iostream>

#include "givens_refinement.h"

#define GIVREF_MAX_ITERATIONS 100

namespace SVD_Project {
template <typename _MatrixType>
GivRef_SVD<_MatrixType>::GivRef_SVD() {}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::preparation_phase(
    const MatrixType& A, unsigned int computationOptions) {
  // Set computation flags based on provided options
  this->m_computeFullU = (computationOptions & Eigen::ComputeFullU) != 0;
  this->m_computeThinU = (computationOptions & Eigen::ComputeThinU) != 0;
  this->m_computeFullV = (computationOptions & Eigen::ComputeFullV) != 0;
  this->m_computeThinV = (computationOptions & Eigen::ComputeThinV) != 0;

  Eigen::internal::UpperBidiagonalization<_MatrixType> bidiag(A);
  MatrixType B = bidiag.bidiagonal().toDenseMatrix();

  // std::cout << "\bidiag\n" << B;

  m = B.rows();
  n = B.cols();

  // Initialize the Jacobi rotation matrices and working matrix
  left_J = MatrixType::Identity(m, m);
  right_J = MatrixType::Identity(n, n);

  sigm_B = B;
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::qr_iterations_phase() {
  using RealScalar = typename MatrixType::RealScalar;
  Index min_mn = std::min(m, n);
  RealScalar tol = RealScalar(1e-10) * sigm_B.norm();  // Convergence tolerance

  for (int iter = 0; iter < GIVREF_MAX_ITERATIONS; ++iter) {
    performQRIteration();

    // Check for convergence on off-diagonal elements
    RealScalar off_diag_norm = 0;
    for (Index i = 0; i < min_mn - 1; ++i)
      off_diag_norm += std::abs(sigm_B(i, i + 1));
    if (off_diag_norm < tol) break;
  }
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::placeholder_phase() {
  // TODO: placeholder
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::finalizing_output_phase() {
  Index min_mn = std::min(m, n);
  // std::cout << "\nsigm_B\n" << sigm_B;
  // std::cout << "\nright\n" << right_J;
  // std::cout << "\nleft\n" << left_J;

  // Extract singular values
  this->m_singularValues = sigm_B.diagonal().head(min_mn);

  // Form U and V matrices based on computation options
  if (this->m_computeFullU) {
    this->m_matrixU = left_J.transpose();
  } else if (this->m_computeThinU) {
    this->m_matrixU = left_J.transpose().leftCols(min_mn);
  }

  if (this->m_computeFullV) {
    this->m_matrixV = right_J;
  } else if (this->m_computeThinV) {
    this->m_matrixV = right_J.leftCols(min_mn);
  }

  this->m_isInitialized = true;
}

template <typename _MatrixType>
GivRef_SVD<_MatrixType>& GivRef_SVD<_MatrixType>::compute(
    const MatrixType& A, unsigned int computationOptions) {
  preparation_phase(A, computationOptions);
  qr_iterations_phase();
  placeholder_phase();
  finalizing_output_phase();
  return *this;
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::performQRIteration() {
  using Scalar = typename MatrixType::Scalar;

  // Apply Givens rotations to zero out off-diagonal elements
  for (Index i = 0; i < n - 1; ++i) {
    // Right rotation to zero sigm_B(i, i+1)
    Eigen::JacobiRotation<Scalar> rotRight;
    rotRight.makeGivens(sigm_B(i, i), sigm_B(i, i + 1));
    sigm_B.applyOnTheRight(i, i + 1, rotRight);
    right_J.applyOnTheRight(i, i + 1, rotRight);

    // Left rotation to zero sigm_B(i+1, i)
    if (i < m - 1) {
      Eigen::JacobiRotation<Scalar> rotLeft;
      rotLeft.makeGivens(sigm_B(i, i), sigm_B(i + 1, i));
      sigm_B.applyOnTheLeft(i, i + 1, rotLeft.transpose());
      left_J.applyOnTheLeft(i, i + 1, rotLeft.transpose());
    }
  }
}
}  // namespace SVD_Project
#endif  // GIVENS_REFINEMENT_HPP
