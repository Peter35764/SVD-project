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
void GivRef_SVD<_MatrixType>::coordinate_descent_refinement(
    const MatrixType& B_target, const Eigen::VectorXd& true_singular_values) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;

  const RealScalar eta_base = 0.1;
  const RealScalar epsilon = 1e-8;
  const int max_iterations = 200;
  const RealScalar convergence_tol = 1e-7;

  std::vector<RealScalar> theta_left(m - 1, 0.0);
  std::vector<RealScalar> theta_right(n - 1, 0.0);

  Index min_dim = std::min(m, n);

  // insert sing vals
  MatrixType Sigma_true = MatrixType::Zero(m, n);
  for (Index i = 0; i < std::min(min_dim, true_singular_values.size()); ++i) {
    Sigma_true(i, i) = true_singular_values(i);
  }

  for (int iter = 0; iter < max_iterations; ++iter) {
    RealScalar max_change = 0.0;

    // left rotations
    for (Index k = 0; k < m - 1; ++k) {
      // B_tilde
      MatrixType U_current = MatrixType::Identity(m, m);
      MatrixType V_current = MatrixType::Identity(n, n);

      // Apply rotations
      for (Index i = 0; i < m - 1; ++i) {
        Eigen::JacobiRotation<Scalar> rot;
        rot.makeGivens(std::cos(theta_left[i]), std::sin(theta_left[i]));
        U_current.applyOnTheRight(i, i + 1, rot);
      }

      for (Index i = 0; i < n - 1; ++i) {
        Eigen::JacobiRotation<Scalar> rot;
        rot.makeGivens(std::cos(theta_right[i]), std::sin(theta_right[i]));
        V_current.applyOnTheRight(i, i + 1, rot);
      }

      MatrixType B_tilde = U_current * Sigma_true * V_current.transpose();
      MatrixType Error = B_tilde - B_target;

      // Compute grad
      MatrixType dU_dtheta = MatrixType::Zero(m, m);
      dU_dtheta(k, k) = -std::sin(theta_left[k]);
      dU_dtheta(k, k + 1) = -std::cos(theta_left[k]);
      dU_dtheta(k + 1, k) = std::cos(theta_left[k]);
      dU_dtheta(k + 1, k + 1) = -std::sin(theta_left[k]);

      // Apply rotations
      for (Index i = 0; i < m - 1; ++i) {
        if (i != k) {
          Eigen::JacobiRotation<Scalar> rot;
          rot.makeGivens(std::cos(theta_left[i]), std::sin(theta_left[i]));
          if (i < k) {
            dU_dtheta.applyOnTheLeft(i, i + 1, rot);
          } else {
            dU_dtheta.applyOnTheRight(i, i + 1, rot);
          }
        }
      }

      MatrixType gradient_term = dU_dtheta * Sigma_true * V_current.transpose();
      RealScalar gradient = (Error.cwiseProduct(gradient_term)).sum();

      // Update step
      RealScalar eta = eta_base / (std::abs(gradient) + epsilon);

      // Update angle
      RealScalar old_theta = theta_left[k];
      theta_left[k] -= eta * gradient;
      max_change = std::max(max_change, std::abs(theta_left[k] - old_theta));
    }

    // Update right rot
    for (Index k = 0; k < n - 1; ++k) {
      MatrixType U_current = MatrixType::Identity(m, m);
      MatrixType V_current = MatrixType::Identity(n, n);

      for (Index i = 0; i < m - 1; ++i) {
        Eigen::JacobiRotation<Scalar> rot;
        rot.makeGivens(std::cos(theta_left[i]), std::sin(theta_left[i]));
        U_current.applyOnTheRight(i, i + 1, rot);
      }

      for (Index i = 0; i < n - 1; ++i) {
        Eigen::JacobiRotation<Scalar> rot;
        rot.makeGivens(std::cos(theta_right[i]), std::sin(theta_right[i]));
        V_current.applyOnTheRight(i, i + 1, rot);
      }

      MatrixType B_tilde = U_current * Sigma_true * V_current.transpose();
      MatrixType Error = B_tilde - B_target;

      // Compute grad
      MatrixType dV_dtheta = MatrixType::Zero(n, n);
      dV_dtheta(k, k) = -std::sin(theta_right[k]);
      dV_dtheta(k, k + 1) = -std::cos(theta_right[k]);
      dV_dtheta(k + 1, k) = std::cos(theta_right[k]);
      dV_dtheta(k + 1, k + 1) = -std::sin(theta_right[k]);

      for (Index i = 0; i < n - 1; ++i) {
        if (i != k) {
          Eigen::JacobiRotation<Scalar> rot;
          rot.makeGivens(std::cos(theta_right[i]), std::sin(theta_right[i]));
          if (i < k) {
            dV_dtheta.applyOnTheLeft(i, i + 1, rot);
          } else {
            dV_dtheta.applyOnTheRight(i, i + 1, rot);
          }
        }
      }

      MatrixType gradient_term = U_current * Sigma_true * dV_dtheta.transpose();
      RealScalar gradient = (Error.cwiseProduct(gradient_term)).sum();

      RealScalar eta = eta_base / (std::abs(gradient) + epsilon);
      RealScalar old_theta = theta_right[k];
      theta_right[k] -= eta * gradient;
      max_change = std::max(max_change, std::abs(theta_right[k] - old_theta));
    }

    // converged?
    if (max_change < convergence_tol) break;
  }

  // Apply final rotations to get refined U and V
  left_J = MatrixType::Identity(m, m);
  right_J = MatrixType::Identity(n, n);

  for (Index i = 0; i < m - 1; ++i) {
    Eigen::JacobiRotation<Scalar> rot;
    rot.makeGivens(std::cos(theta_left[i]), std::sin(theta_left[i]));
    left_J.applyOnTheRight(i, i + 1, rot);
  }

  for (Index i = 0; i < n - 1; ++i) {
    Eigen::JacobiRotation<Scalar> rot;
    rot.makeGivens(std::cos(theta_right[i]), std::sin(theta_right[i]));
    right_J.applyOnTheRight(i, i + 1, rot);
  }
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
  Eigen::VectorXd computed_sv =
      sigm_B.diagonal().head(std::min(m, n)).cwiseAbs();
  coordinate_descent_refinement(sigm_B, computed_sv);
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
