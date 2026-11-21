#pragma once

#include <Eigen/LU>

#include "newton_polar.h"

namespace SVD_Project {
template <typename MatrixType>
MatrixType Newton_Polar<MatrixType>::compute(const MatrixType& A,
                                             Index max_iter,
                                             RealScalar tol) {
  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n &&
               "Newton polar decomposition requires square matrix");

  MatrixType X = A;
  if (norm1(X) == RealScalar(0)) return X;

  const MatrixType I = MatrixType::Identity(m, n);

  for (Index k = 0; k < max_iter; ++k) {
    const RealScalar nX = norm1(X);

    MatrixType Y = X.adjoint().partialPivLu().solve(I);   
    if (!Y.allFinite()) return X;

    MatrixType delta = RealScalar(0.5) * (Y - X);    
    X += delta;

    if (norm1(delta) <= tol * nX) break;
  }

  return X;
}

template <typename _MatrixType>
Newton_Polar_SVD<_MatrixType>::Newton_Polar_SVD(
    const _MatrixType& matrix, unsigned int computationOptions) {
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
Newton_Polar_SVD<_MatrixType>& Newton_Polar_SVD<_MatrixType>::compute(
    const _MatrixType& A, unsigned int computationOptions) {
  typedef typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>
      SingularValuesType;

  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n && "Newton_Polar_SVD requires square matrix");

  Newton_Polar<MatrixType> np;
  MatrixType Up = np.compute(A);   

  MatrixType H = Up.adjoint() * A;
  H = Scalar(0.5) * (H + H.adjoint());

  Eigen::SelfAdjointEigenSolver<MatrixType> es(H);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("Newton_Polar_SVD: eigensolver failed");

  const auto evals = es.eigenvalues();
  const auto V     = es.eigenvectors();

  std::vector<Index> idx(evals.size());
  std::iota(idx.begin(), idx.end(), Index(0));
  std::sort(idx.begin(), idx.end(),
            [&](Index i, Index j) { return evals[i] > evals[j]; });

  MatrixType         Vsorted(V.rows(), V.cols());
  SingularValuesType S(evals.size());
  for (Index i = 0; i < evals.size(); ++i) {
    Vsorted.col(i) = V.col(idx[i]);
    S(i)           = std::max(RealScalar(0), evals[idx[i]]);
  }

  this->allocate(m, n, computationOptions);

  this->m_matrixU        = Up * Vsorted;
  this->m_matrixV        = Vsorted;
  this->m_singularValues = S;
  this->m_isInitialized  = true;

  return *this;
}

}  // namespace SVD_Project