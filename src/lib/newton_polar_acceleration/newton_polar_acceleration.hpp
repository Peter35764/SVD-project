#pragma once

namespace SVD_Project {
template <typename MatrixType>
typename Newton_Polar_Acceleration<MatrixType>::RealScalar
Newton_Polar_Acceleration<MatrixType>::norm1(const MatrixType& M) {
  return M.cwiseAbs().colwise().sum().maxCoeff();
}

template <typename MatrixType>
typename Newton_Polar_Acceleration<MatrixType>::RealScalar
Newton_Polar_Acceleration<MatrixType>::normInf(const MatrixType& M) {
  return M.cwiseAbs().rowwise().sum().maxCoeff();
}

template <typename MatrixType>
MatrixType Newton_Polar_Acceleration<MatrixType>::compute(
    const MatrixType& A, Index max_iter, RealScalar tol) {
  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n &&
               "Newton polar decomposition requires square matrix");

  m_steps        = 0;
  m_scaled_steps = 0;

  MatrixType X = A;
  if (norm1(X) == RealScalar(0)) return X;

  const MatrixType I = MatrixType::Identity(m, n);

  bool use_scaling = true;

  for (Index k = 0; k < max_iter; ++k) {
    const RealScalar nX1   = norm1(X);
    const RealScalar nXinf = use_scaling ? normInf(X) : RealScalar(0);

    MatrixType Z;
    {
      Eigen::PartialPivLU<MatrixType> lu(X);
      Z = lu.solve(I);
    }
    if (!Z.allFinite()) {
      Eigen::FullPivLU<MatrixType> lu(X);
      if (!lu.isInvertible()) break;
      Z = lu.solve(I);
      if (!Z.allFinite()) break;
    }

    Z.adjointInPlace();
    const MatrixType& Y = Z;

    RealScalar gamma = RealScalar(1);
    if (use_scaling) {
      MatrixType  R   = X - Y;
      RealScalar  rho = std::sqrt(norm1(R) * normInf(R));

      if (rho <= m_tau) {
        use_scaling = false;      
      } else {
        const RealScalar nY1   = norm1(Y);
        const RealScalar nYinf = normInf(Y);
        gamma = std::sqrt(std::sqrt((nY1 * nYinf) / (nX1 * nXinf)));
        if (!std::isfinite(gamma) || gamma <= RealScalar(0))
          gamma = RealScalar(1);
        ++m_scaled_steps;
      }
    }

    MatrixType X_new = RealScalar(0.5) *
                       (gamma * X + (RealScalar(1) / gamma) * Y);
    if (!X_new.allFinite()) break;

    X_new -= X;                      
    const RealScalar nDelta = norm1(X_new);
    X += X_new;        

    ++m_steps;

    if (nDelta <= tol * nX1) break;
  }

  return X;
}

template <typename _MatrixType>
Newton_Polar_SVD_Acceleration<_MatrixType>::Newton_Polar_SVD_Acceleration(
    const _MatrixType& matrix, unsigned int computationOptions) {
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
Newton_Polar_SVD_Acceleration<_MatrixType>&
Newton_Polar_SVD_Acceleration<_MatrixType>::compute(
    const _MatrixType& A, unsigned int computationOptions) {
  typedef typename Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>
      SingularValuesType;

  const Index m = A.rows();
  const Index n = A.cols();
  eigen_assert(m == n &&
               "Newton_Polar_SVD_Acceleration requires square matrix");

  Newton_Polar_Acceleration<MatrixType> np;
  np.setScalingThreshold(m_tau);
  MatrixType Up = np.compute(A);
  m_steps        = np.steps();
  m_scaled_steps = np.scaledSteps();

  MatrixType H = Up.adjoint() * A;
  H = Scalar(0.5) * (H + H.adjoint());

  Eigen::SelfAdjointEigenSolver<MatrixType> es(H);
  if (es.info() != Eigen::Success)
    throw std::runtime_error(
        "Newton_Polar_SVD_Acceleration: eigensolver failed");

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