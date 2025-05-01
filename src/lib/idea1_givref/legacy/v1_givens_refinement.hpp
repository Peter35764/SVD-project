#ifndef V1_GIVENS_REFINEMENT_HPP
#define V1_GIVENS_REFINEMENT_HPP

#include <cassert>

namespace SVD_Project {
// Interface of tests forces this structure
template <typename _MatrixType>
v1_GivRef_SVD<_MatrixType>::v1_GivRef_SVD(const _MatrixType &matrix,
                                    unsigned int computationOptions)
    : m_divOstream(nullptr), has_true_sigm(false) {
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
v1_GivRef_SVD<_MatrixType>::v1_GivRef_SVD(
    const _MatrixType &matrix,
    const typename Base::SingularValuesType &singularValues,
    unsigned int computationOptions)
    : m_singularValues(singularValues),
      m_divOstream(nullptr),
      has_true_sigm(
          true) {  // we DO need to diff constructors, even if its dirty
  // Runtime assertion
  assert(singularValues.rows() == std::min(matrix.rows(), matrix.cols()) &&
         "Singular value vector size and matrix size do not match");
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
v1_GivRef_SVD<_MatrixType>::v1_GivRef_SVD(
    const _MatrixType &matrix,
    const typename Base::SingularValuesType &singularValues, std::ostream *os,
    unsigned int computationOptions)
    : m_singularValues(singularValues),
      m_divOstream(os),
      has_true_sigm(true) {  // see note above
  assert(singularValues.rows() == std::min(matrix.rows(), matrix.cols()) &&
         "Singular value vector size and matrix size do not match");
  compute(matrix, computationOptions);
}

// Actual compute
template <typename _MatrixType>
v1_GivRef_SVD<_MatrixType> &v1_GivRef_SVD<_MatrixType>::compute(
    const _MatrixType &matrix, unsigned int computationOptions) {
  if (has_true_sigm && m_singularValues.size() > 0) {
    assert(m_singularValues.rows() == std::min(matrix.rows(), matrix.cols()) &&
           "Singular value vector size and matrix size do not match");
  }

  setupMatrices(matrix);  // Setup and init

  // QR iters and old tol
  using Scalar = typename _MatrixType::Scalar;
  Scalar eps = std::numeric_limits<Scalar>::epsilon();
  Scalar tol = eps * std::pow(Scalar(10), Scalar(0.125));  // As per paper
  int max_iter = 100;

  iterQRtillConv(tol, max_iter);

  // Finalize the decomposition
  fixFormatResults();

  return *this;
}

template <typename _MatrixType>
v1_GivRef_SVD<_MatrixType> &v1_GivRef_SVD<_MatrixType>::compute(
    const _MatrixType &matrix, std::ostream *os,
    unsigned int computationOptions) {
  m_divOstream = os;
  return compute(matrix, computationOptions);
}

template <typename _MatrixType>
void v1_GivRef_SVD<_MatrixType>::setDivergenceOstream(std::ostream *os) {
  m_divOstream = os;
}

// Setup and init
template <typename _MatrixType>
void v1_GivRef_SVD<_MatrixType>::setupMatrices(const _MatrixType &matrix) {
  m = matrix.rows();
  n_cols = matrix.cols();
  n = std::min(m, n_cols);
  trigonom_i = 0;
  iter_num = 0;

  // Init rot mats
  left_J = MatrixDynamic::Zero(m, m);
  right_J = MatrixDynamic::Zero(n_cols, n_cols);
  // Then set them as identity matrices
  left_J.setIdentity();
  right_J.setIdentity();

  // Bidiagonalize the input matrix
  auto bid = Eigen::internal::UpperBidiagonalization<_MatrixType>(matrix);
  B = bid.bidiagonal();
  sigm_B = B;

  if (has_true_sigm) {  // thats why we diff them
    // but we DO eat provided singular values if available
    true_sigm_B = MatrixDynamic::Zero(m, n_cols);
    true_sigm_B.diagonal() = m_singularValues;
  } else {
    // If we somehow fuck up - we rollback for jacobi svd, implying that we
    // DO NOT want to use it mainly. Again, NOT for direct use, main method is
    // providing sing values
    Eigen::JacobiSVD<_MatrixType> svd(
        matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    true_sigm_B = MatrixDynamic::Zero(m, n_cols);
    true_sigm_B.diagonal() = svd.singularValues().head(n);
  }

  // Alloc for rotations
  int max_iter = 100;
  int total_rotations = 2 * max_iter * (n - 1);
  Cosines.resize(total_rotations);
  Sines.resize(total_rotations);
  Tans.resize(total_rotations);
  NewCosines.resize(total_rotations);
  NewSines.resize(total_rotations);
}

// Main steps QR till conv/stop criterion
template <typename _MatrixType>
void v1_GivRef_SVD<_MatrixType>::iterQRtillConv(
    typename _MatrixType::Scalar tol, int max_iter) {
  using Scalar = typename _MatrixType::Scalar;

  Scalar divergence = (sigm_B - true_sigm_B).norm();  // init div

  if (m_divOstream) {  // output if we do have stream
    *m_divOstream << divergence << std::endl;
  }

  for (int i = 0; i < max_iter; i++) {  // until conv or max_iter
    Impl_QR_zero_iter();
    iter_num++;

    divergence = (sigm_B - true_sigm_B).norm();  // update div
    if (m_divOstream) {
      *m_divOstream << divergence << std::endl;
    }

    // Convergence and zeroing if appl
    if (isConvergedSafely(tol, max_iter)) {
      break;  // has converged
    }
  }

  // Store the rotation parameters
  NewCosines = Cosines;
  NewSines = Sines;
}

// Format/fix the SVD results according to impl nuances
template <typename _MatrixType>
void v1_GivRef_SVD<_MatrixType>::fixFormatResults() {
  revert_negative_singular();  // ensure positivity

  // Set the final decomposition matrices
  m_matrixU = left_J.transpose();
  m_matrixV = right_J;
  m_singularValues = sigm_B.diagonal().head(n);
}

// Convergence
template <typename _MatrixType>
bool v1_GivRef_SVD<_MatrixType>::isConvergedSafely(
    typename _MatrixType::Scalar tol, int max_iter) {
  if (n < 2) return true;  // Trivial for 1×1 matrices

  using Scalar = typename _MatrixType::Scalar;
  const Scalar eps = std::numeric_limits<Scalar>::epsilon();

  Scalar sigma_min;
  Scalar sigma_max;

  if (has_true_sigm) {
    // Use true singular values for threshold calculations when available
    sigma_min = std::numeric_limits<Scalar>::max();
    sigma_max = 0;
    for (Index i = 0; i < n; i++) {
      sigma_min = std::min(sigma_min, std::abs(true_sigm_B(i, i)));
      sigma_max = std::max(sigma_max, std::abs(true_sigm_B(i, i)));
    }
  } else {
    // Use current estimates when true values not available
    sigma_min = std::numeric_limits<Scalar>::max();
    sigma_max = 0;
    for (Index i = 0; i < n; i++) {
      sigma_min = std::min(sigma_min, std::abs(sigm_B(i, i)));
      sigma_max = std::max(sigma_max, std::abs(sigm_B(i, i)));
    }
  }

  // LAPACK threshold method
  Scalar unfl = std::numeric_limits<Scalar>::min();
  const int maxiter = 6;  // LAPACK default
  Scalar thresh;

  if (tol >= 0) {
    // (1.6)
    thresh = std::max(tol * sigma_min,
                      Scalar(maxiter) * (Scalar(n) * (Scalar(n) * unfl)));
  } else {
    // (1.7)
    thresh = std::max(std::abs(tol) * sigma_max,
                      Scalar(maxiter) * (Scalar(n) * (Scalar(n) * unfl)));
  }

  // paper's more sophisticated criterion
  VectorDynamic mu(n);
  mu(0) = std::abs(sigm_B(0, 0));
  for (Index j = 0; j < n - 1; j++) {
    Scalar e_j_sq = sigm_B(j, j + 1) * sigm_B(j, j + 1);
    Scalar s_jp1 = std::abs(sigm_B(j + 1, j + 1));
    mu(j + 1) = s_jp1 * (mu(j) / (mu(j) + e_j_sq));
  }

  VectorDynamic lambda(n);
  lambda(n - 1) = std::abs(sigm_B(n - 1, n - 1));
  for (Index j = n - 2; j >= 0; j--) {
    Scalar e_j_sq = sigm_B(j, j + 1) * sigm_B(j, j + 1);
    Scalar s_j = std::abs(sigm_B(j, j));
    lambda(j) = s_j * (lambda(j + 1) / (lambda(j + 1) + e_j_sq));
  }

  bool all_converged = true;
  for (Index j = 0; j < n - 1; j++) {
    Scalar e_j = std::abs(sigm_B(j, j + 1));

    // Skip if already numerically zero
    if (e_j <= eps * std::max(std::abs(sigm_B(j, j)),
                              std::abs(sigm_B(j + 1, j + 1)))) {
      sigm_B(j, j + 1) = 0;  // Explicitly zero out
      continue;
    }

    // First check LAPACK threshold - simplest approach
    if (e_j <= thresh) {
      sigm_B(j, j + 1) = 0;  // Zero out element
      continue;
    }

    // If not caught by LAPACK threshold, try paper's criteria
    // Criterion 1a and 1b
    bool can_zero_by_1a = (e_j / mu(j) <= tol);
    bool can_zero_by_1b = (e_j / lambda(j + 1) <= tol);

    if (can_zero_by_1a || can_zero_by_1b) {
      sigm_B(j, j + 1) = 0;  // Nuking it
    } else {
      all_converged = false;  // At least one element didnt converge
    }
  }

  return all_converged;
}

template <typename _MatrixType>
std::vector<typename _MatrixType::Scalar> v1_GivRef_SVD<_MatrixType>::ROT(
    typename _MatrixType::Scalar f, typename _MatrixType::Scalar g) {
  using Scalar = typename _MatrixType::Scalar;
  Scalar cs, sn, r;
  if (f == 0) {
    cs = 0;
    sn = 1;
    r = g;
  } else {
    Eigen::JacobiRotation<Scalar> rot;
    rot.makeGivens(f, g);
    cs = rot.c();
    sn = rot.s();
    r = cs * f + sn * g;  // match orig
    if (trigonom_i < Tans.size()) {
      Tans(trigonom_i) =
          std::abs(f) > std::abs(g) ? g / f : f / g;  // match orig
    }
  }
  return std::vector<Scalar>{cs, sn, r};
}

template <typename _MatrixType>
void v1_GivRef_SVD<_MatrixType>::Impl_QR_zero_iter() {
  if (n < 2) return;  // without this it could trigger asserts
  using Scalar = typename _MatrixType::Scalar;
  Scalar oldcs = 1, oldsn = 0;
  Scalar cs = 1, sn, r;
  for (Index i = 0; i < n - 1; i++) {
    auto temp1 = ROT(sigm_B(i, i) * cs, sigm_B(i, i + 1));
    cs = temp1[0];
    sn = temp1[1];
    r = temp1[2];
    Eigen::JacobiRotation<Scalar> rotRight(
        cs, -sn);  // Needs negative sn to match original, we're trying to
                   // keep the og impl idea
    right_J.applyOnTheRight(i, i + 1, rotRight);
    Cosines(trigonom_i) = cs;
    Sines(trigonom_i) = sn;
    trigonom_i++;
    if (i != 0) {
      sigm_B(i - 1, i) = oldsn * r;
    }
    auto temp2 = ROT(oldcs * r, sigm_B(i + 1, i + 1) * sn);
    oldcs = temp2[0];
    oldsn = temp2[1];
    sigm_B(i, i) = temp2[2];
    Eigen::JacobiRotation<Scalar> rotLeft(oldcs,
                                          -oldsn);  // See note above
    rotLeft.transpose();                            // To match og
    left_J.applyOnTheLeft(i, i + 1, rotLeft);
    Cosines(trigonom_i) = oldcs;
    Sines(trigonom_i) = oldsn;
    trigonom_i++;
  }
  Scalar h = sigm_B(n - 1, n - 1) * cs;
  sigm_B(n - 2, n - 1) = h * oldsn;
  sigm_B(n - 1, n - 1) = h * oldcs;
}

template <typename _MatrixType>
void v1_GivRef_SVD<_MatrixType>::revert_negative_singular() {
  for (Index i = 0; i < n; i++) {
    if (sigm_B(i, i) < 0) {
      sigm_B(i, i) = -sigm_B(i, i);
      for (Index j = 0; j < left_J.rows(); j++) {
        left_J(i, j) = -left_J(i, j);
      }
    }
  }
}
}  // namespace SVD_Project
#endif  // V1_GIVENS_REFINEMENT_HPP
