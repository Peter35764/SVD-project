#pragma once

#include "dgesvd.h"
#include <algorithm>
#include <vector>

template <typename _MatrixType>
DGESVD_SVD<_MatrixType>::DGESVD_SVD(const _MatrixType& matrix,
                                   unsigned int computationOptions) {
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
DGESVD_SVD<_MatrixType>& DGESVD_SVD<_MatrixType>::compute(
    const _MatrixType& matrix, unsigned int computationOptions) {
  int m = static_cast<int>(matrix.rows());
  int n = static_cast<int>(matrix.cols());
  int min_mn = std::min(m, n);

  if (m == 0 || n == 0) {
    this->m_singularValues.resize(0);
    this->m_isInitialized = true;
    return *this;
  }

  bool fullU = (computationOptions & Eigen::ComputeFullU) != 0;
  bool thinU = (computationOptions & Eigen::ComputeThinU) != 0;
  bool fullV = (computationOptions & Eigen::ComputeFullV) != 0;
  bool thinV = (computationOptions & Eigen::ComputeThinV) != 0;

  bool wantU = fullU || thinU;
  bool wantV = fullV || thinV;

  char jobu = fullU ? 'A' : (thinU ? 'S' : 'N');
  char jobvt = fullV ? 'A' : (thinV ? 'S' : 'N');

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A_work =
      matrix.template cast<double>();

  Eigen::Matrix<double, Eigen::Dynamic, 1> S(min_mn);

  int ldu = (jobu == 'A' || jobu == 'S') ? m : 1;
  int u_cols = (jobu == 'A') ? m : ((jobu == 'S') ? min_mn : 1);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> U(ldu, u_cols);

  int ldvt = (jobvt == 'A') ? n : ((jobvt == 'S') ? min_mn : 1);
  int vt_cols = n;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> VT(ldvt, vt_cols);

  std::vector<double> superb(min_mn);

  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, A_work.data(), m,
                            S.data(), U.data(), ldu, VT.data(), ldvt, superb.data());

  if (info != 0) {
    throw std::runtime_error("LAPACKE_dgesvd failed with code: " + std::to_string(info));
  }

  this->m_singularValues = S.template cast<Scalar>();

  if (wantU) {
    if (fullU) {
      this->m_matrixU = U.template cast<Scalar>();
      this->m_computeFullU = true;
    } else if (thinU) {
      this->m_matrixU = U.leftCols(min_mn).template cast<Scalar>();
      this->m_computeThinU = true;
    }
  }

  if (wantV) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V = VT.transpose();
    if (fullV) {
      this->m_matrixV = V.template cast<Scalar>();
      this->m_computeFullV = true;
    } else if (thinV) {
      this->m_matrixV = V.leftCols(min_mn).template cast<Scalar>();
      this->m_computeThinV = true;
    }
  }

  this->m_isInitialized = true;
  return *this;
}
