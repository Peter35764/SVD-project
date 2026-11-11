#pragma once

#include "dgesdd.h"
#include <algorithm>

template <typename _MatrixType>
DGESDD_SVD<_MatrixType>::DGESDD_SVD(const _MatrixType& matrix,
                                   unsigned int computationOptions) {
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
DGESDD_SVD<_MatrixType>& DGESDD_SVD<_MatrixType>::compute(
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

  char jobz = 'N';
  if (fullU || fullV) {
    jobz = 'A';
  } else if (thinU || thinV) {
    jobz = 'S';
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A_work =
      matrix.template cast<double>();

  Eigen::Matrix<double, Eigen::Dynamic, 1> S(min_mn);

  int ldu = (jobz == 'A') ? m : ((jobz == 'S') ? m : 1);
  int u_cols = (jobz == 'A') ? m : ((jobz == 'S') ? min_mn : 1);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> U(ldu, u_cols);

  int ldvt = (jobz == 'A') ? n : ((jobz == 'S') ? min_mn : 1);
  int vt_cols = n;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> VT(ldvt, vt_cols);

  int info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, jobz, m, n, A_work.data(), m,
                            S.data(), U.data(), ldu, VT.data(), ldvt);

  if (info != 0) {
    throw std::runtime_error("LAPACKE_dgesdd failed with code: " + std::to_string(info));
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
