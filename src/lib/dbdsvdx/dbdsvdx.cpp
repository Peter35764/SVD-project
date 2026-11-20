#pragma once

#include "dbdsvdx.h"
#include <algorithm>
#include <vector>

template <typename _MatrixType>
DBDSVDX_SVD<_MatrixType>::DBDSVDX_SVD(const MatrixType& matrix,
                                     unsigned int computationOptions) {
  compute(matrix, computationOptions);
}

template <typename _MatrixType>
DBDSVDX_SVD<_MatrixType>& DBDSVDX_SVD<_MatrixType>::compute(
    const MatrixType& matrix, unsigned int computationOptions) {
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

  char jobu = wantU ? 'V' : 'N';
  char jobvt = wantV ? 'V' : 'N';
  char range = 'A';
  double vl = 0.0, vu = 0.0;
  lapack_int il = 0, iu = 0;
  lapack_int ns = 0;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> A_work =
      matrix.template cast<double>();

  Eigen::Matrix<double, Eigen::Dynamic, 1> S(min_mn);

  int ldu = wantU ? m : 1;
  int u_cols = wantU ? min_mn : 1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> U(ldu, u_cols);

  int ldvt = wantV ? min_mn : 1;
  int vt_cols = wantV ? n : 1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> VT(ldvt, vt_cols);

  std::vector<lapack_int> superb(12 * min_mn);

  int info = LAPACKE_dgesvdx(LAPACK_COL_MAJOR, jobu, jobvt, range,
                             m, n, A_work.data(), m,
                             vl, vu, il, iu, &ns,
                             S.data(), U.data(), ldu,
                             VT.data(), ldvt,
                             superb.data());

  if (info != 0) {
    throw std::runtime_error("LAPACKE_dgesvdx (dbdsvdx) failed with code: " + std::to_string(info));
  }

  this->m_singularValues = S.template cast<Scalar>();

  if (wantU) {
    if (fullU && m > min_mn) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qrU(U);
      Eigen::MatrixXd full_U_mat = qrU.householderQ() * Eigen::MatrixXd::Identity(m, m);
      for (int j = 0; j < min_mn; ++j) {
        if (full_U_mat.col(j).dot(U.col(j)) < 0.0) {
          full_U_mat.col(j) = -full_U_mat.col(j);
        }
      }
      this->m_matrixU = full_U_mat.template cast<Scalar>();
      this->m_computeFullU = true;
    } else {
      this->m_matrixU = U.template cast<Scalar>();
      if (fullU && m == min_mn) this->m_computeFullU = true;
      else this->m_computeThinU = true;
    }
  }

  if (wantV) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V = VT.transpose();
    if (fullV && n > min_mn) {
      Eigen::HouseholderQR<Eigen::MatrixXd> qrV(V);
      Eigen::MatrixXd full_V_mat = qrV.householderQ() * Eigen::MatrixXd::Identity(n, n);
      for (int j = 0; j < min_mn; ++j) {
        if (full_V_mat.col(j).dot(V.col(j)) < 0.0) {
          full_V_mat.col(j) = -full_V_mat.col(j);
        }
      }
      this->m_matrixV = full_V_mat.template cast<Scalar>();
      this->m_computeFullV = true;
    } else {
      this->m_matrixV = V.template cast<Scalar>();
      if (fullV && n == min_mn) this->m_computeFullV = true;
      else this->m_computeThinV = true;
    }
  }

  this->m_isInitialized = true;
  return *this;
}
