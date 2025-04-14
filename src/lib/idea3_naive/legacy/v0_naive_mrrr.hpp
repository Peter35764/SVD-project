#ifndef V0_NAIVE_MRRR_HPP
#define V0_NAIVE_MRRR_HPP

#include <lapacke.h>

#include <cmath>

#include "v0_naive_mrrr.h"  // necessary for correct display in ide, does not affect the assembly process and can be removed

namespace SVD_Project {

template <typename _MatrixType>
v0_NaiveMRRR_SVD<_MatrixType>::v0_NaiveMRRR_SVD(
    const _MatrixType &matrix, unsigned int computationOptions) {
  // При создании объекта сразу выполняем вычисление TGK
  compute_tgk(matrix);
}

template <typename _MatrixType>
v0_NaiveMRRR_SVD<_MatrixType> &v0_NaiveMRRR_SVD<_MatrixType>::compute_tgk(
    const _MatrixType &matrix) {
  // Бидиагонализация входной матрицы
  auto bid = Eigen::internal::UpperBidiagonalization(matrix);
  auto B = bid.bidiagonal();
  int n = B.rows();

  // Создаём матрицу TGK(B) размером 2n x 2n и инициализируем её нулями
  Eigen::MatrixXd TGK(2 * n, 2 * n);
  TGK.setZero();

  // Запоминаем множители бидиагонализации
  auto L = bid.householderU();
  auto R = bid.householderV();

  // Создаем базовую матрицу для m_matrixU и m_matrixV
  Eigen::MatrixXd Base(n, n);
  Base.setZero();
  this->m_matrixU = Base;
  this->m_matrixV = Base;

  // Приводим матрицу B к виду, удобному для доступа по индексам
  Eigen::MatrixXd BB = B;
  for (int i = 0; i < n - 1; ++i) {
    TGK(2 * i, 2 * i + 1) = BB(i, i);
    TGK(2 * i + 1, 2 * i + 2) = BB(i, i + 1);
    TGK(2 * i + 1, 2 * i) = BB(i, i);
    TGK(2 * i + 2, 2 * i + 1) = BB(i, i + 1);
  }
  TGK(2 * n - 2, 2 * n - 1) = BB(n - 1, n - 1);
  TGK(2 * n - 1, 2 * n - 2) = BB(n - 1, n - 1);

  int32_t nzc = std::max(1, 2 * n);

  double *d = new double[2 * n];
  for (int i = 0; i < 2 * n; i++) {
    d[i] = TGK.diagonal(0)[i];
  }

  double *e = new double[2 * n];
  for (int i = 0; i < 2 * n - 1; i++) {
    e[i] = TGK.diagonal(-1)[i];
  }

  double *w = new double[2 * n];
  double *z = new double[2 * n * 2 * n];
  int32_t m;
  int32_t isuppz[2 * 2 * n];
  int32_t tryrac = 1;

  int32_t info = LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'A', 2 * n, d, e, 0, 0,
                                0, 0, &m, w, z, 2 * n, nzc, isuppz, &tryrac);
  if (info != 0)
    throw std::runtime_error("LAPACK error: " + std::to_string(info));

  Eigen::VectorXd eigenvalues(2 * n);
  for (int i = 0; i < 2 * n; ++i) {
    eigenvalues(i) = w[2 * n - i - 1];
  }

  Eigen::MatrixXd eigenvectors(2 * n, 2 * n);
  for (int i = 0; i < 2 * n; i++) {
    for (int j = 0; j < 2 * n; j++) {
      eigenvectors(i, j) = i < n ? z[(2 * n - i - 1) * 2 * n + j] * sq
                                 : z[(2 * n - i - 1) * 2 * n + j] * (-sq);
    }
  }
  for (int i = n - 1; i < n + 1; i++) {
    for (int j = 0; j < 2 * n; j++) {
      eigenvectors(i, j) *= -1;
    }
  }

  Eigen::MatrixXd singularValuess(n, n);
  for (int i = 0; i < n; ++i) {
    double sigma = std::abs(eigenvalues(i));
    Eigen::VectorXd q(2 * n);
    for (int j = 0; j < 2 * n; j++) {
      q[j] = eigenvectors(i, j);
    }
    Eigen::VectorXd u(n);
    Eigen::VectorXd v(n);
    for (int j = 0; j < n; ++j) {
      v[j] = q[2 * j];
      u[j] = q[2 * j + 1];
    }
    singularValuess(i, i) = sigma;
    this->m_matrixU.col(i) = u;
    this->m_matrixV.col(i) = v;
  }

  this->m_singularValues = singularValuess.diagonal(0);
  this->m_matrixU = L * this->m_matrixU;
  this->m_matrixV = R * this->m_matrixV;
  this->m_isInitialized = true;
  this->m_computeFullU = true;
  this->m_computeFullV = true;

  delete[] d;
  delete[] e;
  delete[] w;
  delete[] z;

  return *this;
}

};  // namespace SVD_Project

#endif  // V0_NAIVE_MRRR_HPP
