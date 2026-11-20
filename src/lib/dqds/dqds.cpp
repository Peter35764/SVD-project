#pragma once

#include "dqds.h"

template <typename _MatrixType>
DQDS_SVD<_MatrixType>::DQDS_SVD(const _MatrixType& matrix,
                               unsigned int computationOptions) {
  compute(matrix, computationOptions);
  // TODO: computationOptions на данный момент игнорируются.
}

template <typename _MatrixType>
DQDS_SVD<_MatrixType>& DQDS_SVD<_MatrixType>::compute(
    const _MatrixType& matrix, unsigned int computationOptions) {
  auto bid = Eigen::internal::UpperBidiagonalization(
      matrix);  // DQDS принимает на вход бидиагональную матрицу
  typename Eigen::internal::UpperBidiagonalization<
      _MatrixType>::BidiagonalType bid_matrix = bid.bidiagonal();
  // Копируем итоговую бидиагональную матрицу
  auto diagonal =
      bid_matrix.diagonal(0);  // Главная диагональ. diagonal.data() содержит
                               // массив только из элементов диагонали
  auto offdiagonal = bid_matrix.diagonal(1);  // Наддиагональ
  int n = bid.bidiagonal().cols();

  int info;

  if (computationOptions & (Eigen::ComputeFullU | Eigen::ComputeFullV)) {
    this->m_matrixU = bid.householderU();  // U. diagonal.data() содержит U в
                                           // формате Column Major
    this->m_matrixV =
        bid.householderV();  // V. diagonal.data() содержит V в формате Row
                             // Major => при интерпретировании данных
    // как Column Major diagonal.data() содержит V^T.
    info = LAPACKE_dbdsqr(
        LAPACK_COL_MAJOR,
        'U',                     // Upper bidiagonal
        n,                       // N: Order of B
        this->m_matrixV.rows(),  // NCVT: Columns in VT = rows in V
        this->m_matrixU.rows(),  // NRU: Rows in U
        0,                       // NCC: Columns in C
        diagonal.data(),         // [in, out]
        offdiagonal.data(),      // [in, out]
        this->m_matrixV.data(),
        this->m_matrixV.cols(),  // LDVT: Leading dimension of VT
        // т. е. как много элементов необходимо пропустить LAPACK'у, чтобы
        // добраться до начала следующей колонки = rows in VT = cols in V
        this->m_matrixU.data(),  // [in, out]
        this->m_matrixU.rows(),  // LDU
        nullptr,                 // C
        1);
    this->m_computeFullU = true;
    this->m_computeFullV = true;
  } else {
    info = LAPACKE_dbdsqr(
        LAPACK_COL_MAJOR,
        'U',                 // Upper bidiagonal
        n,                   // N: Order of B
        0,                   // NCVT: Columns in VT = rows in V
        0,                   // NRU: Rows in U
        0,                   // NCC: Columns in C
        diagonal.data(),     // [in, out]
        offdiagonal.data(),  // [in, out]
        nullptr,
        1,  // LDVT: Leading dimension of VT
        // т. е. как много элементов необходимо пропустить LAPACK'у, чтобы
        // добраться до начала следующей колонки = rows in VT = cols in V
        nullptr,  // [in, out]
        1,        // LDU
        nullptr,  // C
        1);
  }
  if (info != 0)
    throw std::runtime_error("LAPACK error: " + std::to_string(info));
  this->m_singularValues = diagonal;
  this->m_isInitialized = true;
  return *this;
}
