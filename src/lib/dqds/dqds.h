#pragma once

// guardspirit@protonmail.com
// КМБО-03-22

#include <lapacke.h>

#include <Eigen/SVD>

// Класс, реализующий DQDS с использованием LAPACKE.
// (LAPACKE — обёртка над LAPACK)
// Наследует Eigen::SVDBase.
template <typename _MatrixType>
class DQDS_SVD : public Eigen::SVDBase<DQDS_SVD<_MatrixType>> {
 private:
  typedef Eigen::SVDBase<DQDS_SVD<_MatrixType>> Base;

 public:
  // Конструктор, выполняющий вычисление U, сингулярных значений и V
  DQDS_SVD(const _MatrixType& matrix, unsigned int computationOptions);

  DQDS_SVD<_MatrixType>& compute(const _MatrixType& matrix,
                                 unsigned int computationOptions);
};

//
template <typename _MatrixType>
struct Eigen::internal::traits<DQDS_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "dqds.cpp"
