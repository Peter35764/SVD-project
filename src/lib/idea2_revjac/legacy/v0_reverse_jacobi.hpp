#ifndef V0_REVERSE_JACOBI_HPP
#define V0_REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>
#include <iostream>  // TODO delete if unused

// necessary for correct display in ide (or clangd lsp), does not
// affect the assembly process and can be removed
#include "v0_reverse_jacobi.h"

namespace SVD_Project {

template <typename _MatrixType>
v0_RevJac_SVD<_MatrixType>::v0_RevJac_SVD(const _MatrixType& initial,
                                    const _SingularVectorType& singularValues,
                                    unsigned int computationOptions)
    : m_initialMatrix(initial), m_singularValues(singularValues) {
  Index cols = initial.cols();
  Index rows = initial.rows();

  m_matrixU = MatrixDynamic::Identity(rows, rows);
  m_transposedMatrixV = MatrixDynamic::Identity(cols, cols);
  m_lastRotation = Rotation::Left;
  m_currentMatrix =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;

  this->compute();
}

template <typename _MatrixType>
v0_RevJac_SVD<_MatrixType>& v0_RevJac_SVD<_MatrixType>::compute() {
  for (size_t i = 0; i < MAX_ITERATIONS; i++) {
    iterate();
    if (convergenceReached()) {
      break;
    }
  }
  return *this;
};

template <typename _MatrixType>
void v0_RevJac_SVD<_MatrixType>::iterate() {
  updateDifference();

  calculateBiggestDifference();

  if (m_lastRotation == Rotation::Left) {
    m_transposedMatrixV.applyOnTheRight(
        m_currentI, m_currentJ,
        composeRightRotation(m_currentI, m_currentJ).adjoint());
    m_lastRotation = Rotation::Right;
  } else {
    m_matrixU.applyOnTheLeft(
        m_currentI, m_currentJ,
        composeLeftRotation(m_currentI, m_currentJ).adjoint());
    m_lastRotation = Rotation::Left;
  }
}

template <typename _MatrixType>
bool v0_RevJac_SVD<_MatrixType>::convergenceReached() const {
  return (m_differenceMatrix.norm() < TOLERANCE);
  std::cout << "\nconvergenceReached\n TODO DELETE THIS MESSAGE";
}

template <typename _MatrixType>
void v0_RevJac_SVD<_MatrixType>::updateDifference() {
  m_currentMatrix =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;
  m_differenceMatrix = m_currentMatrix - m_initialMatrix;
}

template <typename _MatrixType>
void v0_RevJac_SVD<_MatrixType>::calculateBiggestDifference() {
  Index maxIndex = m_lastRotation == Rotation::Left ? m_transposedMatrixV.cols()
                                                    : m_matrixU.rows();
  Scalar absBiggestDiff = 0;
  for (Index k = 0; k < maxIndex; k++) {
    for (Index l = 0; l < maxIndex; l++) {
      if (k == l) continue;  // Пропустить диагональ
      Scalar currDiff = std::abs(m_differenceMatrix(k, l));
      if (absBiggestDiff < currDiff) {
        absBiggestDiff = currDiff;
        m_currentI = k;
        m_currentJ = l;
      }
    }
  }
}

template <typename _MatrixType>
Eigen::JacobiRotation<typename v0_RevJac_SVD<_MatrixType>::Scalar>
v0_RevJac_SVD<_MatrixType>::composeLeftRotation(const Index& i,
                                             const Index& j) const {
  Eigen::JacobiRotation<Scalar> rot;
  rot.makeGivens(m_differenceMatrix(i, i),
                 m_differenceMatrix(i, j));  // TODO check div by 0
  return rot;
}

template <typename _MatrixType>
Eigen::JacobiRotation<typename v0_RevJac_SVD<_MatrixType>::Scalar>
v0_RevJac_SVD<_MatrixType>::composeRightRotation(const Index& i,
                                              const Index& j) const {
  Eigen::JacobiRotation<Scalar> rot;
  rot.makeGivens(m_differenceMatrix(i, i),
                 m_differenceMatrix(i, j));  // TODO check div by 0
  return rot;
}

// template <typename _MatrixType>
// Eigen::JacobiRotation<typename v0_RevJac_SVD<_MatrixType>::Scalar>
// v0_RevJac_SVD<_MatrixType>::identityRotation() const {
//   Eigen::JacobiRotation<Scalar> rot;
//   rot.c() = Scalar(1);
//   rot.s() = Scalar(0);
//   return rot;
// }

}  // namespace SVD_Project

#endif  // V0_REVERSE_JACOBI_HPP
