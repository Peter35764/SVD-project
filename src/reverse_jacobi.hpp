#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>

// necessary for correct display in ide (or clangd lsp), does not
// affect the assembly process and can be removed
#include "reverse_jacobi.h"

namespace SVD_Project {

const size_t MAX_ITERATIONS = 1000;

template <typename _MatrixType, typename _SingularVectorType>
RevJac_SVD<_MatrixType, _SingularVectorType>::RevJac_SVD(
    const _MatrixType& initial, const _SingularVectorType& singularValues,
    unsigned int computationOptions)
    : m_initialMatrix(initial), m_singularValues(singularValues) {
  Index cols = initial.cols();
  Index rows = initial.rows();

  m_matrixU = MatrixDynamic::Identity(rows, rows);
  m_transposedMatrixV = MatrixDynamic::Identity(cols, cols);
  m_lastRotation = Rotation::Left;
  m_currentMatrix =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;
}

template <typename _MatrixType, typename _SingularVectorType>
RevJac_SVD<_MatrixType, _SingularVectorType>&
RevJac_SVD<_MatrixType, _SingularVectorType>::compute() {
  for (size_t i = 0; i < MAX_ITERATIONS; i++) {
    iterate();
    if (convergenceReached()) {
      break;
    }
  }
  return *this;
};

template <typename _MatrixType, typename _SingularVectorType>
void RevJac_SVD<_MatrixType, _SingularVectorType>::iterate() {
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

template <typename _MatrixType, typename _SingularVectorType>
bool RevJac_SVD<_MatrixType, _SingularVectorType>::convergenceReached() const {
  // TODO: implement
  return false;
}

template <typename _MatrixType, typename _SingularVectorType>
void RevJac_SVD<_MatrixType, _SingularVectorType>::updateDifference() {
  m_currentMatrix =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;
  m_differenceMatrix = m_currentMatrix - m_initialMatrix;
}

template <typename _MatrixType, typename _SingularVectorType>
void RevJac_SVD<_MatrixType,
                _SingularVectorType>::calculateBiggestDifference() {
  Index maxIndex = m_lastRotation == Rotation::Left ? m_transposedMatrixV.cols()
                                                    : m_matrixU.rows();
  Scalar absBiggestDiff = 0;
  for (Index k = 0; k < maxIndex; k++) {
    for (Index l = k; l < maxIndex; l++) {
      Scalar currDiff = std::abs(m_differenceMatrix(k, l));
      if (absBiggestDiff < currDiff) {
        absBiggestDiff = currDiff;
        m_currentI = k;
        m_currentJ = l;
      }
    }
  }
}

template <typename _MatrixType, typename _SingularVectorType>
Eigen::JacobiRotation<
    typename RevJac_SVD<_MatrixType, _SingularVectorType>::Scalar>
RevJac_SVD<_MatrixType, _SingularVectorType>::composeLeftRotation(
    const Index& i, const Index& j) const {
  Eigen::JacobiRotation<Scalar> rot;
  rot.makeGivens(m_differenceMatrix(i, i), m_differenceMatrix(i, j));
  return rot;
}

template <typename _MatrixType, typename _SingularVectorType>
Eigen::JacobiRotation<
    typename RevJac_SVD<_MatrixType, _SingularVectorType>::Scalar>
RevJac_SVD<_MatrixType, _SingularVectorType>::composeRightRotation(
    const Index& i, const Index& j) const {
  Eigen::JacobiRotation<Scalar> rot;
  rot.makeGivens(m_differenceMatrix(i, i), m_differenceMatrix(i, j));
  return rot;
}

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP
