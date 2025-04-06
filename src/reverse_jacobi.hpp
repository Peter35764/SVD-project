#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

// #include "reverse_jacobi.h" // necessary for correct display in ide, does not affect the assembly process and can be removed
#include <Eigen/src/Jacobi/Jacobi.h>

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
  m_matrixV = MatrixDynamic::Identity(cols, cols);
  m_lastRotation = Rotation::Left;
  m_currentMatrix =
      m_matrixU * m_singularValues.asDiagonal() * m_matrixV.transpose();
}

template <typename _MatrixType, typename _SingularVectorType>
RevJac_SVD<_MatrixType, _SingularVectorType>&
RevJac_SVD<_MatrixType, _SingularVectorType>::compute() {
  for (int i = 0; i < MAX_ITERATIONS; i++) {
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
  biggestDifference(m_currentI, m_currentJ);
  if (m_lastRotation == Rotation::Left) {
    // FIXME: This is kind of a warcrime tbh
    m_matrixV.transposeInPlace();
    m_matrixV.applyOnTheRight(
        composeRightRotation(m_currentI, m_currentJ).adjoint());
    m_matrixV.transposeInPlace();
    m_lastRotation = Rotation::Right;
  } else {
    m_matrixU.applyOnTheLeft(
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
      m_matrixU * m_singularValues.asDiagonal() * m_matrixV.transpose();
  m_differenceMatrix = m_currentMatrix - m_initialMatrix;
}

template <typename _MatrixType, typename _SingularVectorType>
void RevJac_SVD<_MatrixType, _SingularVectorType>::biggestDifference(
    Index& i, Index& j) const {
  Scalar absBiggestDiff = 0;
  for (Index k = 0; k < m_initialMatrix.rows(); k++) {
    for (Index l = 0; k < m_initialMatrix.cols(); l++) {
      Scalar currDiff = std::abs(m_differenceMatrix(k, l));
      if (absBiggestDiff < currDiff) {
        absBiggestDiff = currDiff;
        i = k;
        j = l;
      }
    }
  }
}

template <typename _MatrixType, typename _SingularVectorType>
Eigen::JacobiRotation<
    typename RevJac_SVD<_MatrixType, _SingularVectorType>::Scalar>
RevJac_SVD<_MatrixType, _SingularVectorType>::composeLeftRotation(
    const Index& i, const Index& j) const {
  // TODO: Make sure this is correct code
  return Eigen::JacobiRotation<Scalar>().makeGivens(m_currentMatrix(i, i),
                                                    m_currentMatrix(i, j));
}

template <typename _MatrixType, typename _SingularVectorType>
Eigen::JacobiRotation<
    typename RevJac_SVD<_MatrixType, _SingularVectorType>::Scalar>
RevJac_SVD<_MatrixType, _SingularVectorType>::composeRightRotation(
    const Index& i, const Index& j) const {
  // TODO: Make sure this is correct code
  return Eigen::JacobiRotation<Scalar>().makeGivens(m_currentMatrix(j, j),
                                                    m_currentMatrix(j, i));
}

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP