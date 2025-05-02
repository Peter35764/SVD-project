#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

#include <Eigen/Jacobi>
#include <boost/math/tools/minima.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

#include "reverse_jacobi.h"

namespace SVD_Project {

const size_t MAX_ITERATIONS = 10;
const double TOLERANCE = 1e-10;

template <typename _MatrixType>
RevJac_SVD<_MatrixType>::RevJac_SVD(const _MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    unsigned int computationOptions)
    : m_initialMatrix(initial), m_singularValues(singularValues) {
  // TODO: make this algorithm work for rectangular matrices
  assert(initial.cols() == initial.rows());
  Index cols = initial.cols();
  Index rows = initial.rows();

  m_matrixU = MatrixDynamic::Identity(rows, rows);
  m_transposedMatrixV = MatrixDynamic::Identity(cols, cols);
  m_currentApproximation =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;

  this->compute();
}

template <typename _MatrixType>
RevJac_SVD<_MatrixType>::RevJac_SVD(const _MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    std::ostream* os,
                                    unsigned int computationOptions)
    : m_initialMatrix(initial), m_singularValues(singularValues) {
  // TODO: make this algorithm work for rectangular matrices
  assert(initial.cols() == initial.rows());
  Index cols = initial.cols();
  Index rows = initial.rows();

  m_matrixU = MatrixDynamic::Identity(rows, rows);
  m_transposedMatrixV = MatrixDynamic::Identity(cols, cols);
  m_currentApproximation =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;
  m_divOstream = os;

  this->compute();
}

template <typename _MatrixType>
RevJac_SVD<_MatrixType>::RevJac_SVD(const _MatrixType& initial,
                                    const VectorDynamic& singularValues,
                                    const MatrixDynamic& matrixU,
                                    const MatrixDynamic& matrixV,
                                    std::ostream* os,
                                    unsigned int computationOptions)
    : m_initialMatrix(initial), m_singularValues(singularValues) {
  // TODO: make this algorithm work for rectangular matrices
  assert(initial.cols() == initial.rows());
  Index cols = initial.cols();
  Index rows = initial.rows();

  m_matrixU = matrixU;
  m_transposedMatrixV = matrixV.transpose();
  m_currentApproximation =
      m_matrixU * m_singularValues.asDiagonal() * m_transposedMatrixV;
  m_divOstream = os;

  this->compute();
}

template <typename _MatrixType>
RevJac_SVD<_MatrixType>& RevJac_SVD<_MatrixType>::compute() {
  assert(m_singularValues.rows() ==
             std::min(m_initialMatrix.rows(), m_initialMatrix.cols()) &&
         "Singular value vector size and matrix size do not match");

  for (size_t iter = 0; iter < MAX_ITERATIONS; ++iter) {
    for (Index i = 0; i < m_initialMatrix.rows(); ++i) {
      for (Index j = 0; j < m_initialMatrix.cols(); ++j) {
        iterate(i, j);
      }
    }
    if (m_divOstream) {
      Scalar divergence = (m_currentApproximation - m_initialMatrix).norm();
      *m_divOstream << "Divergence: " << std::to_string(divergence)
                    << std::endl;
    }
    if (convergenceReached()) {
      break;
    }
  }

  return *this;
};

template <typename _MatrixType>
RevJac_SVD<_MatrixType>& RevJac_SVD<_MatrixType>::compute(std::ostream* os) {
  m_divOstream = os;
  return this->compute();
};

template <typename _MatrixType>
void RevJac_SVD<_MatrixType>::iterate(Index i, Index j) {
  // This class is kind of an overkill for its purpose.
  // It represents the function computing the divergence between the current
  // approximation matrix and the initial matrix after applying Jacobi rotation
  // parametrized by the value of the cosign. This class exists in order for us
  // to simplify the creation of this "lambda" function.
  class NormFunction {
   private:
    _MatrixType const& m_initialMatrix;
    MatrixDynamic const& m_currentApproximation;
    RotationType m_rotationType;
    Index i, j;

   public:
    NormFunction(_MatrixType const& initialMatrix,
                 MatrixDynamic const& currentApproximation, Index i, Index j,
                 RotationType rotationType)
        : m_initialMatrix(initialMatrix),
          m_currentApproximation(currentApproximation),
          m_rotationType(rotationType),
          i(i),
          j(j) {}

    Scalar operator()(Scalar c) const {
      Scalar s = std::sqrt(1 - c * c);
      Eigen::JacobiRotation<Scalar> rotation =
          Eigen::JacobiRotation<Scalar>(c, s);
      MatrixDynamic currentApproximation = m_currentApproximation;

      if (m_rotationType == RotationType::Left) {
        currentApproximation.applyOnTheLeft(i, j, rotation.transpose());
      } else {
        currentApproximation.applyOnTheRight(i, j, rotation);
      }

      return (currentApproximation - m_initialMatrix).norm();
    }
  };

  // Calculate first the left and then the right rotation.
  for (auto rotType : {RotationType::Left, RotationType::Left}) {
    NormFunction minimizedFunction =
        NormFunction(m_initialMatrix, m_currentApproximation, i, j, rotType);
    auto result = boost::math::tools::brent_find_minima<NormFunction, Scalar>(
        minimizedFunction, 0, 1, std::numeric_limits<Scalar>::digits);

    Scalar c = result.first;
    Scalar s = std::sqrt(1 - c * c);
    auto rotation = Eigen::JacobiRotation<Scalar>(c, s);

    if (rotType == RotationType::Left) {
      m_matrixU.applyOnTheLeft(i, j, rotation.transpose());
      m_currentApproximation.applyOnTheLeft(i, j, rotation.transpose());
    } else {
      m_transposedMatrixV.applyOnTheRight(i, j, rotation);
      m_currentApproximation.applyOnTheRight(i, j, rotation);
    }
  }
};

template <typename _MatrixType>
bool RevJac_SVD<_MatrixType>::convergenceReached() const {
  // return ((m_currentApproximation - m_initialMatrix).norm() < TOLERANCE);
  return false;
}

}  // namespace SVD_Project

#endif  // REVERSE_JACOBI_HPP
