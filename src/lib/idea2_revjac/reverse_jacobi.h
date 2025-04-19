#ifndef REVERSE_JACOBI_H
#define REVERSE_JACOBI_H

#include <Eigen/SVD>
#include <ostream>

namespace SVD_Project {

template <typename _MatrixType>
class RevJac_SVD : public Eigen::SVDBase<RevJac_SVD<_MatrixType>> {
  typedef Eigen::SVDBase<RevJac_SVD> Base;

  typedef typename _MatrixType::Scalar Scalar;
  typedef typename _MatrixType::Index Index;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;

  enum Rotation { Left, Right };

 public:
  RevJac_SVD() = default;

  RevJac_SVD(const _MatrixType& initial, const VectorDynamic& singularValues,
             unsigned int computationOptions = 0);
  RevJac_SVD(const _MatrixType& initial, const VectorDynamic& singularValues,
             std::ostream* os, unsigned int computationOptions = 0);

  RevJac_SVD& compute();
  RevJac_SVD& compute(std::ostream* os);

  const MatrixDynamic& matrixU() const { return m_matrixU; }
  const MatrixDynamic& matrixV() const {
    return m_transposedMatrixV.transpose();
  }
  const VectorDynamic& singularValues() const { return m_singularValues; }

 private:
  void iterate();
  bool convergenceReached() const;
  void updateDifference();
  void calculateBiggestDifference();
  Eigen::JacobiRotation<Scalar> composeLeftRotation(const Index& i,
                                                    const Index& j) const;
  Eigen::JacobiRotation<Scalar> composeRightRotation(const Index& i,
                                                     const Index& j) const;
  // Eigen::JacobiRotation<Scalar> identityRotation() const;

  MatrixDynamic m_matrixU;
  MatrixDynamic m_transposedMatrixV;
  const VectorDynamic& m_singularValues;
  const _MatrixType& m_initialMatrix;
  MatrixDynamic m_currentMatrix;
  _MatrixType m_differenceMatrix;
  Rotation m_lastRotation;
  Index m_currentI, m_currentJ;

  std::ostream* m_divOstream;
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::RevJac_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "reverse_jacobi.hpp"

#endif  // REVERSE_JACOBI_H
