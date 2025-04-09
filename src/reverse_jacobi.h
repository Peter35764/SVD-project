#ifndef REVERSE_JACOBI_H
#define REVERSE_JACOBI_H

#include <Eigen/SVD>

namespace SVD_Project {

template <typename _MatrixType, typename _SingularVectorType>
class RevJac_SVD
    : public Eigen::SVDBase<RevJac_SVD<_MatrixType, _SingularVectorType>> {
  typedef Eigen::SVDBase<RevJac_SVD> Base;
  typedef typename _MatrixType::Scalar Scalar;
  typedef typename _MatrixType::Index Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;
  enum Rotation { Left, Right };

 public:
  RevJac_SVD(const _MatrixType& initial,
             const _SingularVectorType& singularValues,
             unsigned int computationOptions = 0);
  RevJac_SVD& compute();

  const MatrixDynamic& matrixU() const { return m_matrixU; }
  const MatrixDynamic& matrixV() const {
    return m_transposedMatrixV.transpose();
  }
  const _SingularVectorType& singularValues() const { return m_singularValues; }

 private:
  MatrixDynamic m_matrixU;
  MatrixDynamic m_transposedMatrixV;
  const _SingularVectorType& m_singularValues;
  const _MatrixType& m_initialMatrix;
  MatrixDynamic m_currentMatrix;
  _MatrixType m_differenceMatrix;
  Rotation m_lastRotation;
  Index m_currentI, m_currentJ;

  void iterate();
  bool convergenceReached() const;
  void updateDifference();
  void calculateBiggestDifference();
  Eigen::JacobiRotation<Scalar> composeLeftRotation(const Index& i,
                                                    const Index& j) const;
  Eigen::JacobiRotation<Scalar> composeRightRotation(const Index& i,
                                                     const Index& j) const;
  Eigen::JacobiRotation<Scalar> identityRotation() const;
};

}  // namespace SVD_Project

template <typename _MatrixType, typename _SingularVectorType>
struct Eigen::internal::traits<
    SVD_Project::RevJac_SVD<_MatrixType, _SingularVectorType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "reverse_jacobi.hpp"

#endif  // REVERSE_JACOBI_H
