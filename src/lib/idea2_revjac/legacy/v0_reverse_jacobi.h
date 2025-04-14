#ifndef V0_REVERSE_JACOBI_H
#define V0_REVERSE_JACOBI_H

#include <Eigen/SVD>

namespace SVD_Project {

template <typename _MatrixType>
class v0_RevJac_SVD : public Eigen::SVDBase<v0_RevJac_SVD<_MatrixType>> {
  typedef Eigen::SVDBase<v0_RevJac_SVD> Base;
  typedef typename _MatrixType::Scalar Scalar;
  typedef typename _MatrixType::Index Index;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;
  enum Rotation { Left, Right };

  using _SingularVectorType = VectorDynamic;

 public:
  const int MAX_ITERATIONS = 1000;
  const double TOLERANCE = 1e-10;

  v0_RevJac_SVD(const _MatrixType& initial,
                const _SingularVectorType& singularValues,
                unsigned int computationOptions = 0);
  v0_RevJac_SVD& compute();

  const MatrixDynamic& matrixU() const { return m_matrixU; }
  const MatrixDynamic& matrixV() const {
    return m_transposedMatrixV;  // Crashes if .transpose() used.
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
  // Eigen::JacobiRotation<Scalar> identityRotation() const;
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::v0_RevJac_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "v0_reverse_jacobi.hpp"

#endif  // V0_REVERSE_JACOBI_H
