#ifndef REVERSE_JACOBI_H
#define REVERSE_JACOBI_H

#include <Eigen/src/Jacobi/Jacobi.h>

#include <Eigen/SVD>

namespace SVD_Project {

template <typename _MatrixType>
class RevJac_SVD : public Eigen::SVDBase<RevJac_SVD<_MatrixType>> {
  typedef Eigen::SVDBase<RevJac_SVD> Base;
  typedef typename _MatrixType::Scalar Scalar;
  typedef typename _MatrixType::Index Index;
  enum Rotation { Left, Right };

 public:
  RevJac_SVD(const _MatrixType &initial,
             const Base::SingularValuesType &singularValues,
             unsigned int computationOptions = 0);
  RevJac_SVD &compute();

  const typename Base::MatrixUType &matrixU() const { return m_matrixU; }
  const typename Base::MatrixVType &matrixV() const { return m_matrixV; }
  const typename Base::SingularValuesType &singularValues() const {
    return m_singularValues;
  }

 private:
  typename Base::MatrixUType m_matrixU;
  typename Base::MatrixVType m_matrixV;
  const typename Base::SingularValuesType &m_singularValues;
  const _MatrixType &m_initialMatrix;
  _MatrixType &m_currentMatrix;
  _MatrixType m_differenceMatrix;
  Rotation m_lastRotation;
  Index m_currentI, m_currentJ;

  void iterate();
  bool convergenceReached() const;
  void updateDifference();
  void biggestDifference(Index &i, Index &j) const;
  Eigen::JacobiRotation<Scalar> composeLeftRotation(const Index &i,
                                                    const Index &j) const;
  Eigen::JacobiRotation<Scalar> composeRightRotation(const Index &i,
                                                     const Index &j) const;
};

}  // namespace SVD_Project

// template<typename _MatrixType>
// struct Eigen::internal::traits<SVD_Project::RevJac_SVD<_MatrixType>>
//     : Eigen::internal::traits<_MatrixType>
// {
//     typedef Eigen::internal::traits<_MatrixType>::MatrixType MatrixType;
//     typedef Eigen::internal::traits<MatrixType>::Scalar Scalar;
//     typedef Eigen::internal::traits<MatrixType>::Index Index;
// };

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::RevJac_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "reverse_jacobi.hpp"

#endif  // REVERSE_JACOBI_H