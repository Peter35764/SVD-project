#ifndef REVERSE_JACOBI_H
#define REVERSE_JACOBI_H

#include <Eigen/SVD>
#include <iostream>
#include <ostream>

namespace SVD_Project {

template <typename _MatrixType>
class RevJac_SVD : public Eigen::SVDBase<RevJac_SVD<_MatrixType>> {
 public:
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;
  using Base = Eigen::SVDBase<RevJac_SVD<_MatrixType>>;

  using Base::matrixU;
  using Base::matrixV;
  using Base::singularValues;

 protected:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;
  std::ostream* m_divOstream = nullptr;
  void calculateSymmetrizingRotation(Scalar w, Scalar x, Scalar y, Scalar z,
                                     Scalar& c, Scalar& s);

 public:
  RevJac_SVD() = default;

  ~RevJac_SVD() = default;

  RevJac_SVD(const MatrixType& initial, const VectorDynamic& singularValues,
             unsigned int computationOptions = 0);
  RevJac_SVD(const MatrixType& initial, const VectorDynamic& singularValues,
             std::ostream* os, unsigned int computationOptions = 0);
  RevJac_SVD(const MatrixType& initial, const VectorDynamic& singularValues,
             const MatrixDynamic& matrixU, const MatrixDynamic& matrixV,
             unsigned int computationOptions = 0);
  RevJac_SVD(const MatrixType& initial, const VectorDynamic& singularValues,
             const MatrixDynamic& matrixU, const MatrixDynamic& matrixV,
             std::ostream* os, unsigned int computationOptions = 0);

  RevJac_SVD& Compute(const MatrixType& initial,
                      unsigned int computationOptions = 0);
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::RevJac_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "reverse_jacobi.hpp"

#endif  // REVERSE_JACOBI_H
