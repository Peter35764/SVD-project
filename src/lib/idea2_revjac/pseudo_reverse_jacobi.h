#ifndef PSEUDO_REVERSE_JACOBI_H
#define PSEUDO_REVERSE_JACOBI_H

#include <Eigen/SVD>
#include <ostream>

namespace SVD_Project {

template <typename _MatrixType>
class PseudoRevJac_SVD : public Eigen::SVDBase<PseudoRevJac_SVD<_MatrixType>> {
 public:
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;
  using Base = Eigen::SVDBase<PseudoRevJac_SVD<_MatrixType>>;

  using Base::matrixU;
  using Base::matrixV;
  using Base::singularValues;

 protected:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;
  std::ostream* m_divOstream = nullptr;
  enum RotationType { Left, Right };

 public:
  PseudoRevJac_SVD() = default;

  ~PseudoRevJac_SVD() = default;

  PseudoRevJac_SVD(const MatrixType& initial,
                   const VectorDynamic& singularValues,
                   unsigned int computationOptions = 0);
  PseudoRevJac_SVD(const MatrixType& initial,
                   const VectorDynamic& singularValues, std::ostream* os,
                   unsigned int computationOptions = 0);
  PseudoRevJac_SVD(const MatrixType& initial,
                   const VectorDynamic& singularValues,
                   const MatrixDynamic& matrixU, const MatrixDynamic& matrixV,
                   unsigned int computationOptions = 0);
  PseudoRevJac_SVD(const MatrixType& initial,
                   const VectorDynamic& singularValues,
                   const MatrixDynamic& matrixU, const MatrixDynamic& matrixV,
                   std::ostream* os, unsigned int computationOptions = 0);

  PseudoRevJac_SVD& Compute(const MatrixType& initial,
                            unsigned int computationOptions = 0);
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::PseudoRevJac_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "pseudo_reverse_jacobi.hpp"

#endif  // PSEUDO_REVERSE_JACOBI_H
