#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/Dense>
// #include <Eigen/Eigenvalues>
// #include <algorithm>
// #include <cmath>
// #include <stdexcept>
// #include <vector>

namespace SVD_Project {
template <typename _MatrixType>
class GivRef_SVD;
}

namespace Eigen {
namespace internal {

template <typename _MatrixType>
struct traits<SVD_Project::GivRef_SVD<_MatrixType>>
    : public traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

}  // namespace internal
}  // namespace Eigen

namespace SVD_Project {

/**
 * @brief Implicit zero-shift QR with modifications
 *
 * @tparam _MatrixType matrix type (example Eigen::Matrix<double,
 * Eigen::Dynamic, Eigen::Dynamic>).
 */
template <typename _MatrixType>
class GivRef_SVD : public Eigen::SVDBase<GivRef_SVD<_MatrixType>> {
 public:
  using Base = Eigen::SVDBase<GivRef_SVD<_MatrixType>>;
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using Index = Eigen::Index;

  GivRef_SVD();

  /**
   * @brief A constructor that calculates the SVD immediately upon object
   * creation.
   *
   * @param B input bidiagonal matrix.
   * @param computationOptions Calculation flags (example, Eigen::ComputeFullU |
   * Eigen::ComputeFullV).
   */
  GivRef_SVD(const MatrixType& B,
             unsigned int computationOptions = Eigen::ComputeThinU |
                                               Eigen::ComputeThinV) {
    compute(B, computationOptions);
  }

  ~GivRef_SVD() = default;

  /**
   * @brief Calculates the SVD of the initial matrix A.
   *
   * @param A
   * @param computationOptions Calculation flags (example, Eigen::ComputeFullU |
   * Eigen::ComputeFullV).
   * @return Reference to class
   */
  GivRef_SVD& compute(const MatrixType& A,
                      unsigned int computationOptions = Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);

  // getters for resulting vectors (from Eigen::SVDBase)
  using Base::matrixU;
  using Base::matrixV;
  using Base::singularValues;

 protected:
};

}  // namespace SVD_Project

#include "givens_refinement.hpp"

#endif  // GIVENS_REFINEMENT_H
