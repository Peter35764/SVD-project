#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/Dense>
#include <vector>
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
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;

  GivRef_SVD();

  // TODO
  GivRef_SVD(const MatrixType& A,
             unsigned int computationOptions = Eigen::ComputeThinU |
                                               Eigen::ComputeThinV) {}
  /**
   * @brief A constructor that calculates the SVD immediately upon object
   * creation.
   *
   * @param B input bidiagonal matrix.
   * @param computationOptions Calculation flags (example, Eigen::ComputeFullU |
   * Eigen::ComputeFullV).
   */
  GivRef_SVD(const MatrixType& A,
             const VectorDynamic& singularValues = VectorDynamic(),
             std::ostream* os = nullptr,
             unsigned int computationOptions = Eigen::ComputeThinU |
                                               Eigen::ComputeThinV)
      : m_divOstream(os), singVals(singularValues) {
    compute(A, computationOptions);
  }

  GivRef_SVD(const MatrixType& A, const VectorDynamic& singularValues,
             unsigned int computationOptions)
      : GivRef_SVD(A, singularValues, nullptr,
                   computationOptions) {  // Делегирование
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
  /*
   * @brief Performs one QR iteration of the SVD algorithm using Givens
   * rotations
   */
  void performQRIteration();

  void preparation_phase(const MatrixType& A, unsigned int computationOptions);
  void qr_iterations_phase();

  void coordinate_descent_refinement(const MatrixType& B_target);
  void finalizing_output_phase();

  std::ostream* m_divOstream = nullptr;

  MatrixType left_J;        // Left Jacobi rotation matrix
  MatrixType right_J;       // Right Jacobi rotation matrix
  MatrixType sigm_B;        // Working copy of the bidiagonal matrix
  MatrixType m_original_B;  // refinement target
  Index m;                  // Number of rows
  std::vector<RealScalar> m_qr_theta_left;
  std::vector<RealScalar> m_qr_theta_right;
  Index n;  // Number of columns

  const VectorDynamic singVals;
};

}  // namespace SVD_Project

#include "givens_refinement.hpp"

#endif  // GIVENS_REFINEMENT_H
