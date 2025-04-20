#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/SVD>
#include <ostream>
#include <vector>

namespace SVD_Project {

template <typename _MatrixType>
class GivRef_SVD : public Eigen::SVDBase<GivRef_SVD<_MatrixType>> {
  typedef Eigen::SVDBase<GivRef_SVD> Base;

  typedef typename _MatrixType::Scalar Scalar;
  typedef typename _MatrixType::Index Index;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixDynamic;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorDynamic;

 public:
  GivRef_SVD() = default;
  // This structure is required for tests
  explicit GivRef_SVD(const _MatrixType &matrix,
                      unsigned int computationOptions = 0);
  explicit GivRef_SVD(const _MatrixType &matrix,
                      const typename Base::SingularValuesType &singularValues,
                      unsigned int computationOptions = 0);
  explicit GivRef_SVD(const _MatrixType &matrix,
                      const typename Base::SingularValuesType &singularValues,
                      std::ostream *os, unsigned int computationOptions = 0);

  GivRef_SVD &compute(const _MatrixType &matrix,
                      unsigned int computationOptions = 0);
  GivRef_SVD &compute(const _MatrixType &matrix, std::ostream *os,
                      unsigned int computationOptions = 0);

  void setDivergenceOstream(std::ostream *os);
  const typename Base::MatrixUType &matrixU() const { return m_matrixU; }
  const typename Base::MatrixVType &matrixV() const { return m_matrixV; }
  const typename Base::SingularValuesType &singularValues() const {
    return m_singularValues;
  }

 private:
  // Introducing helpers for compute decomposition
  void setupMatrices(const _MatrixType &matrix);
  void iterQRtillConv(Scalar tol, int max_iter);
  void fixFormatResults();

  std::vector<Scalar> ROT(Scalar f, Scalar g);
  void Impl_QR_zero_iter();
  void revert_negative_singular();
  bool isConvergedSafely(Scalar tol,
                         int max_iter);  // cant be const since we want to
                                         // actually null the elements on check

  MatrixDynamic m_matrixU, m_matrixV;
  VectorDynamic m_singularValues;
  MatrixDynamic left_J;
  MatrixDynamic right_J;
  MatrixDynamic sigm_B;
  MatrixDynamic true_sigm_B;
  MatrixDynamic B;
  VectorDynamic Cosines;
  VectorDynamic Sines;
  VectorDynamic Tans;
  VectorDynamic NewCosines;
  VectorDynamic NewSines;
  Index n;
  Index m;       // rows
  Index n_cols;  // init cols
  Index trigonom_i;
  Index iter_num;
  bool has_true_sigm;  // to diff constructors, whether true_sigm_B is available

  std::ostream *m_divOstream;
};

}  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::GivRef_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "givens_refinement.hpp"

#endif  // GIVENS_REFINEMENT_H
