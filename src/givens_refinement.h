#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/SVD>
#include <vector>

namespace SVD_Project {

template <typename _MatrixType>
class GivRef_SVD : public Eigen::SVDBase<GivRef_SVD<_MatrixType>> {
    typedef Eigen::SVDBase<GivRef_SVD> Base;
    typedef typename _MatrixType::Scalar Scalar;
    typedef typename _MatrixType::Index Index;

  public:
    // This structure is required for tests
    explicit GivRef_SVD(const _MatrixType &matrix,
                        unsigned int computationOptions = 0);
    GivRef_SVD &compute(const _MatrixType &matrix,
                        unsigned int computationOptions = 0);

    const typename Base::MatrixUType &matrixU() const { return m_matrixU; }
    const typename Base::MatrixVType &matrixV() const { return m_matrixV; }
    const typename Base::SingularValuesType &singularValues() const {
        return m_singularValues;
    }

  private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> SquareMatrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> EVector;

    SquareMatrix m_matrixU, m_matrixV;
    EVector m_singularValues;

    SquareMatrix left_J;
    SquareMatrix right_J;
    SquareMatrix sigm_B;
    SquareMatrix true_sigm_B;
    SquareMatrix B;

    EVector Cosines;
    EVector Sines;
    EVector Tans;
    EVector NewCosines;
    EVector NewSines;

    Index n;
    Index trigonom_i;
    Index iter_num;

    std::vector<Scalar> ROT(Scalar f, Scalar g);
    void Impl_QR_zero_iter();
    void revert_negative_singular();
    void initialize(const _MatrixType &matrix, unsigned int computationOptions);
    bool isConvergedSafely(Scalar tol, int max_iter) const;
};

} // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::GivRef_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
    typedef _MatrixType MatrixType;
};

#include "givens_refinement.hpp"

#endif // GIVENS_REFINEMENT_H
