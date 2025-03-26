#ifndef GIVENS_REFINEMENT_HPP
#define GIVENS_REFINEMENT_HPP

#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>
#include <vector>

namespace SVD_Project {

// Interface of tests forces this structure
template <typename _MatrixType>
GivRef_SVD<_MatrixType>::GivRef_SVD(const _MatrixType &matrix,
                                    unsigned int computationOptions) {
    initialize(matrix, computationOptions);
}

template <typename _MatrixType>
GivRef_SVD<_MatrixType> &
GivRef_SVD<_MatrixType>::compute(const _MatrixType &matrix,
                                 unsigned int computationOptions) {
    initialize(matrix, computationOptions);
    return *this;
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::initialize(const _MatrixType &matrix,
                                         unsigned int computationOptions) {
    int m = matrix.rows();
    int n_cols = matrix.cols();
    n = std::min(m, n_cols);
    trigonom_i = 0;
    iter_num = 0;

    left_J = SquareMatrix::Identity(m, m);
    right_J = SquareMatrix::Identity(n_cols, n_cols);

    auto bid = Eigen::internal::UpperBidiagonalization<_MatrixType>(matrix);
    B = bid.bidiagonal();
    sigm_B = B;

    Eigen::JacobiSVD<_MatrixType> svd(matrix, Eigen::ComputeFullU |
                                                  Eigen::ComputeFullV);
    true_sigm_B = SquareMatrix::Zero(m, n_cols);
    true_sigm_B.diagonal() = svd.singularValues().head(n);

    int max_iter = 100; // TODO
    int total_rotations = 2 * max_iter * (n - 1);
    Cosines.resize(total_rotations);
    Sines.resize(total_rotations);
    Tans.resize(total_rotations);
    NewCosines.resize(total_rotations);
    NewSines.resize(total_rotations);

    for (int i = 0; i < max_iter; i++) {
        Impl_QR_zero_iter();
        iter_num++;
    }

    NewCosines = Cosines;
    NewSines = Sines;
    revert_negative_singular();

    m_matrixU = left_J.transpose();
    m_matrixV = right_J;
    m_singularValues = sigm_B.diagonal().head(n);
}

template <typename _MatrixType>
std::vector<typename _MatrixType::Scalar>
GivRef_SVD<_MatrixType>::ROT(typename _MatrixType::Scalar f,
                             typename _MatrixType::Scalar g) {
    using Scalar = typename _MatrixType::Scalar;
    Scalar cs, sn, r;

    if (f == 0) {
        cs = 0;
        sn = 1;
        r = g;
    } else if (std::abs(f) > std::abs(g)) {
        Scalar t = g / f;
        Scalar tt = std::sqrt(1 + t * t);
        cs = 1 / tt;
        sn = t * cs;
        r = f * tt;
        if (trigonom_i < Tans.size()) {
            Tans(trigonom_i) = t;
        }
    } else {
        Scalar t = f / g;
        Scalar tt = std::sqrt(1 + t * t);
        sn = 1 / tt;
        cs = t * sn;
        r = g * tt;
        if (trigonom_i < Tans.size()) {
            Tans(trigonom_i) = t;
        }
    }

    return std::vector<Scalar>{cs, sn, r};
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::Impl_QR_zero_iter() {
    if (n < 2)
        return; // without this it could trigger asserts

    using Scalar = typename _MatrixType::Scalar;
    Scalar oldcs = 1, oldsn = 0;
    Scalar cs = 1, sn, r;

    for (Index i = 0; i < n - 1; i++) {
        auto temp1 = ROT(sigm_B(i, i) * cs, sigm_B(i, i + 1));
        cs = temp1[0];
        sn = temp1[1];
        r = temp1[2];

        SquareMatrix Temp_J_R =
            SquareMatrix::Identity(right_J.rows(), right_J.cols());
        Temp_J_R(i, i) = cs;
        Temp_J_R(i, i + 1) = -sn;
        Temp_J_R(i + 1, i) = sn;
        Temp_J_R(i + 1, i + 1) = cs;
        right_J = right_J * Temp_J_R;

        Cosines(trigonom_i) = cs;
        Sines(trigonom_i) = sn;
        trigonom_i++;

        if (i != 0) {
            sigm_B(i - 1, i) = oldsn * r;
        }

        auto temp2 = ROT(oldcs * r, sigm_B(i + 1, i + 1) * sn);
        oldcs = temp2[0];
        oldsn = temp2[1];
        sigm_B(i, i) = temp2[2];

        SquareMatrix Temp_J_L =
            SquareMatrix::Identity(left_J.rows(), left_J.cols());
        Temp_J_L(i, i) = oldcs;
        Temp_J_L(i, i + 1) = oldsn;
        Temp_J_L(i + 1, i) = -oldsn;
        Temp_J_L(i + 1, i + 1) = oldcs;
        left_J = Temp_J_L * left_J;

        Cosines(trigonom_i) = oldcs;
        Sines(trigonom_i) = oldsn;
        trigonom_i++;
    }

    Scalar h = sigm_B(n - 1, n - 1) * cs;
    sigm_B(n - 2, n - 1) = h * oldsn;
    sigm_B(n - 1, n - 1) = h * oldcs;
}

template <typename _MatrixType>
void GivRef_SVD<_MatrixType>::revert_negative_singular() {
    for (Index i = 0; i < n; i++) {
        if (sigm_B(i, i) < 0) {
            sigm_B(i, i) = -sigm_B(i, i);
            for (Index j = 0; j < left_J.rows(); j++) {
                left_J(i, j) = -left_J(i, j);
            }
        }
    }
}

} // namespace SVD_Project

#endif // GIVENS_REFINEMENT_HPP
