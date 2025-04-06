#ifndef V0_GIVENS_REFINEMENT_HPP
#define V0_GIVENS_REFINEMENT_HPP

#include "v0_givens_refinement.h"  // necessary for correct display in ide, does not affect the assembly process and can be removed

namespace SVD_Project {

// Interface of tests forces this structure
template <typename _MatrixType>
v0_GivRef_SVD<_MatrixType>::v0_GivRef_SVD(const _MatrixType& matrix,
                                          unsigned int computationOptions) {
    initialize(matrix, computationOptions);
}

template <typename _MatrixType>
v0_GivRef_SVD<_MatrixType>& v0_GivRef_SVD<_MatrixType>::compute(
    const _MatrixType& matrix, unsigned int computationOptions) {
    initialize(matrix, computationOptions);
    return *this;
}

template <typename _MatrixType>
bool v0_GivRef_SVD<_MatrixType>::isConvergedSafely(
    typename _MatrixType::Scalar tol, int max_iter) const {
    if (n < 2)
        return true; // Trivial for 1×1 matrices

    using Scalar = typename _MatrixType::Scalar;

    // Protection against potentional underflow issues, mentioned in paper
    const Scalar underflow_threshold = std::numeric_limits<Scalar>::min();
    tol = std::max(tol, Scalar(max_iter) * underflow_threshold);

    // Compute mu, see (4.3)
    EVector mu(n);
    mu(0) = std::abs(sigm_B(0, 0));
    for (Index j = 0; j < n - 1; j++) {
        Scalar e_j_sq = sigm_B(j, j + 1) * sigm_B(j, j + 1); // e_j^2
        Scalar s_jp1 = std::abs(sigm_B(j + 1, j + 1));       // s_j+1 = sjp
        mu(j + 1) = s_jp1 * (mu(j) / (mu(j) + e_j_sq));
    }

    // Compute λ_j, see (4.4)
    EVector lambda(n);
    lambda(n - 1) = std::abs(sigm_B(n - 1, n - 1));
    for (Index j = n - 2; j >= 0; j--) {
        Scalar e_j_sq = sigm_B(j, j + 1) * sigm_B(j, j + 1); // e_j^2
        Scalar s_j = std::abs(sigm_B(j, j));                 // |s_j|
        lambda(j) =
            s_j * (lambda(j + 1) / (lambda(j + 1) + e_j_sq)); // Direct impl
    }

    // Apply Conv.Crit 1a and 1b
    bool all_converged = true;
    for (Index j = 0; j < n - 1; j++) {
        Scalar e_j = std::abs(sigm_B(j, j + 1));
        Scalar e_j_sq = e_j * e_j;

        // Without this check could get near-zero div
        if (e_j <= std::numeric_limits<Scalar>::epsilon() *
                       std::max(std::abs(sigm_B(j, j)),
                                std::abs(sigm_B(j + 1, j + 1)))) {
            continue;
        }

        // Criterion 1a: If e_j^2/mu_j <= tol we can set e_j to 0
        bool can_zero_by_1a = (e_j_sq / mu(j) <= tol);

        // Criterion 1b: If |e_j|^2/lambda_j+1 <= tol we can set e_j to 0
        bool can_zero_by_1b = (e_j_sq / lambda(j + 1) <= tol);

        if (!(can_zero_by_1a || can_zero_by_1b)) { // any criterion suffices ig
            all_converged = false;
            break;
        }
    }

    return all_converged;
}

template <typename _MatrixType>
void v0_GivRef_SVD<_MatrixType>::initialize(const _MatrixType& matrix,
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

    // In the initialize method, replace the for-loop with:
    Scalar tol = Scalar(1e-10); // tolerance
    int max_iter = 100;         // Safety limit

    int total_rotations = 2 * max_iter * (n - 1);
    Cosines.resize(total_rotations);
    Sines.resize(total_rotations);
    Tans.resize(total_rotations);
    NewCosines.resize(total_rotations);
    NewSines.resize(total_rotations);

    for (int i = 0; i < max_iter; i++) {
        Impl_QR_zero_iter();
        iter_num++;

        if (isConvergedSafely(tol, max_iter)) {
            break;
        }
    }

    NewCosines = Cosines;
    NewSines = Sines;
    revert_negative_singular();

    m_matrixU = left_J.transpose();
    m_matrixV = right_J;
    m_singularValues = sigm_B.diagonal().head(n);
}

template <typename _MatrixType>
std::vector<typename _MatrixType::Scalar> v0_GivRef_SVD<_MatrixType>::ROT(
    typename _MatrixType::Scalar f, typename _MatrixType::Scalar g) {
    using Scalar = typename _MatrixType::Scalar;
    Scalar cs, sn, r;

    if (f == 0) {
        cs = 0;
        sn = 1;
        r = g;
    } else {
        Eigen::JacobiRotation<Scalar> rot;
        rot.makeGivens(f, g);
        cs = rot.c();
        sn = rot.s();
        r = cs * f + sn * g; // match orig
        if (trigonom_i < Tans.size()) {
            Tans(trigonom_i) =
                std::abs(f) > std::abs(g) ? g / f : f / g; // match orig
        }
    }

    return std::vector<Scalar>{cs, sn, r};
}

template <typename _MatrixType>
void v0_GivRef_SVD<_MatrixType>::Impl_QR_zero_iter() {
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

        Eigen::JacobiRotation<Scalar> rotRight(
            cs, -sn); // Needs negative sn to match original, we're trying to
                      // keep the og impl idea
        right_J.applyOnTheRight(i, i + 1, rotRight);

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

        Eigen::JacobiRotation<Scalar> rotLeft(oldcs,
                                              -oldsn); // See note above
        rotLeft.transpose();                           // To match og
        left_J.applyOnTheLeft(i, i + 1, rotLeft);

        Cosines(trigonom_i) = oldcs;
        Sines(trigonom_i) = oldsn;
        trigonom_i++;
    }

    Scalar h = sigm_B(n - 1, n - 1) * cs;
    sigm_B(n - 2, n - 1) = h * oldsn;
    sigm_B(n - 1, n - 1) = h * oldcs;
}

template <typename _MatrixType>
void v0_GivRef_SVD<_MatrixType>::revert_negative_singular() {
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

#endif  // V0_GIVENS_REFINEMENT_HPP
