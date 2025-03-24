#ifndef GIVENS_REFINEMENT_HPP
#define GIVENS_REFINEMENT_HPP

#include <iostream>
#include <vector>

namespace SVD_Project {

template<typename _MatrixType, typename _FPNum, int N>
GivRef_SVD<_MatrixType, _FPNum, N>::GivRef_SVD(const _MatrixType &matrix,
                                               unsigned int computationOptions)
{
    compute(matrix, computationOptions);
}

template<typename _MatrixType, typename _FPNum, int N>
GivRef_SVD<_MatrixType, _FPNum, N> &GivRef_SVD<_MatrixType, _FPNum, N>::compute(
    const _MatrixType &matrix, unsigned int computationOptions)
{
    // Шаг 1 - Бидиагональная матрица

    auto bid = Eigen::internal::UpperBidiagonalization(matrix);
    auto B = bid.bidiagonal().toDenseMatrix();

    std::cout << B;

    // Шаг 2 - Повороты Гивенса

    return *this;
}

// _FPNum -- floating-point number
template<typename _MatrixType, typename _FPNum, int N>
std::vector<_FPNum> GivRef_SVD<_MatrixType, _FPNum, N>::ROT(_FPNum f, _FPNum g)
{
    // TODO return std::vector<_FPNum>{cos, sin, r};
    return std::vector<_FPNum>{0, 1, 2};
}

} // namespace SVD_Project

#endif // GIVENS_REFINEMENT_HPP
