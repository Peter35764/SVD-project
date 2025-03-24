#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

/* Test of all matrix norms presented in Eigen worth looking into:
 *
 * - norm() - for vectors, the l2 norm of *this, and for matrices the Frobenius
 *   norm. In both cases, it consists in the square root of the sum of the
 *   square of all the matrix entries. For vectors, this is also equals to the
 *   square root of the dot product of *this with itself.
 *
 * - operatorNorm() - Computes the l2 operator norm.
 *
 * - squaredNorm() - for vectors, the squared l2 norm of *this, and for matrices
 *   the squared Frobenius norm. In both cases, it consists in the sum of the
 *   square of all the matrix entries. For vectors, this is also equals to the
 *   dot product of *this with itself.
 *
 * - lpNorm() - the coefficient-wise l^p norm of *this, that is, returns the
 *   p-th root of the sum of the p-th powers of the absolute values of the
 *   coefficients of *this. If p is the special value Eigen::Infinity, this
 *   function returns the l^infty norm, that is the maximum of the absolute values
 *   of the coefficients of *this.
 *
 * - stableNorm() - the l2 norm of *this avoiding underflow and overflow.
 *   This version use a blockwise two passes algorithm: 1 - find the absolute
 *   largest coefficient s 2 - compute s*abs(*this/s) in a standard way
 *
 * - blueNorm() - the l2 norm of *this using the Blue's algorithm.
 *   A Portable Fortran Program to Find the Euclidean Norm of a Vector,
 *   ACM TOMS, Vol 4, Issue 1, 1978.
 *
 * - hypotNorm() - the l2 norm of *this avoiding undeflow and overflow.
 *   This version use a concatenation of hypot() calls, and it is very slow.
 */

Eigen::MatrixXd randomOrthogonalMatrix(size_t n)
{
    Eigen::MatrixXd M(n, n);
    M.setRandom(); // Fill with random values between -1 and 1
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    return qr.householderQ(); // Returns Q, an orthogonal matrix in QR decomposition
}

int main()
{
    size_t n = 100;
    double small_singular_value = 1e-5;

    Eigen::MatrixXd U = randomOrthogonalMatrix(n);
    Eigen::MatrixXd V = randomOrthogonalMatrix(n);

    Eigen::MatrixXd S_well_cond = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd A_well_cond = U * S_well_cond * V.transpose();

    Eigen::MatrixXd S_ill_cond = S_well_cond;
    // Set each 4th singular value of matrix being composed to a very small value
    // in order to make it ill conditioned
    for (size_t i = 0; i < n; i += 4) {
        S_ill_cond(i, i) = small_singular_value;
    }
    Eigen::MatrixXd A_ill = U * S_ill_cond * V.transpose();

    std::cout << "Well cond. norm():\t" << S_well_cond.norm() << std::endl;
    std::cout << "Ill cond. norm():\t" << S_ill_cond.norm() << std::endl;

    std::cout << "Well cond. opetarorNorm():\t" << S_well_cond.operatorNorm() << std::endl;
    std::cout << "Ill cond. opetarorNorm():\t" << S_ill_cond.operatorNorm() << std::endl;

    std::cout << "Well cond. squaredNorm():\t" << S_well_cond.squaredNorm() << std::endl;
    std::cout << "Ill cond. squaredNorm():\t" << S_ill_cond.squaredNorm() << std::endl;

    std::cout << "Well cond. lpNorm<3>():\t" << S_well_cond.lpNorm<3>() << std::endl;
    std::cout << "Ill cond. lpNorm<3>():\t" << S_ill_cond.lpNorm<3>() << std::endl;

    std::cout << "Well cond. lpNorm<5>():\t" << S_well_cond.lpNorm<5>() << std::endl;
    std::cout << "Ill cond. lpNorm<5>():\t" << S_ill_cond.lpNorm<5>() << std::endl;

    std::cout << "Well cond. stableNorm():\t" << S_well_cond.stableNorm() << std::endl;
    std::cout << "Ill cond. stableNorm():\t" << S_ill_cond.stableNorm() << std::endl;

    std::cout << "Well cond. blueNorm():\t" << S_well_cond.blueNorm() << std::endl;
    std::cout << "Ill cond. blueNorm():\t" << S_ill_cond.blueNorm() << std::endl;

    std::cout << "Well cond. hypotNorm():\t" << S_well_cond.hypotNorm() << std::endl;
    std::cout << "Ill cond. hypotNorm():\t" << S_ill_cond.hypotNorm() << std::endl;

    return 0;
}