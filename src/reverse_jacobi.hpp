#ifndef REVERSE_JACOBI_HPP
#define REVERSE_JACOBI_HPP

#include "reverse_jacobi.h" // necessary for correct display in ide, does not affect the assembly process and can be removed
#include <boost/math/tools/minima.hpp>

namespace SVD_Project {

const size_t ITERATIONS = 1000;

template<typename Sub_SVD>
requires std::derived_from<Sub_SVD, Eigen::SVDBase<Sub_SVD>> RevJac_SVD<Sub_SVD>::RevJac_SVD(
    const MatrixType &initial, unsigned int computationOptions)
    : Sub_SVD(
          initial,
          computationOptions
              & ~(Eigen::ComputeThinU | Eigen::ComputeThinV | Eigen::ComputeFullU
                  | Eigen::ComputeFullV))
{
    assert(initial.cols() == initial.rows());
    this->m_matrixV = MatrixType::Identity(initial.rows(), initial.cols());
    MatrixType temp = this->m_singularValues.asDiagonal();

    //std::cout << "initial:" << std::endl << initial << std::endl;
    //std::cout << "singular:" << std::endl << temp << std::endl;

    // TODO : Consider either lowering the number of iterations, or adding an
    // early exit condition to speed up the algorithm
    for (int i = 0; i < ITERATIONS; i++) {
        //while ((temp - initial).norm() > tolerance) {
        for (int p = 0; p < initial.rows(); p++) {
            for (int q = p + 1; q < initial.cols(); q++) {
                // Подбираем c и s (c^2 + s^2 = 1), что:
                // [ c s ]^T   [ temp_{pp} temp_{pq} ]   [ c s ]   [ *            initial_{pq} ]
                // [     ]   * [                     ] * [     ] = [                           ]
                // [-s c ]     [ temp_{qp} temp_{qq} ]   [-s c ]   [ initial_{pq} *            ]
                // temp_{pq}*(c^2 - s^2) + (temp_{pp} - temp_{qq})cs = initial_{pq}
                //std::cout << temp(p, q) << " " << temp(p, p) << " " << temp(q, q) << " " << initial(p, q) <<  std::endl;
                struct MinFronebius
                {
                    int p, q;
                    MatrixType const &initial;
                    MatrixType const &temp;

                    MinFronebius(int p, int q, MatrixType const &initial, MatrixType const &temp)
                        : p(p)
                        , q(q)
                        , initial(initial)
                        , temp(temp)
                    {}

                    Scalar operator()(Scalar const &c)
                    {
                        Scalar s = sqrt(1 - c * c);
                        auto rotation = Eigen::JacobiRotation(c, s);
                        MatrixType temp2 = temp;

                        temp2.applyOnTheLeft(p, q, rotation.transpose());
                        temp2.applyOnTheRight(p, q, rotation);
                        return (temp2 - initial).norm();
                    }
                };
                auto result = boost::math::tools::brent_find_minima<MinFronebius, Scalar>(
                    MinFronebius(p, q, initial, temp), 0, 1, std::numeric_limits<Scalar>::digits);
                Scalar c = result.first;
                Scalar s = sqrt(1 - c * c);
                auto rotation = Eigen::JacobiRotation(c, s);
                temp.applyOnTheLeft(p, q, rotation.transpose());
                temp.applyOnTheRight(p, q, rotation);
                this->m_matrixV.applyOnTheLeft(p, q, rotation);

                // std::cout << "c = " << c << "; s = " << s << std::endl;
                //std::cout << "DIFF: " << result.first << " " << result.second << " " << (temp - initial).norm() << std::endl;
            }
        }
    }
    this->m_matrixU = MatrixType::Identity(initial.rows(), initial.cols());

    //this->m_matrixU = this->m_matrixV.transpose();

    this->m_computeFullU = true;
    this->m_computeFullV = true;

    // std::cout << (temp - initial).norm() << "\n";
}

} // namespace SVD_Project

#endif // REVERSE_JACOBI_HPP
