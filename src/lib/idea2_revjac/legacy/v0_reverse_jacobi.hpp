#ifndef V0_REVERSE_JACOBI_HPP
#define V0_REVERSE_JACOBI_HPP

#include <Eigen/Core>
#include <Eigen/Jacobi>
#include <Eigen/SVD>
#include <boost/math/tools/minima.hpp>
#include <iostream>

// necessary for correct display in ide (or clangd lsp), does not
// affect the assembly process and can be removed
#include "v0_reverse_jacobi.h"

namespace SVD_Project {

template <typename _MatrixType>
v0_RevJac_SVD<_MatrixType>::v0_RevJac_SVD(
    const _MatrixType& initial, const _SingularVectorType& singularValues,
    unsigned int computationOptions)
    : m_initialMatrix(initial), m_singularValues(singularValues) {
  assert(m_initialMatrix.cols() == m_initialMatrix.rows());
  this->m_matrixV =
      _MatrixType::Identity(m_initialMatrix.rows(), m_initialMatrix.cols());
  this->m_matrixU =
      _MatrixType::Identity(m_initialMatrix.rows(), m_initialMatrix.cols());
}

template <typename _MatrixType>
v0_RevJac_SVD<_MatrixType>& v0_RevJac_SVD<_MatrixType>::compute() {
  _MatrixType temp = this->m_singularValues.asDiagonal();
  // std::cout << "m_initialMatrix:" << std::endl << initial << std::endl;
  // std::cout << "singular:" << std::endl << temp << std::endl;
  for (int i = 0; i < MAX_ITERATIONS; i++) {
    // while ((temp - m_initialMatrix).norm() > tolerance) {
    for (int p = 0; p < m_initialMatrix.rows(); p++) {
      for (int q = p + 1; q < m_initialMatrix.cols(); q++) {
        // Подбираем c и s (c^2 + s^2 = 1), что:
        // [ c s ]^T   [ temp_{pp} temp_{pq} ]   [ c s ]   [ *
        // m_initialMatrix_{pq} ] [ ]   * [                     ] * [     ] = [
        // ]
        // [-s c ]     [ temp_{qp} temp_{qq} ]   [-s c ]   [
        // m_initialMatrix_{pq} * ] temp_{pq}*(c^2 - s^2) + (temp_{pp} -
        // temp_{qq})cs = m_initialMatrix_{pq} std::cout << temp(p, q) << " " <<
        // temp(p, p) << " " << temp(q, q) << " " << m_initialMatrix(p, q) <<
        // std::endl;
        struct MinFronebius {
          int p, q;
          _MatrixType const& initial;
          _MatrixType const& temp;

          MinFronebius(int p, int q, _MatrixType const& initial,
                       _MatrixType const& temp)
              : p(p), q(q), initial(initial), temp(temp) {}

          Scalar operator()(Scalar const& c) {
            Scalar s = sqrt(1 - c * c);
            auto rotation = Eigen::JacobiRotation(c, s);
            _MatrixType temp2 = temp;

            temp2.applyOnTheLeft(p, q, rotation.transpose());
            temp2.applyOnTheRight(p, q, rotation);
            return (temp2 - initial).norm();
          }
        };
        auto result =
            boost::math::tools::brent_find_minima<MinFronebius, Scalar>(
                MinFronebius(p, q, m_initialMatrix, temp), 0, 1,
                std::numeric_limits<Scalar>::digits);
        Scalar c = result.first;
        Scalar s = sqrt(1 - c * c);
        auto rotation = Eigen::JacobiRotation(c, s);
        temp.applyOnTheLeft(p, q, rotation.transpose());
        temp.applyOnTheRight(p, q, rotation);
        this->m_matrixV.applyOnTheLeft(p, q, rotation);
        std::cout << "c = " << c << "; s = " << s << std::endl;
        // std::cout << "DIFF: " << result.first << " " << result.second << " "
        // << (temp - m_initialMatrix).norm() << std::endl;
      }
    }
  }
  // this->m_matrixU = this->m_matrixV.transpose();
  this->m_computeFullU = true;
  this->m_computeFullV = true;
  std::cout << (temp - m_initialMatrix).norm() << "\n";
  return *this;
}

}  // namespace SVD_Project

#endif  // V0_REVERSE_JACOBI_HPP
