#ifndef GIVENS_REFINEMENT_HPP
#define GIVENS_REFINEMENT_HPP

#include "givens_refinement.h"

namespace SVD_Project {

template <typename _MatrixType>
GivRef_SVD<_MatrixType>::GivRef_SVD() {}

template <typename _MatrixType>
GivRef_SVD<_MatrixType>& GivRef_SVD<_MatrixType>::compute(
    const MatrixType& B, unsigned int computationOptions) {
  Index m = B.rows();
  Index n = B.cols();
  Index r = std::min(m, n);

  // TODO

  this->m_isInitialized = true;
  return *this;
}

}  // namespace SVD_Project

#endif  // GIVENS_REFINEMENT_HPP
