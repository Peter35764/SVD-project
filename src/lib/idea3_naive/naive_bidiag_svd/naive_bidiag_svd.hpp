#ifndef NAIVE_BIDIAG_SVD_HPP
#define NAIVE_BIDIAG_SVD_HPP

#include "naive_bidiag_svd.h"

namespace SVD_Project {

template <typename _MatrixType>
NaiveBidiagSVD<_MatrixType>::NaiveBidiagSVD() {
  // Конструктор по умолчанию – никаких специальных действий не требуется.
}

template <typename _MatrixType>
NaiveBidiagSVD<_MatrixType>& NaiveBidiagSVD<_MatrixType>::Compute(const MatrixType& B, unsigned int computationOptions) {
  Index m = B.rows();
  Index n = B.cols();
  Index r = std::min(m, n);

  // Устанавливаем флаги вычисления U и V.
  this->m_computeFullU = (computationOptions & Eigen::ComputeFullU) != 0;
  this->m_computeThinU = (computationOptions & Eigen::ComputeThinU) != 0;
  this->m_computeFullV = (computationOptions & Eigen::ComputeFullV) != 0;
  this->m_computeThinV = (computationOptions & Eigen::ComputeThinV) != 0;
  eigen_assert(!(this->m_computeFullU && this->m_computeThinU) && "Нельзя запрашивать и полное, и тонкое U одновременно.");
  eigen_assert(!(this->m_computeFullV && this->m_computeThinV) && "Нельзя запрашивать и полное, и тонкое V одновременно.");

  // Выделяем память для сингулярных значений.
  this->m_singularValues.resize(r);

  // Выделяем память для матриц U и V в зависимости от запроса.
  if (this->m_computeFullU)
    this->m_matrixU.resize(m, m);
  else if (this->m_computeThinU)
    this->m_matrixU.resize(m, r);
  else
    this->m_matrixU.resize(m, 0);

  if (this->m_computeFullV)
    this->m_matrixV.resize(n, n);
  else if (this->m_computeThinV)
    this->m_matrixV.resize(n, r);
  else
    this->m_matrixV.resize(n, 0);

  // Определяем, какую нормальную матрицу использовать: если m >= n, то M = B^T B, иначе M = B B^T.
  bool useBtb = (m >= n);
  Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic> M;
  if (useBtb) {
    M = B.transpose() * B; // M имеет размер n x n.
  } else {
    M = B * B.transpose(); // M имеет размер m x m.
  }

  // Решаем симметричную задачу собственных значений.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic> > eigenSolver(M);
  if (eigenSolver.info() != Eigen::Success) {
    throw std::runtime_error("Ошибка при вычислении собственных значений нормальной матрицы.");
  }

  // Получаем собственные значения и собственные векторы.
  Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> evals = eigenSolver.eigenvalues();
  Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic> evecs = eigenSolver.eigenvectors();
  Index eigDim = evals.size();

  // Сортируем собственные значения и соответствующие векторы по убыванию.
  std::vector<Index> indices(eigDim);
  for (Index i = 0; i < eigDim; ++i) {
    indices[i] = i;
  }
  std::sort(indices.begin(), indices.end(), [&evals](Index a, Index b) {
    return evals(a) > evals(b);
  });
  Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> evals_sorted(eigDim);
  Eigen::Matrix<RealScalar, Eigen::Dynamic, Eigen::Dynamic> evecs_sorted(evecs.rows(), eigDim);
  for (Index i = 0; i < eigDim; ++i) {
    evals_sorted(i) = evals(indices[i]);
    evecs_sorted.col(i) = evecs.col(indices[i]);
  }

  // Вычисляем сингулярные значения: sigma_j = sqrt(max(eigenvalue, 0))
  for (Index j = 0; j < r; ++j) {
    RealScalar lambda = (evals_sorted(j) < RealScalar(0)) ? RealScalar(0) : evals_sorted(j);
    this->m_singularValues(j) = std::sqrt(lambda);
  }

  // Вычисляем матрицы U и V в зависимости от выбранного варианта.
  if (useBtb) {
    // Если M = B^T B, то собственные векторы являются правыми сингулярными векторами.
    if (this->m_computeFullV)
      this->m_matrixV = evecs_sorted;
    else if (this->m_computeThinV)
      this->m_matrixV = evecs_sorted.leftCols(r);

    // Вычисляем левую часть: u_j = (1/σ_j) * B * v_j, для σ_j > 0.
    if (this->m_computeFullU || this->m_computeThinU) {
      for (Index j = 0; j < r; ++j) {
        RealScalar sigma = this->m_singularValues(j);
        if (sigma > RealScalar(0)) {
          Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> ucol = B * this->m_matrixV.col(j);
          ucol /= ucol.norm();
          this->m_matrixU.col(j) = ucol;
        } else {
          // Если σ_j == 0, заполняем соответствующий столбец нулями.
          this->m_matrixU.col(j).setZero();
        }
      }
      // Если требуется полный U (m > r), дополняем оставшиеся столбцы до ортонормированного базиса.
      if (this->m_computeFullU && r < m) {
        CompleteBasisForU(B);
      }
    }
  } else {
    // Если M = B B^T, то собственные векторы являются левыми сингулярными векторами.
    if (this->m_computeFullU)
      this->m_matrixU = evecs_sorted;
    else if (this->m_computeThinU)
      this->m_matrixU = evecs_sorted.leftCols(r);

    // Вычисляем правую часть: v_j = (1/σ_j) * B^T * u_j.
    if (this->m_computeFullV || this->m_computeThinV) {
      for (Index j = 0; j < r; ++j) {
        RealScalar sigma = this->m_singularValues(j);
        if (sigma > RealScalar(0)) {
          Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> vcol = B.transpose() * this->m_matrixU.col(j);
          vcol /= vcol.norm();
          this->m_matrixV.col(j) = vcol;
        } else {
          this->m_matrixV.col(j).setZero();
        }
      }
      if (this->m_computeFullV && r < n) {
        CompleteBasisForV(B);
      }
    }
  }

  this->m_isInitialized = true;
  return *this;
}

template <typename _MatrixType>
void NaiveBidiagSVD<_MatrixType>::CompleteBasisForU(const MatrixType& B) {
  // Дополнение матрицы U до полного ортонормированного базиса.
  // Пусть первые r столбцов уже вычислены. Для оставшихся столбцов воспользуемся
  // алгоритмом Householder QR.
  Index m = B.rows();
  Index r = this->m_singularValues.size();
  // Если U уже имеет m столбцов, то ничего не делаем.
  if (this->m_matrixU.cols() < m) return;

  // Используем HouseholderQR для вычисления полного Q.
  Eigen::HouseholderQR< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > qr(this->m_matrixU.leftCols(r));
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q = qr.householderQ();
  // Записываем оставшиеся столбцы из Q.
  this->m_matrixU.block(0, r, m, m - r) = Q.block(0, r, m, m - r);
}

template <typename _MatrixType>
void NaiveBidiagSVD<_MatrixType>::CompleteBasisForV(const MatrixType& B) {
  // Дополнение матрицы V до полного ортонормированного базиса.
  Index n = B.cols();
  Index r = this->m_singularValues.size();
  if (this->m_matrixV.cols() < n) return;

  Eigen::HouseholderQR< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > qr(this->m_matrixV.leftCols(r));
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q = qr.householderQ();
  this->m_matrixV.block(0, r, n, n - r) = Q.block(0, r, n, n - r);
}

}  // namespace SVD_Project

#endif  // NAIVE_BIDIAG_SVD_HPP
