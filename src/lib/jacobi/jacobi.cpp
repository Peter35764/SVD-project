#pragma once

#include "jacobi.h"

template <typename MatrixType>
void JTS_SVD<MatrixType>::allocate(Eigen::Index rows, Eigen::Index cols,
                                   unsigned int computationOptions) {
  eigen_assert(rows >= 0 && cols >= 0);
  // eigen_assert((rows >= cols) && "JTS_SVD: rows < cols, now unsupported");

  if (m_isAllocated && rows == m_rows && cols == m_cols &&
      computationOptions == m_computationOptions) {
    return;
  }

  m_rows = rows;
  m_cols = cols;
  m_info = Eigen::Success;
  m_isInitialized = false;
  m_isAllocated = true;
  m_computationOptions = computationOptions;
  m_computeFullU = (computationOptions & Eigen::ComputeFullU) != 0;
  m_computeThinU = (computationOptions & Eigen::ComputeThinU) != 0;
  m_computeFullV = (computationOptions & Eigen::ComputeFullV) != 0;
  m_computeThinV = (computationOptions & Eigen::ComputeThinV) != 0;
  eigen_assert(!(m_computeFullU && m_computeThinU) &&
               "JTS_SVD: you can't ask for both full and thin U");
  eigen_assert(!(m_computeFullV && m_computeThinV) &&
               "JTS_SVD: you can't ask for both full and thin V");
  eigen_assert(EIGEN_IMPLIES(m_computeThinU || m_computeThinV,
                             MatrixType::ColsAtCompileTime == Eigen::Dynamic) &&
               "JTS_SVD: thin U and V are only available when your matrix has "
               "a dynamic number of columns.");

  m_diagSize = (std::min)(m_rows, m_cols);
  m_singularValues.resize(m_diagSize);
  if (RowsAtCompileTime == Eigen::Dynamic)
    m_matrixU.resize(m_rows, m_computeFullU   ? m_rows
                             : m_computeThinU ? m_diagSize
                                              : 0);
  if (ColsAtCompileTime == Eigen::Dynamic)
    m_matrixV.resize(m_cols, m_computeFullV   ? m_cols
                             : m_computeThinV ? m_diagSize
                                              : 0);
  m_workMatrix.resize(m_rows, m_cols);
}

template <typename MatrixType>
JTS_SVD<MatrixType>& JTS_SVD<MatrixType>::compute(
    const MatrixType& matrix, unsigned int computationOptions,
    const params_t& params) {
  allocate(matrix.rows(), matrix.cols(), computationOptions);

  /*** шаг 1. Просто инициализация, без QR разложения, как в эйгене ***/
  m_workMatrix = matrix.block(0, 0, m_rows, m_cols);
  if (m_computeFullU) m_matrixU.setIdentity(m_rows, m_rows);
  if (m_computeThinU) m_matrixU.setIdentity(m_rows, m_diagSize);
  if (m_computeFullV) m_matrixV.setIdentity(m_cols, m_cols);
  if (m_computeThinV) m_matrixV.setIdentity(m_cols, m_diagSize);

  /** два случая
   * базовый, который описан в статье, где строк не меньше чем колонок
   * и тот где меньше
   * второй случай можно свести к первому, просто транспонировав матрицу
   * но чтобы не трогать типы, придется обработать отдельно
   */
  if (matrix.rows() >= matrix.cols()) {
    return compute_basic(matrix, params);
  } else {
    return compute_transposed(matrix, params);
  }
}

template <typename MatrixType>
JTS_SVD<MatrixType>& JTS_SVD<MatrixType>::compute_basic(
    const MatrixType& matrix, const params_t& params) {
  const int SWEEPS_MAX =
      (int64_t)m_diagSize * (m_diagSize - 1) / 2 * params.sweeps_factor + 1;
  const RealScalar threshold =
      RealScalar(params.epsilon) * matrix.squaredNorm();

  /*** шаг 2. Основа ***/
  for (int sweeps_cur = 0; sweeps_cur < SWEEPS_MAX; ++sweeps_cur) {
    // пока пусть будет так, хотя было бы неплохо обойтись только эйгеном
    using pivot = typename std::tuple<RealScalar, int, int>;
    std::vector<pivot> pivots;
    pivots.reserve(m_diagSize * (m_diagSize - 1) / 2);

    for (int i = 0; i < m_diagSize - 1; ++i) {
      for (int j = i + 1; j < m_diagSize; ++j) {
        // простое умножение падает
        // https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
        Scalar b = std::abs(m_workMatrix.col(i).dot(m_workMatrix.col(j)));
        pivots.emplace_back(b, i, j);
      }
    }

    int count_left = pivots.size() * params.tau;
    count_left += count_left < pivots.size();  // чтобы хотя бы один нашелся
    // https://en.cppreference.com/w/cpp/algorithm/nth_element
    std::nth_element(pivots.begin(), pivots.begin() + count_left, pivots.end(),
                     std::greater<pivot>());
    pivots.resize(count_left);
    std::sort(pivots.begin(), pivots.end(), std::greater<pivot>());
    if (std::get<0>(pivots[0]) < threshold) {
      break;
    }

    /**  в статье предлагается сначала поместить тройки в очередь
     *  и только после того как они были забиты, производить подсчеты
     *  поскольку в очереди FIFO то можно по этому принципу сделать и без
     * очереди
     */
    for (const auto& [_, i, j] : pivots) {
      Scalar gamma = (m_workMatrix.col(j).squaredNorm() -
                      m_workMatrix.col(i).squaredNorm()) /
                     (2 * m_workMatrix.col(i).dot(m_workMatrix.col(j)));
      Scalar t = 1 / (std::abs(gamma) + std::sqrt(gamma * gamma + 1));
      t *=
          gamma != 0 ? gamma / std::abs(gamma) : 0;  // для работы с комлексными
      Scalar c = 1 / std::sqrt(1 + t * t);
      Scalar s = t * c;

      // нет уверенности, что с комплекснозначными заработает
      Eigen::JacobiRotation<Scalar> J(c, s);

      m_workMatrix.applyOnTheRight(i, j, J);
      if (computeV()) m_matrixV.applyOnTheRight(i, j, J);
    }
  }

  /*** шаг 3. Заполняем сингулярные значения ***/
  for (Eigen::Index i = 0; i < m_diagSize; ++i) {
    RealScalar a = m_workMatrix.col(i).norm();
    m_singularValues.coeffRef(i) = a;
    if (computeU()) m_matrixU.col(i) = m_workMatrix.col(i) / a;
  }

  /*** шаг 4. Сортировка, оставлено без изменений из эйгена ***/
  m_nonzeroSingularValues = m_diagSize;
  for (Eigen::Index i = 0; i < m_diagSize; ++i) {
    Eigen::Index pos;
    RealScalar maxRemainingSingularValue =
        m_singularValues.tail(m_diagSize - i).maxCoeff(&pos);
    if (maxRemainingSingularValue == RealScalar(0)) {
      m_nonzeroSingularValues = i;
      break;
    }
    if (pos) {
      pos += i;
      std::swap(m_singularValues.coeffRef(i), m_singularValues.coeffRef(pos));
      if (computeU()) m_matrixU.col(pos).swap(m_matrixU.col(i));
      if (computeV()) m_matrixV.col(pos).swap(m_matrixV.col(i));
    }
  }

  m_isInitialized = true;
  return *this;
}

/** A^T = (BJ(VJ)^T)^T
 * A = VJ (BJ)^T
 * A = VJ (BJ)^T
 * A = VJ (USJ)^T
 * A = VJ J^T S^T U^T
 * A = VJ J^T (US)^T
 *
 * Код который будет здесь по большей части является копией compute_basic
 */
template <typename MatrixType>
JTS_SVD<MatrixType>& JTS_SVD<MatrixType>::compute_transposed(
    const MatrixType& matrix, const params_t& params) {
  const int SWEEPS_MAX =
      (int64_t)m_diagSize * (m_diagSize - 1) / 2 * params.sweeps_factor + 1;
  const RealScalar threshold =
      RealScalar(params.epsilon) * matrix.squaredNorm();

  /*** шаг 2. Основа ***/
  for (int sweeps_cur = 0; sweeps_cur < SWEEPS_MAX; ++sweeps_cur) {
    // пока пусть будет так, хотя было бы неплохо обойтись только эйгеном
    using pivot = typename std::tuple<RealScalar, int, int>;
    std::vector<pivot> pivots;
    pivots.reserve(m_diagSize * (m_diagSize - 1) / 2);

    for (int i = 0; i < m_diagSize - 1; ++i) {
      for (int j = i + 1; j < m_diagSize; ++j) {
        // простое умножение падает
        // https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
        Scalar b = std::abs(m_workMatrix.row(i).dot(m_workMatrix.row(j)));
        pivots.emplace_back(b, i, j);
      }
    }

    int count_left = pivots.size() * params.tau;
    count_left += count_left < pivots.size();  // чтобы хотя бы один нашелся
    // https://en.cppreference.com/w/cpp/algorithm/nth_element
    std::nth_element(pivots.begin(), pivots.begin() + count_left, pivots.end(),
                     std::greater<pivot>());
    pivots.resize(count_left);
    std::sort(pivots.begin(), pivots.end(), std::greater<pivot>());

    if (std::get<0>(pivots[0]) < threshold) {
      break;
    }

    /**  в статье предлагается сначала поместить тройки в очередь
     *  и только после того как они были забиты, производить подсчеты
     *  поскольку в очереди FIFO то можно по этому принципу сделать и без
     * очереди
     */
    for (const auto& [_, i, j] : pivots) {
      Scalar gamma = (m_workMatrix.row(j).squaredNorm() -
                      m_workMatrix.row(i).squaredNorm()) /
                     (2 * m_workMatrix.row(i).dot(m_workMatrix.row(j)));
      Scalar t = 1 / (std::abs(gamma) + std::sqrt(gamma * gamma + 1));
      t *=
          gamma != 0 ? gamma / std::abs(gamma) : 0;  // для работы с комлексными
      Scalar c = 1 / std::sqrt(1 + t * t);
      Scalar s = t * c;

      // нет уверенности, что с комплекснозначными заработает
      Eigen::JacobiRotation<Scalar> J(c, s);
      m_workMatrix.applyOnTheLeft(i, j, J.transpose());
      if (computeU()) m_matrixU.applyOnTheRight(i, j, J);
    }
  }

  /*** шаг 3. Заполняем сингулярные значения ***/
  for (Eigen::Index i = 0; i < m_diagSize; ++i) {
    RealScalar a = m_workMatrix.row(i).norm();
    m_singularValues.coeffRef(i) = a;
    if (computeV()) m_matrixV.col(i) = m_workMatrix.row(i) / a;
  }

  /*** шаг 4. Сортировка, оставлено без изменений из эйгена ***/
  m_nonzeroSingularValues = m_diagSize;
  for (Eigen::Index i = 0; i < m_diagSize; ++i) {
    Eigen::Index pos;
    RealScalar maxRemainingSingularValue =
        m_singularValues.tail(m_diagSize - i).maxCoeff(&pos);
    if (maxRemainingSingularValue == RealScalar(0)) {
      m_nonzeroSingularValues = i;
      break;
    }
    if (pos) {
      pos += i;
      std::swap(m_singularValues.coeffRef(i), m_singularValues.coeffRef(pos));
      if (computeU()) m_matrixU.col(pos).swap(m_matrixU.col(i));
      if (computeV()) m_matrixV.col(pos).swap(m_matrixV.col(i));
    }
  }

  m_isInitialized = true;
  return *this;
}
