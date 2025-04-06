#ifndef MRRR_H
#define MRRR_H

#include <Eigen/SVD>

// Предварительное объявление класса-шаблона MRRR_SVD
template <typename _MatrixType>
class MRRR_SVD;

// Специализация traits для MRRR_SVD в пространстве Eigen::internal
namespace Eigen {
namespace internal {
template <typename _MatrixType>
struct traits<MRRR_SVD<_MatrixType>> : public traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};
}  // namespace internal
}  // namespace Eigen

// Объявление класса-шаблона MRRR_SVD с полями и методами
template <typename _MatrixType>
class MRRR_SVD : public Eigen::SVDBase<MRRR_SVD<_MatrixType>> {
 public:
  // Конструктор: принимает исходную матрицу и опции вычисления
  MRRR_SVD(const _MatrixType& matrix, unsigned int computationOptions);

  // Метод для вычисления разложения TGK
  MRRR_SVD<_MatrixType>& compute_tgk(const _MatrixType& matrix);

  // Деструктор по необходимости (если потребуется)
  ~MRRR_SVD() = default;

  // Геттеры для доступа к результатам
  const Eigen::MatrixXd& matrixU() const { return m_matrixU; }
  const Eigen::MatrixXd& matrixV() const { return m_matrixV; }
  const Eigen::VectorXd& singularValues() const { return m_singularValues; }

 private:
  Eigen::MatrixXd m_matrixU;         // Матрица U из SVD
  Eigen::MatrixXd m_matrixV;         // Матрица V из SVD
  Eigen::VectorXd m_singularValues;  // Вектор сингулярных значений
  bool m_isInitialized = false;      // Флаг инициализации
  bool m_computeFullU = false;       // Флаг вычисления полной матрицы U
  bool m_computeFullV = false;       // Флаг вычисления полной матрицы V
};

#include "mrrr.hpp"  // Подключаем реализацию

#endif  // MRRR_H
