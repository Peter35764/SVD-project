#ifndef TGK_INVERSE_H
#define TGK_INVERSE_H

#include <Eigen/SVD>
#include <vector>

namespace SVD_Project {

/**
 * @brief  TGKInv_SVD — SVD для бидиагональной матрицы через
 *         форму Голуба-Кахана + обратные итерации.
 *
 * @tparam _MatrixType  ожидается Eigen-матрица **n×n**, у которой
 *         заполнены только главная и наддиагональ.
 *
 * Конструктор принимает саму бидиагональную матрицу *B* и
 * вектор «почти точных» сингулярных значений *sigma* (их даёт,
 * например, DQDS).  Метод `compute()` выполняет:
 *  1. Построение трёхдиагональной TGK-формы;
 *  2. Обратные итерации для ±σᵢ;
 *  3. Ортонормализацию столбцов;
 *  4. Раскладку нечётных/чётных компонент ⇒ U,V.
 *
 * После чего стандартные геттеры `matrixU()/matrixV()` и
 * `singularValues()` (унаследованы от `Eigen::SVDBase`) возвращают
 * результат.
 */
template <typename _MatrixType>
class TGKInv_SVD : public Eigen::SVDBase<TGKInv_SVD<_MatrixType>> {
  using Base          = Eigen::SVDBase<TGKInv_SVD>;
  using Scalar        = typename _MatrixType::Scalar;
  using Index         = typename _MatrixType::Index;
  using MatrixDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

 public:
  TGKInv_SVD() = default;

  TGKInv_SVD(const _MatrixType& bidiag,
             const VectorDynamic& sigma,
             unsigned /* computationOptions */ = 0);

  TGKInv_SVD& compute();                     ///< Запуск алгоритма
  TGKInv_SVD& compute(std::ostream* dbg);    ///< То же + поток отладочных логов

  /* Доступ к результатам (SVDBase уже даёт matrixU(), matrixV(), singularValues()) */

 private:                 // ----- внутренние вспомогательные вещи -----
  // служебные подфункции вынесены внутрь .hpp
  MatrixDynamic  m_matrixU, m_matrixV;
  VectorDynamic  m_sigma;
  _MatrixType    m_B;          ///< входная бидиагональная матрица
  std::ostream*  m_dbg = nullptr;
};

} // namespace SVD_Project


/*  Специализация Eigen-traits — так принято во всех остальных файлах  */
template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::TGKInv_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  using MatrixType = _MatrixType;
};

#include "tgk_inverse.hpp"   // реализация шаблона

#endif // TGK_INVERSE_H