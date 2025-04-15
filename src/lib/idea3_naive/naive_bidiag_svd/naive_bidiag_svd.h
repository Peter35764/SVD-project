#ifndef NAIVE_BIDIAG_SVD_H
#define NAIVE_BIDIAG_SVD_H

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cmath>

// Форвард-декларация класса NaiveBidiagSVD в пространстве имён SVD_Project.
namespace SVD_Project {
  template<typename _MatrixType>
  class NaiveBidiagSVD;
}

// Специализация трейтов для NaiveBidiagSVD, необходимая для корректной работы с Eigen::SVDBase.
namespace Eigen {
namespace internal {

template<typename _MatrixType>
struct traits<SVD_Project::NaiveBidiagSVD<_MatrixType>> : public traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

} // namespace internal
} // namespace Eigen

namespace SVD_Project {

/**
 * @brief Класс для вычисления SVD бидиагональной матрицы с использованием нормальных уравнений.
 *
 * На вход подаётся бидиагональная матрица B следующего вида:
 * \f[
 *    B = \operatorname{diag}(\alpha_1, \dots, \alpha_n) + \operatorname{diag}_{+1}(\beta_1, \dots, \beta_{n-1}),
 * \f]
 * для которой выполняется разложение:
 * \f[
 *    B = U \Sigma V^*.
 * \f]
 *
 * Метод работает следующим образом:
 * - Если число строк m \>= числа столбцов n, вычисляется нормальная матрица \f$M=B^TB\f$ (размер n×n),
 *   иначе – \f$M=BB^T\f$ (размер m×m).
 * - Затем решается симметричная задача собственных значений M с помощью Eigen::SelfAdjointEigenSolver.
 * - Полученные собственные значения \f$\lambda_j\f$ преобразуются в сингулярные значения \f$\sigma_j=\sqrt{\lambda_j}\f$
 *   (при этом собственные значения сортируются по убыванию).
 * - Если использовалось \f$B^TB\f$, то собственные векторы и есть правые сингулярные векторы V, а левый набор
 *   вычисляется по формуле \f$u_j=\frac{Bv_j}{\sigma_j}\f$.
 * - Если использовалось \f$BB^T\f$, то процесс происходит наоборот.
 * - Дополнительно, если запрошено полное SVD, то если размерность (m или n) больше ранга, оставшиеся столбцы U (или V)
 *   дополняются до унитарной матрицы с помощью QR-разложения.
 *
 * @tparam _MatrixType Тип матрицы (например, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>).
 */
template <typename _MatrixType>
class NaiveBidiagSVD : public Eigen::SVDBase< NaiveBidiagSVD<_MatrixType> > {
 public:
  using MatrixType  = _MatrixType;
  using Scalar      = typename MatrixType::Scalar;
  using RealScalar  = typename Eigen::NumTraits<Scalar>::Real;
  using Index       = Eigen::Index;
  using Base        = Eigen::SVDBase< NaiveBidiagSVD<_MatrixType> >;

  /// Конструктор по умолчанию.
  NaiveBidiagSVD();

  /**
   * @brief Конструктор, вычисляющий SVD сразу при создании объекта.
   *
   * @param B Входная бидиагональная матрица.
   * @param computationOptions Флаги вычисления (например, Eigen::ComputeFullU | Eigen::ComputeFullV).
   */
  NaiveBidiagSVD(const MatrixType& B, unsigned int computationOptions = Eigen::ComputeThinU | Eigen::ComputeThinV) {
    Compute(B, computationOptions);
  }

  /// Деструктор (по умолчанию).
  ~NaiveBidiagSVD() = default;

  /**
   * @brief Вычисляет SVD бидиагональной матрицы B.
   *
   * @param B Входная бидиагональная матрица.
   * @param computationOptions Флаги вычисления (например, Eigen::ComputeFullU | Eigen::ComputeFullV).
   * @return Ссылка на текущий объект.
   */
  NaiveBidiagSVD& Compute(const MatrixType& B, unsigned int computationOptions = Eigen::ComputeThinU | Eigen::ComputeThinV);

  // Методы доступа к результатам (унаследованы от Eigen::SVDBase)
  using Base::matrixU;
  using Base::matrixV;
  using Base::singularValues;

 protected:
  /**
   * @brief Дополняет матрицу U до полного ортонормированного базиса, если требуется полный SVD.
   *
   * @param B Входная матрица (используется только для определения размерности).
   */
  void CompleteBasisForU(const MatrixType& B);

  /**
   * @brief Дополняет матрицу V до полного ортонормированного базиса, если требуется полный SVD.
   *
   * @param B Входная матрица (используется только для определения размерности).
   */
  void CompleteBasisForV(const MatrixType& B);
};

}  // namespace SVD_Project

#include "naive_bidiag_svd.hpp"

#endif  // NAIVE_BIDIAG_SVD_H
