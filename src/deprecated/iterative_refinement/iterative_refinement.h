#ifndef ITERATIVE_REFINEMENT_H
#define ITERATIVE_REFINEMENT_H

#include <cmath>
#include <iostream>
#include <vector>

#if __has_include(<Eigen/LU>)
#include <Eigen/LU>
#elif __has_include(<eigen3/Eigen/LU>)
#include <eigen3/Eigen/LU>
#endif

#if __has_include(<Eigen/SVD>)
#include <Eigen/SVD>
#elif __has_include(<eigen3/Eigen/SVD>)
#include <eigen3/Eigen/SVD>
#endif

/*Название статьи:  Uchino Yuki, Terao Takeshi, Ozaki Katsuhisa. 2022.08.05 –
Acceleration of Iterative Refinement for Singular Value Decomposition.
(https://www.researchgate.net/publication/362642883_Acceleration_of_Iterative_Refinement_for_Singular_Value_Decomposition)
* Обязательные условия: 1)в матрицах колво строк больше или равному количеству
столбцов *2)каждое сингулярное значение больше последующего (d1>d2>...>dn)
* 3)каждое приближенное сингулярное значение отлично друг от друга (di != dj для
i!=j)
*
* Выполнил Едренников Д.А. КМБО-01-20
*/

// Заметка по конструктору: AOIR_SVD(Eigen::Matrix<T, M, N> A)
// Принимает матрицу размера M N при этом M>N.
// Конструктор может быть использован самостоятельно для вычисления левых и
// правых сингулярных векторов и собственных чисел матрицы. Дабы, получить левые
// и правые сингулярные вектора и собственных числа используют следующие методы
// класса: matrixV(), matrixU(), singularValues(), соответственно. Данные
// функции возвращают матрицы размер N N, M M, M N.

template <typename T, int M, int N>
class AOIR_SVD;

template <typename T, int M, int N>
class AOIR_SVD {
 private:
  Eigen::Matrix<T, M, M> U;
  Eigen::Matrix<T, M, N> S;
  Eigen::Matrix<T, N, N> V;

  void Set_U(Eigen::Matrix<T, M, M> A) { U = A; }

  void Set_V(Eigen::Matrix<T, N, N> A) { V = A; }

  void Set_S(Eigen::Matrix<T, M, N> A) { S = A; }

 protected:
  AOIR_SVD MSVD_SVD(const Eigen::Matrix<T, M, N>& A,
                    const Eigen::Matrix<T, M, M>& Ui,
                    const Eigen::Matrix<T, N, N>& Vi) {
    const int m = A.rows();  // Получаем размеры изначальной матрицы
    const int n = A.cols();

    if (A.rows() < A.cols()) {
      std::cout << "Attention! Number of the rows must be greater or equal  "
                   "than  number of the columns";
      AOIR_SVD ANS;
      return ANS;
    }

    using matrix_dd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using matrix_nn = Eigen::Matrix<double, N, N>;  // V
    using matrix_mn = Eigen::Matrix<double, M, N>;  // A
    using matrix_mm = Eigen::Matrix<double, M, M>;  // U

    matrix_mn Ad = A.template cast<double>();  // Согласно нашему алгоритму все
                                               // почти все вычисления должны
    matrix_mm Ud =
        Ui.template cast<double>();  // производится с двойной точностью,
                                     // поэтому переделываем тип
    matrix_nn Vd = Vi.template cast<double>();  // всех матриц в double
    // Далее все тип всех элементов также будет double. Но в конце, все матрицу
    // будут приведены к изначальному типу.

    matrix_mn Udmn = Ud.block(0, 0, m, n);

    matrix_mn P = Ad * Vd;
    matrix_nn Q = Ad.transpose() * Udmn;

    matrix_nn ViT = Vd.transpose();
    matrix_mm UiT = Ud.transpose();
    std::vector<double> r(n, 0.0);
    std::vector<double> s(n, 0.0);
    std::vector<double> t(n, 0.0);
    matrix_nn Sigma_n(n, n);
    Sigma_n.setZero();

    for (int i = 0; i < n; i++) {
      r[i] = 1.0 - UiT.row(i) * Ud.col(i);
      s[i] = 1.0 - ViT.row(i) * Vd.col(i);
      t[i] = UiT.row(i) * P.col(i);
      Sigma_n(i, i) = t[i] / (1 - ((r[i] + s[i]) * 0.5));
    };

    matrix_mn Sigma(m, n);  // Матрица сингулярных значений
    Sigma.setZero();
    Sigma.block(0, 0, n, n) = Sigma_n.transpose();

    matrix_mn Cgamma = P - Udmn * Sigma_n;
    matrix_nn Cdelta = Q - Vd * Sigma_n;

    matrix_nn Calpha = Udmn.transpose() * Cgamma;
    matrix_nn Cbetta = ViT * Cdelta;

    matrix_nn D = Sigma_n * Calpha + Cbetta * Sigma_n;
    matrix_nn E = Calpha * Sigma_n + Sigma_n * Cbetta;

    matrix_nn G(n, n);  // Матрица, необходимая для более точного вычисления
                        // правых сингулярных векторов
    matrix_nn F11(n, n);

    double temp1;  // Временные переменные, необходимые для более быстрого
                   // вычисления элементов матриц
    double temp2;  // G и F11

    for (int i = 0; i < n; i++)  // Вычисление элементов матриц G и F11
    {
      temp1 = Sigma_n(i, i) * Sigma_n(i, i);

      for (int j = 0; j < n; j++) {
        temp2 = Sigma_n(j, j) * Sigma_n(j, j);

        G(i, j) =
            D(i, j) /
            (temp2 -
             temp1);  // Значение диагональных элементов не должно равняться
        F11(i, j) =
            E(i, j) / (temp2 - temp1);  // бесконечности. Оно будет пересчитано
      }
    }

    for (int i = 0; i < n;
         i++)  // Пересчет диагональных элементов для матриц G и F11
    {
      G(i, i) = s[i] * 0.5;
      F11(i, i) = r[i] * 0.5;
    }

    matrix_dd F12(n, m - n);
    F12 = Sigma_n.inverse() * P.transpose() * Ud.block(0, n, m, m - n);
    matrix_dd F21 =
        Ud.block(0, n - 1, m, m - n).transpose() * Cgamma * Sigma_n.inverse();
    matrix_dd F22 = 0.5 * (matrix_dd::Constant(m - n, m - n, 1.0) -
                           Ud.block(0, n - 1, m, m - n).transpose() *
                               Ud.block(0, n, m, m - n));

    matrix_mm F(m, m);  // Матрица, необходимая для более точного вычисления
                        // левых сингулярных векторов
    F.block(0, 0, n, n) = F11;
    F.block(0, n, n, m - n) = F12;
    F.block(n, 0, m - n, n) = F21;
    F.block(n, n, m - n, m - n) = F22;

    matrix_mm U =
        Ud +
        Ud * F;  // Вычисление уточнённых значений левых сингулярных векторов
    matrix_nn V =
        Vd +
        Vd * G;  // Вычисление уточнённых значений правых сингулярных векторов

    AOIR_SVD<T, M, N> ANS;
    ANS.Set_V(V.template cast<T>());  // Приведение матриц к изначальному типу
    ANS.Set_U(U.template cast<T>());
    ANS.Set_S(Sigma.template cast<T>());
    return ANS;
  }

 public:
  AOIR_SVD() {};

  AOIR_SVD(Eigen::Matrix<T, M, N> A) {
    Eigen::BDCSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
        A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Неточный расчёт левых и правых сингулярных векторов с помощью функции из
    // библиотеки Eigen
    AOIR_SVD<T, M, N> temp;
    temp = MSVD_SVD(A, svd.matrixU(),
                    svd.matrixV());  // Уточнение результата нашей функцией
    this->U = temp.matrixU();
    this->V = temp.matrixV();
    this->S = temp.singularValues();
  }

  Eigen::Matrix<T, N, N> matrixV() { return V; }

  Eigen::Matrix<T, M, M> matrixU() { return U; }

  Eigen::Matrix<T, M, N> singularValues() { return S; }
};

#endif  // ITERATIVE_REFINEMENT_H
