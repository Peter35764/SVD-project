#pragma once

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

  void Set_U(Eigen::Matrix<T, M, M> A);

  void Set_V(Eigen::Matrix<T, N, N> A);

  void Set_S(Eigen::Matrix<T, M, N> A);

 protected:
  AOIR_SVD MSVD_SVD(const Eigen::Matrix<T, M, N>& A,
                    const Eigen::Matrix<T, M, M>& Ui,
                    const Eigen::Matrix<T, N, N>& Vi);

 public:
  AOIR_SVD();

  AOIR_SVD(Eigen::Matrix<T, M, N> A);

  Eigen::Matrix<T, N, N> matrixV();

  Eigen::Matrix<T, M, M> matrixU();

  Eigen::Matrix<T, M, N> singularValues();
};

#include "iterative_refinement.cpp"
