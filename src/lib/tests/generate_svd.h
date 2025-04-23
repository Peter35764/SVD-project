#ifndef GENERATE_SVD_H
#define GENERATE_SVD_H
//
//  SVD
//
//  Created by Victoria Koreshkova on 29.03.2024.
//
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

#include "ctime"

// rows - число строк
// cols - число столбцов
// p = min(rows, cols)
// generate_function(int n) - функция, генерирующая сингулярные значения. Должна
// генерировать n чисел. sigmas - массив сингулярных значений U - матрица U V -
// матрица V RNG - std::default_random_generator, генератор случайных чисел с
// заданным seed distribution -  std::uniform_real_distribution(a, b), заданное
// равномерное распределение includeBoundaries - bool, включать ли в сингулярные
// числа границы промежутка

// В файле generate_svd.cpp вызывается конструктор SVDGenerator из файла
// generate_svd.h для создания матриц U,V,SIGMA. Выводится произведение матриц U
// и транспонированной U. Функция generate из файла generate_svd.h  используется
// для заполнения матрицы SIGMA и получения матриц U,V. В нее передается
// количество  сингулярных значений. В файле generate_svd.h описан класс и все
// необходимые параметры и функции. В файле generate_svd.cpp создаются и
// заполняются матрицы U,V,SIGMA посредством обращения к необходимым методам и
// параметрам  в generate_svd.h Конструктор создает необходимые матрицы. В него
// предаются параметры: число строк, число столбцов, генератор случайных чисел с
// параметром, заданное распределение и флаг необходимости включения в
// сингулярные значения границ распределения.

template <typename T, int M = Eigen::Dynamic, int N = Eigen::Dynamic>
class SVDGenerator {
 private:
  using MatrixUType = Eigen::Matrix<T, M, M>;
  using MatrixVType = Eigen::Matrix<T, N, N>;
  using MatrixSType = Eigen::Matrix<T, M, N>;
  using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  bool generatedFLG = false;
  MatrixUType U;
  MatrixVType V;
  MatrixSType S;
  VectorDynamic sigmas;
  MatrixDynamic initialMatrix;  // Итоговая матрица.
  std::default_random_engine RNG;
  std::uniform_real_distribution<T> distribution;
  bool includeBoundaries;
  int rows;
  int cols;
  int p;

  void set_sing_vals(const VectorDynamic& sigmas1) {
    assert(p == sigmas1.size());
    sigmas = sigmas1;
    std::sort(sigmas.data(), sigmas.data() + sigmas.size(), std::greater<T>());
  }

  VectorDynamic gen_rand_nums(int n) {
    VectorDynamic tmp(n);
    for (int i = 0; i < n; ++i) tmp[i] = distribution(RNG);
    if (includeBoundaries) {
      tmp[0] = distribution.a();
      tmp[1] = distribution.b();
    }
    return tmp;
  }

 public:
  SVDGenerator(int rows1, int cols1, const std::default_random_engine RNG_src,
               const std::uniform_real_distribution<T> dist_src,
               bool includeBoundaries = false) {
    assert(rows1 > 0);
    assert(cols1 > 0);

    rows = rows1;
    cols = cols1;
    p = std::min(rows, cols);

    U = MatrixUType::Zero(rows, rows);
    V = MatrixVType::Zero(cols, cols);
    S = MatrixSType::Zero(rows, cols);
    initialMatrix = MatrixSType::Zero(rows, cols);

    RNG = RNG_src;
    distribution = dist_src;
    VectorDynamic sigmas1 = VectorDynamic(p);
    std::fill(sigmas1.begin(), sigmas1.end(), T(0));
    set_sing_vals(sigmas1);
  }

  MatrixUType getMatrixU() {
    if (!generatedFLG) generate(p);
    return U;
  }

  MatrixVType getMatrixV() {
    if (!generatedFLG) generate(p);
    return V;
  }

  MatrixSType getMatrixS() {
    if (!generatedFLG) generate(p);
    return S;
  }

  MatrixDynamic getInitialMatrix() {
    if (!generatedFLG) generate(p);
    return initialMatrix;
  }

  void generate(int nonzeros) {
    assert(nonzeros <= p);
    generatedFLG = true;
    std::fill(sigmas.begin(), sigmas.end(), T(0));
    VectorDynamic nonzero_sigmas = gen_rand_nums(nonzeros);
    std::copy(nonzero_sigmas.begin(), nonzero_sigmas.end(), sigmas.begin());
    std::sort(sigmas.begin(), sigmas.end(), std::greater<T>());
    // U,V-ортогональные матрицы, SIGMA - диагональная матрица

    /*Создается две случайные матрицы нужных размеров - T1 и T2,элементы -
    случайные числа от 0.1 до 10. QR разложение раскладывает матрицу на
    произведение двух: ортогональной Q и верхнедиагональной R С помощью этого
    разложения случайная матрица превращается в ортогональную и далее эта
    ортогональная матрица Q используется как U или V */

    MatrixDynamic T_1(rows, rows), T_2(cols, cols), Q_1(rows, rows),
        Q_2(cols, cols), R_1(rows, rows), R_2(cols, cols);
    // Сингулярные значения нумеруются в порядке убывания
    // Тут на всякий случай сортируем массив сингулярных чисел, чтобы элменты
    // шли в порядке убывания l1 >= l2 >=.....>= lk >= 0

    S.setZero();

    for (int i = 0; i < p; i++) S(i, i) = sigmas[i];

    // Заполнение матриц T_1 и T_2 случайными элементами от 0.1 до 10
    T HI = static_cast<T>(10);
    T LO = static_cast<T>(0.1);
    T range = HI - LO;

    T_1 = MatrixUType::Random(rows, rows);
    T_1 = (T_1 + MatrixUType::Constant(rows, rows, static_cast<T>(1))) * range /
          2;
    T_1 = (T_1 + MatrixUType::Constant(rows, rows, LO));

    T_2 = MatrixVType::Random(cols, cols);
    T_2 = (T_2 + MatrixVType::Constant(cols, cols, static_cast<T>(1))) * range /
          2;
    T_2 = (T_2 + MatrixVType::Constant(cols, cols, LO));

    // QR_разложение матриц T1 и T2
    Q_1 = (Eigen::FullPivHouseholderQR<Eigen::Matrix<T, N, M>>(T_1)).matrixQ();
    Q_2 = (Eigen::FullPivHouseholderQR<Eigen::Matrix<T, N, M>>(T_2)).matrixQ();
    U = Q_1;
    V = Q_2.transpose();

    initialMatrix = U * S * V;
  }
};

#endif  // GENERATE_SVD_H
