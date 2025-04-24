#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/dense>
#include <random>

#include "generate_svd.h"

template <typename T, int N = Eigen::Dynamic>
class QR_zero_shift {
 private:
  using SquareMatrix = Eigen::Matrix<T, N, N>;

  SquareMatrix left_J;       // Произведение всех поворпотов Гивенса слева от B
  SquareMatrix right_J;      // Произведение всех поворпотов Гивенса справа от B
  SquareMatrix sigm_B;       // Матрица с сингулярными значениями B
  SquareMatrix true_sigm_B;  // Матрица с точными сингулярными значениями B
  SquareMatrix B;            // Изначальная Бидиагональная матрица

  Eigen::VectorXd Cosines;  // Вектор с изначальными значениями косинусов
  Eigen::VectorXd Sines;    // Вектор с изначальными значениями косинусов
  Eigen::VectorXd Tans;     // Вектор с изначальными значениями тангенсов

  Eigen::VectorXd NewCosines;  // Вектор со значениями косинусов после сдвигов
  Eigen::VectorXd NewSines;    // Вектор со значениями синусов после сдвигов

  int trigonom_i;  // Значение для перебора векторов с тригонометрическими
                   // функциями

  int iter_num;  // Количество итераций алгоритма QR with zero shift
  int n;         // Размер матриц

  //(Brain) ROT (Субфункция, просчитывающая поворот Гивенса)
  // ( cs  sn ) (f) = (r)
  // ( -sn cs ) (g)   (0)
  std::vector<T> ROT(T f, T g) {
    T cs;
    T sn;
    T r;
    if (f == 0) {
      cs = 0;
      sn = 1;
      r = g;

    } else if (abs(f) > abs(g)) {
      T t = g / f;
      T tt = sqrt(1 + pow(t, 2));
      cs = 1 / tt;
      sn = t * cs;
      r = f * tt;
      Tans(trigonom_i) = t;
    } else {
      T t = f / g;
      T tt = sqrt(1 + pow(t, 2));
      sn = 1 / tt;
      cs = t * sn;
      r = g * tt;
      Tans(trigonom_i) = t;
    }

    std::vector<T> result{cs, sn, r};
    return result;
  }

  // Генерация бидиагональной матрицы
  void Generate_bidiagonal(T min_dist, T max_dist, int rand_num) {
    // Генерация матриц
    std::default_random_engine RNG(rand_num);
    std::uniform_real_distribution<T> dist(min_dist, max_dist);
    SVDGenerator<T> SVD(n, n, RNG, dist, true);
    SVD.generate(n);

    true_sigm_B = SVD.MatrixS();

    SquareMatrix SVD_matr =
        SVD.MatrixU() * true_sigm_B * SVD.MatrixV().transpose();
    auto bid = Eigen::internal::UpperBidiagonalization<SquareMatrix>(
        SVD_matr);  // Бидиагонализация
    B = bid.bidiagonal();

    sigm_B = B;
  }

  // Одна итерация Implicit Zero-Shift QR Algorithm
  void Impl_QR_zero_iter() {
    T oldcs = 1;
    T oldsn;
    T cs = 1;
    T sn;
    T r;

    for (int i = 0; i != n - 1; i++) {
      // Домножение на повороты справа
      std::vector<T> temp1 = ROT(sigm_B(i, i) * cs, sigm_B(i, i + 1));
      cs = temp1[0];
      sn = temp1[1];
      r = temp1[2];

      //(cs, -sn)
      //(sn, cs)
      SquareMatrix Temp_J_R = SquareMatrix::Identity(n, n);
      Temp_J_R(i, i) = cs;
      Temp_J_R(i, i + 1) = -sn;
      Temp_J_R(i + 1, i) = sn;
      Temp_J_R(i + 1, i + 1) = cs;

      right_J = right_J * Temp_J_R;

      Cosines(trigonom_i) = cs;
      Sines(trigonom_i) = sn;
      trigonom_i++;

      if (i != 0) {
        sigm_B(i - 1, i) = oldsn * r;
      }

      // Домножение на повороты слева
      std::vector<T> temp2 = ROT(oldcs * r, sigm_B(i + 1, i + 1) * sn);
      oldcs = temp2[0];
      oldsn = temp2[1];
      sigm_B(i, i) = temp2[2];

      //(cs, sn) Знак у синуса меняется ввиду транспозиции в разложении СВД
      //(-sn, cs)
      SquareMatrix Temp_J_L = SquareMatrix::Identity(n, n);
      Temp_J_L(i, i) = oldcs;
      Temp_J_L(i, i + 1) = oldsn;
      Temp_J_L(i + 1, i) = -oldsn;
      Temp_J_L(i + 1, i + 1) = oldcs;

      left_J = Temp_J_L * left_J;

      Cosines(trigonom_i) = oldcs;
      Sines(trigonom_i) = oldsn;
      trigonom_i++;
    }
    T h = sigm_B(n - 1, n - 1) * cs;
    sigm_B(n - 2, n - 1) = h * oldsn;
    sigm_B(n - 1, n - 1) = h * oldcs;
  }

  // Переносим знак у сингулярного значения на соответсвующий вектор в левой
  // матрице. (ЛОМАЕТСЯ ОРТОГОНАЛЬНОСТЬ!)
  void revert_negative_singular() {
    for (int i = 0; i != n; i++)
      if (sigm_B(i, i) < 0) {
        sigm_B(i, i) = -sigm_B(i, i);
        for (int j = 0; j != n; j++) left_J(i, j) = -left_J(i, j);
      }
  }

  // Нужна для работы функции "shifts", Подсчёт нормы ||B-B*||, где B -
  // изначальная матрица, B* - матрица составленная перемножением поворотов.
  T differance_norm(
      Eigen::VectorXd sines, Eigen::VectorXd cosines,
      bool true_sigm = false)  // Булево значение нужно для понимания того, из
                               // каких сингулярных значений составляется B*
                               // (настоящих или посчитанных)
  {
    T cs;
    T sn;
    SquareMatrix temp_right_J = SquareMatrix::Identity(n, n);
    SquareMatrix temp_left_J = SquareMatrix::Identity(n, n);
    for (int i = 0; i != (n - 1) * iter_num; i++) {
      // Составляем матрицу поворота справа
      cs = cosines(2 * i);
      sn = sines(2 * i);
      SquareMatrix Temp_J_R = SquareMatrix::Identity(n, n);
      Temp_J_R(i % (n - 1), i % (n - 1)) = cs;
      Temp_J_R(i % (n - 1), i % (n - 1) + 1) = -sn;
      Temp_J_R(i % (n - 1) + 1, i % (n - 1)) = sn;
      Temp_J_R(i % (n - 1) + 1, i % (n - 1) + 1) = cs;

      temp_right_J = temp_right_J * Temp_J_R;

      // Составляем матрицу поворота слева
      cs = cosines(2 * i + 1);
      sn = sines(2 * i + 1);
      SquareMatrix Temp_J_L = SquareMatrix::Identity(n, n);
      Temp_J_L(i % (n - 1), i % (n - 1)) = cs;
      Temp_J_L(i % (n - 1), i % (n - 1) + 1) = sn;
      Temp_J_L(i % (n - 1) + 1, i % (n - 1)) = -sn;
      Temp_J_L(i % (n - 1) + 1, i % (n - 1) + 1) = cs;

      temp_left_J = Temp_J_L * temp_left_J;
    }
    SquareMatrix temp_B;
    if (true_sigm) {
      // Для настоящих значений нужно поменять знак на соответсвующий при том же
      // посчитанном (ненастоящим) сингулярном значпении
      SquareMatrix temp_true_sigm = true_sigm_B;
      for (int i = 0; i != n; i++) {
        if (sigm_B(i, i) < 0) temp_true_sigm(i, i) = -temp_true_sigm(i, i);
      }
      temp_B = temp_left_J.transpose() * temp_true_sigm *
               temp_right_J.transpose();  // Составляем B*

    } else {
      SquareMatrix temp_sigm = sigm_B;
      temp_B = temp_left_J.transpose() * sigm_B *
               temp_right_J.transpose();  // Составляем B*
    }
    return ((B - temp_B).norm());
  }

 public:
  QR_zero_shift(
      int n1, T min_dist, T max_dist,
      int rand_num)  // Конструктор использующий генератор бидиагональных матриц
  {
    assert(n1 > 1);

    n = n1;
    iter_num = 0;

    left_J = SquareMatrix::Identity(n, n);
    right_J = SquareMatrix::Identity(n, n);

    Cosines = Eigen::VectorXd();
    Sines = Eigen::VectorXd();
    NewCosines = Eigen::VectorXd();
    NewSines = Eigen::VectorXd();
    trigonom_i = 0;

    Generate_bidiagonal(min_dist, max_dist, rand_num);
  }

  QR_zero_shift(SquareMatrix b)  // Конструктор с задачей матрицы B,
                                 // использующий СВД из Eigen
  {
    n = b.rows();
    iter_num = 0;

    left_J = SquareMatrix::Identity(n, n);
    right_J = SquareMatrix::Identity(n, n);

    auto bid = Eigen::internal::UpperBidiagonalization<SquareMatrix>(b);
    B = bid.bidiagonal();
    sigm_B = B;

    auto B_svd = Eigen::JacobiSVD<SquareMatrix>(B);
    true_sigm_B = B_svd.singularValues().matrix().asDiagonal();

    Cosines = Eigen::VectorXd();
    Sines = Eigen::VectorXd();
    NewCosines = Eigen::VectorXd();
    NewSines = Eigen::VectorXd();
    trigonom_i = 0;
  }

  // Основная функция задающая количество итераций
  // алгоритма."null_superdiagonal" - , "null_superdiagonal" - зануление
  // супердиагонали полученный матрицы с сингулярными значениями
  void Implicit_QR_with_zero_shift(
      int m, bool null_superdiagonal = false,
      bool revert_negative = false)  // m - количество итераций алгоритма
  {
    Cosines.resize(2 * m * (n - 1));
    Sines.resize(2 * m * (n - 1));
    NewCosines.resize(2 * m * (n - 1));
    NewSines.resize(2 * m * (n - 1));
    Tans.resize(2 * m * (n - 1));

    for (int i = 0; i != m; i++) {
      iter_num++;
      Impl_QR_zero_iter();
    }
    if (null_superdiagonal) {
      for (int i = 0; i != n - 1; i++) {
        sigm_B(i, i + 1) = 0;
      }
    }
    NewCosines = Cosines;
    NewSines = Sines;
    if (revert_negative) revert_negative_singular();
  }

  // Подсчёт нормы || B - B* || , где B - изначальная матрица, B* - матрица
  // составленная перемножением поворотов.
  T differance_norm(bool true_sigm = false) {
    T cs;
    T sn;
    SquareMatrix temp_right_J = SquareMatrix::Identity(n, n);
    SquareMatrix temp_left_J = SquareMatrix::Identity(n, n);
    for (int i = 0; i != (n - 1) * iter_num; i++) {
      // Составление поварота справа
      cs = NewCosines(2 * i);
      sn = NewSines(2 * i);
      SquareMatrix Temp_J_R = SquareMatrix::Identity(n, n);
      Temp_J_R(i % (n - 1), i % (n - 1)) = cs;
      Temp_J_R(i % (n - 1), i % (n - 1) + 1) = -sn;
      Temp_J_R(i % (n - 1) + 1, i % (n - 1)) = sn;
      Temp_J_R(i % (n - 1) + 1, i % (n - 1) + 1) = cs;

      temp_right_J = temp_right_J * Temp_J_R;

      // Составление поварота слева
      cs = NewCosines(2 * i + 1);
      sn = NewSines(2 * i + 1);
      SquareMatrix Temp_J_L = SquareMatrix::Identity(n, n);
      Temp_J_L(i % (n - 1), i % (n - 1)) = cs;
      Temp_J_L(i % (n - 1), i % (n - 1) + 1) = sn;
      Temp_J_L(i % (n - 1) + 1, i % (n - 1)) = -sn;
      Temp_J_L(i % (n - 1) + 1, i % (n - 1) + 1) = cs;

      temp_left_J = Temp_J_L * temp_left_J;
    }
    SquareMatrix temp_B;
    if (true_sigm) {
      // Для настоящих значений нужно поменять знак на соответсвующий при том же
      // посчитанном (ненастоящим) сингулярном значпении
      SquareMatrix temp_true_sigm = true_sigm_B;
      for (int i = 0; i != n; i++) {
        if (sigm_B(i, i) < 0) temp_true_sigm(i, i) = -temp_true_sigm(i, i);
      }
      temp_B =
          temp_left_J.transpose() * temp_true_sigm * temp_right_J.transpose();

    } else {
      SquareMatrix temp_sigm = sigm_B;
      temp_B = temp_left_J.transpose() * sigm_B * temp_right_J.transpose();
    }
    return ((B - temp_B).norm());
  }

  // Идея сдвигов: для каждого угла, для каждого поворота изменяем значение
  // синуса на сдвиг, изменяем соответсвенно косинус и пересчитываем норму, если
  // норма меньше, то сохраняем новое значение
  T shifts(Eigen::VectorXd shifts, bool true_sigm = false) {
    T new_norm;
    T old_norm = differance_norm();
    for (int i = 0; i != trigonom_i; i++)
      for (T shift : shifts) {
        Eigen::VectorXd temp_new_sines = NewSines;
        Eigen::VectorXd temp_new_cosines = NewCosines;

        // Пересчёт синусов и косинусов
        temp_new_sines(i) = NewSines(i) + shift;
        temp_new_cosines(i) = (NewCosines(i) < 0)
                                  ? -sqrt(1 - pow(temp_new_sines(i), 2))
                                  : sqrt(1 - pow(temp_new_sines(i), 2));

        // Пересчитывем нормы с новыми синусами и косинусами
        new_norm = differance_norm(temp_new_sines, temp_new_cosines, true_sigm);
        // Записываем новые значения, если норма уменьшилась
        if (new_norm < old_norm) {
          old_norm = new_norm;
          NewSines = temp_new_sines;
          NewCosines = temp_new_cosines;
        }
      }
    // Возвращаем финальную норму
    return old_norm;
  }

  // Строим матрицу слева перемножением поворотов с новыми значениями синусов и
  // косинусов
  SquareMatrix reconstruct_left() {
    T cs;
    T sn;
    SquareMatrix temp_left_J = SquareMatrix::Identity(n, n);
    for (int i = 0; i != (n - 1) * iter_num; i++) {
      cs = NewCosines(2 * i + 1);
      sn = NewSines(2 * i + 1);
      SquareMatrix Temp_J_L = SquareMatrix::Identity(n, n);
      Temp_J_L(i % (n - 1), i % (n - 1)) = cs;
      Temp_J_L(i % (n - 1), i % (n - 1) + 1) = sn;
      Temp_J_L(i % (n - 1) + 1, i % (n - 1)) = -sn;
      Temp_J_L(i % (n - 1) + 1, i % (n - 1) + 1) = cs;

      temp_left_J = Temp_J_L * temp_left_J;
    }
    return temp_left_J;
  }

  // Строим матрицу справа перемножением поворотов с новыми значениями синусов и
  // косинусов
  SquareMatrix reconstruct_right() {
    T cs;
    T sn;
    SquareMatrix temp_right_J = SquareMatrix::Identity(n, n);
    for (int i = 0; i != (n - 1) * iter_num; i++) {
      cs = NewCosines(2 * i);
      sn = NewSines(2 * i);
      SquareMatrix Temp_J_R = SquareMatrix::Identity(n, n);
      Temp_J_R(i % (n - 1), i % (n - 1)) = cs;
      Temp_J_R(i % (n - 1), i % (n - 1) + 1) = -sn;
      Temp_J_R(i % (n - 1) + 1, i % (n - 1)) = sn;
      Temp_J_R(i % (n - 1) + 1, i % (n - 1) + 1) = cs;

      temp_right_J = temp_right_J * Temp_J_R;
    }
    return temp_right_J;
  }

  // Геттеры
  SquareMatrix Get_B() { return B; }
  SquareMatrix Get_left_J() { return left_J; }
  SquareMatrix Get_right_J() { return right_J; }
  SquareMatrix Get_sigm_B() { return sigm_B; }
  SquareMatrix Get_true_sigm_B() { return true_sigm_B; }
  Eigen::VectorXd Get_cosines() { return Cosines; }
  Eigen::VectorXd Get_sines() { return Sines; }
  Eigen::VectorXd Get_t() { return Tans; }
  int Get_size() { return n; }
  int Get_iter_num() { return iter_num; }
};
