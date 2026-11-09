#pragma once

#include <Eigen/Jacobi>
#include <Eigen/SVD>
#include <iostream>

// Сделано Черниковым Тимофеем КМБО-03-21
// github: chernikovtimofey

template <typename _MatrixType>
class GolubKahanSVD;

// Такая структура присутствует для всех классов Eigen и определяет данные,
// известные на этапе компиляции
template <typename _MatrixType>
struct Eigen::internal::traits<GolubKahanSVD<_MatrixType>>
    : public Eigen::internal::traits<_MatrixType> {
 public:
  typedef _MatrixType MatrixType;
};

// Класс разложения матрицы в SCD по алгоритму Голаба-Кохана для разложение
// действительных матриц. Левую и правую унитарные матрицы можно получить
// методами matrixU() и matrixV() соответсвенно Вектор с сингулярными значениями
// с помощью singularValues()
template <typename _MatrixType>
class GolubKahanSVD : public Eigen::SVDBase<GolubKahanSVD<_MatrixType>> {
 private:
  typedef Eigen::SVDBase<GolubKahanSVD> Base;

 public:
  using Base::cols;
  using Base::computeU;
  using Base::computeV;
  using Base::rows;

  // Тип матрицы на разложение и тип ее элементов
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;

  // Тип левой и правой унитарных матриц, тип вектора с сингулярными значениями
  typedef typename Base::MatrixUType MatrixUType;
  typedef typename Base::MatrixVType MatrixVType;
  typedef typename Base::SingularValuesType SingularValuesType;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  GolubKahanSVD() : m_isTranspose(false), m_compU(false), m_compV(false) {}

  GolubKahanSVD(Eigen::Index rows, Eigen::Index cols,
                unsigned int computationOptions = 0) {
    allocate(rows, cols, computationOptions);
  }

  GolubKahanSVD(const MatrixType& matrix, unsigned int computationOptions = 0) {
    compute(matrix, computationOptions);
  }

  // computationOptions - параметр, определяющий тип разожение (тонкое, полное)
  // Eigen::ComputeFull[U, V] - полное разложение
  // Eigen::ComputeThin[U, V] - тонкое разложение
  GolubKahanSVD& compute(const MatrixType& matrix,
                         unsigned int computationOptions);

  GolubKahanSVD& compute(const MatrixType& matrix) {
    return compute(matrix, this->m_computationOptions);
  }

 private:
  void allocate(Eigen::Index rows, Eigen::Index cols,
                unsigned int computationOptions);
  void algorithm();
  template <typename SubMatrixType>
  void step(SubMatrixType submatrix);
  template <typename SubMatrixType>
  Scalar cal_shift(SubMatrixType submatrix);
  template <typename HouseholderU, typename HouseholderV>
  void copyUV(const HouseholderU& householderU,
              const HouseholderV& householderV, MatrixX& naiveU,
              MatrixX& naiveV);

 protected:
  MatrixX m_naiveU, m_naiveV;
  MatrixX m_computed;
  bool m_isTranspose, m_compU, m_compV;

  using Base::m_computeFullU;
  using Base::m_computeFullV;
  using Base::m_computeThinU;
  using Base::m_computeThinV;
  using Base::m_diagSize;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_matrixU;
  using Base::m_matrixV;
  using Base::m_nonzeroSingularValues;
  using Base::m_singularValues;
};


#include "golub_kahan.cpp"
