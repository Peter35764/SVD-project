#ifndef V0_NAIVE_MRRR_H
#define V0_NAIVE_MRRR_H

#include <Eigen/SVD>

namespace SVD_Project {

// Объявление класса-шаблона v0_NaiveMRRR_SVD с полями и методами
template<typename _MatrixType>
class v0_NaiveMRRR_SVD : public Eigen::SVDBase<v0_NaiveMRRR_SVD<_MatrixType>>
{
public:
 const double sq = std::sqrt(2);
 // Конструктор: принимает исходную матрицу и опции вычисления
 v0_NaiveMRRR_SVD(const _MatrixType& matrix, unsigned int computationOptions);

 // Метод для вычисления разложения TGK
 v0_NaiveMRRR_SVD<_MatrixType>& compute_tgk(const _MatrixType& matrix);

 // Деструктор по необходимости (если потребуется)
 ~v0_NaiveMRRR_SVD() = default;

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

};  // namespace SVD_Project

template <typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::v0_NaiveMRRR_SVD<_MatrixType>>
    : public Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

#include "v0_naive_mrrr.hpp"  // Подключаем реализацию

#endif  // V0_NAIVE_MRRR_H
