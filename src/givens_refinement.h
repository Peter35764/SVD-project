#ifndef GIVENS_REFINEMENT_H
#define GIVENS_REFINEMENT_H

#include <Eigen/Core>
#include <Eigen/SVD>

namespace SVD_Project {

// _FPNum   - floating-point number
// N        - Size of square matrix
template<typename _MatrixType, typename _FPNum = double, int N = Eigen::Dynamic>
class GivRef_SVD : public Eigen::SVDBase<GivRef_SVD<_MatrixType>>
{
    typedef Eigen::SVDBase<GivRef_SVD> Base;

public:
    GivRef_SVD(const _MatrixType &matrix, unsigned int computationOptions = 0);

    GivRef_SVD &compute(const _MatrixType &matrix, unsigned int computationOptions = 0);

private:
    using SquareMatrix = Eigen::Matrix<typename _MatrixType::Scalar, N, N>;

    SquareMatrix left_J;      // Произведение всех поворпотов Гивенса слева от B
    SquareMatrix right_J;     // Произведение всех поворпотов Гивенса справа от B
    SquareMatrix sigm_B;      // Матрица с сингулярными значениями B
    SquareMatrix true_sigm_B; // Матрица с точными сингулярными значениями B
    SquareMatrix B;           // Изначальная Бидиагональная матрица

    Eigen::VectorXd Cosines; // Вектор с изначальными значениями косинусов
    Eigen::VectorXd Sines;   // Вектор с изначальными значениями косинусов
    Eigen::VectorXd Tans;    // Вектор с изначальными значениями тангенсов

    Eigen::VectorXd NewCosines; // Вектор со значениями косинусов после сдвигов
    Eigen::VectorXd NewSines;   // Вектор со значениями синусов после сдвигов

    int trigonom_i; // Значение для перебора векторов с тригонометрическими функциями
    int iter_num;   // Количество итераций алгоритма QR with zero shift
    int n;          // Размер матриц

    std::vector<_FPNum> ROT(_FPNum f, _FPNum g);
};

} // namespace SVD_Project

// traits - типозависимые свойства
template<typename _MatrixType>
struct Eigen::internal::traits<SVD_Project::GivRef_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType>
{
    typedef _MatrixType MatrixType;
};

#include "givens_refinement.hpp"

#endif // GIVENS_REFINEMENT_H
