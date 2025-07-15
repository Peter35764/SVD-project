#include <iostream>
#include <Eigen/Dense> // Включает <Eigen/SVD> для BDCSVD
#include <vector>      // Для std::vector, если понадобится
#include <string>      // Для std::string
#include <iomanip>     // Для std::fixed, std::setprecision, std::scientific
#include <limits>      // Для std::numeric_limits

#include "iterative_refinement_4.h" // Ваш адаптированный класс AOIR_SVD_4

int main() {
    using namespace std;
    using namespace Eigen;

    using Real = float; 

    Matrix<Real, Dynamic, Dynamic> A(10, 9);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 20, 21, 22, 23, 24, 25, 26, 27,
         28, 29, 30, 31, 32, 33, 34, 35, 36,
         37, 38, 39, 40, 41, 42, 43, 44, 45,
         46, 47, 48, 49, 50, 51, 52, 53, 54,
         55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 66, 67, 68, 68, 70, 71, 72,
         73, 74, 75, 76, 77, 78, 79, 80, 81,
         3, 9, 4.98942, 0.324235, 443534.0, 345.0, 56.543853, 450.435234, 43.34353221; // Добавил .0 к целым числам для ясности типа float

    // Используем BDCSVD для получения "эталонных" сингулярных чисел для этого теста
    BDCSVD<Matrix<Real, Dynamic, Dynamic>> svd_eigen_ref(A, ComputeThinU | ComputeThinV); // Используем ComputeThinU | ComputeThinV для эффективности
    Eigen::Matrix<Real, Eigen::Dynamic, 1> true_sv_for_this_test = svd_eigen_ref.singularValues();

    // Вызов конструктора AOIR_SVD_4 с новым интерфейсом
    SVD_Project::AOIR_SVD_4<Real> refined_svd_solver(A, true_sv_for_this_test, "log_testing_algo4_standalone");

    cout << fixed << setprecision(6); // Общие настройки вывода для времени
    cout << "Refined SVD by AOIR_SVD_4:" << endl;
    cout << "------------------------------------" << endl;
    cout << "Iterations taken: " << refined_svd_solver.iterations_taken() << endl;
    cout << "Time taken: " << refined_svd_solver.time_taken_s() << " s" << endl;
    
    cout << scientific << setprecision(std::numeric_limits<Real>::max_digits10); // Для вывода ошибок с высокой точностью
    cout << "Achieved Sigma Relative Error: " << refined_svd_solver.achieved_sigma_relative_error() << endl;
    cout << "Achieved U Ortho Error (||I-UU^T||): " << refined_svd_solver.achieved_U_ortho_error() << endl;
    cout << "Achieved V Ortho Error (||I-VV^T||): " << refined_svd_solver.achieved_V_ortho_error() << endl;
    cout << endl;

    cout << "Refined U (first 5x5 block or less):" << endl;
    cout << refined_svd_solver.matrixU().topLeftCorner(min(5, (int)A.rows()), min(5, (int)A.rows())) << "\n\n";

    cout << "Refined V (first 5x5 block or less):" << endl;
    cout << refined_svd_solver.matrixV().topLeftCorner(min(5, (int)A.cols()), min(5, (int)A.cols())) << "\n\n";
    
    cout << "Refined S (diagonal matrix form, first 5x5 block or less):" << endl;
    cout << refined_svd_solver.singularValues().topLeftCorner(min(5, (int)A.rows()), min(5, (int)A.cols())) << "\n\n";

    cout << "Reconstructed A from refined SVD (A = U*S*V^T):" << endl;
    cout << fixed << setprecision(3); // Меньшая точность для вывода матрицы A
    cout << refined_svd_solver.matrixU() * refined_svd_solver.singularValues() * refined_svd_solver.matrixV().transpose() << "\n\n";


    // Для сравнения, можно вывести оригинальное SVD от Eigen
    cout << "Original Eigen SVD for comparison:" << endl;
    cout << "------------------------------------" << endl;
    Matrix<Real, Dynamic, Dynamic> S_eigen_diag_matrix(A.rows(), A.cols());
    S_eigen_diag_matrix.setZero();
    Eigen::Index num_singular_values_eigen = svd_eigen_ref.singularValues().size();
    for(Eigen::Index i = 0; i < num_singular_values_eigen; ++i) {
        if (i < S_eigen_diag_matrix.rows() && i < S_eigen_diag_matrix.cols()) {
           S_eigen_diag_matrix(i,i) = svd_eigen_ref.singularValues()(i);
        }
    }
    cout << "Reconstructed A from Eigen SVD (A = U*S*V^T):" << endl;
    cout << svd_eigen_ref.matrixU() * S_eigen_diag_matrix * svd_eigen_ref.matrixV().transpose() << "\n";

    return 0;
}