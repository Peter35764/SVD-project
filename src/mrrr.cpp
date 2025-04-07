#include "mrrr.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
    const int n = 4;
    MatrixXd A(n, n);
    
    // Заполнение тестовой матрицы
    A << 1, 1, 1, 1,
         0, 0, 0, 1,
         1, 0, 1, 0,
         1, 1, 0, 1;
    
    cout << "Original matrix A:\n" << A << endl << endl;
    
    try {
        // Используем ComputeFullU | ComputeFullV в качестве опций вычисления
        MRRR_SVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        
        auto U = svd.matrixU();
        auto V = svd.matrixV();
        auto singularValues = svd.singularValues();
        
        cout << "Left singular vectors U:\n" << U << endl << endl;
        cout << "Singular values vector:\n" << singularValues << endl << endl;
        cout << "Right singular vectors V:\n" << V << endl << endl;
        
        // Восстановление матрицы A через U * S.asDiagonal() * V.transpose()
        MatrixXd reconstructed = U * singularValues.asDiagonal() * V.transpose();
        cout << "Reconstructed matrix:\n" << reconstructed << endl << endl;
        
        // Вычисление ошибки восстановления (норма Фробениуса)
        MatrixXd error = A - reconstructed;
        cout << "Reconstruction error (Frobenius norm): " 
             << error.norm() << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
