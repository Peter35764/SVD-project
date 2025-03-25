#include "mrrr.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
    const int n = 4;
    MatrixXd A(n, n);
    
    // Заполнение тестовой матрицы (можно раскомментировать случайное заполнение)
    A << 1, 1, 1, 1,
         0, 0, 0, 1,
         1, 0, 1, 0,
         1, 1, 0, 1;
    
    /* Для случайного заполнения:
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    A = MatrixXd::NullaryExpr(n, n, [&](){ return dis(gen); });
    */
    
    cout << "Original matrix A:\n" << A << endl << endl;
    
    try {
        MRRR_SVD<int> svd(A);
        
        auto U = svd.matrixU();
        auto V = svd.matrixV();
        auto S = svd.singularValues();
        
        cout << "Left singular vectors U:\n" << U << endl << endl;
        cout << "Singular values matrix S:\n" << S << endl << endl;
        cout << "Right singular vectors V:\n" << V << endl << endl;
        
        // Проверка разложения
        MatrixXd reconstructed = U * S * V.transpose();
        cout << "Reconstructed matrix:\n" << reconstructed << endl << endl;
        
        // Вычисление ошибки восстановления
        MatrixXd error = A - reconstructed;
        cout << "Reconstruction error (Frobenius norm): " 
             << error.norm() << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
