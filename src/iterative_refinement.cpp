#include "iterative_refinement.h"

//Проверка кода

int main()
{
    using namespace std;
    using namespace Eigen;
    Eigen::Matrix<float, Dynamic, Dynamic> A(10,9);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 68, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81,
        3, 9, (float)4.98942, (float)0.324235,  443534, 345, (float)56.543853, (float)450.435234, (float)43.34353221;

   BDCSVD<MatrixXf> svd(A, ComputeFullU | ComputeFullV);
   AOIR_SVD<float,10,9> Ans(A);

   cout << Ans.matrixU() * Ans.singularValues() * Ans.matrixV().transpose() << "\n" << "\n";
   cout << Ans.matrixU() << "\n";
   cout << Ans.matrixV() << "\n";
   cout << Ans.singularValues() << "\n" << "\n";

   
    Array<float,1, Dynamic> sigm(9);
    sigm  = svd.singularValues();
    Eigen::Matrix<float, Dynamic, Dynamic > I(10, 9);
    I.setZero();
    I.block(0,0,9,9) = sigm.matrix().asDiagonal();
    Eigen::Matrix<float, Dynamic, Dynamic > U = svd.matrixU();
    cout << U*I* svd.matrixV().transpose() << "\n" << "\n";

    return 0;
}
