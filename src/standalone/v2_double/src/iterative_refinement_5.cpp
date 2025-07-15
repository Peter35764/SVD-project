#include <iostream>
#include <Eigen/SVD>
#include "iterative_refinement_5.h"

int main() {
    using namespace std;
    using namespace Eigen;
    using namespace SVD_Project;


    using Real = double;

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
         3, 9, 4.98942, 0.324235, 443534, 345, 56.543853, 450.435234, 43.34353221;


    MatrixXd A_double = A.template cast<double>();
    BDCSVD<MatrixXd> svd(A_double, ComputeFullU | ComputeFullV);


    SVD_Project::AOIR_SVD_5<Real> Ans(A);

    cout << "Refined A ≈ U*S*V^T:\n"
         << Ans.matrixU() * Ans.singularValues() * Ans.matrixV().transpose() << "\n\n";

    cout << "U:\n" << Ans.matrixU() << "\n\n";
    cout << "V:\n" << Ans.matrixV() << "\n\n";
    cout << "S:\n" << Ans.singularValues() << "\n\n";


    Array<Real, 1, Dynamic> sigm(9);
    sigm = svd.singularValues().template cast<Real>();

    Matrix<Real, Dynamic, Dynamic> I(10, 9);
    I.setZero();
    I.block(0, 0, 9, 9) = sigm.matrix().asDiagonal();


    Matrix<Real, Dynamic, Dynamic> U = svd.matrixU().template cast<Real>();
    Matrix<Real, Dynamic, Dynamic> V = svd.matrixV().template cast<Real>();

    cout << "Original SVD A ≈ U*S*V^T:\n" << U * I * V.transpose() << "\n";

    return 0;
}
