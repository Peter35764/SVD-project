#include <iostream>
#include <iomanip>

// если это делать по нормальному, то у меня компилятор немного ругается
// было лень менять все на нечто нормальное
#include "jacobi.h"

/*
 * теперь здесь ничего не будет
 * только то как я проверял, работает ли это дело
 */
int main()
{
    using namespace std;

    // PermutationMatrix<Dynamic> P(3);
    // // P.setIdentity();
    // for (int i = 0; i < 3; ++i)
    // {
    //     P.indices()[i] = 3 - i - 1;
    // }
    // auto P2 = P * P.transpose();
    // for (int i = 0; i < 3; i++)
    //     cout << P2.indices()[i] << "\n";

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m(9, 10);
    m << 1,  2,  3,  4,  5,  6,  7,  8,  9,
         10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 20, 21, 22, 23, 24, 25, 26, 27,
         28, 29, 30, 31, 32, 33, 34, 35, 36,
         37, 38, 39, 40, 41, 42, 43, 44, 45,
         46, 47, 48, 49, 50, 51, 52, 53, 54,
         55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 66, 67, 68, 68, 70, 71, 72,
         73, 74, 75, 76, 77, 78, 79, 80, 81,
         82, 83, 84, 85, 86, 87, 88, 89, 90;
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m2 = m.transpose();

    params_t params;  
    params.epsilon = 1e-18;  
    JTS_SVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> jts_svd;
    jts_svd.compute(m, Eigen::ComputeFullU | Eigen::ComputeFullV, params);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> S(9, 10);
    S.setZero();

    auto singVector = jts_svd.singularValues();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> U = jts_svd.matrixU();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V = jts_svd.matrixV();

    for (int i = 0; i < 9; i++)
    {
        S(i, i) = singVector(i);
    }
    cout << jts_svd.computeU() << " " << jts_svd.computeV() << "\n";
    cout << S << "\n\n";

    cout << U << "\n\n" << S << "\n\n" << V << "\n\n";
    cout << U * S * V.transpose() << "\n\n";
    cout << U * S << "\n\n";
    

    for (int i = 0; i < U.cols(); i++)
    {
        for (int j = 0; j < U.cols(); j++)
        {
            cout << setprecision(4) << fixed << U.col(i).dot(U.col(j)) << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
    cout << "got the problem\n" 
         << "this is result of V^T * V\n"
         << V.transpose() * V << "\n\n"
         << "and this is result of V * V^T\n"
         << V * V.transpose() << "\n\n";

    char c;
    cin >> c;
    return 0;
}
