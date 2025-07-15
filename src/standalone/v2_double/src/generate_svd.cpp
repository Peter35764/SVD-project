// generate_svd.cpp
#include "generate_svd.h" 
#include <iostream>
#include <vector>
#include <iomanip>     
#include <boost/multiprecision/cpp_dec_float.hpp>

using float128 = boost::multiprecision::cpp_dec_float_100;

int main()
{
    using T = float128;

    const int rows = 4;
    const int cols = 4;
    const T sigma_min = T("0.5"); 
    const T sigma_max = T("2.0"); 

    // <<< ИСПРАВЛЕНИЕ ЗДЕСЬ >>>
    SVD_Project::SVDGenerator<T> SVD_instance(rows, cols, sigma_min, sigma_max);

    SVD_instance.generate();

    const int output_precision = std::numeric_limits<T>::max_digits10; 
    std::cout << std::fixed << std::setprecision(output_precision);
    std::cout << "U * U_transpose (should be Identity):" << std::endl;
    std::cout << SVD_instance.MatrixU() * SVD_instance.MatrixU().transpose() << std::endl;
    std::cout << "\nV * V_transpose (should be Identity):" << std::endl;
    std::cout << SVD_instance.MatrixV() * SVD_instance.MatrixV().transpose() << std::endl;
    std::cout << "\nS matrix:" << std::endl;
    std::cout << SVD_instance.MatrixS() << std::endl;
    std::cout << "\nMatrix A:" << std::endl;
    std::cout << SVD_instance.MatrixA() << std::endl;
    std::cout << "\nSingular values vector:" << std::endl;
    std::cout << SVD_instance.SingularValues() << std::endl;

    return 0;
}