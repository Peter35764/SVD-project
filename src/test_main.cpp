// test_main.cpp
#include "testing.h"
#include <chrono>
#include <iostream>

int main() {
    auto start = std::chrono::high_resolution_clock::now();


    //генерируеся таблицу в файле "jacobi_test_table.txt" теста метода Eigen::JacobiSVD
    //с соотношением сингулярных чисел:  1.01, 1.2, 2, 5, 10, 50      ---    6
    //причем каждое соотношение относится к двум интервалам сингулярных чисел: 
    //                      маленьких {0,1}, больших {1,100} (это не параметризованно)   ---   2
    //с матрицами размеров: {3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}   ---   6
    //6*2*6 = 72 - всего столько строк будет в таблице
    //размер выборки для усреднения: 20
    svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>("jacobi_test_table.txt",
                                 {1.01, 1.2, 2, 5, 10, 50},
                                 {{3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}}, 
                                 20);
    svd_test_func<double, SVDGenerator, DQDS_SVD>("dqds_test_table.txt",
                                 {1.01, 1.2, 2, 5, 10, 50},
                                 {{3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}}, 
                                 20);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGlobal = end - start;
    std::cout << "\nFull execution time = " << durationGlobal.count() << " seconds.\n";

    char c;
    std::cin >> c;
    return 0;
}
