#include <iostream>
#include <Eigen/SVD> 
#include "iterative_refinement_1.h" 
#include <boost/multiprecision/cpp_dec_float.hpp>

using float128 = boost::multiprecision::cpp_dec_float_100;

int main() {
    using namespace std;
    using namespace Eigen;
    using namespace SVD_Project;
    using Real = float128; 

    Matrix<Real, Dynamic, Dynamic> A(10, 9);
    A << Real("1"), Real("2"), Real("3"), Real("4"), Real("5"), Real("6"), Real("7"), Real("8"), Real("9"),
         Real("10"), Real("11"), Real("12"), Real("13"), Real("14"), Real("15"), Real("16"), Real("17"), Real("18"),
         Real("19"), Real("20"), Real("21"), Real("22"), Real("23"), Real("24"), Real("25"), Real("26"), Real("27"),
         Real("28"), Real("29"), Real("30"), Real("31"), Real("32"), Real("33"), Real("34"), Real("35"), Real("36"),
         Real("37"), Real("38"), Real("39"), Real("40"), Real("41"), Real("42"), Real("43"), Real("44"), Real("45"),
         Real("46"), Real("47"), Real("48"), Real("49"), Real("50"), Real("51"), Real("52"), Real("53"), Real("54"),
         Real("55"), Real("56"), Real("57"), Real("58"), Real("59"), Real("60"), Real("61"), Real("62"), Real("63"),
         Real("64"), Real("65"), Real("66"), Real("67"), Real("68"), Real("68"), Real("70"), Real("71"), Real("72"),
         Real("73"), Real("74"), Real("75"), Real("76"), Real("77"), Real("78"), Real("79"), Real("80"), Real("81"),
         Real("3"), Real("9"), Real("4.98942"), Real("0.324235"), Real("443534"), Real("345"), Real("56.543853"), Real("450.435234"), Real("43.34353221");

    cout << "Running AOIR_SVD_1 with Real = float128..." << endl;
    SVD_Project::AOIR_SVD_1<Real> Ans(A, "error_algo1_main_test"); 

    const int print_precision = std::numeric_limits<Real>::max_digits10;
    cout << std::scientific << std::setprecision(print_precision);

    cout << "\nRefined SVD Results (from AOIR_SVD_1<float128>):\n";
    cout << "---------------------------------------------------\n";
    cout << "Refined U:\n" << Ans.matrixU() << "\n\n";
    cout << "Refined V:\n" << Ans.matrixV() << "\n\n";
    cout << "Refined S (matrix form):\n" << Ans.singularValues() << "\n\n";
    cout << "\nFinished." << endl;

    return 0;
}