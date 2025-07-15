#ifndef GENERATE_SVD_H
#define GENERATE_SVD_H

#include <Eigen/Dense>
#include <ctime>
#include <random>
#include <cassert>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace SVD_Project {

template <typename T, int M_Rows = Eigen::Dynamic, int N_Cols = Eigen::Dynamic>
class SVDGenerator {
private:
    using MatrixUType = Eigen::Matrix<T, M_Rows, M_Rows>; 
    using MatrixVType = Eigen::Matrix<T, N_Cols, N_Cols>; 
    using MatrixSType = Eigen::Matrix<T, M_Rows, N_Cols>; 
    using DynamicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using EigenSingValVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    bool generatedFLG = false;
    MatrixUType U_true; 
    MatrixVType V_true; 
    MatrixSType S_diag_matrix; 
    EigenSingValVector singular_values_vec; 

    T sigma_min_limit;
    T sigma_max_limit;

    int rows_A; 
    int cols_A; 
    int p_rank; 

public:
    SVDGenerator(int r, int c, T min_sv, T max_sv) : 
        rows_A(r), cols_A(c), p_rank(std::min(r,c)),
        sigma_min_limit(min_sv), sigma_max_limit(max_sv)
    {
        assert(rows_A > 0);
        assert(cols_A > 0);
        assert(max_sv >= min_sv);
        assert(min_sv > T("0.0"));

        U_true = MatrixUType::Zero(rows_A, rows_A);
        V_true = MatrixVType::Zero(cols_A, cols_A);
        S_diag_matrix = MatrixSType::Zero(rows_A, cols_A);
        singular_values_vec.resize(p_rank); 
        singular_values_vec.fill(T("0.0"));
    }

    MatrixUType MatrixU() {
        if (!generatedFLG) generate();
        return U_true;
    }

    MatrixVType MatrixV() {
        if (!generatedFLG) generate();
        return V_true;
    }

    MatrixSType MatrixS() { 
        if (!generatedFLG) generate();
        return S_diag_matrix;
    }

    EigenSingValVector SingularValues() {
        if(!generatedFLG) generate();
        return singular_values_vec;
    }

    DynamicMatrix MatrixA() {
        if(!generatedFLG) generate();
        return U_true * S_diag_matrix * V_true.transpose();
    }

    void generate() {
        if (generatedFLG) return;
        generatedFLG = true;

        singular_values_vec.fill(T("0.0")); 

        if (p_rank > 1) {
            T p_minus_1 = T(p_rank - 1);
            if (boost::multiprecision::abs(p_minus_1) < std::numeric_limits<T>::epsilon()){ // Avoid division by zero if p_rank was 1 but somehow passed p_rank > 1
                 if (p_rank > 0) singular_values_vec(0) = sigma_max_limit; // only one value
            } else {
                T step = (sigma_max_limit - sigma_min_limit) / p_minus_1;
                for (int i = 0; i < p_rank; ++i) {
                    T i_T = T(i);
                    singular_values_vec(i) = sigma_max_limit - i_T * step; 
                    if (singular_values_vec(i) < T("0.0")) singular_values_vec(i) = T("0.0");
                }
            }
        } else if (p_rank == 1) {
            singular_values_vec(0) = sigma_max_limit;
        }

        S_diag_matrix.setZero();
        for(int i = 0; i < p_rank; i++) {
            S_diag_matrix(i,i) = singular_values_vec(i);
        }

        DynamicMatrix T_1_rand(rows_A, rows_A), T_2_rand(cols_A, cols_A);
        DynamicMatrix Q_1_ortho, Q_2_ortho;

        T_1_rand = MatrixUType::Random(rows_A,rows_A); 
        T_2_rand = MatrixVType::Random(cols_A,cols_A); 
        
        Eigen::HouseholderQR<DynamicMatrix> qr1(T_1_rand);
        Eigen::HouseholderQR<DynamicMatrix> qr2(T_2_rand);
        Q_1_ortho = qr1.householderQ(); 
        Q_2_ortho = qr2.householderQ(); 

        if (Q_1_ortho.cols() < rows_A) {
            MatrixUType temp_Q1 = MatrixUType::Identity(rows_A, rows_A);
            temp_Q1.leftCols(Q_1_ortho.cols()) = Q_1_ortho;
            U_true = temp_Q1;
        } else {
            U_true = Q_1_ortho;
        }

        if (Q_2_ortho.cols() < cols_A) {
            MatrixVType temp_Q2 = MatrixVType::Identity(cols_A, cols_A);
            temp_Q2.leftCols(Q_2_ortho.cols()) = Q_2_ortho;
            V_true = temp_Q2.transpose(); 
        } else {
            V_true = Q_2_ortho.transpose(); 
        }
    }
};

} 

#endif // GENERATE_SVD_H