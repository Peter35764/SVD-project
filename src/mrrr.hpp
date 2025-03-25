#ifndef MRRR_HPP
#define MRRR_HPP

#include "mrrr.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <cstdint>
#include <lapacke.h>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <ostream>
#include <tuple>
#include <vector>

double sq = sqrt(2);

template<typename T>
void MRRR_SVD<T>::Set_U(const Eigen::MatrixXd& A) {
    U = A;
}

template<typename T>
void MRRR_SVD<T>::Set_V(const Eigen::MatrixXd& A) {
    V = A;
}

template<typename T>
void MRRR_SVD<T>::Set_S(const Eigen::MatrixXd& A) {
    S = A;
}

template<typename T>
MRRR_SVD<T> MRRR_SVD<T>::compute_bsvd(const Eigen::MatrixXd& matrix) {
    auto bid = Eigen::internal::UpperBidiagonalization(matrix);
    auto B = bid.bidiagonal();
    auto L = bid.householderU();
    auto R = bid.householderV();
    
    int n = B.rows();        
    Eigen::MatrixXd TGK(2*n,2*n);
    TGK.setZero();
    Eigen::MatrixXd BB = B;
    
    std::cout << "B: " << std::endl << B.diagonal(0) << std::endl << B.diagonal(1) << std::endl;
    std::cout << "BB: " << std::endl << BB << std::endl;
    
    int nn = 2*n;
    for (int i = 0; i < n - 1; ++i) {
        TGK(2*i,2*i+1) = BB(i,i); TGK(2*i+1,2*i+2) = BB(i,i+1);
        TGK(2*i+1,2*i) = BB(i,i); TGK(2*i+2,2*i+1) = BB(i,i+1);
    }
    TGK(nn-2,nn-1) = BB(n-1,n-1);
    TGK(nn-1,nn-2) = BB(n-1,n-1);
    
    std::cout << "TGK: " << std::endl << TGK << std::endl;
    
    int32_t nzc = std::max(1,nn);
    double *d = new double[2*n];
    for (int i = 0; i < 2*n; i++) {
        d[i] = TGK.diagonal(0)[i];
    }
    
    double *e = new double [2*n];
    for (int i = 0; i < 2*n-1; i++) {
        e[i] = TGK.diagonal(-1)[i];
    }
    
    double *w = new double[nn];
    double *z = new double[nn*nn];
    int32_t m;
    int32_t isuppz [2*nn];
    int32_t tryrac = 1;
    
    int32_t info = LAPACKE_dstemr(
        LAPACK_COL_MAJOR,
        'V',
        'A',
        nn,
        d,
        e,
        0,
        0,
        0,
        0,
        &m,
        w,
        z,
        nn,
        nzc,
        isuppz,
        &tryrac
    );
    
    if (info != 0)
        throw std::runtime_error("LAPACK error: " + std::to_string(info));
    
    Eigen::VectorXd eigenvalues(nn);
    for (int i = 0; i < nn; ++i) {
        eigenvalues(i) = w[nn-i-1];
    }
    
    std::cout << "eigenvalues: " << std::endl << eigenvalues << std::endl;
    
    Eigen::MatrixXd eigenvectors(nn,nn);
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) {
            eigenvectors(i,j) = i < n ? z[(nn-i-1)*nn+j]*sq : z[(nn-i-1)*nn+j] * (-1)*sq;
        }
    }
    
    for (int i = n-1; i < n+1; i++) {
        for (int j = 0; j < nn; j++) {
            eigenvectors(i,j) *= -1;
        }
    }
    
    std::cout << "eigenvectors: " << std::endl << eigenvectors << std::endl;
    
    Eigen::MatrixXd matU(n,n);
    Eigen::MatrixXd matV(n,n);
    Eigen::MatrixXd singularValuess(n,n);
    int ind = 0;
    
    for (int i = 0; i < n; ++i) {
        ind = i;
        double sigma = std::abs(eigenvalues(ind));
        if (sigma < 1e-15) { 
            sigma = 0; ind = -i - 1;
        } 
        
        Eigen::VectorXd q(nn);
        for (int j = 0; j < nn; j++) {
            q[j] = eigenvectors(ind,j);
        }
        
        std::cout << "q: " << std::endl << q << std::endl;
        
        Eigen::VectorXd u(n);
        Eigen::VectorXd v(n);
        for (int j = 0; j < n; ++j) {
            v[j] = q[2*j];
            u[j] = q[2*j+1];
        }
        
        singularValuess(i,i) = sigma;
        matU.col(i) = u;
        matV.col(i) = v;
    }
    
    std::cout << "matU: " << std::endl << matU << std::endl;
    std::cout << "matV: " << std::endl << matV << std::endl;
    
    Set_S(singularValuess);
    Set_U(L*matU);
    Set_V(R*matV);
    
    delete[] d; delete[] e; delete[] w; delete[] z; 
    return *this;
}

template<typename T>
MRRR_SVD<T>::MRRR_SVD() {}

template<typename T>
MRRR_SVD<T>::MRRR_SVD(const Eigen::MatrixXd& A) {
    compute_bsvd(A);
}

template<typename T>
Eigen::MatrixXd MRRR_SVD<T>::matrixV() {
    return V;
}

template<typename T>
Eigen::MatrixXd MRRR_SVD<T>::matrixU() {
    return U;
}

template<typename T>
Eigen::MatrixXd MRRR_SVD<T>::singularValues() {
    return S;
}

#endif // MRRR_HPP
