#ifndef ITERATIVE_REFINEMENT_5_HPP
#define ITERATIVE_REFINEMENT_5_HPP

#include "iterative_refinement_5.h"
#include <Eigen/Dense>
#include <limits>

#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <chrono>


#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>


extern "C" {
#include <openblas/cblas.h>
}



namespace SVD_Project {


using float100 = boost::multiprecision::cpp_dec_float_100;


template<typename T_Scalar> using Matrix = Eigen::Matrix<T_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template<typename T_Scalar> using Vector = Eigen::Matrix<T_Scalar, Eigen::Dynamic, 1>;


using LowPrec = double;

using MatrixLowPrec = Eigen::Matrix<LowPrec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;



template<typename T>
AOIR_SVD_5<T>::AOIR_SVD_5() {}



template<typename T>
AOIR_SVD_5<T>::AOIR_SVD_5(const Matrix<T>& A, const std::string& history_filename_base) {



    Eigen::MatrixXd A_double = A.template cast<double>();
    if (A.rows() == 0 || A.cols() == 0) {
         throw std::runtime_error("Input matrix A is empty in AOIR_SVD_5 constructor.");
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd_double(A_double, Eigen::ComputeFullU | Eigen::ComputeFullV);


    Matrix<T> initial_U = svd_double.matrixU().template cast<T>();
    Matrix<T> initial_V = svd_double.matrixV().template cast<T>();


    const int m_init = A.rows();
    const int n_init = A.cols();


    if (initial_U.cols() < m_init) {
        Matrix<T> U_ext(m_init, m_init); U_ext.setZero();
        U_ext.leftCols(initial_U.cols()) = initial_U; initial_U = U_ext;
    } else if (initial_U.cols() > m_init) {
         initial_U = initial_U.leftCols(m_init);
    }


    if (initial_V.rows() != n_init || initial_V.cols() != n_init) {
        if (initial_V.rows() == n_init && initial_V.cols() == std::min(m_init, n_init)) {
            Matrix<T> V_ext = Matrix<T>::Identity(n_init, n_init);
            V_ext.leftCols(initial_V.cols()) = initial_V;
            initial_V = V_ext;
            std::cerr << "Warning: Initial V was not square (" << initial_V.rows() << "x" << initial_V.cols()
                      << "), extended to " << n_init << "x" << n_init << "." << std::endl;
        } else {
            throw std::runtime_error("Initial V matrix from BDCSVD has unexpected dimensions!");
        }
    }



    const T noise_level = T("1e-4");
    if (boost::multiprecision::abs(noise_level) > std::numeric_limits<T>::epsilon()) {
        Matrix<T> noise_U = Matrix<T>::Random(initial_U.rows(), initial_U.cols());
        Matrix<T> noise_V = Matrix<T>::Random(initial_V.rows(), initial_V.cols());
        initial_U += noise_level * noise_U;
        initial_V += noise_level * noise_V;
    }


    AOIR_SVD_5<T> result = MSVD_SVD(A, initial_U, initial_V, history_filename_base);


    this->U = result.matrixU();
    this->V = result.matrixV();
    this->S = result.singularValues();
}




template<typename T> void AOIR_SVD_5<T>::Set_U(const Matrix<T>& U_in) { this->U = U_in; }
template<typename T> void AOIR_SVD_5<T>::Set_V(const Matrix<T>& V_in) { this->V = V_in; }

template<typename T> void AOIR_SVD_5<T>::Set_S(const Matrix<T>& Sigma_full) { this->S = Sigma_full; }
template<typename T> Matrix<T> AOIR_SVD_5<T>::matrixU() const { return this->U; }
template<typename T> Matrix<T> AOIR_SVD_5<T>::matrixV() const { return this->V; }
template<typename T> Matrix<T> AOIR_SVD_5<T>::singularValues() const { return this->S; }





template<typename T>
AOIR_SVD_5<T> AOIR_SVD_5<T>::MSVD_SVD(
    const Matrix<T>& A,
    const Matrix<T>& U_init_param,
    const Matrix<T>& V_init_param,
    const std::string& history_filename_base
) {
    const int m = A.rows();
    const int n = A.cols();
    const int m_minus_n = m - n;

    if (m < n) { throw std::runtime_error("MSVD_SVD requires m >= n"); }

    const T eps = std::numeric_limits<T>::epsilon();

    const T threshold = boost::multiprecision::sqrt(eps);


    const T ortho_tolerance = T("1e-98");
    const T sigma_tolerance = T("1e-98");
    int iter_count = 0;

    Matrix<T> U = U_init_param; Matrix<T> V = V_init_param;
    Matrix<T> Sigma = Matrix<T>::Zero(m, n);
    Matrix<T> Sigma_n = Matrix<T>::Zero(n, n);
    Vector<T> sigma(n);


    Matrix<T> Sigma_prev_diag = Matrix<T>::Zero(n, n);
    T sigma_relative_error = std::numeric_limits<T>::infinity();


    std::ofstream history_file;
    std::string history_filename;
    bool log_history = !history_filename_base.empty();
    if (log_history) {
        history_filename = history_filename_base + "_conv5_full.csv";
        history_file.open(history_filename);
        if (history_file) {
             history_file << "Iteration,SigmaRelError,U_OrthoError,V_OrthoError,StepTime_us\n";
        } else {
             log_history = false;
             std::cerr << "Warning: Could not open history file: " << history_filename << std::endl;
        }
    }


    while (true) {
        auto iter_start_time = std::chrono::high_resolution_clock::now();


        Matrix<T> R_full = Matrix<T>::Identity(m, m) - U.transpose() * U;
        Matrix<T> S_full = Matrix<T>::Identity(n, n) - V.transpose() * V;
        T u_ortho_error = R_full.norm();
        T v_ortho_error = S_full.norm();


        if (iter_count > 0) {
             T norm_prev = Sigma_prev_diag.norm();

             if (norm_prev > eps) {
                 sigma_relative_error = (Sigma_n - Sigma_prev_diag).norm() / norm_prev;
             } else {
                 sigma_relative_error = (Sigma_n - Sigma_prev_diag).norm();
             }
        } else {
             sigma_relative_error = std::numeric_limits<T>::infinity();
        }


        if (iter_count > 0) {
            bool ortho_converged = (u_ortho_error <= ortho_tolerance && v_ortho_error <= ortho_tolerance);
            bool sigma_converged = (sigma_relative_error <= sigma_tolerance);

            if (ortho_converged && sigma_converged) {
                break;
            }
        }




        Matrix<T> U1 = U.leftCols(n);
        Matrix<T> U2;
        if (m > n && m_minus_n > 0) {
             U2 = U.rightCols(m_minus_n);
        }


        Matrix<T> P; P.noalias() = A * V;
        Matrix<T> Q; Q.noalias() = A.transpose() * U1;



        Vector<T> r(n), s(n), t(n);

        Sigma_prev_diag = Sigma_n;
        Sigma_n.setZero();
        for (int i = 0; i < n; ++i) {
             r(i) = R_full(i,i);
             s(i) = S_full(i,i);
             t(i) = U1.col(i).dot(P.col(i));

             T denom_sigma = T("1.0") - T("0.5") * (r(i) + s(i));
             Sigma_n(i, i) = (boost::multiprecision::abs(denom_sigma) < eps * T("10.0")) ? t(i) : (t(i) / denom_sigma);
             sigma(i) = Sigma_n(i, i);
        }


        Sigma.setZero();
        Sigma.topLeftCorner(n, n) = Sigma_n;


        Matrix<T> Cy; Cy.noalias() = P - U1 * Sigma_n;
        Matrix<T> C8; C8.noalias() = Q - V * Sigma_n;


        Matrix<T> Ca, Cb;
        {

             const CBLAS_LAYOUT layout = CblasColMajor;
             const CBLAS_TRANSPOSE transA = CblasTrans;
             const CBLAS_TRANSPOSE transB = CblasNoTrans;
             const int M_blas = n; const int N_blas = n; const int K_blas = m;
             const double alpha_blas = 1.0; const double beta_blas = 0.0;
             MatrixLowPrec U1_low = U1.template cast<LowPrec>();
             MatrixLowPrec Cy_low = Cy.template cast<LowPrec>();
             MatrixLowPrec Ca_low_blas(M_blas, N_blas);
             assert(U1_low.rows() == K_blas && U1_low.cols() == M_blas);
             assert(Cy_low.rows() == K_blas && Cy_low.cols() == N_blas);
             assert(Ca_low_blas.rows() == M_blas && Ca_low_blas.cols() == N_blas);
             cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                         U1_low.data(), U1_low.outerStride(),
                         Cy_low.data(), Cy_low.outerStride(),
                         beta_blas,
                         Ca_low_blas.data(), Ca_low_blas.outerStride());
             Ca = Ca_low_blas.template cast<T>();
        }
        {

             const CBLAS_LAYOUT layout = CblasColMajor;
             const CBLAS_TRANSPOSE transA = CblasTrans;
             const CBLAS_TRANSPOSE transB = CblasNoTrans;
             const int M_blas = n; const int N_blas = n; const int K_blas = n;
             const double alpha_blas = 1.0; const double beta_blas = 0.0;
             MatrixLowPrec V_low = V.template cast<LowPrec>();
             MatrixLowPrec C8_low = C8.template cast<LowPrec>();
             MatrixLowPrec Cb_low_blas(M_blas, N_blas);
             assert(V_low.rows() == K_blas && V_low.cols() == M_blas);
             assert(C8_low.rows() == K_blas && C8_low.cols() == N_blas);
             assert(Cb_low_blas.rows() == M_blas && Cb_low_blas.cols() == N_blas);
             cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                         V_low.data(), V_low.outerStride(),
                         C8_low.data(), C8_low.outerStride(),
                         beta_blas,
                         Cb_low_blas.data(), Cb_low_blas.outerStride());
             Cb = Cb_low_blas.template cast<T>();
        }


        Matrix<T> D = Sigma_n * Ca + Cb * Sigma_n;
        Matrix<T> E = Ca * Sigma_n + Sigma_n * Cb;


        Matrix<T> G = Matrix<T>::Zero(n, n);
        Matrix<T> F11 = Matrix<T>::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            T sigma2_i = sigma(i) * sigma(i);

            G(i, i) = T("0.5") * s(i); F11(i, i) = T("0.5") * r(i);
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    T sigma2_j = sigma(j) * sigma(j); T denom = sigma2_j - sigma2_i;

                    if (boost::multiprecision::abs(denom) > threshold * boost::multiprecision::abs(sigma2_j)) {
                         G(i, j) = D(i, j) / denom; F11(i, j) = E(i, j) / denom;
                    } else {
                         G(i, j) = T("0.0"); F11(i, j) = T("0.0");
                    }
                }
            }
        }


        Matrix<T> F = Matrix<T>::Zero(m, m);
        F.topLeftCorner(n, n) = F11;


        if (m > n && m_minus_n > 0) {

            Matrix<T> SigmaInv = Matrix<T>::Identity(n,n);
            bool invertible = true;
            for(int i=0; i<n; ++i) {

                if (boost::multiprecision::abs(sigma(i)) < threshold) {
                    invertible = false;
                    break;
                }

                SigmaInv(i,i) = T("1.0") / sigma(i);
            }
            Matrix<T> F12, F21, F22;
            if (invertible) {

                Matrix<T> PTU2; PTU2.noalias() = P.transpose() * U2;
                F12.noalias() = -SigmaInv * PTU2;


                Matrix<T> U2Cy_high;
                {

                    const CBLAS_LAYOUT layout = CblasColMajor;
                    const CBLAS_TRANSPOSE transA = CblasTrans;
                    const CBLAS_TRANSPOSE transB = CblasNoTrans;
                    const int M_blas = m_minus_n;
                    const int N_blas = n;
                    const int K_blas = m;
                    const double alpha_blas = 1.0; const double beta_blas = 0.0;
                    MatrixLowPrec U2_low = U2.template cast<LowPrec>();
                    MatrixLowPrec Cy_low = Cy.template cast<LowPrec>();
                    MatrixLowPrec U2Cy_low_blas(M_blas, N_blas);
                    assert(U2_low.rows() == K_blas && U2_low.cols() == M_blas);
                    assert(Cy_low.rows() == K_blas && Cy_low.cols() == N_blas);
                    assert(U2Cy_low_blas.rows() == M_blas && U2Cy_low_blas.cols() == N_blas);
                    cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                                U2_low.data(), U2_low.outerStride(),
                                Cy_low.data(), Cy_low.outerStride(),
                                beta_blas,
                                U2Cy_low_blas.data(), U2Cy_low_blas.outerStride());
                    U2Cy_high = U2Cy_low_blas.template cast<T>();
                }
                F21.noalias() = U2Cy_high * SigmaInv;
            } else {

                F12 = Matrix<T>::Zero(n, m_minus_n);
                F21 = Matrix<T>::Zero(m_minus_n, n);
            }


            F22.noalias() = T("0.5") * (Matrix<T>::Identity(m_minus_n, m_minus_n) - U2.transpose() * U2);


            F.block(0, n, n, m_minus_n) = F12;
            F.block(n, 0, m_minus_n, n) = F21;
            F.block(n, n, m_minus_n, m_minus_n) = F22;
        }


        {
            Matrix<T> UpdateU_high;
            {

                const CBLAS_LAYOUT layout = CblasColMajor;
                const CBLAS_TRANSPOSE transA = CblasNoTrans;
                const CBLAS_TRANSPOSE transB = CblasNoTrans;
                const int M_blas = m; const int N_blas = m; const int K_blas = m;
                const double alpha_blas = 1.0; const double beta_blas = 0.0;
                MatrixLowPrec U_low = U.template cast<LowPrec>();
                MatrixLowPrec F_low = F.template cast<LowPrec>();
                MatrixLowPrec UpdateU_low_blas(M_blas, N_blas);
                assert(U_low.rows() == M_blas && U_low.cols() == K_blas);
                assert(F_low.rows() == K_blas && F_low.cols() == N_blas);
                assert(UpdateU_low_blas.rows() == M_blas && UpdateU_low_blas.cols() == N_blas);
                cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                            U_low.data(), U_low.outerStride(), F_low.data(), F_low.outerStride(),
                            beta_blas, UpdateU_low_blas.data(), UpdateU_low_blas.outerStride());
                UpdateU_high = UpdateU_low_blas.template cast<T>();
            }
            U += UpdateU_high;
        }
        {
            Matrix<T> UpdateV_high;
             {

                const CBLAS_LAYOUT layout = CblasColMajor;
                const CBLAS_TRANSPOSE transA = CblasNoTrans;
                const CBLAS_TRANSPOSE transB = CblasNoTrans;
                const int M_blas = n; const int N_blas = n; const int K_blas = n;
                const double alpha_blas = 1.0; const double beta_blas = 0.0;
                MatrixLowPrec V_low = V.template cast<LowPrec>();
                MatrixLowPrec G_low = G.template cast<LowPrec>();
                MatrixLowPrec UpdateV_low_blas(M_blas, N_blas);
                assert(V_low.rows() == M_blas && V_low.cols() == K_blas);
                assert(G_low.rows() == K_blas && G_low.cols() == N_blas);
                assert(UpdateV_low_blas.rows() == M_blas && UpdateV_low_blas.cols() == N_blas);
                cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                            V_low.data(), V_low.outerStride(), G_low.data(), G_low.outerStride(),
                            beta_blas, UpdateV_low_blas.data(), UpdateV_low_blas.outerStride());
                UpdateV_high = UpdateV_low_blas.template cast<T>();
            }
            V += UpdateV_high;
        }



        auto iter_end_time = std::chrono::high_resolution_clock::now();
        auto iter_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(iter_end_time - iter_start_time).count();



        if (log_history && history_file) {
             const int log_precision = std::numeric_limits<T>::max_digits10;
             history_file << iter_count << ","
                          << std::scientific << std::setprecision(log_precision) << sigma_relative_error << ","
                          << std::scientific << std::setprecision(log_precision) << u_ortho_error << ","
                          << std::scientific << std::setprecision(log_precision) << v_ortho_error << ","
                          << iter_duration_us << "\n";
        }


        iter_count++;

    }



    if (log_history && history_file && history_file.is_open()) {

        Matrix<T> R_final = Matrix<T>::Identity(m, m) - U.transpose() * U;
        Matrix<T> S_final = Matrix<T>::Identity(n, n) - V.transpose() * V;
        T u_ortho_final = R_final.norm();
        T v_ortho_final = S_final.norm();

        const int log_precision = std::numeric_limits<T>::max_digits10;
        history_file << iter_count << ","
                     << std::scientific << std::setprecision(log_precision) << sigma_relative_error << ","
                     << std::scientific << std::setprecision(log_precision) << u_ortho_final << ","
                     << std::scientific << std::setprecision(log_precision) << v_ortho_final << ","
                     << 0 << "\n";
    }




    if (log_history && history_file) {
        history_file.close();
    }


    AOIR_SVD_5<T> result;
    result.Set_U(U);
    result.Set_V(V);


    result.Set_S(Sigma);
    return result;
}


}

#endif