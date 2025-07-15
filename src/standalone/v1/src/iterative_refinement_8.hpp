#ifndef ITERATIVE_REFINEMENT_8_HPP
#define ITERATIVE_REFINEMENT_8_HPP

#include "iterative_refinement_8.h"
#include <Eigen/Dense>
#include <limits>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <cassert>
#include <type_traits>


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

using MatrixLP = Eigen::Matrix<LowPrec, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;






template<typename T>
AOIR_SVD_8<T>::AOIR_SVD_8() {}



template<typename T>
AOIR_SVD_8<T>::AOIR_SVD_8(const Matrix<T>& A, const std::string& history_filename_base) {



    Eigen::MatrixXd A_double = A.template cast<double>();
    if (A.rows() == 0 || A.cols() == 0) {
         throw std::runtime_error("Input matrix A is empty in AOIR_SVD_8 constructor.");
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


    AOIR_SVD_8<T> result = MSVD_SVD(A, initial_U, initial_V, history_filename_base);


    this->U = result.matrixU();
    this->V = result.matrixV();
    this->S = result.singularValues();
}



template<typename T> void AOIR_SVD_8<T>::Set_U(const Matrix<T>& U_in) { this->U = U_in; }
template<typename T> void AOIR_SVD_8<T>::Set_V(const Matrix<T>& V_in) { this->V = V_in; }

template<typename T> void AOIR_SVD_8<T>::Set_S(const Matrix<T>& Sigma_full) { this->S = Sigma_full; }
template<typename T> Matrix<T> AOIR_SVD_8<T>::matrixU() const { return this->U; }
template<typename T> Matrix<T> AOIR_SVD_8<T>::matrixV() const { return this->V; }
template<typename T> Matrix<T> AOIR_SVD_8<T>::singularValues() const { return this->S; }






template<typename T>
AOIR_SVD_8<T> AOIR_SVD_8<T>::MSVD_SVD(
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


    const T sqrt2 = boost::multiprecision::sqrt(T("2.0"));
    const T inv_sqrt2 = T("1.0") / sqrt2;
    const T one = T("1.0");
    const T two = T("2.0");
    const T half = T("0.5");
    const T ten = T("10.0");


    std::ofstream history_file;
    std::string history_filename;
    bool log_history = !history_filename_base.empty();
    if (log_history) {
        history_filename = history_filename_base + "_conv8_full.csv";
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
        Matrix<T> S_full_ortho = Matrix<T>::Identity(n, n) - V.transpose() * V;
        T u_ortho_error = R_full.norm();
        T v_ortho_error = S_full_ortho.norm();


        Matrix<T> U1 = U.leftCols(n);
        Matrix<T> P; P.noalias() = A * V;
        Matrix<T> Q; Q.noalias() = A.transpose() * U1;



        Vector<T> r(n), t(n);
        Sigma_prev_diag = Sigma_n;
        Sigma_n.setZero();
        for (int i = 0; i < n; ++i) {
             T u_norm_sq = U.col(i).squaredNorm();
             T v_norm_sq = V.col(i).squaredNorm();
             r(i) = one - (u_norm_sq + v_norm_sq) / two;
             t(i) = U1.col(i).dot(P.col(i));
             T denom_sigma = one - r(i);
             Sigma_n(i, i) = (boost::multiprecision::abs(denom_sigma) < eps * ten) ? t(i) : (t(i) / denom_sigma);
             sigma(i) = Sigma_n(i, i);
        }


        Sigma.setZero();
        Sigma.topLeftCorner(n, n) = Sigma_n;


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
            if (ortho_converged && sigma_converged) { break; }
        }




        Matrix<T> U2;
        if (m > n && m_minus_n > 0) {
             U2 = U.rightCols(m_minus_n);
        } else {
             U2.resize(m, 0);
        }


        Matrix<T> P1; P1.noalias() = Q - V * Sigma_n;
        Matrix<T> P2; P2.noalias() = P - U1 * Sigma_n;


        Matrix<T> P3, P4;
        {
            const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
            const int M_blas = n; const int N_blas = n; const int K_blas = n;
            const double alpha_blas = 1.0; const double beta_blas = 0.0;
            MatrixLP V_low = V.template cast<LowPrec>(); MatrixLP P1_low = P1.template cast<LowPrec>(); MatrixLP P3_low_blas(M_blas, N_blas);
            assert(V_low.rows() == K_blas && V_low.cols() == M_blas); assert(P1_low.rows() == K_blas && P1_low.cols() == N_blas); assert(P3_low_blas.rows() == M_blas && P3_low_blas.cols() == N_blas);
            cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas, V_low.data(), V_low.outerStride(), P1_low.data(), P1_low.outerStride(), beta_blas, P3_low_blas.data(), P3_low_blas.outerStride());
            P3 = P3_low_blas.template cast<T>();
        }
        {
            const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
            const int M_blas = n; const int N_blas = n; const int K_blas = m;
            const double alpha_blas = 1.0; const double beta_blas = 0.0;
            MatrixLP U1_low = U1.template cast<LowPrec>(); MatrixLP P2_low = P2.template cast<LowPrec>(); MatrixLP P4_low_blas(M_blas, N_blas);
            assert(U1_low.rows() == K_blas && U1_low.cols() == M_blas); assert(P2_low.rows() == K_blas && P2_low.cols() == N_blas); assert(P4_low_blas.rows() == M_blas && P4_low_blas.cols() == N_blas);
            cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas, U1_low.data(), U1_low.outerStride(), P2_low.data(), P2_low.outerStride(), beta_blas, P4_low_blas.data(), P4_low_blas.outerStride());
            P4 = P4_low_blas.template cast<T>();
        }


        Matrix<T> Q1 = (half * (P3 + P4));
        Matrix<T> Q2 = (half * (P3 - P4));


        Matrix<T> Q3, Q4;
        if (m > n && m_minus_n > 0) {

            Matrix<T> PTU2; PTU2.noalias() = P.transpose() * U2;
            Q3 = inv_sqrt2 * PTU2;


            Matrix<T> U2P2_high;
            {
                const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
                const int M_blas = m_minus_n; const int N_blas = n; const int K_blas = m;
                const double alpha_blas = 1.0; const double beta_blas = 0.0;
                MatrixLP U2_low = U2.template cast<LowPrec>(); MatrixLP P2_low = P2.template cast<LowPrec>(); MatrixLP U2P2_low_blas(M_blas, N_blas);
                assert(U2_low.rows() == K_blas && U2_low.cols() == M_blas); assert(P2_low.rows() == K_blas && P2_low.cols() == N_blas); assert(U2P2_low_blas.rows() == M_blas && U2P2_low_blas.cols() == N_blas);
                cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas, U2_low.data(), U2_low.outerStride(), P2_low.data(), P2_low.outerStride(), beta_blas, U2P2_low_blas.data(), U2P2_low_blas.outerStride());
                U2P2_high = U2P2_low_blas.template cast<T>();
            }
            Q4 = inv_sqrt2 * U2P2_high;
        } else {
             Q3.resize(n, 0); Q3.setZero();
             Q4.resize(0, n); Q4.setZero();
        }


        Matrix<T> E1 = Matrix<T>::Zero(n, n); Matrix<T> E2 = Matrix<T>::Zero(n, n);
        Matrix<T> E3 = Matrix<T>::Zero(n, m_minus_n); Matrix<T> E4 = Matrix<T>::Zero(m_minus_n, n);
        Matrix<T> E5 = Matrix<T>::Zero(m_minus_n, m_minus_n);

        for(int i=0; i<n; ++i) {
            E1(i,i) = r(i) / two;
            for(int j=0; j<n; ++j) {
                if (i == j) {
                     E2(i,j) = T("0.0");
                     continue;
                }
                T sigma_i = sigma(i); T sigma_j = sigma(j);
                T denom1 = sigma_j - sigma_i;
                T denom2 = sigma_i + sigma_j;

                if (boost::multiprecision::abs(denom1) > eps * (boost::multiprecision::abs(sigma_i) + boost::multiprecision::abs(sigma_j))) {
                    E1(i,j) = Q1(i,j) / denom1;
                }

                if (denom2 > eps) {
                    E2(i,j) = Q2(i,j) / denom2;
                }
            }
        }

        if (m > n && m_minus_n > 0) {

            Matrix<T> SigmaInv = Matrix<T>::Identity(n,n);
            bool invertible = true;
            for(int i=0; i<n; ++i) {
                if (boost::multiprecision::abs(sigma(i)) < threshold) { invertible = false; break; }
                SigmaInv(i,i) = one / sigma(i);
            }
            if (invertible) {
                 E3.noalias() = SigmaInv * Q3;
                 E4.noalias() = Q4 * SigmaInv;
            }


            Matrix<T> R22 = R_full.bottomRightCorner(m_minus_n, m_minus_n);
            E5.noalias() = half * R22;
        }







        {
            Matrix<T> E1pE2 = E1 + E2;
            Matrix<T> UpdateV_high;
            {
                const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasNoTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
                const int M_blas = n; const int N_blas = n; const int K_blas = n;
                const double alpha_blas = 1.0; const double beta_blas = 0.0;
                MatrixLP V_low = V.template cast<LowPrec>(); MatrixLP E1pE2_low = E1pE2.template cast<LowPrec>(); MatrixLP UpdateV_low_blas(M_blas, N_blas);
                assert(V_low.rows() == M_blas && V_low.cols() == K_blas); assert(E1pE2_low.rows() == K_blas && E1pE2_low.cols() == N_blas); assert(UpdateV_low_blas.rows() == M_blas && UpdateV_low_blas.cols() == N_blas);
                cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas, V_low.data(), V_low.outerStride(), E1pE2_low.data(), E1pE2_low.outerStride(), beta_blas, UpdateV_low_blas.data(), UpdateV_low_blas.outerStride());
                UpdateV_high = UpdateV_low_blas.template cast<T>();
            }
            V += UpdateV_high;
        }


        {
            Matrix<T> E1mE2 = E1 - E2;
            Matrix<T> Term1_high, Term2_high;
            {
                const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasNoTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
                const int M_blas = m; const int N_blas = n; const int K_blas = n;
                const double alpha_blas = 1.0; const double beta_blas = 0.0;
                MatrixLP U1_low_upd = U1.template cast<LowPrec>(); MatrixLP E1mE2_low = E1mE2.template cast<LowPrec>(); MatrixLP Term1_low_blas(M_blas, N_blas);
                assert(U1_low_upd.rows() == M_blas && U1_low_upd.cols() == K_blas); assert(E1mE2_low.rows() == K_blas && E1mE2_low.cols() == N_blas); assert(Term1_low_blas.rows() == M_blas && Term1_low_blas.cols() == N_blas);
                cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas, U1_low_upd.data(), U1_low_upd.outerStride(), E1mE2_low.data(), E1mE2_low.outerStride(), beta_blas, Term1_low_blas.data(), Term1_low_blas.outerStride());
                Term1_high = Term1_low_blas.template cast<T>();
            }
            if (m > n && m_minus_n > 0) {

                Matrix<T> sqrt2_U2 = sqrt2 * U2;
                Matrix<T> E4_T = E4;
                {
                    const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasNoTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
                    const int M_blas = m; const int N_blas = n; const int K_blas = m_minus_n;
                    const double alpha_blas = 1.0; const double beta_blas = 0.0;
                    MatrixLP sqrt2_U2_low = sqrt2_U2.template cast<LowPrec>();
                    MatrixLP E4_low = E4_T.template cast<LowPrec>();
                    MatrixLP Term2_low_blas(M_blas, N_blas);
                    assert(sqrt2_U2_low.rows() == M_blas && sqrt2_U2_low.cols() == K_blas);
                    assert(E4_low.rows() == K_blas && E4_low.cols() == N_blas);
                    assert(Term2_low_blas.rows() == M_blas && Term2_low_blas.cols() == N_blas);
                    cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                                sqrt2_U2_low.data(), sqrt2_U2_low.outerStride(),
                                E4_low.data(), E4_low.outerStride(),
                                beta_blas,
                                Term2_low_blas.data(), Term2_low_blas.outerStride());
                    Term2_high = Term2_low_blas.template cast<T>();
                }
            } else {
                Term2_high = Matrix<T>::Zero(m, n);
            }
            U1 += (Term1_high + Term2_high);
        }


        if (m > n && m_minus_n > 0) {
             Matrix<T> Term3_high, Term4_high;
             {
                const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasNoTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
                const int M_blas = m; const int N_blas = m_minus_n; const int K_blas = m_minus_n;
                const double alpha_blas = 1.0; const double beta_blas = 0.0;
                MatrixLP U2_low_upd = U2.template cast<LowPrec>(); MatrixLP E5_low = E5.template cast<LowPrec>(); MatrixLP Term3_low_blas(M_blas, N_blas);
                assert(U2_low_upd.rows() == M_blas && U2_low_upd.cols() == K_blas); assert(E5_low.rows() == K_blas && E5_low.cols() == N_blas); assert(Term3_low_blas.rows() == M_blas && Term3_low_blas.cols() == N_blas);
                cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas, U2_low_upd.data(), U2_low_upd.outerStride(), E5_low.data(), E5_low.outerStride(), beta_blas, Term3_low_blas.data(), Term3_low_blas.outerStride());
                Term3_high = Term3_low_blas.template cast<T>();
             }
             {
                 Matrix<T> sqrt2_U1 = sqrt2 * U1;
                 Matrix<T> E3_T = E3;
                 {
                     const CBLAS_LAYOUT layout = CblasColMajor; const CBLAS_TRANSPOSE transA = CblasNoTrans; const CBLAS_TRANSPOSE transB = CblasNoTrans;
                     const int M_blas = m; const int N_blas = m_minus_n; const int K_blas = n;
                     const double alpha_blas = 1.0; const double beta_blas = 0.0;
                     MatrixLP sqrt2_U1_low = sqrt2_U1.template cast<LowPrec>();
                     MatrixLP E3_low = E3_T.template cast<LowPrec>();
                     MatrixLP Term4_low_blas(M_blas, N_blas);
                     assert(sqrt2_U1_low.rows() == M_blas && sqrt2_U1_low.cols() == K_blas);
                     assert(E3_low.rows() == K_blas && E3_low.cols() == N_blas);
                     assert(Term4_low_blas.rows() == M_blas && Term4_low_blas.cols() == N_blas);
                     cblas_dgemm(layout, transA, transB, M_blas, N_blas, K_blas, alpha_blas,
                                 sqrt2_U1_low.data(), sqrt2_U1_low.outerStride(),
                                 E3_low.data(), E3_low.outerStride(),
                                 beta_blas,
                                 Term4_low_blas.data(), Term4_low_blas.outerStride());
                     Term4_high = Term4_low_blas.template cast<T>();
                 }
             }
             U2 += (Term3_high - Term4_high);
        }

        U.leftCols(n) = U1;
        if (m > n && m_minus_n > 0) {
             U.rightCols(m_minus_n) = U2;
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
        Matrix<T> S_final_ortho = Matrix<T>::Identity(n, n) - V.transpose() * V;
        T u_ortho_final = R_final.norm();
        T v_ortho_final = S_final_ortho.norm();
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


    AOIR_SVD_8<T> result;
    result.Set_U(U);
    result.Set_V(V);

    result.Set_S(Sigma);
    return result;
}


}

#endif