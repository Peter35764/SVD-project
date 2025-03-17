#ifndef TESTING_H
#define TESTING_H

#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <cassert>
#include "generate_svd.h"
#include "dqds.h"

template<typename T, template <typename> class gen_cl, template <typename> class svd_cl>
void svd_test_func(std::string fileName, const std::vector<T>& SigmaMaxMinRatiosVec, 
                  const std::vector<std::pair<int,int>>& MatSizesVec, const int n) {
    auto printTable = [](std::ostream& out, const std::vector<std::vector<std::string>>& data) {
        if (data.empty()) return;
        std::vector<size_t> widths;
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (widths.size() <= i) widths.push_back(0);
                widths[i] = std::max(widths[i], row[i].size());
            }
        }
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << std::left << std::setw(widths[i] + 3) << row[i];
            }
            out << "\n";
        }
    };

    auto num2str = [](T value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    };

    T generalSum = 0;
    for (const auto& MatSize : MatSizesVec) {
        generalSum += (MatSize.first * MatSize.second);
    }

    using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorDynamic = Eigen::Vector<T, Eigen::Dynamic>;

    const std::vector<std::pair<T,T>> Intervals = {{0,1}, {1,100}};
    std::random_device rd;
    std::default_random_engine gen(rd());
    
    std::vector<std::vector<std::string>> table = {{"Dimension", "Sigma-max/min-ratio", "SV interval", 
                                                   "AVG ||I-U_t*U||", "AVG ||I-U*U_t||", "AVG ||I-V_t*V||",
                                                   "AVG ||I-V*V_t||", "AVG relative err. sigma"}};

    T ProgressCoeff = n * Intervals.size() * SigmaMaxMinRatiosVec.size() * generalSum / 100.0;
    T progress = 0;

    MatrixDynamic U_true, S_true, V_true, U_calc, V_calc, V_calc_transpose;
    VectorDynamic SV_calc;

    for (const auto& MatSize : MatSizesVec) {
        const int N = MatSize.first;
        const int M = MatSize.second;
        int minNM = std::min(N, M);

        U_true.resize(N,N); U_calc.resize(N,N);
        S_true.resize(N,M); SV_calc.resize(minNM);
        V_true.resize(M,M); V_calc.resize(M,M); V_calc_transpose.resize(M,M);

        for (const auto& SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
            for (const auto& interval : Intervals) {
                assert((interval.first < interval.second, "\nError: left boundary >= right boundary\n"));
                assert((interval.first * SigmaMaxMinRatio <= interval.second, 
                       "\nError: no sigma values exist with such ratio in such interval\n"));

                std::uniform_real_distribution<T> distrSigmaMin(interval.first, interval.second/SigmaMaxMinRatio);
                T sigma_min = distrSigmaMin(gen);
                T sigma_max = SigmaMaxMinRatio * sigma_min;

                std::uniform_real_distribution<T> distr(sigma_min, sigma_max);
                assert((minNM >= 2, "\nError: no columns or rows allowed\n"));

                T avg_dev_UUt = 0, avg_dev_UtU = 0, avg_dev_VVt = 0, avg_dev_VtV = 0, avg_relErr_sigma = 0;

                for (int i = 1; i <= n; ++i) {
                    gen_cl<T> svd_gen(N, M, gen, distr, true);
                    svd_gen.generate(minNM);

                    U_true = svd_gen.MatrixU(); S_true = svd_gen.MatrixS(); V_true = svd_gen.MatrixV();
                    svd_cl<MatrixDynamic> svd_func((U_true * S_true * V_true.transpose()).eval(), 
                                                 Eigen::ComputeFullU | Eigen::ComputeFullV);
                    U_calc = svd_func.matrixU(); SV_calc = svd_func.singularValues(); V_calc = svd_func.matrixV();
                    avg_dev_UUt += (MatrixDynamic::Identity(N,N) - U_calc*U_calc.transpose()).squaredNorm()/n;
                    avg_dev_UtU += (MatrixDynamic::Identity(N,N) - U_calc.transpose()*U_calc).squaredNorm()/n;
                    avg_dev_VVt += (MatrixDynamic::Identity(M,M) - V_calc*V_calc.transpose()).squaredNorm()/n;
                    avg_dev_VtV += (MatrixDynamic::Identity(M,M) - V_calc.transpose()*V_calc).squaredNorm()/n;
                    avg_relErr_sigma += (S_true.diagonal() - SV_calc).cwiseQuotient(S_true.diagonal()).cwiseAbs().maxCoeff()/n;

                    progress += M*N/ProgressCoeff;
                    std::cout << "\n" + num2str(progress) + "% was done.\n";
                }

                table.emplace_back(std::vector<std::string>{num2str(N) + "x" + num2str(M), num2str(SigmaMaxMinRatio), 
                                   "[" + num2str(interval.first) + ", " + num2str(interval.second) + "]",
                                   num2str(avg_dev_UUt), num2str(avg_dev_UtU), 
                                   num2str(avg_dev_VVt), num2str(avg_dev_VtV), 
                                   num2str(avg_relErr_sigma)});
            }
        }
    }

    std::ofstream file(fileName);
    if (file) {
        printTable(file, table);
        file.close();
    } else {
        std::cerr << "Error while creating/opening file!\n";
    }
}

#endif // TESTING_H
