#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <random>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <limits>

#include <array>


#include <boost/multiprecision/cpp_dec_float.hpp>

#include "config.h"
#include "iterative_refinement_8.h"
#include "generate_svd.h"


using float128 = boost::multiprecision::cpp_dec_float_100;

const std::array<float128, 1> sigma_ratio = {float128("5000")};
constexpr std::array<std::pair<int, int>, 1> matrix_size = {{{20, 15}}};

constexpr int matrix_num_for_sample_averaging = 1;

std::mutex cout_mutex;



template<typename Derived>
Eigen::Vector<typename Derived::Scalar, Eigen::Dynamic> extract_singular_values(const Derived& sv) {

    if (sv.rows() == 0 || sv.cols() == 0) {

        return Eigen::Vector<typename Derived::Scalar, Eigen::Dynamic>();
    }
    return sv.diagonal().head(std::min(sv.rows(), sv.cols()));
}



template<typename MatrixType>
class AOIRWrapper_8 {
public:
    AOIRWrapper_8(const MatrixType& A,
                  const std::string& history_filename_base = "",

                  unsigned int flags = Eigen::ComputeFullU | Eigen::ComputeFullV) {

        impl = SVD_Project::AOIR_SVD_8<typename MatrixType::Scalar>(A, history_filename_base);
    }

    auto matrixU() const        { return impl.matrixU(); }
    auto matrixV() const        { return impl.matrixV(); }

    auto singularValues() const { return extract_singular_values(impl.singularValues()); }
private:
    SVD_Project::AOIR_SVD_8<typename MatrixType::Scalar> impl;
};



template<typename T, template <typename> class gen_cl, template <typename> class svd_cl>
void svd_test_func(std::string fileName,
                   const std::vector<T>& SigmaMaxMinRatiosVec,
                   const std::vector<std::pair<int,int>>& MatSizesVec,
                   const int n,
                   const std::string& algorithmName,
                   int lineNumber) {


    auto printTable = [](std::ostream& out, const std::vector<std::vector<std::string>>& data){
        if (data.empty()) return;
        std::vector<size_t> widths;

        for (const auto &row : data) {
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


    auto printCSV = [](std::ostream& out, const std::vector<std::vector<std::string>>& data) {
        for (const auto& row : data) {
            bool first = true;
            for (const auto& cell : row) {
                if (!first) out << ",";
                std::string cellFormatted = cell;

                size_t pos = cellFormatted.find('"');
                while (pos != std::string::npos) {
                    cellFormatted.replace(pos, 1, "\"\"");
                    pos = cellFormatted.find('"', pos + 2);
                }

                if (cellFormatted.find(',') != std::string::npos ||
                    cellFormatted.find('"') != std::string::npos ||
                    cellFormatted.find('\n') != std::string::npos) {
                    cellFormatted = "\"" + cellFormatted + "\"";
                }
                out << cellFormatted;
                first = false;
            }
            out << "\n";
        }
    };


    auto num2str = [](T value){
        std::ostringstream oss;


        const int print_precision = std::numeric_limits<T>::max_digits10;
        oss << std::scientific << std::setprecision(print_precision) << value;
        return oss.str();
    };


    using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorDynamic = Eigen::Vector<T, Eigen::Dynamic>;


    const std::vector<std::pair<T,T>> Intervals = {{T("1e-5"), T("1.0")}};


    std::random_device rd;
    std::mt19937 gen(rd());


    std::vector<std::vector<std::string>> table = {
        {"Dimension", "Sigma-max/min-ratio", "SV interval",
         "AVG ||I-U*U_t||", "AVG ||I-U_t*U||", "AVG ||I-V*V_t||",
         "AVG ||I-V_t*V||", "AVG relative err. sigma"}
    };


    const size_t total_combinations = MatSizesVec.size() * SigmaMaxMinRatiosVec.size() * Intervals.size();
    size_t combinations_done = 0;


    MatrixDynamic U_true, S_true_mat, V_true, U_calc, V_calc, A_gen;
    VectorDynamic SV_calc, SV_true;


    for (const auto& MatSize : MatSizesVec) {
        const int N = MatSize.first;
        const int M = MatSize.second;
        const int minNM = std::min(N, M);


        U_true.resize(N, N); U_calc.resize(N, N);
        S_true_mat.resize(N, M);
        SV_calc.resize(minNM); SV_true.resize(minNM);
        V_true.resize(M, M); V_calc.resize(M, M);
        A_gen.resize(N, M);

        for (const auto& SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
            for (const auto& interval : Intervals) {


                 if (interval.first <= T("0.0") || interval.second <= interval.first || interval.first * SigmaMaxMinRatio > interval.second) {

                     std::lock_guard<std::mutex> lock(cout_mutex);
                     std::cerr << "⚠️ Skipping invalid interval/ratio combination for ["
                               << interval.first << ", " << interval.second
                               << "] with ratio " << SigmaMaxMinRatio << "\n";
                     continue;
                 }
                 T upper_bound = interval.second / SigmaMaxMinRatio;
                 if (interval.first > upper_bound) {
                     std::lock_guard<std::mutex> lock(cout_mutex);
                     std::cerr << "⚠️ Skipping invalid interval/ratio (lower > upper/ratio) for ["
                               << interval.first << ", " << interval.second
                               << "] with ratio " << SigmaMaxMinRatio << "\n";
                     continue;
                 }


                std::uniform_real_distribution<double> distrSigmaMin_double(
                    interval.first.template convert_to<double>(),
                    upper_bound.template convert_to<double>()
                );



                T avg_dev_UUt = T("0.0"), avg_dev_UtU = T("0.0"), avg_dev_VVt = T("0.0");
                T avg_dev_VtV = T("0.0"), avg_relErr_sigma = T("0.0");


                for (int iter = 0; iter < n; ++iter) {



                    T target_sigma_min = interval.first;
                    T target_sigma_max = target_sigma_min * SigmaMaxMinRatio;



                    gen_cl<T> svd_gen(N, M, target_sigma_min, target_sigma_max);

                    svd_gen.generate();

                    U_true = svd_gen.MatrixU();
                    S_true_mat = svd_gen.MatrixS();
                    V_true = svd_gen.MatrixV();


                    A_gen.noalias() = U_true * S_true_mat * V_true.transpose();


                    std::ostringstream history_base_stream;

                     const int name_precision = 3;
                     history_base_stream << "log_A1_" << N << "x" << M
                                       << "_r" << std::fixed << std::setprecision(name_precision) << SigmaMaxMinRatio
                                       << "_i" << std::scientific << std::setprecision(name_precision) << interval.first
                                       << "_n" << iter;
                    std::string history_base_name = history_base_stream.str();

                    std::replace(history_base_name.begin(), history_base_name.end(), '.', '_');
                    std::replace(history_base_name.begin(), history_base_name.end(), ',', '_');
                    std::replace(history_base_name.begin(), history_base_name.end(), '+', 'p');
                    std::replace(history_base_name.begin(), history_base_name.end(), '-', 'm');



                    svd_cl<MatrixDynamic> svd_func(A_gen, history_base_name, Eigen::ComputeFullU | Eigen::ComputeFullV);


                    U_calc = svd_func.matrixU();
                    SV_calc = svd_func.singularValues();
                    V_calc = svd_func.matrixV();


                    MatrixDynamic I_N = MatrixDynamic::Identity(N, N);
                    MatrixDynamic I_M = MatrixDynamic::Identity(M, M);
                    T dev_UUt = (I_N - U_calc * U_calc.transpose()).norm();
                    T dev_UtU = (I_N - U_calc.transpose() * U_calc).norm();
                    T dev_VVt = (I_M - V_calc * V_calc.transpose()).norm();
                    T dev_VtV = (I_M - V_calc.transpose() * V_calc).norm();


                    SV_true = S_true_mat.diagonal().head(minNM);



                    T relErr_sigma_sum = T("0.0");
                    int valid_sigma_count = 0;
                    for (int k = 0; k < minNM; ++k) {
                        T true_sv = (k < SV_true.size()) ? SV_true(k) : T("0.0");
                        T calc_sv = (k < SV_calc.size()) ? SV_calc(k) : T("0.0");

                        if (boost::multiprecision::abs(true_sv) > std::numeric_limits<T>::epsilon() * T("100.0")) {
                            relErr_sigma_sum += boost::multiprecision::abs(calc_sv - true_sv) / boost::multiprecision::abs(true_sv);
                            valid_sigma_count++;
                        } else if (boost::multiprecision::abs(calc_sv) > std::numeric_limits<T>::epsilon() * T("100.0")) {

                        }

                    }
                    T relErr_sigma_avg_iter = (valid_sigma_count > 0) ? (relErr_sigma_sum / valid_sigma_count) : T("0.0");


                    avg_dev_UUt += dev_UUt; avg_dev_UtU += dev_UtU; avg_dev_VVt += dev_VVt;
                    avg_dev_VtV += dev_VtV; avg_relErr_sigma += relErr_sigma_avg_iter;

                }


                if (n > 0) {
                     avg_dev_UUt /= n; avg_dev_UtU /= n; avg_dev_VVt /= n;
                     avg_dev_VtV /= n; avg_relErr_sigma /= n;
                 } else {

                     avg_dev_UUt = T("0.0"); avg_dev_UtU = T("0.0"); avg_dev_VVt = T("0.0");
                     avg_dev_VtV = T("0.0"); avg_relErr_sigma = T("0.0");
                 }


                combinations_done++;
                double percent = (total_combinations > 0) ? (static_cast<double>(combinations_done) * 100.0 / total_combinations) : 100.0;
                int barWidth = 50;
                int pos = (total_combinations > 0) ? static_cast<int>(barWidth * combinations_done / total_combinations) : barWidth;
                pos = std::min(pos, barWidth);

                std::ostringstream progressStream;
                progressStream << algorithmName << ": "
                               << std::fixed << std::setprecision(2) << percent
                               << "% [";
                for (int j = 0; j < barWidth; ++j) {
                    if (j < pos) progressStream << "=";
                    else progressStream << " ";
                }
                progressStream << "]";

                {
                    std::lock_guard<std::mutex> lock(cout_mutex);

                    std::cout << "\033[" << lineNumber << ";0H"
                              << progressStream.str()
                              << "\033[K"
                              << std::flush;
                }


                table.push_back(std::vector<std::string>{
                    std::to_string(N) + "x" + std::to_string(M),
                    num2str(SigmaMaxMinRatio),
                    "[" + num2str(interval.first) + ", " + num2str(interval.second) + "]",
                    num2str(avg_dev_UUt), num2str(avg_dev_UtU),
                    num2str(avg_dev_VVt), num2str(avg_dev_VtV),
                    num2str(avg_relErr_sigma)
                });

            }
        }
    }


    std::ofstream file(fileName);
    if (file) {
         printTable(file, table);
         file.close();
    } else {
         std::lock_guard<std::mutex> lock(cout_mutex);
         std::cerr << "Error opening file " << fileName << "!\n";
    }

    std::string csvFileName = fileName;
    size_t pos_dot = csvFileName.rfind(".txt");
    if (pos_dot != std::string::npos) {
         csvFileName.replace(pos_dot, 4, ".csv");
    } else {
         csvFileName += ".csv";
    }
    std::ofstream csv_file(csvFileName);
    if (csv_file) {
         printCSV(csv_file, table);
         csv_file.close();
    } else {
         std::lock_guard<std::mutex> lock(cout_mutex);
         std::cerr << "Error opening file " << csvFileName << "!\n";
    }
};



const std::vector<float128> sigma_ratio_vec(sigma_ratio.begin(), sigma_ratio.end());
const std::vector<std::pair<int, int>> matrix_size_vec(matrix_size.begin(), matrix_size.end());



int main() {
    auto start = std::chrono::high_resolution_clock::now();


    std::cout << "\033[2J\033[H";
    std::cout << "\033[?25l";

    std::vector<std::pair<std::string, double>> test_times;
    int next_line_for_progress = 1;


    std::string algo_name_8 = "Iterative Refinement 8 SVD (float128)";
    std::string file_name_8 = "iterative_refinement_8_float128_table.txt";
    auto t_start_8 = std::chrono::high_resolution_clock::now();



    svd_test_func<float128, SVDGenerator, AOIRWrapper_8>(file_name_8,
        sigma_ratio_vec, matrix_size_vec,
        matrix_num_for_sample_averaging, algo_name_8, next_line_for_progress++);

    auto t_end_8 = std::chrono::high_resolution_clock::now();
    double duration_algo_8 = std::chrono::duration<double>(t_end_8 - t_start_8).count();
    test_times.emplace_back(algo_name_8, duration_algo_8);



    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGlobal = end - start;


    std::cout << "\033[" << (next_line_for_progress + 1) << ";0H";


    std::cout << "\nExecution time for " << algo_name_8 << " = "
              << std::fixed << std::setprecision(4) << duration_algo_8 << " seconds.\n";

    std::cout << "Total execution time = "
              << std::fixed << std::setprecision(4) << durationGlobal.count() << " seconds.\n";



    std::string time_filename = "test_times_algo1_float128.txt";
    std::ofstream timeFile(time_filename);
    if (timeFile) {
        timeFile << algo_name_8 << " : " << duration_algo_8 << " seconds\n";

        timeFile << "Total execution time : " << durationGlobal.count() << " seconds\n";
        timeFile.close();
    } else {

        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error while creating/opening " << time_filename << "!\n";
    }

    std::cout << "\033[?25h";
    return 0;
}