#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>

#include "config.h"
#include "dqds.h"
#include "generate_svd.h"
#include "givens_refinement.h"
#include "mrrr.h"
#include "reverse_jacobi.h"

//Александр Нам, КМБО-04-20
//Any questions: alexnam16@gmail.com

// Функция возвращает матрицу-таблицу формата:
// | размерность | sigma_max/sigma_min | диап. синг. чисел |
// | sum_n(norm[I - U.transpose*U])/n | sum_n(norm[I - U*U.transpose])/n |
// | sum_n(norm[I - V.transpose*V])/n | sum_n(norm[I - V*V.transpose])/n |
// | max(abs((sigma_true_i - sigma_calc_i)/sigma_true_i)) |
// Размер выборки фиксированного размера матриц определяется задаваемым параметром 'n'.

// Функция написана с теми условиями, что:
//    1. Класс SVD разложения наследуется от класса Eigen::SVDBase, причем должен существоваться конструктор класса,
//       который вторым параметром принимает настройку вычислений матриц U и V, т.е. thin или full.
//    2. Генерация случайных матриц происходит с помощью SVDGenerator из generate_svd.h
//    3. В функцию передаётся std::vector соотношений максимального и минимального сингулярного числа
//    4. В функцию передаётся std::vector<std::pair<int,int>> размеров матриц для исследования
//    5. В функцию передаётся int n размер выборки фиксированного размера матриц для подсчёта средних
//    6. Функция работает достаточно долго, особенно для матриц больших размеров, поэтому выводится прогресс в процентах
//    7. Результат исследования не печатается в консоль, а сохраняется в файл, название выбирается первым параметром

// Функция принимает параметрами: 
// - fileName: имя текстового файла, куда будет сохранен результат, т.е. таблица
// - SigmaMaxMinRatiosVec: вектор соотношений максимального и минимального сингулярных чисел;
//                        нужен т.к. ошибка может сильно отличаться у разных соотношений сингулярных чисел;
// - MatSizesVec: вектор размеров матриц для теста;
// - n: количество матриц, которые генерируются с одинаковыми параметрами для усреднения выборки и подсчёта средних
// - algorithmName: название алгоритма, используется в выводе прогресса
// - lineNumber: номер строки в терминале, которую будет обновлять данный алгоритм

std::counting_semaphore<THREADS> thread_semaphore(THREADS);
std::mutex cout_mutex;

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
                if (widths.size() <= i)
                    widths.push_back(0);
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
                if (!first)
                    out << ",";
                std::string cellFormatted = cell;
                if (cellFormatted.find(',') != std::string::npos) {
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
        oss << value;
        return oss.str();
    };

    T generalSum = 0;
    for (const auto& MatSize : MatSizesVec){
        generalSum += (MatSize.first * MatSize.second);
    } 
    
    using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorDynamic = Eigen::Vector<T, Eigen::Dynamic>;

    const std::vector<std::pair<T,T>> Intervals = {{0,1}, {1,100}};
    
	std::random_device rd; //случайное число - значение сида
    std::default_random_engine gen(rd()); //генерация последовательности псевдослучайных цифр 
    //название столбцов таблицы
    std::vector<std::vector<std::string>> table = {
        {"Dimension", "Sigma-max/min-ratio", "SV interval", 
         "AVG ||I-U_t*U||", "AVG ||I-U*U_t||", "AVG ||I-V_t*V||",
         "AVG ||I-V*V_t||", "AVG relative err. sigma"}
    };

    T ProgressCoeff = n * Intervals.size() * SigmaMaxMinRatiosVec.size() * generalSum / 100.0; //наибольшее значение прогресса
    T progress = 0; //инициализация аккумулятора прогресса

    MatrixDynamic U_true, S_true, V_true, U_calc, V_calc, V_calc_transpose;
    VectorDynamic SV_calc; //аналог S_true, но не матрица, а вектор сингулярных значений

    for (const auto& MatSize : MatSizesVec) {
        const int N = MatSize.first;
        const int M = MatSize.second;
        int minNM = std::min(N, M);

        U_true.resize(N, N); U_calc.resize(N, N);
        S_true.resize(N, M); SV_calc.resize(minNM);
        V_true.resize(M, M); V_calc.resize(M, M); V_calc_transpose.resize(M, M);     

        for (const auto& SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
            for (const auto& interval : Intervals) {
                assert((interval.first < interval.second) && "Error: left boundary >= right boundary");
                assert((interval.first * SigmaMaxMinRatio <= interval.second) &&
                       "Error: no sigma values exist with such ratio in such interval");

                std::uniform_real_distribution<T> distrSigmaMin(interval.first, interval.second / SigmaMaxMinRatio);
                T sigma_min = distrSigmaMin(gen); 
                T sigma_max = SigmaMaxMinRatio * sigma_min;

                std::uniform_real_distribution<T> distr(sigma_min, sigma_max);
                assert((minNM >= 2) && "Error: no columns or rows allowed");

                T avg_dev_UUt = 0, avg_dev_UtU = 0, avg_dev_VVt = 0, avg_dev_VtV = 0, avg_relErr_sigma = 0;

                for (int i = 1; i <= n; ++i) {   
                    gen_cl<T> svd_gen(N, M, gen, distr, true);
                    svd_gen.generate(minNM);

                    U_true = svd_gen.MatrixU();
                    S_true = svd_gen.MatrixS();
                    V_true = svd_gen.MatrixV();
                    svd_cl<MatrixDynamic> svd_func((U_true * S_true * V_true.transpose()).eval(), 
                                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
                    U_calc = svd_func.matrixU();
                    SV_calc = svd_func.singularValues();
                    V_calc = svd_func.matrixV();
                    avg_dev_UUt += (MatrixDynamic::Identity(N, N) - U_calc * U_calc.transpose()).squaredNorm() / n;
                    avg_dev_UtU += (MatrixDynamic::Identity(N, N) - U_calc.transpose() * U_calc).squaredNorm() / n;
                    avg_dev_VVt += (MatrixDynamic::Identity(M, M) - V_calc * V_calc.transpose()).squaredNorm() / n;
                    avg_dev_VtV += (MatrixDynamic::Identity(M, M) - V_calc.transpose() * V_calc).squaredNorm() / n;
                    avg_relErr_sigma += (S_true.diagonal() - SV_calc)
                                         .cwiseQuotient(S_true.diagonal())
                                         .cwiseAbs().maxCoeff() / n;

                    progress += static_cast<T>(M * N) / ProgressCoeff;
                    double percent = static_cast<double>(progress);
                    int barWidth = 50;
                    int pos = barWidth * static_cast<int>(percent) / 100;

                    std::ostringstream progressStream;
                    progressStream << algorithmName << ": " 
                                   << std::fixed << std::setprecision(4) << percent 
                                   << "% [";
                    for (int j = 0; j < barWidth; ++j) {
                        if (j < pos) progressStream << "=";
                        else if (j == pos) progressStream << ">";
                        else progressStream << " ";
                    }
                    progressStream << "]";

                    {
                        std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cout << "\033[" << lineNumber << ";0H" << progressStream.str() << "\033[0K" << std::flush;
                    }
                }

                table.emplace_back(std::vector<std::string>{
                    num2str(N) + "x" + num2str(M), 
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
        std::cerr << "Error while creating/opening file!\n";
    }
    
    std::string csvFileName = fileName;
    size_t pos = csvFileName.rfind(".txt");
    if (pos != std::string::npos)
        csvFileName.replace(pos, 4, ".csv");
    else
        csvFileName += ".csv";
    
    std::ofstream csv_file(csvFileName);
    if (csv_file) {
        printCSV(csv_file, table);
        csv_file.close();
    } else {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error while creating/opening file " << csvFileName << "!\n";
    }
};

template<typename _MatrixType>
using RevJac_DQDS_SVD = SVD_Project::RevJac_SVD<DQDS_SVD<_MatrixType>>;

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    // Очистка консоли и перемещение курсора в верхний левый угол
    std::cout << "\033[2J\033[H";

    std::vector<std::pair<std::string, double>> test_times;
    std::mutex test_times_mutex;

    //генерируеся таблица в файле "jacobi_test_table.txt" теста метода Eigen::JacobiSVD
    //с соотношением сингулярных чисел:  1.01, 1.2, 2, 5, 10, 50       ---    6
    //причем каждое соотношение относится к двум интервалам сингулярных чисел: 
    //                      маленьких {0,1}, больших {1,100} (это не параметризованно)   ---   2
    //с матрицами размеров: {3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}   ---   6
    //6*2*6 = 72 - всего столько строк будет в таблице
    //размер выборки для усреднения: 20

#define sigma_ratio {1.01, 1.2, 2, 5, 10, 50} // 0.1, 1, 100
#define matrix_size {{3, 3}, {5, 5}, {10, 10}, {20, 20}, {50, 50}, {100, 100}} // , {500, 500}
#define matrix_num_for_sample_averaging 20

    int flush_string = 1;

    //Запускаем параллельное тестирование алгоритмов
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
    }
    thread_semaphore.acquire();
    std::thread t1([&]() {
        std::string algo_name = "JacobiSVD";
        std::string file_name = "example_JacobiSVD_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>(file_name,
                                                              sigma_ratio,
                                                              matrix_size,
                                                              matrix_num_for_sample_averaging,
                                                              algo_name,
                                                              1);
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        {
            std::lock_guard<std::mutex> lock(test_times_mutex);
            test_times.emplace_back(algo_name, duration);
        }
        thread_semaphore.release();
    });

    // idea 1
    thread_semaphore.acquire();
    std::thread t2([&]() {
        std::string algo_name = "GivRef_SVD";
        std::string file_name = "idea_1_GivRef_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, SVD_Project::GivRef_SVD>(file_name,
                                                                     sigma_ratio,
                                                                     {{3, 3},
                                                                      {5, 5},
                                                                      {10, 10},
                                                                      {20, 20},
                                                                      {50, 50}},
                                                                     matrix_num_for_sample_averaging,
                                                                     algo_name,
                                                                     2);
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        {
            std::lock_guard<std::mutex> lock(test_times_mutex);
            test_times.emplace_back(algo_name, duration);
        }
        thread_semaphore.release();        
    });

    // idea 2
    thread_semaphore.acquire();
    std::thread t3([&]() {
        std::string algo_name = "RevJac_SVD";
        std::string file_name = "idea_1_RevJac_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, RevJac_DQDS_SVD>(file_name,
                                                             sigma_ratio,
                                                             {{3, 3}, {5, 5}, {10, 10}},
                                                             matrix_num_for_sample_averaging,
                                                             algo_name,
                                                             3);
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        {
            std::lock_guard<std::mutex> lock(test_times_mutex);
            test_times.emplace_back(algo_name, duration);
        }
        thread_semaphore.release();
    });

    // idea 3
    thread_semaphore.acquire();
    std::thread t4([&]() {
        std::string algo_name = "MRRR";
        std::string file_name = "idea_1_MRRR_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, MRRR_SVD>(file_name,
                                                      sigma_ratio,
                                                      {{3, 3}, {5, 5}, {10, 10}, {20, 20}},
                                                      matrix_num_for_sample_averaging,
                                                      algo_name,
                                                      4);
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        {
            std::lock_guard<std::mutex> lock(test_times_mutex);
            test_times.emplace_back(algo_name, duration);
        }
        thread_semaphore.release();
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGlobal = end - start;

    // Перемещаем курсор на первую строку для вывода итогового времени
    std::cout << "\033[3;0H";
    std::cout << "\nFull execution time = " << durationGlobal.count() << " seconds.\n";
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
    }
    std::ofstream timeFile("individual_test_times.txt");
    if (timeFile) {
        timeFile << "Max threads: " << THREADS << "\n\n";

        timeFile << "Total execution time: " << durationGlobal.count() << " seconds\n\n";

		timeFile << "Individual algorithm execution times:\n";
		for (const auto& entry : test_times) {
			timeFile << entry.first << " : " << entry.second << " seconds\n";
		}
		timeFile.close();
    } else {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error while creating/opening individual_test_times.txt!\n";
    }

    char c;
    std::cin >> c;
    return 0;
}
