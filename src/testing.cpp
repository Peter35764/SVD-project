#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <iomanip> 
#include <filesystem> 

#include "config.h"
#include "dqds.h"
#include "generate_svd.h"
#include "givens_refinement.h"
#include "legacy/v0_givens_refinement.h"
#include "mrrr.h"
#include "reverse_jacobi.h"

// Александр Нам, КМБО-04-20
// Any questions: alexnam16@gmail.com

// Функция возвращает матрицу-таблицу формата:
// | размерность | sigma_max/sigma_min | диап. синг. чисел |
// | sum_n(norm[I - U.transpose*U])/n | sum_n(norm[I - U*U.transpose])/n |
// | sum_n(norm[I - V.transpose*V])/n | sum_n(norm[I - V*V.transpose])/n |
// | max(abs((sigma_true_i - sigma_calc_i)/sigma_true_i)) |
// Размер выборки фиксированного размера матриц определяется задаваемым
// параметром 'n'.

// Функция написана с теми условиями, что:
//    1. Класс SVD разложения наследуется от класса Eigen::SVDBase, причем
//    должен существоваться конструктор класса,
//       который вторым параметром принимает настройку вычислений матриц U и V,
//       т.е. thin или full.
//    2. Генерация случайных матриц происходит с помощью SVDGenerator из
//    generate_svd.h
//    3. В функцию передаётся std::vector соотношений максимального и
//    минимального сингулярного числа
//    4. В функцию передаётся std::vector<std::pair<int,int>> размеров матриц
//    для исследования
//    5. В функцию передаётся int n размер выборки фиксированного размера матриц
//    для подсчёта средних
//    6. Функция работает достаточно долго, особенно для матриц больших
//    размеров, поэтому выводится прогресс в процентах
//    7. Результат исследования не печатается в консоль, а сохраняется в файл,
//    название выбирается первым параметром

// Функция принимает параметрами:
// - fileName: имя текстового файла, куда будет сохранен результат, т.е. таблица
// - SigmaMaxMinRatiosVec: вектор соотношений максимального и минимального
// сингулярных чисел;
//                        нужен т.к. ошибка может сильно отличаться у разных
//                        соотношений сингулярных чисел;
// - MatSizesVec: вектор размеров матриц для теста;
// - n: количество матриц, которые генерируются с одинаковыми параметрами для
// усреднения выборки и подсчёта средних
// - algorithmName: название алгоритма, используется в выводе прогресса
// - lineNumber: номер строки в терминале, которую будет обновлять данный
// алгоритм

template <typename Derived, typename T>
T lp_norm(const Eigen::MatrixBase<Derived> &M, T p) {
  T sum = 0;
  for (int i = 0; i < M.rows(); ++i) {
    for (int j = 0; j < M.cols(); ++j) {
      sum += std::pow(std::abs(M(i, j)), p);
    }
  }
  return std::pow(sum, T(1) / p);
}

enum class MetricType {
  U_DEVIATION1,
  U_DEVIATION2,
  V_DEVIATION1,
  V_DEVIATION2,
  ERROR_SIGMA,
  RECON_ERROR,
  MAX_DEVIATION
};

template <typename T>
struct MetricSetting {
    MetricType type;    // Тип метрики
    std::string name;   // Название метрики 
    bool relative;      // true: относительная ошибка, false: абсолютная
    T p;                // Параметр p для Lp нормы
    bool enabled;       // Флаг, управляющий выводом метрики в таблицу

    static std::string generateName(const std::string& baseName, bool relative, MetricType type) {
        if (type == MetricType::ERROR_SIGMA || type == MetricType::RECON_ERROR ||
            type == MetricType::U_DEVIATION1 || type == MetricType::U_DEVIATION2 ||
            type == MetricType::V_DEVIATION1 || type == MetricType::V_DEVIATION2 ||
            type == MetricType::MAX_DEVIATION)
        {
            return baseName + (relative ? " (rel)" : " (abs)");
        }
        return baseName;
    }

    MetricSetting(MetricType t,
                  const std::string& baseName,
                  bool rel,
                  T p_val,
                  bool en)
      : type(t),
        name(generateName(baseName, rel, t)), 
        relative(rel),
        p(p_val),
        enabled(en)
    {
        if (!(p_val > T(0))) {
            throw std::invalid_argument("Ошибка: параметр p должен быть больше 0");
        }
    }
};

std::counting_semaphore<THREADS> thread_semaphore(THREADS);
std::mutex cout_mutex;

template<typename T, template<typename> class gen_cl, template<typename> class svd_cl>
void svd_test_func(
    std::string fileName,
    const std::vector<T>& SigmaMaxMinRatiosVec,
    const std::vector<std::pair<int, int>>& MatSizesVec,
    const int n,
    const std::string &algorithmName,
    int lineNumber,
    const std::vector<MetricSetting<T>>& metricsSettings)
{
    auto printTable = [](std::ostream &out, const std::vector<std::vector<std::string>> &data) {
        if (data.empty())
            return;
        std::vector<size_t> widths;
        for (const auto &row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (i >= widths.size())
                    widths.push_back(row[i].size());
                else
                    widths[i] = std::max(widths[i], row[i].size());
            }
        }
        for (const auto &row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << std::left << std::setw(widths[i] + 3) << row[i];
                if (i < row.size() - 1)
                    out << "\t";
            }
            out << "\n";
        }
    };

    auto printCSV = [](std::ostream &out, const std::vector<std::vector<std::string>> &data) {
        for (size_t r = 0; r < data.size(); ++r) {
            bool first = true;
            for (size_t i = 0; i < data[r].size(); ++i) {
                if (!first)
                    out << ",";
                std::string cellFormatted = data[r][i];
                if (cellFormatted.find(',') != std::string::npos) {
                    cellFormatted = "\"" + cellFormatted + "\"";
                }
                out << cellFormatted;
                first = false;
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
    for (const auto &MatSize : MatSizesVec) {
        generalSum += (MatSize.first * MatSize.second);
    }
    for (size_t r = 0; r < data.size(); ++r) {
      for (size_t i = 0; i < data[r].size(); ++i) {
        out << std::left << std::setw(widths[i] + 3) << data[r][i];
      }
      out << "\n";
    }

    MatrixDynamic U_true, S_true, V_true, U_calc, V_calc;
    VectorDynamic SV_calc;

    std::random_device rd;
    std::default_random_engine gen(rd());

    for (const auto &MatSize : MatSizesVec) {
        const int N = MatSize.first;
        const int M = MatSize.second;
        int minNM = std::min(N, M);

        U_true.resize(N, N);
        U_calc.resize(N, N);
        S_true.resize(N, M);
        SV_calc.resize(minNM);
        V_true.resize(M, M);
        V_calc.resize(M, M);

        for (const auto &SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
            for (const auto &interval : Intervals) {
                assert((interval.first < interval.second) && "Error: left boundary >= right boundary");
                assert((interval.first * SigmaMaxMinRatio <= interval.second) &&
                       "Error: no sigma values exist with such ratio in such interval");

                std::uniform_real_distribution<T> distrSigmaMin(interval.first, interval.second / SigmaMaxMinRatio);
                T sigma_min = distrSigmaMin(gen);
                T sigma_max = SigmaMaxMinRatio * sigma_min;

                std::uniform_real_distribution<T> distr(sigma_min, sigma_max);
                assert((minNM >= 2) && "Error: no columns or rows allowed");

                T avg_dev_UUt = 0, avg_dev_UtU = 0, avg_dev_VVt = 0, avg_dev_VtV = 0;
                T avg_dev_UUt_rel = 0, avg_dev_UtU_rel = 0, avg_dev_VVt_rel = 0, avg_dev_VtV_rel = 0;
                
                T avg_sigma_error_rel = 0, avg_sigma_error_abs = 0;
                T avg_recon_error_rel = 0, avg_recon_error_abs = 0;
                T avg_max_deviation = 0;
                T avg_max_deviation_rel = 0;

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

                    T dev_UUt = lp_norm(MatrixDynamic::Identity(N, N) - U_calc * U_calc.transpose(), metricsSettings[0].p);
                    T dev_UtU = lp_norm(MatrixDynamic::Identity(N, N) - U_calc.transpose() * U_calc, metricsSettings[1].p);
                    T dev_VVt = lp_norm(MatrixDynamic::Identity(M, M) - V_calc * V_calc.transpose(), metricsSettings[2].p);
                    T dev_VtV = lp_norm(MatrixDynamic::Identity(M, M) - V_calc.transpose() * V_calc, metricsSettings[3].p);

                    T id_norm_U = lp_norm(MatrixDynamic::Identity(N, N), metricsSettings[0].p);
                    T id_norm_V = lp_norm(MatrixDynamic::Identity(M, M), metricsSettings[2].p);
                    T dev_UUt_rel = dev_UUt / id_norm_U;
                    T dev_UtU_rel = dev_UtU / id_norm_U;
                    T dev_VVt_rel = dev_VVt / id_norm_V;
                    T dev_VtV_rel = dev_VtV / id_norm_V;

                    avg_dev_UUt += dev_UUt / n;
                    avg_dev_UUt_rel += dev_UUt_rel / n;
                    avg_dev_UtU += dev_UtU / n;
                    avg_dev_UtU_rel += dev_UtU_rel / n;
                    avg_dev_VVt += dev_VVt / n;
                    avg_dev_VVt_rel += dev_VVt_rel / n;
                    avg_dev_VtV += dev_VtV / n;
                    avg_dev_VtV_rel += dev_VtV_rel / n;

                    T sigma_err_rel = lp_norm((S_true.diagonal() - SV_calc)
                                              .cwiseQuotient(S_true.diagonal()), metricsSettings[4].p);
                    T sigma_err_abs = lp_norm(S_true.diagonal() - SV_calc, metricsSettings[5].p);
                    avg_sigma_error_rel += sigma_err_rel / n;
                    avg_sigma_error_abs += sigma_err_abs / n;

                    MatrixDynamic A = U_true * S_true * V_true.transpose();
                    MatrixDynamic S_calc_diag = MatrixDynamic::Zero(minNM, minNM);
                    S_calc_diag.diagonal() = SV_calc;
                    MatrixDynamic A_calc = U_calc * S_calc_diag * V_calc.transpose();
                    
                    T recon_err_abs = lp_norm(A - A_calc, metricsSettings[6].p);
                    T recon_err_rel = lp_norm(A - A_calc, metricsSettings[7].p) / lp_norm(A, metricsSettings[7].p);
                    avg_recon_error_abs += recon_err_abs / n;
                    avg_recon_error_rel += recon_err_rel / n;

                    T max_dev = std::max({dev_UUt, dev_UtU, dev_VVt, dev_VtV});
                    T max_dev_rel = std::max({dev_UUt_rel, dev_UtU_rel, dev_VVt_rel, dev_VtV_rel});
                    avg_max_deviation += max_dev / n;
                    avg_max_deviation_rel += max_dev_rel / n;

                    progress += static_cast<T>(M * N) / ProgressCoeff;
                    double percent = static_cast<double>(progress);
                    int barWidth = 50;
                    int pos = barWidth * static_cast<int>(percent) / 100;

                    std::ostringstream progressStream;
                    progressStream << algorithmName << ": " << std::fixed << std::setprecision(4)
                                   << percent << "% [";
                    for (int j = 0; j < barWidth; ++j) {
                        if (j < pos)
                            progressStream << "=";
                        else if (j == pos)
                            progressStream << ">";
                        else
                            progressStream << " ";
                    }
                    progressStream << "]";

                    {
                        std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cout << "\033[" << lineNumber << ";0H" << progressStream.str()
                                  << "\033[0K" << std::flush;
                    }
                } // конец итераций

                std::vector<std::string> row;
                row.push_back(num2str(N) + "x" + num2str(M));
                row.push_back(num2str(SigmaMaxMinRatio));
                row.push_back("[" + num2str(interval.first) + ", " + num2str(interval.second) + "]");
                for (const auto &ms : metricsSettings) {
                    if (!ms.enabled) continue;
                    switch(ms.type) {
                        case MetricType::U_DEVIATION1:
                            row.push_back(num2str(ms.relative ? avg_dev_UUt_rel : avg_dev_UUt));
                            break;
                        case MetricType::U_DEVIATION2:
                            row.push_back(num2str(ms.relative ? avg_dev_UtU_rel : avg_dev_UtU));
                            break;
                        case MetricType::V_DEVIATION1:
                            row.push_back(num2str(ms.relative ? avg_dev_VVt_rel : avg_dev_VVt));
                            break;
                        case MetricType::V_DEVIATION2:
                            row.push_back(num2str(ms.relative ? avg_dev_VtV_rel : avg_dev_VtV));
                            break;
                        case MetricType::ERROR_SIGMA:
                            if (ms.relative)
                                row.push_back(num2str(avg_sigma_error_rel));
                            else
                                row.push_back(num2str(avg_sigma_error_abs));
                            break;
                        case MetricType::RECON_ERROR:
                            if (ms.relative)
                                row.push_back(num2str(avg_recon_error_rel));
                            else
                                row.push_back(num2str(avg_recon_error_abs));
                            break;
                        case MetricType::MAX_DEVIATION:
                            row.push_back(num2str(ms.relative ? avg_max_deviation_rel : avg_max_deviation));
                            break;
                    }
                }
                table.push_back(row);
            }
        }
        out << cellFormatted;
        first = false;
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
  for (const auto &MatSize : MatSizesVec) {
    generalSum += (MatSize.first * MatSize.second);
  }

  using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  const std::vector<std::pair<T, T>> Intervals = {{0, 1}, {1, 100}};

  T ProgressCoeff =
      n * Intervals.size() * SigmaMaxMinRatiosVec.size() * generalSum / 100.0;
  T progress = 0;

  std::vector<std::vector<std::string>> table;
  {
    std::vector<std::string> header;
    header.push_back("Dimension");
    header.push_back("Sigma-max/min-ratio");
    header.push_back("SV interval");
    for (const auto &ms : metricsSettings) {
      if (ms.enabled) header.push_back(ms.name);
    }
    table.push_back(header);
  }

  MatrixDynamic U_true, S_true, V_true, U_calc, V_calc;
  VectorDynamic SV_calc;

  std::random_device rd;
  std::default_random_engine gen(rd());

  for (const auto &MatSize : MatSizesVec) {
    const int N = MatSize.first;
    const int M = MatSize.second;
    int minNM = std::min(N, M);

    U_true.resize(N, N);
    U_calc.resize(N, N);
    S_true.resize(N, M);
    SV_calc.resize(minNM);
    V_true.resize(M, M);
    V_calc.resize(M, M);

    for (const auto &SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
      for (const auto &interval : Intervals) {
        assert((interval.first < interval.second) &&
               "Error: left boundary >= right boundary");
        assert((interval.first * SigmaMaxMinRatio <= interval.second) &&
               "Error: no sigma values exist with such ratio in such interval");

        std::uniform_real_distribution<T> distrSigmaMin(
            interval.first, interval.second / SigmaMaxMinRatio);
        T sigma_min = distrSigmaMin(gen);
        T sigma_max = SigmaMaxMinRatio * sigma_min;

        std::uniform_real_distribution<T> distr(sigma_min, sigma_max);
        assert((minNM >= 2) && "Error: no columns or rows allowed");

        T avg_dev_UUt = 0, avg_dev_UtU = 0, avg_dev_VVt = 0, avg_dev_VtV = 0;
        T avg_sigma_error = 0;
        T avg_recon_error = 0;
        T avg_max_deviation = 0;

        for (int i = 1; i <= n; ++i) {
          gen_cl<T> svd_gen(N, M, gen, distr, true);
          svd_gen.generate(minNM);

          U_true = svd_gen.MatrixU();
          S_true = svd_gen.MatrixS();
          V_true = svd_gen.MatrixV();
          svd_cl<MatrixDynamic> svd_func(
              (U_true * S_true * V_true.transpose()).eval(),
              Eigen::ComputeFullU | Eigen::ComputeFullV);
          U_calc = svd_func.matrixU();
          SV_calc = svd_func.singularValues();
          V_calc = svd_func.matrixV();

          T dev_UUt = lp_norm(
              MatrixDynamic::Identity(N, N) - U_calc * U_calc.transpose(),
              metricsSettings[0].p);
          T dev_UtU = lp_norm(
              MatrixDynamic::Identity(N, N) - U_calc.transpose() * U_calc,
              metricsSettings[1].p);
          T dev_VVt = lp_norm(
              MatrixDynamic::Identity(M, M) - V_calc * V_calc.transpose(),
              metricsSettings[2].p);
          T dev_VtV = lp_norm(
              MatrixDynamic::Identity(M, M) - V_calc.transpose() * V_calc,
              metricsSettings[3].p);

          avg_dev_UUt += dev_UUt / n;
          avg_dev_UtU += dev_UtU / n;
          avg_dev_VVt += dev_VVt / n;
          avg_dev_VtV += dev_VtV / n;

          T sigma_err = 0;
          if (metricsSettings[4].relative)
            sigma_err = lp_norm(
                (S_true.diagonal() - SV_calc).cwiseQuotient(S_true.diagonal()),
                metricsSettings[4].p);
          else
            sigma_err =
                lp_norm(S_true.diagonal() - SV_calc, metricsSettings[4].p);
          avg_sigma_error += sigma_err / n;

          MatrixDynamic A = U_true * S_true * V_true.transpose();
          MatrixDynamic S_calc_diag = MatrixDynamic::Zero(minNM, minNM);
          S_calc_diag.diagonal() = SV_calc;
          MatrixDynamic A_calc = U_calc * S_calc_diag * V_calc.transpose();
          T recon_err = 0;
          if (metricsSettings[5].relative)
            recon_err = lp_norm(A - A_calc, metricsSettings[5].p) /
                        lp_norm(A, metricsSettings[5].p);
          else
            recon_err = lp_norm(A - A_calc, metricsSettings[5].p);
          avg_recon_error += recon_err / n;

          T max_dev = std::max({dev_UUt, dev_UtU, dev_VVt, dev_VtV});
          avg_max_deviation += max_dev / n;

          progress += static_cast<T>(M * N) / ProgressCoeff;
          double percent = static_cast<double>(progress);
          int barWidth = 50;
          int pos = barWidth * static_cast<int>(percent) / 100;

          std::ostringstream progressStream;
          progressStream << algorithmName << ": " << std::fixed
                         << std::setprecision(4) << percent << "% [";
          for (int j = 0; j < barWidth; ++j) {
            if (j < pos)
              progressStream << "=";
            else if (j == pos)
              progressStream << ">";
            else
              progressStream << " ";
          }
          progressStream << "]";

          {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\033[" << lineNumber << ";0H" << progressStream.str()
                      << "\033[0K" << std::flush;
          }
        }  // конец итераций

        std::vector<std::string> row;
        row.push_back(num2str(N) + "x" + num2str(M));
        row.push_back(num2str(SigmaMaxMinRatio));
        row.push_back("[" + num2str(interval.first) + ", " +
                      num2str(interval.second) + "]");
        for (const auto &ms : metricsSettings) {
          if (!ms.enabled) continue;
          switch (ms.type) {
            case MetricType::U_DEVIATION1:
              row.push_back(num2str(avg_dev_UUt));
              break;
            case MetricType::U_DEVIATION2:
              row.push_back(num2str(avg_dev_UtU));
              break;
            case MetricType::V_DEVIATION1:
              row.push_back(num2str(avg_dev_VVt));
              break;
            case MetricType::V_DEVIATION2:
              row.push_back(num2str(avg_dev_VtV));
              break;
            case MetricType::ERROR_SIGMA:
              row.push_back(num2str(avg_sigma_error));
              break;
            case MetricType::RECON_ERROR:
              row.push_back(num2str(avg_recon_error));
              break;
            case MetricType::MAX_DEVIATION:
              row.push_back(num2str(avg_max_deviation));
              break;
          }
        }
        table.push_back(row);
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
}

#define sigma_ratio \
  {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100}  // SigmaMaxMinRatiosVec
#define matrix_size \
  {{3, 3}, {5, 5}}// {10, 10}, {30, 30}, {70, 70}}  // MatSizesVec
#define matrix_num_for_sample_averaging 20        // n

int main()
{
    namespace fs = std::filesystem; // для удобства использования

    auto start = std::chrono::high_resolution_clock::now();

    // Создание папки для результатов тестов с текущей датой и временем
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* ptm = std::localtime(&now_time);
    std::ostringstream oss;
    oss << "TestingResultsBundle<" << std::put_time(ptm, "%d-%m-%Y-%H%M%S") << ">";
    std::string folderName = oss.str();
    fs::create_directory(folderName);

    std::cout << "\033[2J\033[H";

    std::vector<std::pair<std::string, double>> test_times;
    std::mutex test_times_mutex;

    std::vector<double> sigmaRatios = sigma_ratio; // macro sigma_ratio: {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100}
    std::vector<std::pair<int, int>> matrixSizes = matrix_size; // macro matrix_size: {{3, 3}, {5, 5}}
    int sampleCount = matrix_num_for_sample_averaging; // макрос sample count
	//генерируется таблица в файле "jacobi_test_table.txt" теста метода Eigen::JacobiSVD
    //с соотношением сингулярных чисел:  1.01, 1.2, 2, 5, 10, 50       ---    6
    //причем каждое соотношение относится к двум интервалам сингулярных чисел:
    //                      маленьких {0,1}, больших {1,100} (это не параметризованно)   ---   2
    //с матрицами размеров: {3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}   ---   6
    //6*2*6 = 72 - всего столько строк будет в таблице
    //размер выборки для усреднения: 20
    // Инициализация настроек метрик
    std::vector<MetricSetting<double>> metricsSettings = {
        MetricSetting<double>(MetricType::U_DEVIATION1, "AVG ||I-U_t*U||", false, 0.7, true),
        MetricSetting<double>(MetricType::U_DEVIATION1, "AVG ||I-U_t*U||", true, 0.7, true),
        MetricSetting<double>(MetricType::U_DEVIATION2, "AVG ||I-U*U_t||", false, 0.7, true),
        MetricSetting<double>(MetricType::U_DEVIATION2, "AVG ||I-U*U_t||", true, 0.7, true),
        MetricSetting<double>(MetricType::V_DEVIATION1, "AVG ||I-V_t*V||", false, 0.7, true),
        MetricSetting<double>(MetricType::V_DEVIATION1, "AVG ||I-V_t*V||", true, 0.7, true),
        MetricSetting<double>(MetricType::V_DEVIATION2, "AVG ||I-V*V_t||", false, 0.7, true),
        MetricSetting<double>(MetricType::V_DEVIATION2, "AVG ||I-V*V_t||", true, 0.7, true),
        MetricSetting<double>(MetricType::ERROR_SIGMA, "AVG err. sigma", true, 0.7, true),
        MetricSetting<double>(MetricType::ERROR_SIGMA, "AVG err. sigma", false, 0.7, true),
        MetricSetting<double>(MetricType::RECON_ERROR, "AVG recon error", false, 0.7, true),
        MetricSetting<double>(MetricType::RECON_ERROR, "AVG recon error", true, 0.7, true),
        MetricSetting<double>(MetricType::MAX_DEVIATION, "AVG max deviation", false, 0.7, true),
        MetricSetting<double>(MetricType::MAX_DEVIATION, "AVG max deviation", true, 0.7, true)
    };

    int flush_string = 1;
    {
      std::lock_guard<std::mutex> lock(test_times_mutex);
      test_times.emplace_back(algo_name, duration);
    }

    // idea 1 
    thread_semaphore.acquire();
    std::thread t1([&]() {
        std::string algo_name = "JacobiSVD";
        std::string file_name = folderName + "/" + "reference_JacobiSVD_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>(
            file_name,
            sigma_ratio,
            matrix_size,
            matrix_num_for_sample_averaging,
            algo_name,
            flush_string++,
            metricsSettings);
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        {
            std::lock_guard<std::mutex> lock(test_times_mutex);
            test_times.emplace_back(algo_name, duration);
        }
        thread_semaphore.release();
    });
    
    thread_semaphore.acquire();
    std::thread t2([&]() {
        std::string algo_name = "GivRef_SVD";
        std::string file_name = folderName + "/" + "idea_1_GivRef_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, SVD_Project::GivRef_SVD>(
            file_name,
            sigma_ratio,
            matrix_size,
            matrix_num_for_sample_averaging,
            algo_name,
            flush_string++,
            metricsSettings);
        auto t_end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t_end - t_start).count();
        {
            std::lock_guard<std::mutex> lock(test_times_mutex);
            test_times.emplace_back(algo_name, duration);
        }
        thread_semaphore.release();
    });

    // idea 2 
    // thread_semaphore.acquire();
    // std::thread t3([&]() {
    //     std::string algo_name = "RevJac_SVD";
    //     std::string file_name = folderName + "/" + "idea_2_RevJac_table.txt";
    //     auto t_start = std::chrono::high_resolution_clock::now();
    //     svd_test_func<double, SVDGenerator, SVD_Project::RevJac_SVD>(
    //         file_name,
    //         sigma_ratio,
    //         matrix_size,
    //         matrix_num_for_sample_averaging,
    //         algo_name,
    //         flush_string++,
    //         metricsSettings);
    //     auto t_end = std::chrono::high_resolution_clock::now();
    //     double duration = std::chrono::duration<double>(t_end - t_start).count();
    //     {
    //         std::lock_guard<std::mutex> lock(test_times_mutex);
    //         test_times.emplace_back(algo_name, duration);
    //     }
    //     thread_semaphore.release();
    // });
    
    thread_semaphore.acquire();
    std::thread t4([&]() {
        std::string algo_name = "MRRR";
        std::string file_name = folderName + "/" + "idea_3_MRRR_table.txt";
        auto t_start = std::chrono::high_resolution_clock::now();
        svd_test_func<double, SVDGenerator, MRRR_SVD>(
            file_name,
            sigma_ratio,
            matrix_size,
            matrix_num_for_sample_averaging,
            algo_name,
            flush_string++,
            metricsSettings);
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
    t4.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGlobal = end - start;

    std::cout << "\033[5;0H";
    std::cout << "\nFull execution time = " << durationGlobal.count() << " seconds.\n";
    {
      std::lock_guard<std::mutex> lock(test_times_mutex);
      test_times.emplace_back(algo_name, duration);
    }
    std::ofstream timeFile(folderName + "/" + "individual_test_times.txt");
    if (timeFile) {
        // Вывод настроек теста из локальных переменных
        timeFile << "=== Test Settings ===\n";
        timeFile << "Sigma ratios: ";
        for (const auto& ratio : sigmaRatios) {
            timeFile << ratio << " ";
        }
        timeFile << "\n";

        timeFile << "Matrix sizes: ";
        for (const auto& size : matrixSizes) {
            timeFile << size.first << "x" << size.second << " ";
        }
        timeFile << "\n";

    timeFile << "Matrix sizes: ";
    for (const auto &size : matrixSizes) {
      timeFile << size.first << "x" << size.second << " ";
    }
    timeFile << "\n";

    timeFile << "Sample count: " << sampleCount << "\n";

    timeFile << "Metrics Settings:\n";
    for (const auto &ms : metricsSettings) {
      timeFile << "  " << ms.name << " ("
               << (ms.relative ? "relative" : "absolute") << "), p = " << ms.p
               << "\n";
    }
    timeFile << "\n=== Execution Times ===\n";

    timeFile << "Max threads: " << THREADS << "\n\n";
    timeFile << "Total execution time: " << durationGlobal.count()
             << " seconds\n\n";
    timeFile << "Individual algorithm execution times:\n";
    for (const auto &entry : test_times) {
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
