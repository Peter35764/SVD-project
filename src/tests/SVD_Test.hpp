#ifndef SVD_TEST_HPP
#define SVD_TEST_HPP

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <thread>

#include "SVD_Test.h"
#include "config.h"

namespace SVD_Project {

std::counting_semaphore<THREADS> thread_semaphore(THREADS);
std::mutex cout_mutex;

template <typename FloatingPoint, typename MatrixType>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::Lpq_norm(
    const Eigen::MatrixBase<MatrixType> &M, FloatingPoint p, FloatingPoint q) {
  auto abs_p = M.array().abs().pow(p);
  auto inner_sums = abs_p.rowwise().sum().pow(q / p);
  return std::pow(inner_sums.sum(), FloatingPoint(1) / q);
}

template <typename FloatingPoint, typename MatrixType>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::Lp_norm(
    const Eigen::MatrixBase<MatrixType> &M, FloatingPoint p) {
  return Lpq_norm(M, p, p);
}

template <typename FloatingPoint, typename MatrixType>
SVD_Test<FloatingPoint, MatrixType>::MetricSettings::MetricSettings(
    MetricType type_, FloatingPoint p_, bool is_relative_, std::string name_,
    bool enabled_)
    : type(type_),
      p(p_),
      is_relative(is_relative_),
      name(generateName(name_, is_relative_, type_)),
      enabled(enabled_) {
  if (!(p_ > FloatingPoint(0))) {
    throw std::invalid_argument(
        "ERROR: metric value must be > 0 (TODO make for 0 and inf)");
  }
}

template <typename FloatingPoint, typename MatrixType>
std::string SVD_Test<FloatingPoint, MatrixType>::MetricSettings::generateName(
    const std::string &baseName, bool relative, MetricType type) {
  return baseName + (relative ? " (rel)" : " (abs)");
}

template <typename FloatingPoint, typename MatrixType>
template <template <typename> class gen_cl, template <typename> class svd_cl>
void SVD_Test<FloatingPoint, MatrixType>::svd_test_func(
    std::string fileName,
    const std::vector<FloatingPoint> &SigmaMaxMinRatiosVec,
    const std::vector<std::pair<int, int>> &MatSizesVec, const int n,
    const std::string &algorithmName, int lineNumber,
    const std::vector<MetricSettings> &metricsSettings) {
  ++flush;
  FloatingPoint generalProgressSum = 0;

  for (const auto &MatSize : MatSizesVec) {
    generalProgressSum += (MatSize.first * MatSize.second);
  }

  // диапазоны генерации сингулярных чисел
  const std::vector<std::pair<FloatingPoint, FloatingPoint>> Intervals = {
      {0, 1}, {1, 100}};

  FloatingPoint ProgressCoeff = n * Intervals.size() *
                                SigmaMaxMinRatiosVec.size() *
                                generalProgressSum / 100.0;
  FloatingPoint progress = 0;

  std::vector<std::vector<std::string>> table;

  std::vector<std::string> header;
  header.push_back("Dimension");
  header.push_back("Sigma-max/Sigma-min");
  header.push_back("SV interval");
  for (const auto &ms : metricsSettings) {
    if (ms.enabled) header.push_back(ms.name);
  }

  table.push_back(header);

  MatrixDynamic U_true, S_true,
      V_true;                    // Точное и правильное сингулярное разложение
  MatrixDynamic U_calc, V_calc;  // Посчитанные сингулярные вектора
  VectorDynamic S_calc;          // Посчитанные сингулярные значения

  std::random_device rd;
  std::default_random_engine gen(rd());

  for (const auto &MatSize : MatSizesVec) {
    const int N = MatSize.first;
    const int M = MatSize.second;
    int minNM = std::min(N, M);

    auto Identity_U_norm = [N = N](FloatingPoint P) {
      return lp_norm(MatrixDynamic::Identity(N, N), P);
    };
    auto Identity_V_norm = [M = M](FloatingPoint P) {
      return lp_norm(MatrixDynamic::Identity(M, M), P);
    };

    U_true.resize(N, N);
    U_calc.resize(N, N);
    S_true.resize(N, M);
    S_calc.resize(minNM);
    V_true.resize(M, M);
    V_calc.resize(M, M);

    for (const auto &SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
      for (const auto &interval : Intervals) {
        assert((interval.first < interval.second) &&
               "Error: left boundary >= right boundary");
        assert((interval.first * SigmaMaxMinRatio <= interval.second) &&
               "Error: no sigma values exist with such ratio in such interval");

        // Определение допустимого промежутка сингулярных значений
        std::uniform_real_distribution<FloatingPoint> distrSigmaMin(
            interval.first, interval.second / SigmaMaxMinRatio);

        FloatingPoint sigma_min = distrSigmaMin(gen);
        FloatingPoint sigma_max = SigmaMaxMinRatio * sigma_min;

        std::uniform_real_distribution<FloatingPoint> distr(sigma_min,
                                                            sigma_max);
        assert((minNM >= 2) && "Error: no columns or rows allowed");

        // контейнер для хранения метрик
        std::map<MetricSettings, FloatingPoint> results;

        // ============ Подсчет метрик ============
        for (size_t i = 1; i <= n; ++i) {
          // Генерация разложения
          gen_cl<FloatingPoint> svd_gen(N, M, gen, distr, true);
          svd_gen.generate(minNM);

          // "Истиннное" разложение
          U_true = svd_gen.MatrixU();
          S_true = svd_gen.MatrixS();
          V_true = svd_gen.MatrixV();

          // Запуск тестируемого алгоритма
          svd_cl<MatrixDynamic> svd_func(
              (U_true * S_true * V_true.transpose()).eval(),
              Eigen::ComputeFullU | Eigen::ComputeFullV);

          // "Получившееся" разложение
          U_calc = svd_func.matrixU();
          S_calc = svd_func.singularValues();
          V_calc = svd_func.matrixV();

          for (MetricSettings metric_settings : metricsSettings) {
            results[metric_settings] =
                count_metrics(metric_settings, N, M, U_calc, V_calc, S_calc,
                              U_true, V_true, S_true);
          }

          // Подсчет прогресса
          progress += static_cast<FloatingPoint>(M * N) / ProgressCoeff;
          double percent = static_cast<double>(progress);
          int barWidth = 50;  // Ширина прогресс-бара
          int pos = barWidth * static_cast<int>(percent) / 100;

          // Печать прогресса
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

        // average values
        for (MetricSettings &res : results) {
          results[res] /= n;
        }

        // Составление txt версии
        std::vector<std::string> row;
        row.push_back(num2str(N) + "x" + num2str(M));
        row.push_back(num2str(SigmaMaxMinRatio));
        row.push_back("[" + num2str(interval.first) + ", " +
                      num2str(interval.second) + "]");
        for (const auto &ms : metricsSettings) {
          if (!ms.enabled) continue;
          row.push_back(num2str(results[ms]));
        }
        table.push_back(row);
      }
    }
  }

  // Создание .txt файла
  std::ofstream file(fileName);
  if (file) {
    printTable(file, table);
    file.close();
  } else {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Error while creating/opening file!\n";
  }

  // Составление названия файла .csv
  std::string csvFileName = fileName;
  size_t pos = csvFileName.rfind(".txt");
  if (pos != std::string::npos)
    csvFileName.replace(pos, 4, ".csv");
  else
    csvFileName += ".csv";

  // Создание .csv файла
  std::ofstream csv_file(csvFileName);
  if (csv_file) {
    printCSV(csv_file, table);
    csv_file.close();
  } else {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Error while creating/opening file " << csvFileName << "!\n";
  }
}

template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::printTable(
    std::ostream &out, const std::vector<std::vector<std::string>> &data) {
  if (data.empty()) return;
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
      if (i < row.size() - 1) out << "\t";
    }
    out << "\n";
  }
}

template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::printCSV(
    std::ostream &out, const std::vector<std::vector<std::string>> &data) {
  for (size_t r = 0; r < data.size(); ++r) {
    bool first = true;
    for (size_t i = 0; i < data[r].size(); ++i) {
      if (!first) out << ",";
      std::string cellFormatted = data[r][i];
      if (cellFormatted.find(',') != std::string::npos) {
        cellFormatted = "\"" + cellFormatted + "\"";
      }
      out << cellFormatted;
      first = false;
    }
    out << "\n";
  }
}

template <typename FloatingPoint, typename MatrixType>
std::string SVD_Test<FloatingPoint, MatrixType>::num2str(FloatingPoint value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
};

template <typename FloatingPoint, typename MatrixType>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::count_metrics(
    MetricSettings metric_settings, size_t Usize, size_t Vsize,
    MatrixDynamic U_calc, MatrixDynamic V_calc, VectorDynamic S_calc,
    MatrixDynamic U_true, MatrixDynamic V_true, VectorDynamic S_true) {
  FloatingPoint ans = 0;
  VectorDynamic abs_err = {};
  VectorDynamic error = {};

  MatrixDynamic A_true = U_true * S_true * V_true.transpose();
  MatrixDynamic S_calc_diag = MatrixDynamic::Zero(std::min(Usize, Vsize));
  S_calc_diag.diagonal() = S_calc;
  MatrixDynamic A_calc = U_calc * S_calc_diag * V_calc.transpose();

  switch (metric_settings.type) {
    case MetricType::U_DEVIATION1:
      ans = Lp_norm(
          MatrixDynamic::Identity(Usize, Usize) - U_calc * U_calc.transpose(),
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Identity_U_norm(metric_settings.p);
      }
      break;

    case MetricType::U_DEVIATION2:
      ans = Lp_norm(
          MatrixDynamic::Identity(Usize, Usize) - U_calc.transpose() * U_calc,
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Identity_U_norm(metric_settings.p);
      }
      break;

    case MetricType::V_DEVIATION1:
      ans = Lp_norm(
          MatrixDynamic::Identity(Vsize, Vsize) - V_calc * V_calc.transpose(),
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Identity_V_norm(metric_settings.p);
      }
      break;

    case MetricType::V_DEVIATION2:
      ans = Lp_norm(
          MatrixDynamic::Identity(Vsize, Vsize) - V_calc.transpose() * V_calc,
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Identity_V_norm(metric_settings.p);
      }
      break;

    case MetricType::ERROR_SIGMA:
      abs_err = S_true.diagonal() - S_calc;
      error = metric_settings.is_relative
                  ? (S_true.diagonal().array() == 0)
                        .select(0, abs_err.cwiseQuotient(S_true.diagonal()))
                  : abs_err;
      ans = Lp_norm(error, metric_settings.p);
      break;

    case MetricType::RECON_ERROR:
      ans = Lp_norm(A_true - A_calc, metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Lp_norm(A_true, metric_settings.p);
      }
      break;

    case MetricType::MAX_DEVIATION:
      if (metric_settings.is_relative) {
        ans = std::max(
            {count_metrics({U_DEVIATION1, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true),
             count_metrics({U_DEVIATION2, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true),
             count_metrics({V_DEVIATION1, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true),
             count_metrics({V_DEVIATION2, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true)});
      } else {
        ans = std::max(
            {count_metrics({U_DEVIATION1, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true),
             count_metrics({U_DEVIATION2, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true),
             count_metrics({V_DEVIATION1, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true),
             count_metrics({V_DEVIATION2, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true)});
      }
      break;

    default:
      throw std::runtime_error("ERROR: No such metric!");
      break;
  }
  return ans;
}

};  // namespace SVD_Project

#endif  // SVD_TEST_HPP
