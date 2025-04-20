#ifndef SVD_TEST_HPP
#define SVD_TEST_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <semaphore>
#include <string>

#define TESTING_BUNDLE_NAME \
  "TestingResultsBundle-" << std::put_time(ptm, "%d-%m-%Y-%H%M%S")

namespace SVD_Project {
inline std::string createTestingBundleFolder() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm* ptm = std::localtime(&now_time);
  std::ostringstream oss;
  oss << TESTING_BUNDLE_NAME;
  std::string folderName = oss.str();
  std::filesystem::create_directory(folderName);
  return folderName;
}
}

#include "config.h"
#include "../SVD_project.h"

namespace SVD_Project {

// Traits для определения, требует ли алгоритм передачи спектра.
// По умолчанию алгоритм не требует передачи спектра.
template <typename SVDClass>
struct requires_sigma : std::false_type {};

template <typename Matrix>
struct requires_sigma<RevJac_SVD<Matrix>> : std::true_type {};

template <typename Matrix>
struct requires_sigma<v0_RevJac_SVD<Matrix>> : std::true_type {};

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd(const Matrix &A, const Vector &sigma, unsigned int options,
                    bool solve_with_sigmas) {
  if constexpr (requires_sigma<SVDClass>::value) {
    if (solve_with_sigmas) {
      return SVDClass(A, sigma, options);
    } else {
      throw std::invalid_argument(
          "Алгоритм требует передачи спектра, но solve_with_sigmas == false");
    }
  } else {
    return SVDClass(A, options);
  }
}

//-----------------------------------------------------------------------------
// Исходный код тестирования SVD
//-----------------------------------------------------------------------------

template <typename FloatingPoint, typename MatrixType>
template <typename Derived>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::Lpq_norm(
    const Eigen::MatrixBase<Derived> &M, FloatingPoint p, FloatingPoint q) {
  auto abs_p = M.array().abs().pow(p);
  auto row_sums = abs_p.rowwise().sum();
  auto inner_sums = row_sums.array().pow(q / p);
  return std::pow(inner_sums.sum(), FloatingPoint(1) / q);
}

template <typename FloatingPoint, typename MatrixType>
template <typename Derived>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::Lp_norm(
    const Eigen::MatrixBase<Derived> &M, FloatingPoint p) {
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
    const std::string &baseName, bool relative, MetricType /*type*/) {
  return baseName + (relative ? " (rel)" : " (abs)");
}

template <typename FloatingPoint, typename MatrixType>
SVD_Test<FloatingPoint, MatrixType>::SVD_Test() {}

template <typename FloatingPoint, typename MatrixType>
SVD_Test<FloatingPoint, MatrixType>::SVD_Test(
    const std::vector<svd_test_funcSettings> &vec_settings) {
  run_tests_parallel(vec_settings);
}

template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::run_tests_parallel(
    const std::vector<svd_test_funcSettings> &vec_settings) {
  size_t numAlgos = vec_settings.size();
  std::mutex dur_mutex;
  std::vector<std::pair<std::string, double>> test_times;
  std::vector<std::thread> threads;

  std::counting_semaphore<THREADS> sem(THREADS);

  auto overall_start = std::chrono::high_resolution_clock::now();

  for (const auto &s : vec_settings) {
    sem.acquire();
    threads.emplace_back([this, s, &test_times, &dur_mutex, &sem]() {
      auto t_start = std::chrono::high_resolution_clock::now();

      if (s.algorithmName == "SVD_Project::GivRef_SVD") {
        this->svd_test_func<SVDGenerator, SVD_Project::GivRef_SVD>(s);
      } else if (s.algorithmName == "SVD_Project::v0_GivRef_SVD") {
        this->svd_test_func<SVDGenerator, SVD_Project::v0_GivRef_SVD>(s);
      } else if (s.algorithmName == "SVD_Project::NaiveMRRR_SVD") {
        this->svd_test_func<SVDGenerator, SVD_Project::NaiveMRRR_SVD>(s);
      } else if (s.algorithmName == "SVD_Project::v0_NaiveMRRR_SVD") {
        this->svd_test_func<SVDGenerator, SVD_Project::v0_NaiveMRRR_SVD>(s);
      } else if (s.algorithmName == "SVD_Project::RevJac_SVD") {
        this->svd_test_func<SVDGenerator, SVD_Project::RevJac_SVD>(s);
      } else if (s.algorithmName == "SVD_Project::v0_RevJac_SVD") {
        this->svd_test_func<SVDGenerator, SVD_Project::v0_RevJac_SVD>(s);
      } else if (s.algorithmName == "Eigen::JacobiSVD") {
        this->svd_test_func<SVDGenerator, Eigen::JacobiSVD>(s);
      }

      auto t_end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration<double>(t_end - t_start).count();
      {
        std::lock_guard<std::mutex> lock(dur_mutex);
        test_times.emplace_back(s.algorithmName, duration);
      }
      sem.release();
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  auto overall_end = std::chrono::high_resolution_clock::now();
  double overall_duration =
      std::chrono::duration<double>(overall_end - overall_start).count();

  std::cout << "\033[2J\033[H";
  int output_line = static_cast<int>(numAlgos) + 2;
  std::cout << "\033[" << output_line << ";0H"
            << "Full execution time = " << overall_duration << " seconds.\n";

  std::filesystem::path p(vec_settings.front().fileName);
  std::string folderName = p.parent_path().string();

  std::ofstream timeFile(folderName + "/individual_test_times.txt");
  if (timeFile) {
    timeFile << "=== Test Settings ===\n";

    timeFile << "Sigma ratios: ";
    for (const auto &r : vec_settings.front().SigmaMaxMinRatiosVec) {
      timeFile << r << " ";
    }
    timeFile << "\n";

    timeFile << "Matrix sizes: ";
    for (const auto &sz : vec_settings.front().MatSizesVec) {
      timeFile << sz.first << "x" << sz.second << " ";
    }
    timeFile << "\n";

    timeFile << "Sample count: " << vec_settings.front().n << "\n";

    timeFile << "Metrics Settings:\n";
    for (const auto &ms : vec_settings.front().metricsSettings) {
      timeFile << "  " << ms.name << " ("
               << (ms.is_relative ? "relative" : "absolute")
               << "), p = " << ms.p
               << ", enabled = " << (ms.enabled ? "true" : "false") << "\n";
    }

    timeFile << "\n=== Execution Times ===\n";
    timeFile << "Total overall time: " << overall_duration << " seconds\n";
    timeFile << "Individual algorithm execution times:\n";
    for (const auto &entry : test_times) {
      timeFile << entry.first << " : " << entry.second << " seconds\n";
    }
    timeFile.close();
  } else {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Error while creating/opening individual_test_times.txt!\n";
  }
}

template <typename FloatingPoint, typename MatrixType>
template <template <typename> class gen_cl, template <typename> class svd_cl>
void SVD_Test<FloatingPoint, MatrixType>::svd_test_func(
    svd_test_funcSettings settings) {
  svd_test_func<gen_cl, svd_cl>(
      settings.fileName, settings.SigmaMaxMinRatiosVec, settings.MatSizesVec,
      settings.n, settings.algorithmName, settings.lineNumber,
      settings.metricsSettings, settings.solve_with_sigmas);
}

template <typename FloatingPoint, typename MatrixType>
template <template <typename> class gen_cl, template <typename> class svd_cl>
void SVD_Test<FloatingPoint, MatrixType>::svd_test_func(
    std::string fileName,
    const std::vector<FloatingPoint> &SigmaMaxMinRatiosVec,
    const std::vector<std::pair<int, int>> &MatSizesVec, int n,
    const std::string &algorithmName, int lineNumber,
    const std::vector<MetricSettings> &metricsSettings,
    bool solve_with_sigmas) {
  ++flush;
  FloatingPoint generalProgressSum = 0;
  for (const auto &MatSize : MatSizesVec) {
    generalProgressSum += (MatSize.first * MatSize.second);
  }
  // Диапазоны для генерации сингулярных значений.
  const std::vector<std::pair<FloatingPoint, FloatingPoint>> Intervals = {
      {0, 1}, {1, 100}};
  FloatingPoint ProgressCoeff = n * Intervals.size() *
                                SigmaMaxMinRatiosVec.size() *
                                generalProgressSum / 100.0;
  FloatingPoint progress = 0;

  std::vector<std::vector<std::string>> table;
  // Формируем заголовок таблицы.
  std::vector<std::string> header;
  header.push_back("Dimension");
  header.push_back("Sigma-max/Sigma-min");
  header.push_back("SV interval");
  for (const auto &ms : metricsSettings) {
    if (ms.enabled) header.push_back(ms.name);
  }
  table.push_back(header);

  // Определяем типы для динамических матриц и векторов.
  using MatrixDynamic =
      Eigen::Matrix<FloatingPoint, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<FloatingPoint, Eigen::Dynamic, 1>;

  MatrixDynamic U_true, S_true, V_true;  // Истинное SVD-разложение.
  MatrixDynamic U_calc, V_calc;          // Вычисленные сингулярные векторы.
  VectorDynamic S_calc;                  // Вычисленные сингулярные значения.

  std::random_device rd;
  std::default_random_engine gen(rd());

  // Цикл по размерам матриц.
  for (const auto &MatSize : MatSizesVec) {
    const int N = MatSize.first;
    const int M = MatSize.second;
    int minNM = std::min(N, M);

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

        std::uniform_real_distribution<FloatingPoint> distrSigmaMin(
            interval.first, interval.second / SigmaMaxMinRatio);
        FloatingPoint sigma_min = distrSigmaMin(gen);
        FloatingPoint sigma_max = SigmaMaxMinRatio * sigma_min;
        // Определение допустимого промежутка сингулярных значений
        std::uniform_real_distribution<FloatingPoint> distr(sigma_min,
                                                            sigma_max);
        assert((minNM >= 2) && "Error: no columns or rows allowed");

        // Контейнер для накопления результатов по метрикам.
        std::map<MetricSettings, FloatingPoint> results;
        for (const auto &ms : metricsSettings) {
          results[ms] = FloatingPoint(0);
        }

        // Повторяем тест n раз для усреднения.
        for (size_t i = 1; i <= n; ++i) {
          gen_cl<FloatingPoint> svd_gen(N, M, gen, distr, true);
          svd_gen.generate(minNM);

          U_true = svd_gen.MatrixU();
          S_true = svd_gen.MatrixS();
          V_true = svd_gen.MatrixV();

          MatrixDynamic A = (U_true * S_true * V_true.transpose()).eval();

          bool local_solve_with_sigmas = solve_with_sigmas;
          VectorDynamic sigma_to_pass = local_solve_with_sigmas
                                            ? S_true.diagonal().eval()
                                            : VectorDynamic();
          auto svd_func = create_svd<svd_cl<MatrixDynamic>>(
              A, sigma_to_pass, Eigen::ComputeFullU | Eigen::ComputeFullV,
              local_solve_with_sigmas);

          U_calc = svd_func.matrixU();
          S_calc = svd_func.singularValues();
          V_calc = svd_func.matrixV();

          for (const auto &ms : metricsSettings) {
            results[ms] += count_metrics(ms, N, M, U_calc, V_calc, S_calc,
                                         U_true, V_true, S_true);
          }

          progress += static_cast<FloatingPoint>(M * N) / ProgressCoeff;
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
            std::cout << "\033[" << lineNumber << ";0H"
                      << progressStream.str() << "\033[0K"
                      << std::flush;
          }
        } // Конец итераций для данного набора параметров
        for (auto &pair : results) {
          pair.second /= n;
        }

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

// Реализация статического метода compareMatrices.
// compareMatrices должна принимать название алгоритма, размеры матрицы
// и поток. Она выводит изначальную матрицу, собранную матрицу и процент совпавших знаков.
template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::compareMatrices(const std::string &algoName, int rows, int cols, std::ostream &out) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<FloatingPoint> distr(-100, 100);
  MatrixDynamic A(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      A(i, j) = distr(gen);

  MatrixDynamic U_calc, V_calc;
  VectorDynamic S_calc;

  // Для алгоритмов.
  if (algoName == "Eigen::JacobiSVD") {
    Eigen::JacobiSVD<MatrixDynamic> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_calc = svd.matrixU();
    S_calc = svd.singularValues();
    V_calc = svd.matrixV();
  } else if (algoName == "SVD_Project::GivRef_SVD") {
    SVD_Project::GivRef_SVD<MatrixDynamic> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_calc = svd.matrixU();
    S_calc = svd.singularValues();
    V_calc = svd.matrixV();
  } else if (algoName == "SVD_Project::v0_GivRef_SVD") {
    SVD_Project::v0_GivRef_SVD<MatrixDynamic> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_calc = svd.matrixU();
    S_calc = svd.singularValues();
    V_calc = svd.matrixV();
  } else if (algoName == "SVD_Project::NaiveMRRR_SVD") {
    SVD_Project::NaiveMRRR_SVD<MatrixDynamic> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_calc = svd.matrixU();
    S_calc = svd.singularValues();
    V_calc = svd.matrixV();
  } else if (algoName == "SVD_Project::v0_NaiveMRRR_SVD") {
    SVD_Project::v0_NaiveMRRR_SVD<MatrixDynamic> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U_calc = svd.matrixU();
    S_calc = svd.singularValues();
    V_calc = svd.matrixV();
  } else if (algoName == "SVD_Project::RevJac_SVD") {
    {
      Eigen::JacobiSVD<MatrixDynamic> svd_ref(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      VectorDynamic sigma_to_pass = svd_ref.singularValues();
      SVD_Project::RevJac_SVD<MatrixDynamic> svd(A, sigma_to_pass, Eigen::ComputeFullU | Eigen::ComputeFullV);
      U_calc = svd.matrixU();
      S_calc = svd.singularValues();
      V_calc = svd.matrixV();
    }
  } else if (algoName == "SVD_Project::v0_RevJac_SVD") {
    {
      Eigen::JacobiSVD<MatrixDynamic> svd_ref(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      VectorDynamic sigma_to_pass = svd_ref.singularValues();
      SVD_Project::v0_RevJac_SVD<MatrixDynamic> svd(A, sigma_to_pass, Eigen::ComputeFullU | Eigen::ComputeFullV);
      U_calc = svd.matrixU();
      S_calc = svd.singularValues();
      V_calc = svd.matrixV();
    }
  } else {
    throw std::invalid_argument("Unknown algorithm name: " + algoName);
  }

  MatrixDynamic S = MatrixDynamic::Zero(rows, cols);
  int min_dim = std::min(rows, cols);
  for (int i = 0; i < min_dim; ++i)
    S(i, i) = S_calc(i);

  MatrixDynamic A_rec = U_calc * S * V_calc.transpose();

  auto sign = [](FloatingPoint val) -> int {
    return (val > 0) - (val < 0);
  };

  int count = 0;
  int total = rows * cols;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      if (sign(A(i, j)) == sign(A_rec(i, j))) ++count;

  FloatingPoint percent = 100.0 * count / total;
  out << "Algorithm: " << algoName << "\n";
  out << "Original Matrix:\n" << A << "\n";
  out << "Reconstructed Matrix:\n" << A_rec << "\n";
  out << "Percentage of matching signs: " << percent << "%\n";
}

// Реализация метода printTable.
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

// Реализация метода printCSV.
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
}

template <typename FloatingPoint, typename MatrixType>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::count_metrics(
    MetricSettings metric_settings, size_t Usize, size_t Vsize,
    const MatrixDynamic &U_calc, const MatrixDynamic &V_calc,
    const VectorDynamic &S_calc, const MatrixDynamic &U_true,
    const MatrixDynamic &V_true, const MatrixDynamic &S_true) {
  FloatingPoint ans = 0;
  VectorDynamic abs_err;
  VectorDynamic error;

  MatrixDynamic A_true = U_true * S_true * V_true.transpose();
  MatrixDynamic S_calc_diag =
      MatrixDynamic::Zero(std::min(Usize, Vsize), std::min(Usize, Vsize));
  S_calc_diag.diagonal() = S_calc;
  MatrixDynamic A_calc = U_calc * S_calc_diag * V_calc.transpose();

  switch (metric_settings.type) {
    case U_DEVIATION1:
      ans = Lp_norm(
          (MatrixDynamic::Identity(Usize, Usize) - U_calc * U_calc.transpose())
              .eval(),
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Lp_norm(MatrixDynamic::Identity(Usize, Usize).eval(),
                       metric_settings.p);
      }
      break;
    case U_DEVIATION2:
      ans = Lp_norm(
          (MatrixDynamic::Identity(Usize, Usize) - U_calc.transpose() * U_calc)
              .eval(),
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Lp_norm(MatrixDynamic::Identity(Usize, Usize).eval(),
                       metric_settings.p);
      }
      break;
    case V_DEVIATION1:
      ans = Lp_norm(
          (MatrixDynamic::Identity(Vsize, Vsize) - V_calc * V_calc.transpose())
              .eval(),
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Lp_norm(MatrixDynamic::Identity(Vsize, Vsize).eval(),
                       metric_settings.p);
      }
      break;
    case V_DEVIATION2:
      ans = Lp_norm(
          (MatrixDynamic::Identity(Vsize, Vsize) - V_calc.transpose() * V_calc)
              .eval(),
          metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Lp_norm(MatrixDynamic::Identity(Vsize, Vsize).eval(),
                       metric_settings.p);
      }
      break;
    case ERROR_SIGMA:
      abs_err = S_true.diagonal() - S_calc;
      error = metric_settings.is_relative
                  ? (S_true.diagonal().array() == 0)
                        .select(0, abs_err.cwiseQuotient(S_true.diagonal()))
                  : abs_err;
      ans = Lp_norm(error.eval(), metric_settings.p);
      break;
    case RECON_ERROR:
      ans = Lp_norm((A_true - A_calc).eval(), metric_settings.p);
      if (metric_settings.is_relative) {
        ans /= Lp_norm(A_true.eval(), metric_settings.p);
      }
      break;
    case MAX_DEVIATION:
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
  }
  return ans;
}

}  // namespace SVD_Project

#endif  // SVD_TEST_HPP
