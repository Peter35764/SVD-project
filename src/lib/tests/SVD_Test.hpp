#ifndef SVD_TEST_HPP
#define SVD_TEST_HPP

#include <Eigen/SVD>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "../SVD_project.h"
#include "SVD_Test_config.h"
#include "config.h"

#define TESTING_BUNDLE_NAME "TestBundle-" << std::put_time(ptm, "%d-%m-%Y-%H%M")

namespace SVD_Project {

std::string genNameForBundleFolder() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm *ptm = std::localtime(&now_time);
  std::ostringstream oss;
  oss << TESTING_BUNDLE_NAME;
  std::string folderName = oss.str();
  return folderName;
}

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd_impl(const Matrix &A, const Vector &sigma,
                         unsigned int options, bool solve_with_sigmas,
                         std::ostream *os, const Vector *true_sigma_values,
                         std::true_type) {
  const Vector *sigma_to_use =
      (true_sigma_values != nullptr) ? true_sigma_values : &sigma;

  if (solve_with_sigmas || (true_sigma_values != nullptr)) {
    if constexpr (std::is_constructible_v<SVDClass, const Matrix &,
                                          const Vector &, std::ostream *,
                                          unsigned int>) {
      if (os)
        return SVDClass(A, *sigma_to_use, os, options);
      else
        return SVDClass(A, *sigma_to_use, options);
    } else if constexpr (std::is_constructible_v<SVDClass, const Matrix &,
                                                 const Vector &,
                                                 unsigned int>) {
      return SVDClass(A, *sigma_to_use, options);
    } else {
      throw std::runtime_error(
          "ERROR: SVD class requires sigma but no suitable constructor found.");
    }
  } else {
    if constexpr (std::is_constructible_v<SVDClass, const Matrix &,
                                          std::ostream *, unsigned int>) {
      if (os)
        return SVDClass(A, os, options);
      else
        return SVDClass(A, options);
    } else if constexpr (std::is_constructible_v<SVDClass, const Matrix &,
                                                 unsigned int>) {
      return SVDClass(A, options);
    } else {
      throw std::runtime_error(
          "ERROR: SVD class requires sigma but no suitable constructor found "
          "and sigma not provided.");
    }
  }
}

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd_impl(const Matrix &A, const Vector &, unsigned int options,
                         bool, std::ostream *os,
                         const Vector *true_sigma_values,  // Added parameter
                         std::false_type) {
  if constexpr (std::is_constructible_v<SVDClass, const Matrix &,
                                        std::ostream *, unsigned int>) {
    if (os)
      return SVDClass(A, os, options);
    else
      return SVDClass(A, options);
  } else {
    return SVDClass(A, options);
  }
}

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd(
    const Matrix &A, const Vector &sigma, unsigned int options,
    bool solve_with_sigmas, std::ostream *os = nullptr,
    const Vector *true_sigma_values = nullptr) {  // Added parameter
  return create_svd_impl<SVDClass>(A, sigma, options, solve_with_sigmas, os,
                                   true_sigma_values,
                                   requires_sigma<SVDClass>{});
}
// ==================================================================

//-----------------------------------------------------------------------------
// Исходный код тестирования SVD
//-----------------------------------------------------------------------------

template <typename FloatingPoint, typename MatrixType>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::Lpq_norm(const MatrixType &M,
                                                            FloatingPoint p,
                                                            FloatingPoint q) {
  auto abs_p = M.array().abs().pow(p);
  auto row_sums = abs_p.rowwise().sum();
  auto inner_sums = row_sums.array().pow(q / p);
  return std::pow(inner_sums.sum(), FloatingPoint(1) / q);
}

template <typename FloatingPoint, typename MatrixType>
FloatingPoint SVD_Test<FloatingPoint, MatrixType>::Lp_norm(const MatrixType &M,
                                                           FloatingPoint p) {
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
template <template <typename> class svd_cl>
const SVD_Test<FloatingPoint, MatrixType>::AlgorithmInfo
SVD_Test<FloatingPoint, MatrixType>::createAlgorithmInfoEntry(
    std::string name) {
  return {name,
          [](SVD_Test<FloatingPoint, MatrixType> *instance,
             const svd_test_funcSettings &s) {
            instance->template svd_test_func<SVDGenerator, svd_cl>(s);
          },
          [](const MatrixDynamic &A, unsigned int options,
             std::ostream *divergence_stream,
             const VectorDynamic *true_singular_values)
              -> SVDResult {
            VectorDynamic sigma_to_pass;
            bool needs_sigma_trait =
                requires_sigma<svd_cl<MatrixDynamic>>::value;
            bool have_true_sigma_provided =
                (true_singular_values != nullptr &&
                 true_singular_values->size() > 0);  

            if (have_true_sigma_provided) {
              sigma_to_pass = *true_singular_values;
            } else if (needs_sigma_trait) {
              Eigen::JacobiSVD<MatrixDynamic> svd_ref(
                  A, Eigen::ComputeThinU | Eigen::ComputeThinV);
              sigma_to_pass = svd_ref.singularValues();
            }
            auto svd = create_svd<svd_cl<MatrixDynamic>>(
                A, sigma_to_pass, options,
                have_true_sigma_provided || needs_sigma_trait,
                divergence_stream, true_singular_values);

            return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
          }

  };
};

template <typename FloatingPoint, typename MatrixType>
std::map<std::string,
         typename SVD_Test<FloatingPoint, MatrixType>::SvdRunnerFunc>
SVD_Test<FloatingPoint, MatrixType>::get_svd_runners() {
  std::map<std::string, SvdRunnerFunc> runners;

  if (runners.empty()) {
    for (const auto &algo : algorithmsInfo) {
      if (algo.runner) runners[algo.name] = algo.runner;
    }
  }

  return runners;
}

template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::run_tests_parallel(
    const std::vector<svd_test_funcSettings> &vec_settings) {
  std::cout << "\033[2J\033[H";

  size_t numAlgos = vec_settings.size();
  std::mutex dur_mutex;
  std::vector<std::pair<std::string, double>> test_times;
  std::thread threads[THREADS];

  std::counting_semaphore<THREADS> sem(THREADS);

  auto overall_start = std::chrono::high_resolution_clock::now();

  std::map<std::string,
           typename SVD_Test<FloatingPoint, MatrixType>::SvdRunnerFunc>
      svd_test_runners = get_svd_runners();

  int thread_idx = 0;
  for (const auto &s : vec_settings) {
    std::filesystem::path p(s.fileName);
    auto dir = p.parent_path();
    if (!dir.empty()) {
      std::filesystem::create_directories(dir);
    }
    sem.acquire();
    threads[thread_idx++] = std::thread([this, s, &test_times, &dur_mutex, &sem,
                                         &overall_start, &svd_test_runners]() {
      auto t_start = std::chrono::high_resolution_clock::now();
      auto it = svd_test_runners.find(s.algorithmName);
      if (it != svd_test_runners.end()) {
        it->second(this, s);
      } else {
        {
          std::lock_guard<std::mutex> lock(cout_mutex);
          std::cerr << "\nERROR: Unknown algorithm '" << s.algorithmName
                    << "' in run_tests_parallel.\n";
        }
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

  for (int i = 0; i < thread_idx; ++i) {
    threads[i].join();
  }

  auto overall_end = std::chrono::high_resolution_clock::now();
  double overall_duration =
      std::chrono::duration<double>(overall_end - overall_start).count();

  int output_line = static_cast<int>(numAlgos) + 2;
  std::cout << "\033[" << output_line << ";0H"
            << "Full execution time = " << overall_duration << " seconds.\n";

  if (vec_settings.empty()) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr
        << "Error: Test settings vector is empty, cannot write times file.\n";
    return;
  }

  std::filesystem::path p(vec_settings.front().fileName);
  std::string folderName = p.parent_path().string();
  if (folderName.empty()) {
    folderName = ".";
  }

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
  FloatingPoint ProgressCoeff =
      n * Intervals.size() * SigmaMaxMinRatiosVec.size() * generalProgressSum;
  if (ProgressCoeff == 0) ProgressCoeff = 1;
  FloatingPoint progress = 0;
  FloatingPoint currentProgressCounter = 0;

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

  MatrixDynamic U_true, S_true_mat, V_true;  // Истинное SVD-разложение.
  MatrixDynamic U_calc, V_calc;              // Вычисленные сингулярные векторы.
  VectorDynamic S_calc;  // Вычисленные сингулярные значения.
  VectorDynamic S_true_vec;

  std::random_device rd;
  std::default_random_engine gen(rd());

  // Цикл по размерам матриц.
  for (const auto &MatSize : MatSizesVec) {
    const int N = MatSize.first;
    const int M = MatSize.second;
    int minNM = std::min(N, M);

    U_true.resize(N, N);
    U_calc.resize(N, N);
    S_true_mat.resize(N, M);
    S_calc.resize(minNM);
    S_true_vec.resize(minNM);
    V_true.resize(M, M);
    V_calc.resize(M, M);

    for (const auto &SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
      for (const auto &interval : Intervals) {
        assert((interval.first < interval.second) &&
               "Error: left boundary >= right boundary");
        assert((SigmaMaxMinRatio >= FloatingPoint(1)) &&
               "Error: Sigma ratio must be >= 1");

        if (interval.second < interval.first * SigmaMaxMinRatio &&
            interval.first != 0) {
          continue;
        }

        FloatingPoint sigma_min_lower_bound = interval.first;
        FloatingPoint sigma_min_upper_bound =
            (SigmaMaxMinRatio > 0) ? (interval.second / SigmaMaxMinRatio)
                                   : interval.second;
        if (sigma_min_upper_bound < sigma_min_lower_bound &&
            sigma_min_lower_bound > 0) {
          sigma_min_upper_bound = sigma_min_lower_bound;
        }

        std::uniform_real_distribution<FloatingPoint> distrSigmaMin(
            sigma_min_lower_bound, sigma_min_upper_bound);

        // Контейнер для накопления результатов по метрикам.
        std::map<MetricSettings, FloatingPoint> results;
        for (const auto &ms : metricsSettings) {
          if (ms.enabled) {
            results[ms] = FloatingPoint(0);
          }
        }

        // Повторяем тест n раз для усреднения.
        for (size_t i = 1; i <= n; ++i) {
          FloatingPoint sigma_min = distrSigmaMin(gen);
          FloatingPoint sigma_max = SigmaMaxMinRatio * sigma_min;
          sigma_max = std::min(sigma_max, interval.second);
          if (sigma_min > sigma_max && SigmaMaxMinRatio == 1.0) {
            sigma_min = sigma_max;
          } else if (sigma_min > sigma_max) {
            sigma_min = sigma_max / SigmaMaxMinRatio;
          }

          std::uniform_real_distribution<FloatingPoint> distr(sigma_min,
                                                              sigma_max);

          assert((minNM >= 1) &&
                 "Error: Matrix dimensions must be at least 1x1");

          gen_cl<FloatingPoint> svd_gen(N, M, gen, distr, true);
          U_true = svd_gen.getMatrixU();
          S_true_mat = svd_gen.getMatrixS();
          V_true = svd_gen.getMatrixV();

          S_true_vec = S_true_mat.diagonal().head(minNM).eval();

          MatrixDynamic A = (U_true * S_true_mat * V_true.transpose()).eval();

          auto svd_func = execute_svd_algorithm(
              algorithmName, A, Eigen::ComputeFullU | Eigen::ComputeFullV,
              nullptr, &S_true_vec);

          U_calc = svd_func.U;
          S_calc = svd_func.S;
          V_calc = svd_func.V;

          std::sort(S_calc.data(), S_calc.data() + S_calc.size(),
                    std::greater<FloatingPoint>());

          for (const auto &ms : metricsSettings) {
            if (ms.enabled) {
              results[ms] += count_metrics(ms, N, M, U_calc, V_calc, S_calc,
                                           U_true, V_true, S_true_mat);
            }
          }

          currentProgressCounter += static_cast<FloatingPoint>(M * N);
          progress = currentProgressCounter / ProgressCoeff * 100.0;
          double percent = static_cast<double>(progress);
          int barWidth = 50;
          int pos = static_cast<int>(barWidth * percent / 100.0);

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
            if (lineNumber > 0) {
              std::cout << "\033[" << lineNumber << ";0H"
                        << progressStream.str() << "\033[K" << std::flush;
            } else {
              std::cout << progressStream.str() << "\r" << std::flush;
            }
          }
        }  // Конец итераций для данного набора параметров

        for (auto &pair : results) {
          if (n > 0) {
            pair.second /= n;
          }
        }

        std::vector<std::string> row;
        // Используем num2str для всех числовых колонок, кроме Dimension
        row.push_back(std::to_string(N) + "x" +
                      std::to_string(M));  // Dimension как строка
        row.push_back(num2str(SigmaMaxMinRatio));
        row.push_back("[" + num2str(interval.first) + ", " +
                      num2str(interval.second) + "]");
        for (const auto &ms : metricsSettings) {
          if (!ms.enabled) continue;
          auto it = results.find(ms);
          if (it != results.end()) {
            row.push_back(num2str(it->second));  // Метрики тоже форматируем
          } else {
            row.push_back("N/A");
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
    std::cerr << "Error while creating/opening file: " << fileName << "!\n";
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
  {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::ostringstream progressStream;
    int barWidth = 50;
    progressStream << algorithmName << ": " << std::fixed
                   << std::setprecision(4) << 100.0 << "% [";
    for (int j = 0; j < barWidth; ++j) progressStream << "=";
    progressStream << "]";
    if (lineNumber > 0) {
      std::cout << "\033[" << lineNumber << ";0H" << progressStream.str()
                << "\033[K" << std::endl;
    } else {
      std::cout << progressStream.str() << std::endl;
    }
  }
}

template <typename FloatingPoint, typename MatrixType>
std::map<std::string,
         typename SVD_Test<FloatingPoint, MatrixType>::SvdExecutorFunc>
SVD_Test<FloatingPoint, MatrixType>::get_svd_executors() {
  std::map<std::string, SvdExecutorFunc> executors;

  if (executors.empty()) {
    for (const auto &algo : algorithmsInfo) {
      if (algo.executor) executors[algo.name] = algo.executor;
    }
  }

  return executors;
}

template <typename FloatingPoint, typename MatrixType>
typename SVD_Test<FloatingPoint, MatrixType>::SVDResult
SVD_Test<FloatingPoint, MatrixType>::execute_svd_algorithm(
    const std::string &algoName, const MatrixDynamic &A, unsigned int options,
    std::ostream *divergence_stream,
    const VectorDynamic *true_singular_values) {
  static const auto &executors = get_svd_executors();
  auto it = executors.find(algoName);
  if (it != executors.end()) {
    return it->second(A, options, divergence_stream, true_singular_values);
  } else {
    throw std::invalid_argument(
        "Unknown algorithm name in execute_svd_algorithm: " + algoName);
  }
}

template <typename FloatingPoint, typename MatrixType>
typename SVD_Test<FloatingPoint, MatrixType>::MatrixDynamic
SVD_Test<FloatingPoint, MatrixType>::convertVectorToDiagonalMatrix(
    const VectorDynamic &s_calc) {
  Eigen::Index size = s_calc.size();
  MatrixDynamic S_calc_matrix = MatrixDynamic::Zero(size, size);
  for (Eigen::Index i = 0; i < size; ++i) {
    S_calc_matrix(i, i) = s_calc(i);
  }
  return S_calc_matrix;
}

template <typename FloatingPoint, typename MatrixType>
std::vector<std::string>
SVD_Test<FloatingPoint, MatrixType>::getAlgorithmNames() {
  static const auto &executors = get_svd_executors();
  std::vector<std::string> names;
  names.reserve(executors.size());
  for (auto &kv : executors) {
    names.push_back(kv.first);
  }
  return names;
}

template <typename FloatingPoint, typename MatrixType>
typename SVD_Test<FloatingPoint, MatrixType>::VectorDynamic
SVD_Test<FloatingPoint, MatrixType>::convertSquareMatrixDiagonalToVector(
    const MatrixDynamic &S_calc_matrix) {
  Eigen::Index size = S_calc_matrix.rows();
  assert(S_calc_matrix.cols() == size && "Input matrix must be square.");
  VectorDynamic vector = VectorDynamic::Zero(size);
  for (Eigen::Index i = 0; i < size; ++i) {
    vector(i) = S_calc_matrix(i, i);
  }
  return vector;
}

template <typename FloatingPoint, typename MatrixType>
typename SVD_Test<FloatingPoint, MatrixType>::VectorDynamic
SVD_Test<FloatingPoint, MatrixType>::processSingularValues(
    const VectorDynamic &sv) {
  VectorDynamic result = sv;

  result = result.cwiseAbs();
  std::sort(
      result.data(), result.data() + result.size(),
      [](const FloatingPoint &a, const FloatingPoint &b) { return a > b; });

  return result;
}

template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::compareMatrices(
    const std::string &algoName, int rows, int cols,
    unsigned int computationOptions, std::ostream &out) {
  try {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<FloatingPoint> distr(-100, 100);
    SVDGenerator<FloatingPoint> svd_gen(rows, cols, gen, distr, true);
    int minNM = std::min(rows, cols);
    MatrixDynamic A(svd_gen.getInitialMatrix());  // create random matrix A

    VectorDynamic S_true_vec_ref = svd_gen.getMatrixS().diagonal().eval();
    std::sort(S_true_vec_ref.data(),
              S_true_vec_ref.data() + S_true_vec_ref.size(),
              std::greater<FloatingPoint>());

    SVDResult result = execute_svd_algorithm(algoName, A, computationOptions,
                                             &out, &S_true_vec_ref);

    MatrixDynamic U_calc = result.U;
    VectorDynamic S_calc = result.S;
    MatrixDynamic V_calc = result.V;

    std::sort(S_calc.data(), S_calc.data() + S_calc.size(),
              std::greater<FloatingPoint>());

    MatrixDynamic S_calc_matrix = convertVectorToDiagonalMatrix(S_calc);

    MatrixDynamic A_rec = U_calc * S_calc_matrix * V_calc.transpose();

    auto sign = [](FloatingPoint val) -> int {
      if (val == FloatingPoint(0)) return 0;
      return (val > 0) ? 1 : -1;
    };

    int count = 0;
    int total = rows * cols;

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int sign_A = sign(A(i, j));
        int sign_A_rec = sign(A_rec(i, j));
        if (sign_A == sign_A_rec) {
          count++;
        }
      }
    }

    FloatingPoint signs_percent =
        (total > 0) ? (100.0 * static_cast<FloatingPoint>(count) /
                       static_cast<FloatingPoint>(total))
                    : 0.0;
    FloatingPoint frob_norm = Lp_norm(
        processSingularValues(S_true_vec_ref) - processSingularValues(S_calc),
        2);

    std::cout << "\n=======================================\n";

    std::cout << "Algorithm: " << algoName << "\n\n";
    std::cout << "Original Matrix (" << rows << "x" << cols << "):\n"
              << A << "\n\n";
    std::cout << "Reconstructed Matrix:\n" << A_rec << "\n\n";
    std::cout << "Percentage of matching signs (based on all elements): "
              << std::fixed << std::setprecision(2) << signs_percent << "%\n";
    std::cout << "Frobenius norm of difference between initial and calculated "
                 "singular values: "
              << std::fixed << std::setprecision(3) << frob_norm << "\n";

    std::cout << "=======================================\n";

  } catch (const std::invalid_argument &e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    if (&out != &std::cout &&
        &out != &null_stream) {  // Записываем ошибку в файл
      out << "Error: Invalid argument: " << e.what() << "\n";
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "Runtime error during SVD computation: " << e.what()
              << std::endl;
    if (&out != &std::cout && &out != &null_stream) {
      out << "Error: Runtime error during SVD computation: " << e.what()
          << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "An unexpected error occurred during SVD computation: "
              << e.what() << std::endl;
    if (&out != &std::cout && &out != &null_stream) {
      out << "Error: An unexpected error occurred during SVD computation: "
          << e.what() << "\n";
    }
  }
}

// Реализация метода printTable.
template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::printTable(
    std::ostream &out, const std::vector<std::vector<std::string>> &data) {
  if (data.empty()) return;
  std::vector<size_t> widths;
  for (const auto &row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i >= widths.size()) widths.resize(i + 1, 0);
      widths[i] = std::max(widths[i], row[i].size());
    }
  }

  if (!data.empty()) {
    for (size_t i = 0; i < data[0].size(); ++i) {
      if (i < widths.size()) {
        out << std::string(widths[i] + 1, '-') << " ";
      }
    }
    out << "\n";
  }

  bool is_header = true;
  for (const auto &row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i < widths.size()) {
        out << std::left << std::setw(widths[i] + 1) << row[i];
      } else {
        out << std::left << std::setw(row[i].size() + 1) << row[i];
      }
    }
    out << "\n";
    if (is_header && data.size() > 1) {
      for (size_t i = 0; i < row.size(); ++i) {
        if (i < widths.size()) {
          out << std::string(widths[i] + 1, '-') << " ";
        }
      }
      out << "\n";
      is_header = false;
    }
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
      bool needs_quoting = (cellFormatted.find(',') != std::string::npos ||
                            cellFormatted.find('\"') != std::string::npos);
      if (needs_quoting) {
        size_t pos = cellFormatted.find('\"');
        while (pos != std::string::npos) {
          cellFormatted.replace(pos, 1, "\"\"");
          pos = cellFormatted.find('\"', pos + 2);
        }
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
