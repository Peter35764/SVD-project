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

#include "../SVD_project.h"
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

// Traits для определения, требует ли алгоритм передачи спектра.
// По умолчанию алгоритм не требует передачи спектра.
template <typename SVDClass>
struct requires_sigma : std::false_type {};

template <typename Matrix>
struct requires_sigma<RevJac_SVD<Matrix>> : std::true_type {};

template <typename Matrix>
struct requires_sigma<v0_RevJac_SVD<Matrix>> : std::true_type {};

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd_impl(const Matrix &A, const Vector &sigma,
                         unsigned int options, bool solve_with_sigmas,
                         std::true_type) {
  if (solve_with_sigmas) {
    return SVDClass(A, sigma, options);
  } else {
    Eigen::JacobiSVD<Matrix> svd_ref(A,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
    Vector computed_sigma = svd_ref.singularValues();
    return SVDClass(A, computed_sigma, options);
  }
}

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd_impl(const Matrix &A, const Vector &, unsigned int options,
                         bool, std::false_type) {
  return SVDClass(A, options);
}

template <typename SVDClass, typename Matrix, typename Vector>
SVDClass create_svd(const Matrix &A, const Vector &sigma, unsigned int options,
                    bool solve_with_sigmas) {
  return create_svd_impl<SVDClass>(A, sigma, options, solve_with_sigmas,
                                   requires_sigma<SVDClass>{});
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
std::map<std::string,
         typename SVD_Test<FloatingPoint, MatrixType>::SvdRunnerFunc>
SVD_Test<FloatingPoint, MatrixType>::initialize_svd_runners() {
  std::map<std::string, SvdRunnerFunc> runners;

  if (!runners.size()) {
    runners["SVD_Project::GivRef_SVD"] = [](SVD_Test *instance,
                                            const svd_test_funcSettings &s) {
      instance->template svd_test_func<SVDGenerator, SVD_Project::GivRef_SVD>(
          s);
    };
    runners["SVD_Project::v0_GivRef_SVD"] = [](SVD_Test *instance,
                                               const svd_test_funcSettings &s) {
      instance
          ->template svd_test_func<SVDGenerator, SVD_Project::v0_GivRef_SVD>(s);
    };
    runners["SVD_Project::NaiveMRRR_SVD"] = [](SVD_Test *instance,
                                               const svd_test_funcSettings &s) {
      instance
          ->template svd_test_func<SVDGenerator, SVD_Project::NaiveMRRR_SVD>(s);
    };
    runners["SVD_Project::v0_NaiveMRRR_SVD"] =
        [](SVD_Test *instance, const svd_test_funcSettings &s) {
          instance->template svd_test_func<SVDGenerator,
                                           SVD_Project::v0_NaiveMRRR_SVD>(s);
        };
    runners["SVD_Project::RevJac_SVD"] = [](SVD_Test *instance,
                                            const svd_test_funcSettings &s) {
      instance->template svd_test_func<SVDGenerator, SVD_Project::RevJac_SVD>(
          s);
    };
    runners["SVD_Project::v0_RevJac_SVD"] = [](SVD_Test *instance,
                                               const svd_test_funcSettings &s) {
      instance
          ->template svd_test_func<SVDGenerator, SVD_Project::v0_RevJac_SVD>(s);
    };
    runners["Eigen::JacobiSVD"] = [](SVD_Test *instance,
                                     const svd_test_funcSettings &s) {
      instance->template svd_test_func<SVDGenerator, Eigen::JacobiSVD>(s);
    };
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
  std::vector<std::thread> threads;

  std::counting_semaphore<THREADS> sem(THREADS);

  auto overall_start = std::chrono::high_resolution_clock::now();

  std::map<std::string,
           typename SVD_Test<FloatingPoint, MatrixType>::SvdRunnerFunc>
      svd_test_runners = initialize_svd_runners();

  for (const auto &s : vec_settings) {
    std::filesystem::create_directories(s.fileName);

    sem.acquire();
    threads.emplace_back([this, s, &test_times, &dur_mutex, &sem,
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

  for (auto &t : threads) {
    t.join();
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
          svd_gen.generate(minNM);

          U_true = svd_gen.MatrixU();
          S_true_mat = svd_gen.MatrixS();
          V_true = svd_gen.MatrixV();

          S_true_vec = S_true_mat.diagonal().head(minNM).eval();

          MatrixDynamic A = (U_true * S_true_mat * V_true.transpose()).eval();

          VectorDynamic sigma_to_pass = S_true_vec;
          auto svd_func = create_svd<svd_cl<MatrixDynamic>>(
              A, sigma_to_pass, Eigen::ComputeFullU | Eigen::ComputeFullV,
              solve_with_sigmas);

          U_calc = svd_func.matrixU();
          S_calc = svd_func.singularValues();
          V_calc = svd_func.matrixV();

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
SVD_Test<FloatingPoint, MatrixType>::initialize_svd_executors() {
  std::map<std::string, SvdExecutorFunc> executors;

  auto create_executor = []<template <typename> class SvdImpl>(
                             const MatrixDynamic &A,
                             unsigned int options) -> SVDResult {
    VectorDynamic sigma_to_pass;
    bool needs_sigma = requires_sigma<SvdImpl<MatrixDynamic>>::value;

    if (needs_sigma) {
      Eigen::JacobiSVD<MatrixDynamic> svd_ref(
          A, Eigen::ComputeThinU | Eigen::ComputeThinV);
      sigma_to_pass = svd_ref.singularValues();
    }

    auto svd = create_svd<SvdImpl<MatrixDynamic>>(A, sigma_to_pass, options,
                                                  needs_sigma);

    return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
  };

  executors["Eigen::JacobiSVD"] = [](const MatrixDynamic &A,
                                     unsigned int options) -> SVDResult {
    Eigen::JacobiSVD<MatrixDynamic> svd(A, options);
    return {svd.matrixU(), svd.singularValues(), svd.matrixV()};
  };

  executors["SVD_Project::GivRef_SVD"] =
      [&create_executor](const MatrixDynamic &A,
                         unsigned int options) -> SVDResult {
    return create_executor.template operator()<SVD_Project::GivRef_SVD>(
        A, options);
  };
  executors["SVD_Project::v0_GivRef_SVD"] =
      [&create_executor](const MatrixDynamic &A,
                         unsigned int options) -> SVDResult {
    return create_executor.template operator()<SVD_Project::v0_GivRef_SVD>(
        A, options);
  };
  executors["SVD_Project::NaiveMRRR_SVD"] =
      [&create_executor](const MatrixDynamic &A,
                         unsigned int options) -> SVDResult {
    return create_executor.template operator()<SVD_Project::NaiveMRRR_SVD>(
        A, options);
  };
  executors["SVD_Project::v0_NaiveMRRR_SVD"] =
      [&create_executor](const MatrixDynamic &A,
                         unsigned int options) -> SVDResult {
    return create_executor.template operator()<SVD_Project::v0_NaiveMRRR_SVD>(
        A, options);
  };
  executors["SVD_Project::RevJac_SVD"] =
      [&create_executor](const MatrixDynamic &A,
                         unsigned int options) -> SVDResult {
    return create_executor.template operator()<SVD_Project::RevJac_SVD>(
        A, options);
  };
  executors["SVD_Project::v0_RevJac_SVD"] =
      [&create_executor](const MatrixDynamic &A,
                         unsigned int options) -> SVDResult {
    return create_executor.template operator()<SVD_Project::v0_RevJac_SVD>(
        A, options);
  };

  return executors;
}

template <typename FloatingPoint, typename MatrixType>
std::map<std::string,
         typename SVD_Test<FloatingPoint, MatrixType>::SvdExecutorFunc>
    SVD_Test<FloatingPoint, MatrixType>::svd_executors =
        initialize_svd_executors();

template <typename FloatingPoint, typename MatrixType>
typename SVD_Test<FloatingPoint, MatrixType>::SVDResult
SVD_Test<FloatingPoint, MatrixType>::execute_svd_algorithm(
    const std::string &algoName, const MatrixDynamic &A, unsigned int options) {
  auto it = svd_executors.find(algoName);
  if (it != svd_executors.end()) {
    return it->second(A, options);
  } else {
    throw std::invalid_argument(
        "Unknown algorithm name in execute_svd_algorithm: " + algoName);
  }
}

// Реализация статического метода compareMatrices.
// compareMatrices должна принимать название алгоритма, размеры матрицы
// и поток. Она выводит изначальную матрицу, собранную матрицу и процент
// совпавших знаков.
template <typename FloatingPoint, typename MatrixType>
void SVD_Test<FloatingPoint, MatrixType>::compareMatrices(
    const std::string &algoName, int rows, int cols, std::ostream &out) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<FloatingPoint> distr(-100, 100);
  MatrixDynamic A(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) A(i, j) = distr(gen);

  SVDResult result = execute_svd_algorithm(
      algoName, A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  MatrixDynamic U_calc = result.U;
  VectorDynamic S_calc = result.S;
  MatrixDynamic V_calc = result.V;

  MatrixDynamic S_mat = MatrixDynamic::Zero(rows, cols);
  int min_dim = std::min(rows, cols);
  int num_singular_values = S_calc.size();
  for (int i = 0; i < std::min(min_dim, num_singular_values); ++i) {
    S_mat(i, i) = S_calc(i);
  }

  MatrixDynamic A_rec = U_calc * S_mat * V_calc.transpose();

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

  FloatingPoint percent = (total > 0)
                              ? (100.0 * static_cast<FloatingPoint>(count) /
                                 static_cast<FloatingPoint>(total))
                              : 0.0;

  out << "Algorithm: " << algoName << "\n";
  out << "Original Matrix (" << rows << "x" << cols << "):\n" << A << "\n\n";
  out << "Reconstructed Matrix:\n" << A_rec << "\n\n";
  out << "Percentage of matching signs (based on all elements): " << std::fixed
      << std::setprecision(2) << percent << "%\n";
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
    const MatrixDynamic &V_true, const MatrixDynamic &S_true_mat) {
  FloatingPoint ans = FloatingPoint(0);
  VectorDynamic S_true_vec;

  int minNM = std::min(Usize, Vsize);
  if (S_true_mat.rows() >= minNM && S_true_mat.cols() >= minNM) {
    S_true_vec = S_true_mat.diagonal().head(minNM).eval();
  } else {
    S_true_vec.resize(minNM);
    S_true_vec.setZero();
  }

  MatrixDynamic A_true = U_true * S_true_mat * V_true.transpose();

  MatrixDynamic S_calc_diag = MatrixDynamic::Zero(Usize, Vsize);
  int num_singular_values = S_calc.size();
  for (int i = 0; i < std::min(minNM, num_singular_values); ++i) {
    S_calc_diag(i, i) = S_calc(i);
  }

  MatrixDynamic A_calc = U_calc * S_calc_diag * V_calc.transpose();

  MatrixDynamic IdU = MatrixDynamic::Identity(Usize, Usize);
  MatrixDynamic IdV = MatrixDynamic::Identity(Vsize, Vsize);
  FloatingPoint norm_IdU = Lp_norm(IdU, metric_settings.p);
  FloatingPoint norm_IdV = Lp_norm(IdV, metric_settings.p);
  FloatingPoint norm_A_true = Lp_norm(A_true, metric_settings.p);

  switch (metric_settings.type) {
    case U_DEVIATION1:
      ans = Lp_norm((IdU - U_calc * U_calc.transpose()).eval(),
                    metric_settings.p);
      if (metric_settings.is_relative && norm_IdU != 0) {
        ans /= norm_IdU;
      }
      break;
    case U_DEVIATION2:
      ans = Lp_norm((IdU - U_calc.transpose() * U_calc).eval(),
                    metric_settings.p);
      if (metric_settings.is_relative && norm_IdU != 0) {
        ans /= norm_IdU;
      }
      break;
    case V_DEVIATION1:
      ans = Lp_norm((IdV - V_calc * V_calc.transpose()).eval(),
                    metric_settings.p);
      if (metric_settings.is_relative && norm_IdV != 0) {
        ans /= norm_IdV;
      }
      break;
    case V_DEVIATION2:
      ans = Lp_norm((IdV - V_calc.transpose() * V_calc).eval(),
                    metric_settings.p);
      if (metric_settings.is_relative && norm_IdV != 0) {
        ans /= norm_IdV;
      }
      break;
    case ERROR_SIGMA: {
      int size_to_compare = std::min(S_true_vec.size(), S_calc.size());
      if (size_to_compare > 0) {
        VectorDynamic abs_err =
            (S_true_vec.head(size_to_compare) - S_calc.head(size_to_compare))
                .cwiseAbs();
        if (metric_settings.is_relative) {
          VectorDynamic rel_err = VectorDynamic::Zero(size_to_compare);
          for (int k = 0; k < size_to_compare; ++k) {
            // Проверяем знаменатель на 0 перед делением
            if (std::abs(S_true_vec(k)) >
                std::numeric_limits<FloatingPoint>::
                    epsilon()) {  // Используем эпсилон для сравнения с 0
              rel_err(k) = abs_err(k) / std::abs(S_true_vec(k));
            } else if (std::abs(abs_err(k)) >
                       std::numeric_limits<FloatingPoint>::epsilon()) {
              // Если истинное значение 0, а ошибка не 0, относительная ошибка
              // бесконечна или велика
              rel_err(k) = std::numeric_limits<
                  FloatingPoint>::infinity();  // Или большое число
            } else {
              // Если и истинное значение, и ошибка близки к 0, относительная
              // ошибка 0
              rel_err(k) = 0.0;
            }
          }
          ans = Lp_norm(rel_err, metric_settings.p);
        } else {
          ans = Lp_norm(abs_err, metric_settings.p);
        }
      } else {
        ans = 0;
      }
    } break;
    case RECON_ERROR:
      ans = Lp_norm((A_true - A_calc).eval(), metric_settings.p);
      if (metric_settings.is_relative && norm_A_true != 0) {
        ans /= norm_A_true;
      }
      break;
    case MAX_DEVIATION:
      if (metric_settings.is_relative) {
        ans = std::max(
            {count_metrics({U_DEVIATION1, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat),
             count_metrics({U_DEVIATION2, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat),
             count_metrics({V_DEVIATION1, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat),
             count_metrics({V_DEVIATION2, metric_settings.p, true, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat)});
      } else {
        ans = std::max(
            {count_metrics({U_DEVIATION1, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat),
             count_metrics({U_DEVIATION2, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat),
             count_metrics({V_DEVIATION1, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat),
             count_metrics({V_DEVIATION2, metric_settings.p, false, "", false},
                           Usize, Vsize, U_calc, V_calc, S_calc, U_true, V_true,
                           S_true_mat)});
      }
      break;
    default:
      throw std::runtime_error("ERROR: Unknown metric type specified!");
  }
  // Проверка на NaN или inf перед возвратом, если необходимо
  if (std::isnan(ans) || std::isinf(ans)) {
    // Обработка: вернуть 0, максимальное значение или другое значение по
    // умолчанию
    return std::numeric_limits<FloatingPoint>::max();  // Например
  }
  return ans;
}

}  // namespace SVD_Project

#endif  // SVD_TEST_HPP
