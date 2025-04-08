#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <semaphore>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "dqds.h"
#include "givens_refinement.h"
#include "legacy/v0_givens_refinement.h"
#include "mrrr.h"
#include "reverse_jacobi.h"
#include "tests/SVD_Test.h"
#include "tests/generate_svd.h"

using namespace SVD_Project;

#define sigma_ratio \
  {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100}  // SigmaMaxMinRatiosVec
#define matrix_size \
  {{3, 3}, {5, 5}, {10, 10}}//, {30, 30}}, {70, 70}}  // MatSizesVec
#define matrix_num_for_sample_averaging 20        // n

int main() {
  using MetricType = SVD_Test<double, Eigen::MatrixXd>::MetricType;
  using MetricSettings = SVD_Test<double, Eigen::MatrixXd>::MetricSettings;

  namespace fs = std::filesystem;

  auto start = std::chrono::high_resolution_clock::now();

  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm* ptm = std::localtime(&now_time);
  std::ostringstream oss;
  oss << "TestingResultsBundle<" << std::put_time(ptm, "%d-%m-%Y-%H%M%S")
      << ">";
  std::string folderName = oss.str();
  fs::create_directory(folderName);

  std::cout << "\033[2J\033[H";

  std::vector<std::pair<std::string, double>> test_times;
  std::mutex test_times_mutex;

  std::vector<double> sigmaRatios =
      sigma_ratio;  // macro sigma_ratio: {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100}
  std::vector<std::pair<int, int>> matrixSizes =
      matrix_size;  // macro matrix_size: {{3, 3}, {5, 5}}
  int sampleCount =
      matrix_num_for_sample_averaging;  // макрос sample count
                                        // генерируется таблица в файле
                                        // "jacobi_test_table.txt" теста метода
                                        // Eigen::JacobiSVD
  // с соотношением сингулярных чисел:  1.01, 1.2, 2, 5, 10, 50       ---    6
  // причем каждое соотношение относится к двум интервалам сингулярных чисел:
  //                       маленьких {0,1}, больших {1,100} (это не
  //                       параметризованно)   ---   2
  // с матрицами размеров: {3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}
  // /  ---   6 6*2*6 = 72 - всего столько строк будет в таблице размер выборки
  // для усреднения: 20
  //  Инициализация настроек метрик
  std::vector<MetricSettings> metricsSettings = {
      MetricSettings(MetricType::ERROR_SIGMA, 0.7, true, "AVG err. sigma",
                     true),
      MetricSettings(MetricType::ERROR_SIGMA, 0.7, false, "AVG err. sigma abs",
                     true),
      MetricSettings(MetricType::RECON_ERROR, 0.7, true, "AVG recon error",
                     true),
      MetricSettings(MetricType::RECON_ERROR, 0.7, true, "AVG recon error abs",
                     true),
      MetricSettings(MetricType::MAX_DEVIATION, 0.7, true, "AVG max deviation",
                     true),
      MetricSettings(MetricType::MAX_DEVIATION, 0.7, true,
                     "AVG max deviation abs", true)};

  // int flush_string = 1;
  // {
  //   std::lock_guard<std::mutex> lock(cout_mutex);
  // }

  // // idea 1
  // thread_semaphore.acquire();
  // std::thread t1(
  //     [&]() {
  //       std::string algo_name = "JacobiSVD";
  //       std::string file_name = folderName + "/" +
  //       "reference_JacobiSVD_table.txt"; auto t_start =
  //       std::chrono::high_resolution_clock::now(); svd_test_func<double,
  //       SVDGenerator, Eigen::JacobiSVD>(
  //           file_name,
  //           sigma_ratio,
  //           matrix_size,
  //           matrix_num_for_sample_averaging,
  //           algo_name,
  //           flush_string++,
  //           metricsSettings);
  //       auto t_end = std::chrono::high_resolution_clock::now();
  //       double duration = std::chrono::duration<double>(t_end -
  //       t_start).count();
  //       {
  //         std::lock_guard<std::mutex> lock(test_times_mutex);
  //         test_times.emplace_back(algo_name, duration);
  //       }
  //       thread_semaphore.release();
  //     });

  // thread_semaphore.acquire();
  // std::thread t2(
  //     [&]() {
  //       std::string algo_name = "GivRef_SVD";
  //       std::string file_name = folderName + "/" + "idea_1_GivRef_table.txt";
  //       auto t_start = std::chrono::high_resolution_clock::now();
  //       svd_test_func<double, SVDGenerator, SVD_Project::GivRef_SVD>(
  //           file_name,
  //           sigma_ratio,
  //           matrix_size,
  //           matrix_num_for_sample_averaging,
  //           algo_name,
  //           flush_string++,
  //           metricsSettings);
  //       auto t_end = std::chrono::high_resolution_clock::now();
  //       double duration = std::chrono::duration<double>(t_end -
  //       t_start).count();
  //       {
  //         std::lock_guard<std::mutex> lock(test_times_mutex);
  //         test_times.emplace_back(algo_name, duration);
  //       }
  //       thread_semaphore.release();
  //     });

  // thread_semaphore.acquire();
  // std::thread t3(
  //     [&]() {
  //       std::string algo_name = "v0_GivRef_SVD";
  //       std::string file_name = folderName + "/" + "v0_GivRef_table.txt";
  //       auto t_start = std::chrono::high_resolution_clock::now();
  //       svd_test_func<double, SVDGenerator, SVD_Project::v0_GivRef_SVD>(
  //           file_name,
  //           sigma_ratio,
  //           matrix_size,
  //           matrix_num_for_sample_averaging,
  //           algo_name,
  //           flush_string++,
  //           metricsSettings);
  //       auto t_end = std::chrono::high_resolution_clock::now();
  //       double duration = std::chrono::duration<double>(t_end -
  //       t_start).count();
  //       {
  //         std::lock_guard<std::mutex> lock(test_times_mutex);
  //         test_times.emplace_back(algo_name, duration);
  //       }
  //       thread_semaphore.release();
  //     });
  // // idea 2
  // // thread_semaphore.acquire();
  // // std::thread t3([&]() {
  // //     std::string algo_name = "RevJac_SVD";
  // //     std::string file_name = folderName + "/" +
  // "idea_2_RevJac_table.txt";
  // //     auto t_start = std::chrono::high_resolution_clock::now();
  // //     svd_test_func<double, SVDGenerator, SVD_Project::RevJac_SVD>(
  // //         file_name,
  // //         sigma_ratio,
  // //         matrix_size,
  // //         matrix_num_for_sample_averaging,
  // //         algo_name,
  // //         flush_string++,
  // //         metricsSettings);
  // //     auto t_end = std::chrono::high_resolution_clock::now();
  // //     double duration = std::chrono::duration<double>(t_end -
  // //     t_start).count();
  // //     {
  // //         std::lock_guard<std::mutex> lock(test_times_mutex);
  // //         test_times.emplace_back(algo_name, duration);
  // //     }
  // //     thread_semaphore.release();
  // // });

  // thread_semaphore.acquire();
  // std::thread t4(
  //     [&]() {
  //       std::string algo_name = "MRRR";
  //       std::string file_name = folderName + "/" + "idea_3_MRRR_table.txt";
  //       auto t_start = std::chrono::high_resolution_clock::now();
  //       svd_test_func<double, SVDGenerator, MRRR_SVD>(
  //           file_name,
  //           sigma_ratio,
  //           matrix_size,
  //           matrix_num_for_sample_averaging,
  //           algo_name,
  //           flush_string++,
  //           metricsSettings);
  //       auto t_end = std::chrono::high_resolution_clock::now();
  //       double duration = std::chrono::duration<double>(t_end -
  //       t_start).count();
  //       {
  //         std::lock_guard<std::mutex> lock(test_times_mutex);
  //         test_times.emplace_back(algo_name, duration);
  //       }
  //       thread_semaphore.release();
  //     });

  // t1.join();
  // t2.join();
  // t4.join();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> durationGlobal = end - start;

  std::cout << "\033[5;0H";
  std::cout << "\nFull execution time = " << durationGlobal.count()
            << " seconds.\n";
  // {
  //   std::lock_guard<std::mutex> lock(cout_mutex);
  // }
  std::ofstream timeFile(folderName + "/" + "individual_test_times.txt");
  if (timeFile) {
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

    timeFile << "Sample count: " << sampleCount << "\n";

    timeFile << "Metrics Settings:\n";
    for (const auto& ms : metricsSettings) {
      timeFile << "  " << ms.name << " ("
               << (ms.relative ? "relative" : "absolute") << "), p = " << ms.p
               << "\n";
    }
    timeFile << "\n=== Execution Times ===\n";

    timeFile << "Max threads: " << THREADS << "\n\n";
    timeFile << "Total execution time: " << durationGlobal.count()
             << " seconds\n\n";
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
