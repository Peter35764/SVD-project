#ifndef SVD_TEST_H
#define SVD_TEST_H

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../SVD_project.h"

namespace SVD_Project {
inline std::mutex cout_mutex;
int flush = 0;
}  // namespace SVD_Project

namespace SVD_Project {

// =========== buffer for no output ===========
class NullBuffer : public std::streambuf {
 public:
  int overflow(int c) override { return c; }
};

static NullBuffer null_buffer;
static std::ostream null_stream(&null_buffer);
// ============================================

template <typename FloatingPoint, typename MatrixType>
class SVD_Test {
 public:
  using MatrixDynamic =
      Eigen::Matrix<FloatingPoint, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<FloatingPoint, Eigen::Dynamic, 1>;

  enum MetricType {
    U_DEVIATION1,
    U_DEVIATION2,
    V_DEVIATION1,
    V_DEVIATION2,
    ERROR_SIGMA,
    RECON_ERROR,
    MAX_DEVIATION
  };

  struct MetricSettings {
    MetricType type;   // Тип метрики.
    FloatingPoint p;   // Параметр нормы.
    bool is_relative;  // true, если ошибка относительная, false – абсолютная.
    std::string name;  // Имя метрики (будет дополнено "(rel)" или "(abs)").
    bool enabled;      // Флаг: если true – метрика выводится.

    MetricSettings(MetricType type_, FloatingPoint p_, bool is_relative_,
                   std::string name_, bool enabled_);

    std::string generateName(const std::string &baseName, bool relative,
                             MetricType type);

    // for use in map
    bool operator<(const MetricSettings &other) const {
      return std::tie(type, p, is_relative, name, enabled) <
             std::tie(other.type, other.p, other.is_relative, other.name,
                      other.enabled);
    }
  };

  struct SVDResult {
    MatrixDynamic U;
    VectorDynamic S;
    MatrixDynamic V;
  };

  struct svd_test_funcSettings {
    std::string fileName;  // Имя файла для вывода результатов.
    std::vector<FloatingPoint>
        SigmaMaxMinRatiosVec;  // Коэффициенты sigma_max/sigma_min. Чем больше,
                               // тем хуже обусловленность, 1 = все сигмы равны.
    std::vector<std::pair<int, int>>
        MatSizesVec;  // Размеры тестируемых матриц.
    int n;            // Количество итераций (выборка).
    std::string
        algorithmName;  // Название алгоритма (для отображения прогресса).
    int lineNumber;     // Номер строки вывода прогресса в консоли.
    std::vector<MetricSettings> metricsSettings;  // Настройки метрик.
    bool solve_with_sigmas;  // Флаг управления передачей спектра.
  };

  SVD_Test();

  SVD_Test(const std::vector<svd_test_funcSettings> &vec_settings);

  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(svd_test_funcSettings settings);

  // ======================
  //     Static methods
  // ======================

  static FloatingPoint Lpq_norm(const MatrixType &M, FloatingPoint p,
                                FloatingPoint q);

  static FloatingPoint Lp_norm(const MatrixType &M, FloatingPoint p);

  static void compareMatrices(const std::string &algoName, int rows, int cols,
                              unsigned int computationOptions = 0,
                              std::ostream &out = null_stream);

  static std::vector<std::string> getAlgorithmNames();

 protected:
  using SvdRunnerFunc =
      std::function<void(SVD_Test *, const svd_test_funcSettings &)>;

  using SvdExecutorFunc =
      std::function<SVDResult(const MatrixDynamic &, unsigned int, std::ostream*, const VectorDynamic*)>;


  static std::map<std::string, SvdRunnerFunc> get_svd_runners();
  static std::map<std::string, SvdExecutorFunc> get_svd_executors();

  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(std::string fileName,
                     const std::vector<FloatingPoint> &SigmaMaxMinRatiosVec,
                     const std::vector<std::pair<int, int>> &MatSizesVec, int n,
                     const std::string &algorithmName, int lineNumber,
                     const std::vector<MetricSettings> &metricsSettings,
                     bool solve_with_sigmas);
  void run_tests_parallel(
      const std::vector<svd_test_funcSettings> &vec_settings);

  static SVDResult execute_svd_algorithm(const std::string &algoName,
                                         const MatrixDynamic &A,
                                         unsigned int options,
                                         std::ostream* divergence_stream,
                                         const VectorDynamic* true_singular_values = nullptr);


  FloatingPoint count_metrics(MetricSettings metric_settings, size_t Usize,
                              size_t Vsize, const MatrixDynamic &U_calc,
                              const MatrixDynamic &V_calc,
                              const VectorDynamic &S_calc,
                              const MatrixDynamic &U_true,
                              const MatrixDynamic &V_true,
                              const MatrixDynamic &S_true);

  void printTable(std::ostream &out,
                  const std::vector<std::vector<std::string>> &data);
  void printCSV(std::ostream &out,
                const std::vector<std::vector<std::string>> &data);
  std::string num2str(FloatingPoint value);

  struct AlgorithmInfo {
    std::string name;
    SvdRunnerFunc runner;
    SvdExecutorFunc executor;
  };

  template <template <typename> class svd_cl>
  static const AlgorithmInfo createAlgorithmInfoEntry(std::string name);
  static const std::vector<AlgorithmInfo> algorithmsInfo;

  static MatrixDynamic convertVectorToDiagonalMatrix(
      const VectorDynamic &s_calc);
  static VectorDynamic convertSquareMatrixDiagonalToVector(
      const MatrixDynamic &diagonalMatrix);
};

using SVDT = SVD_Project::SVD_Test<
    double, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

}  // namespace SVD_Project

#include "SVD_Test.hpp"

#endif  // SVD_TEST_H
