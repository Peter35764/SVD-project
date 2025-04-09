#ifndef SVD_TEST_H
#define SVD_TEST_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <semaphore>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <algorithm>

namespace SVD_Project {
  inline std::mutex cout_mutex;
  int flush = 0;
}

namespace SVD_Project {

template <typename FloatingPoint, typename MatrixType>
class SVD_Test {
 public:
  template <typename Derived>
  static FloatingPoint Lpq_norm(const Eigen::MatrixBase<Derived>& M, FloatingPoint p, FloatingPoint q);

  template <typename Derived>
  static FloatingPoint Lp_norm(const Eigen::MatrixBase<Derived>& M, FloatingPoint p);

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
    MetricType type;      // Тип метрики.
    FloatingPoint p;      // Параметр нормы.
    bool is_relative;     // true, если ошибка относительная, false – абсолютная.
    std::string name;     // Имя метрики (будет дополнено "(rel)" или "(abs)").
    bool enabled;         // Флаг: если true – метрика выводится.

    MetricSettings(MetricType type_, FloatingPoint p_, bool is_relative_,
                   std::string name_, bool enabled_);

    std::string generateName(const std::string &baseName, bool relative, MetricType type);

    bool operator<(const MetricSettings &other) const {
      if (type != other.type)
        return type < other.type;
      if (p != other.p)
        return p < other.p;
      if (is_relative != other.is_relative)
        return is_relative < other.is_relative;
      if (name != other.name)
        return name < other.name;
      return enabled < other.enabled;
    }
  };

  using MatrixDynamic = Eigen::Matrix<FloatingPoint, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<FloatingPoint, Eigen::Dynamic, 1>;

  struct svd_test_funcSettings {
    std::string fileName;                          // Имя файла для вывода результатов.
    std::vector<FloatingPoint> SigmaMaxMinRatiosVec; // Коэффициенты sigma_max/sigma_min.
    std::vector<std::pair<int, int>> MatSizesVec;   // Размеры тестируемых матриц.
    int n;                                         // Количество итераций (выборка).
    std::string algorithmName;                     // Название алгоритма (для отображения прогресса).
    int lineNumber;                                // Номер строки вывода прогресса в консоли.
    std::vector<MetricSettings> metricsSettings;   // Настройки метрик.
  };

  SVD_Test();

  SVD_Test(const std::vector<svd_test_funcSettings> &vec_settings);

  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(svd_test_funcSettings settings);

 protected:
  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(
      std::string fileName,
      const std::vector<FloatingPoint> &SigmaMaxMinRatiosVec, // Чем больше, тем хуже обусловленность, 1 = все сигмы равны
      const std::vector<std::pair<int, int>> &MatSizesVec, int n,
      const std::string &algorithmName, int lineNumber,
      const std::vector<MetricSettings> &metricsSettings);

  void run_tests_parallel(const std::vector<svd_test_funcSettings> &vec_settings);

  void printTable(std::ostream &out, const std::vector<std::vector<std::string>> &data);
  void printCSV(std::ostream &out, const std::vector<std::vector<std::string>> &data);
  std::string num2str(FloatingPoint value);

  FloatingPoint count_metrics(MetricSettings metric_settings, size_t Usize,
                              size_t Vsize, const MatrixDynamic &U_calc,
                              const MatrixDynamic &V_calc, const VectorDynamic &S_calc,
                              const MatrixDynamic &U_true, const MatrixDynamic &V_true,
                              const MatrixDynamic &S_true);
};
using SVDT = SVD_Project::SVD_Test<
    double, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

}  // namespace SVD_Project

#include "SVD_Test.hpp"

#endif  // SVD_TEST_H
