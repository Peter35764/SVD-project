#ifndef SVD_TEST_H
#define SVD_TEST_H

// FloatingPoint sum = 0;
// for (size_t n = 0; n < M.rows(); ++n) {
//   FloatingPoint inner_sum = 0;
//   for (size_t m = 0; m < M.cols(); ++m) {
//     inner_sum += std::pow(std::abs(M(n, m), p));
//   }
//   sum += std::pow(inner_sum, q / p);
// }
// return std::pow(sum, FloatingPoint(1) / q);

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

template <typename FloatingPoint, typename MatrixType>
class SVD_Test {
 public:
  // https://en.wikipedia.org/wiki/Matrix_norm#%22Entry-wise%22_matrix_norms
  static FloatingPoint Lpq_norm(const Eigen::MatrixBase<MatrixType> &M,
                                FloatingPoint p, FloatingPoint q);

  static FloatingPoint Lp_norm(const Eigen::MatrixBase<MatrixType> &M,
                               FloatingPoint p);

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
    MetricType type;
    FloatingPoint p;
    bool is_relative;
    std::string name;
    bool enabled;

    MetricSettings(MetricType type_, FloatingPoint p_, bool is_relative_,
                   std::string name_, bool enabled_);

    std::string generateName(const std::string &baseName, bool relative,
                             MetricType type);

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

  using MatrixDynamic =
      Eigen::Matrix<FloatingPoint, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<FloatingPoint, Eigen::Dynamic, 1>;

  struct svd_test_funcSettings {
    std::string fileName;
    std::vector<FloatingPoint> SigmaMaxMinRatiosVec;
    std::vector<std::pair<int, int>> MatSizesVec;
    int n;
    std::string algorithmName;
    int lineNumber;
    std::vector<MetricSettings> metricsSettings;
  };

  SVD_Test();

  SVD_Test(std::vector<svd_test_funcSettings> vec_settings);

  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(svd_test_funcSettings settings);

  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(
      std::string fileName,
      const std::vector<FloatingPoint>
          &SigmaMaxMinRatiosVec,  // Чем больше, тем хуже обусловленность, 1 = все сигмы равны
      const std::vector<std::pair<int, int>> &MatSizesVec, const int n,
      const std::string &algorithmName, int lineNumber,
      const std::vector<MetricSettings> &metricsSettings);

 protected:
  static inline size_t flush = 0;

  void printTable(std::ostream &out,
                  const std::vector<std::vector<std::string>> &data);
  void printCSV(std::ostream &out,
                const std::vector<std::vector<std::string>> &data);
  std::string num2str(FloatingPoint value);

  FloatingPoint count_metrics(MetricSettings metric_settings, size_t Usize,
                              size_t Vsize, const MatrixDynamic &U_calc,
                              const MatrixDynamic &V_calc, const VectorDynamic &S_calc,
                              const MatrixDynamic &U_true, const MatrixDynamic &V_true,
                              const MatrixDynamic &S_true);
};

};  // namespace SVD_Project

#endif  // SVD_TEST_H
