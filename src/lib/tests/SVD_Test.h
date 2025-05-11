/** @file SVD_Test.h
 */

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

/**
 * @brief Класс для тестирования различных алгоритмов сингулярного разложение (SVD).
 * 
 * Предоставляет фреймворк для генерации тестовых матриц, выполнения SVD с разными алгоритмами,
 * вычесление метрик и сравнения результатов.
 *
 * @tparam FloatingPoint Тип данных для плавающей точки (например, `float` или `double`).
 * @tparam MatrixType Тип матрицы (например, `Eigen::Matrix<FloatingPoint, Eigen::Dynamic, Eigen::Dynamic>`).
 */

template <typename FloatingPoint, typename MatrixType>
class SVD_Test {
 public:
  using MatrixDynamic =
      Eigen::Matrix<FloatingPoint, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorDynamic = Eigen::Matrix<FloatingPoint, Eigen::Dynamic, 1>;

  /**
   * @brief  Метрики, используемые при тестировании SVD.
   */
  enum MetricType {
    U_DEVIATION1, ///< Отклонение U от ортогональности : $||I - U U^T||_{p}$
    U_DEVIATION2, ///< Отклонение U от ортогональности : $||I - U^T U||_{p}$
    V_DEVIATION1, ///< Отклонение V от ортогональности : $||I - V V^T||_{p}$
    V_DEVIATION2, ///< Отклонение V от ортогональности : $||I - V^T V||_{p}$
    ERROR_SIGMA,  ///< Ошибка в сингулярных значениях : $|| \Sigma_{true} - \Sigma_{calc} ||_p$ (абсолютная) или $|| (\Sigma_{true} - \Sigma_{calc}) ./ \Sigma_{true} ||_p$ (относительная)
    RECON_ERROR,  ///< Ошибка реконструкции : $||(A - A_{reconstructed})||_{p}$ (абсолютная) или $|| (A - A_{reconstructed}) ./ A ||_{p}$ (относительная)
    MAX_DEVIATION ///< Максимальное из отклонений U и V (использующих норму Lp).
  };

    /**
    * @brief Настройки для конкретной метрики.
    */
  struct MetricSettings {
    MetricType type;   ///< Тип метрики.
    FloatingPoint p;   ///< Параметр нормы.
    bool is_relative;  ///< true, если ошибка относительная, false – абсолютная.
    std::string name;  ///< Имя метрики (будет дополнено "(rel)" или "(abs)").
    bool enabled;      ///< Флаг: если true – метрика выводится.
    
    /**
    * @brief Конструктор для MetricSettings.
    * @param type_ Тип метрики.
    * @param p_ Параметр нормы.
    * @param is_relative_ Является ли метрика относительной.
    * @param name_ Базовое имя метрики.
    * @param enabled_ Включена ли метрика.
    * @throw std::invalid_argument Если `_p` <= 0.
    */
    MetricSettings(MetricType type_, FloatingPoint p_, bool is_relative_,
                   std::string name_, bool enabled_);

    /**
    * @brief Генерирует полное имя метрики.
    * @param baseName Базовое имя.
    * @param relative Флаг относительности.
    * @param type Тип метрики.
    * @return Сгенерированное имя.
    */
    std::string generateName(const std::string &baseName, bool relative,
                             MetricType type);

    /** 
    * @brief Оператор сравнения для использования в ассоциативных контейнерах (например std::map).
    */
    bool operator<(const MetricSettings &other) const {
      return std::tie(type, p, is_relative, name, enabled) <
             std::tie(other.type, other.p, other.is_relative, other.name,
                      other.enabled);
    }
  };

    /**
    * @brief Структура для хранения результатов вычисления SVD.
    */
  struct SVDResult {
    MatrixDynamic U; ///< Вычисленные левые сингулярные векторы.
    VectorDynamic S; ///< Вычисленные сингулярные значения.
    MatrixDynamic V; ///< Вычисленные правые сингулярные векторы.
  };

    /**
    * @brief Настройки для одного запуска тестовой функции SVD.
    */
  struct svd_test_funcSettings {
    std::string fileName;  ///< Имя файла для вывода результатов.
    std::vector<FloatingPoint>
        SigmaMaxMinRatiosVec;  ///< Коэффициенты sigma_max/sigma_min. Чем больше, тем хуже обусловленность, 1 = все сигмы равны.
    std::vector<std::pair<int, int>>
        MatSizesVec;  ///< Размеры тестируемых матриц.
    int n;            ///< Количество итераций (выборка).
    std::string
        algorithmName;  ///< Название алгоритма SVD для тестирования (должно соответствовать имени, зарегистрированному в `algorithmsInfo`).
    int lineNumber;     ///< Номер строки вывода прогресса в консоли.
    std::vector<MetricSettings> metricsSettings;  ///< Настройки метрик.
    bool solve_with_sigmas;  ///< Флаг управления передачей спектра.
  };

    /**
    *@brief Конструктор по умолчанию.
    */
  SVD_Test();

  /**
  * @brief Конструктор, который немедленно запускает тесты на основе предоставленных настроек.
  * @param vec_settings Вектор настроек тестов для различных алгоритмов.
  */
  SVD_Test(const std::vector<svd_test_funcSettings> &vec_settings);

  /**
  * @brief Запускает один тест SVD для конкретного алгоритма и настроек.
  * этот метод является внутренней частью тестового фреймворка и обычно вызывается 
  * из метода run_tests_parallel.
  *
  * @tparam gen_cl Шаблон класса генератора матриц.
  * @tparam svd_cl Шаблон класса SVD.
  * @param settings Настройки теста.
  */
  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(svd_test_funcSettings settings);

  // ======================
  //     Static methods
  // ======================

  /**
   * @brief Вычисляет норму Lpq матрицы
   *
   * Формула нормы:
   * $$||M||_{p,q} = \left( \sum_{i=1}^m \left( \sum_{j=1}^n |M_{ij}|^p \right)^{q/p} \right)^{1/q}$$
   *
   * @param M входная матрица.
   * @param p значение p для нормы. Должно быть > 0.
   * @param q значение q для нормы. Должно быть > 0.
   * @return Вычесленная норма Lpq
   */
  static FloatingPoint Lpq_norm(const MatrixType &M, FloatingPoint p,
                                FloatingPoint q);

  /**
   * @brief Вычисляет норму Lp матрицы.
   *
   * Является частным случаем Lpq нормы, когда p=q.
   * Формула нормы Lp:
   * $$||M||_p = \left( \sum_{i=1}^m \sum_{j=1}^n |M_{ij}|^p \right)^{1/p}$$
   *
   * @param M входная матрица.
   * @param p значение p для нормы. Должно быть > 0.
   * @return Вычесленная норма Lp
   */
  static FloatingPoint Lp_norm(const MatrixType &M, FloatingPoint p);

  /**
  * @brief Сравнивает исходную матрицу с реконструированной матрицей из результатов SVD.
  *
  * Выводит исходную и реконструированную матрицы, а также процент совпадения знаков элементов.
  *
  * @param algoName Имя использованного алгоритма.
  * @param rows Количество строк в матрице.
  * @param cols Количество столбцов в матрице.
  * @param computationOptions Опции, использованные для вычисления SVD. По умолчанию 0.
  * @param out Выходной поток для результатов и деталей сравнения. По умолчанию null_stream.
  */
  static void compareMatrices(const std::string &algoName, int rows, int cols,
                              unsigned int computationOptions = 0,
                              std::ostream &out = null_stream);

  /**
   * @brief Возвращает список имен всех зарегистрированных алгоритмов SVD.
   *
   * @return Вектор строк с именами алгоритмов.
   */
  static std::vector<std::string> getAlgorithmNames();

 protected:
   /**
   * @brief Функциональный объект для запуска тестовой функции SVD.
   *
   * Связывает имя алгоритма с соответствующим методом `svd_test_func`.
   */
  using SvdRunnerFunc =
      std::function<void(SVD_Test *, const svd_test_funcSettings &)>;

  /**
   * @brief Функциональный объект для выполнения SVD-разложения конкретным алгоритмом.
   *
   * Принимает матрицу, опции, необязательный поток для вывода отладки и необязательный
   * вектор истинных сингулярных значений. Возвращает структуру SVDResult.
   */
  using SvdExecutorFunc =
      std::function<SVDResult(const MatrixDynamic &, unsigned int,
                              std::ostream *, const VectorDynamic *)>;

  static std::map<std::string, SvdRunnerFunc> get_svd_runners();
  static std::map<std::string, SvdExecutorFunc> get_svd_executors();

  /** 
  * @brief Запускает один тест SVD для конкретного алгоритма и подробных настроек.
  *
  * Это перегруженная версия svd_test_func, принимающая параметры напрямую.
  *
  * @tparam gen_cl Шаблон класса генератора матриц.
  * @tparam svd_cl Шаблон класса SVD.
  * @param fileName Имя выходного файла.
  * @param SigmaMaxMinRatiosVec Вектор отношений сигма.
  * @param MatSizesVec Вектор размеров матриц.
  * @param n Количество выборок.
  * @param algorithmName Имя алгоритма.
  * @param lineNumber Номер строки для прогресса.
  * @param metricsSettings Настройки метрик.
  * @param solve_with_sigmas Следует ли предоставлять сигмы конструктору.
  */
  template <template <typename> class gen_cl, template <typename> class svd_cl>
  void svd_test_func(std::string fileName,
                     const std::vector<FloatingPoint> &SigmaMaxMinRatiosVec,
                     const std::vector<std::pair<int, int>> &MatSizesVec, int n,
                     const std::string &algorithmName, int lineNumber,
                     const std::vector<MetricSettings> &metricsSettings,
                     bool solve_with_sigmas);

  /**
   * @brief Запускает несколько тестов SVD в параллельных потоках.
   *
   * Каждая настройка из вектора `vec_settings` запускается в отдельном потоке
   * (с ограничением по количеству одновременно запущенных потоков).
   *
   * @param vec_settings Вектор настроек тестов для различных алгоритмов.
   */
  void run_tests_parallel(
      const std::vector<svd_test_funcSettings> &vec_settings);
  /**
   * @brief Выполняет SVD-разложение матрицы, используя указанный алгоритм.
   *
   * Этот метод выступает в роли прослойки, вызывая соответствующий
   * функциональный объект из `get_svd_executors`.
   *
   * @param algoName Имя алгоритма SVD для выполнения.
   * @param A Входная матрица для разложения.
   * @param options Опции вычисления SVD (например, флаги Eigen).
   * @param divergence_stream Поток для вывода отладочной информации алгоритмом (опционально).
   * @param true_singular_values Указатель на вектор истинных сингулярных значений (опционально).
   * @return SVDResult Структура с вычисленными U, S, V.
   * @throw std::invalid_argument Если указано неизвестное имя алгоритма.
   */
  static SVDResult execute_svd_algorithm(
      const std::string &algoName, const MatrixDynamic &A, unsigned int options,
      std::ostream *divergence_stream,
      const VectorDynamic *true_singular_values = nullptr);

  /**
  * @brief Вычисляет значение указанной метрики.
  *
  * @param metric_settings Настройки метрики для вычисления.
  * @param Usize Количество строк в U.
  * @param Vsize Количество строк в V.
  * @param U_calc Вычисленная матрица U.
  * @param V_calc Вычисленная матрица V.
  * @param S_calc Вычисленный вектор сингулярных значений.
  * @param U_true Истинная матрица U.
  * @param V_true Истинная матрица V.
  * @param S_true Истинная матрица S (диагональная).
  * @return Вычисленное значение метрики.
  * @throw std::runtime_error если тип метрики неизвестен.
  */    
  FloatingPoint count_metrics(MetricSettings metric_settings, size_t Usize,
                              size_t Vsize, const MatrixDynamic &U_calc,
                              const MatrixDynamic &V_calc,
                              const VectorDynamic &S_calc,
                              const MatrixDynamic &U_true,
                              const MatrixDynamic &V_true,
                              const MatrixDynamic &S_true);

  /**
   * @brief Печатает двумерный вектор строк в виде форматированной таблицы в указанный поток вывода.
   * @param out Выходной поток.
   * @param data Двумерный вектор строк, представляющий данные таблицы.
   */
  void printTable(std::ostream &out,
                  const std::vector<std::vector<std::string>> &data);

  /**
   * @brief Печатает двумерный вектор строк в формате CSV в указанный поток вывода.
   * @param out Выходной поток.
   * @param data Двумерный вектор строк, представляющий данные CSV.
   */
  void printCSV(std::ostream &out,
                const std::vector<std::vector<std::string>> &data);
  /**
   * @brief Преобразует числовое значение в строку.
   * @param value Числовое значение.
   * @return Строковое представление значения.
   */
  std::string num2str(FloatingPoint value);

  /**
   * @brief Структура для хранения информации об алгоритме SVD.
   *
   * Используется для регистрации алгоритмов в тестовом фреймворке.
   */
  struct AlgorithmInfo {
    std::string name;
    SvdRunnerFunc runner;
    SvdExecutorFunc executor;
  };

  /**
   * @brief Создает запись AlgorithmInfo для заданного класса SVD алгоритма.
   *
   * Этот статический метод используется при инициализации `algorithmsInfo`.
   *
   * @tparam svd_cl Шаблон класса SVD алгоритма.
   * @param name Строковое имя алгоритма.
   * @return Константная структура AlgorithmInfo.
   */
  template <template <typename> class svd_cl>
  static const AlgorithmInfo createAlgorithmInfoEntry(std::string name);
  
  /**
   * @brief Вектор константных структур AlgorithmInfo, содержащий информацию обо всех зарегистрированных алгоритмах.
   *
   * Инициализируется при запуске программы и используется для поиска и вызова алгоритмов по их имени.
   */
  static const std::vector<AlgorithmInfo> algorithmsInfo;

  /**
   * @brief Преобразует вектор сингулярных значений в диагональную матрицу.
   * @param s_calc Вектор сингулярных значений.
   * @return Диагональная матрица с сингулярными значениями на диагонали.
   */
  static MatrixDynamic convertVectorToDiagonalMatrix(
      const VectorDynamic &s_calc);
  /**
   * @brief Извлекает диагональные элементы квадратной матрицы в вектор.
   * @param diagonalMatrix Квадратная матрица.
   * @return Вектор, содержащий диагональные элементы.
   * @throw std::runtime_error Если входная матрица не квадратная.
   */
  static VectorDynamic convertSquareMatrixDiagonalToVector(
      const MatrixDynamic &diagonalMatrix);
};

using SVDT = SVD_Project::SVD_Test<
    double, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

}  // namespace SVD_Project

#include "SVD_Test.hpp"

#endif  // SVD_TEST_H
