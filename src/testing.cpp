// testing.cpp

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "lib/SVD_project.h"

// Александр Нам, КМБО-04-20
// Any questions: alexnam16@gmail.com
// Владислав Букин, КМБО-04-20

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

#define TESTING_BUNDLE_NAME \
  "TestingResultsBundle-" << std::put_time(ptm, "%d-%m-%Y-%H%M%S")

int main() {
  using SVDT = SVD_Project::SVDT;
  namespace fs = std::filesystem;

  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm* ptm = std::localtime(&now_time);
  std::ostringstream oss;
  oss << TESTING_BUNDLE_NAME;
  std::string folderName = oss.str();
  fs::create_directory(folderName);

  std::cout << "\033[2J\033[H";

  // Задание параметров тестирования
  std::vector<double> sigmaRatios = {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100};
  std::vector<std::pair<int, int>> matrixSizes = {{3, 3}, {5, 5}, {10, 10}};
  int sampleCount = 20;

  // Настройки метрик (порядок: MetricType, p, is_relative, базовое имя, флаг
  // включения)
  std::vector<SVDT::MetricSettings> metricsSettings = {
      SVDT::MetricSettings(SVDT::ERROR_SIGMA, 0.7, true, "AVG err. sigma",
                           true),
      SVDT::MetricSettings(SVDT::ERROR_SIGMA, 0.7, false, "AVG err. sigma",
                           true),
      SVDT::MetricSettings(SVDT::RECON_ERROR, 0.7, false, "AVG recon error",
                           true),
      SVDT::MetricSettings(SVDT::RECON_ERROR, 0.7, true, "AVG recon error",
                           true),
      SVDT::MetricSettings(SVDT::MAX_DEVIATION, 0.7, false, "AVG max deviation",
                           true),
      SVDT::MetricSettings(SVDT::MAX_DEVIATION, 0.7, true, "AVG max deviation",
                           true)};

  SVDT::svd_test_funcSettings settingsJacobi{
      folderName + "/reference_JacobiSVD_table.txt",
      sigmaRatios,
      matrixSizes,
      sampleCount,
      "Eigen::JacobiSVD",  // Название алгоритма
      1,                   // Прогресс будет выводиться на строке 1
      metricsSettings,
  	  false};

  // SVDT::svd_test_funcSettings settingsGivRef{
  //     folderName + "/idea_1_GivRef_table.txt",
  //     sigmaRatios,
  //     matrixSizes,
  //     sampleCount,
  //     "SVD_Project::GivRef_SVD",
  //     2,
  //     metricsSettings,
  // 	   false};

  // SVDT::svd_test_funcSettings settingsV0{folderName + "/v0_GivRef_table.txt",
  //                                        sigmaRatios,
  //                                        matrixSizes,
  //                                        sampleCount,
  //                                        "SVD_Project::v0_GivRef_SVD",
  //                                        3,
  //                                        metricsSettings,
  // 										  false};

  // SVDT::svd_test_funcSettings settingsMRRR {
  //     folderName + "/idea_3_MRRR_table.txt",
  //     sigmaRatios,
  //     matrixSizes,
  //     sampleCount,
  //     "MRRR",
  //     4,
  //     metricsSettings
  // };

  // SVDT::svd_test_funcSettings settingsRevJac{folderName +
  // "/RevJac_table.txt",
  //                                            sigmaRatios,
  //                                            matrixSizes,
  //                                            sampleCount,
  //                                            "SVD_Project::RevJac_SVD",
  //                                            4,
  //                                            metricsSettings,
  // 											  true};

  std::vector<SVDT::svd_test_funcSettings> allSettings = {
      settingsJacobi}//, settingsGivRef, settingsV0, settingsRevJac};

  SVDT tester(allSettings);

  std::cout << "\nResults have been saved in folder: " << folderName << "\n";
  std::cin.get();
  return 0;
}
