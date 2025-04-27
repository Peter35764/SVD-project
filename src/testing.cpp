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

int main() {
  using SVDT = SVD_Project::SVDT;
  namespace fs = std::filesystem;

  SVDT::compareMatrices("SVD_Project::v0_GivRef_SVD", 5, 5, std::cout);
  SVDT::compareMatrices("SVD_Project::v0_RevJac_SVD", 5, 5, std::cout);

  auto names = SVDT::getAlgorithmNames();
  std::cout << "Available SVD algorithms:\n";

  SVDT::compareMatrices("Eigen::JacobiSVD", 5, 5, std::cout);
  SVDT::compareMatrices("SVD_Project::NaiveBidiagSVD", 5, 5, std::cout);

  return 0;

  /*
  std::string folderName = SVD_Project::genNameForBundleFolder();

  // Задание параметров тестирования
  std::vector<double> sigmaRatios = {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100};
  std::vector<std::pair<int, int>> matrixSizes = {
      {3, 3}, {5, 5}, {10, 10}, {20, 20}, {50, 50}};
  int sampleCount = 20;

  // Настройки метрик (порядок: MetricType, p, is_relative, базовое имя, флаг
  // // включения)
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
  //     false};

  SVDT::svd_test_funcSettings settingsV0{folderName + "/v0_GivRef_table.txt",
                                         sigmaRatios,
                                         matrixSizes,
                                         sampleCount,
                                         "SVD_Project::v0_GivRef_SVD",
                                         3,
                                         metricsSettings,
                                         false};

  // SVDT::svd_test_funcSettings settingsMRRR{
  //     folderName + "/idea_3_MRRR_table.txt",
  //     sigmaRatios,
  //     matrixSizes,
  //     sampleCount,
  //     "MRRR",
  //     4,
  //     metricsSettings};

  // SVDT::svd_test_funcSettings settingsRevJac{folderName +
  // "/RevJac_table.txt",
  //                                            sigmaRatios,
  //                                            matrixSizes,
  //                                            sampleCount,
  //                                            "SVD_Project::v0_RevJac_SVD",
  //                                            4,
  //                                            metricsSettings,
  //                                            true};

  std::vector<SVDT::svd_test_funcSettings> allSettings = {
      settingsJacobi,
      settingsV0};  //, settingsRevJac, settingsGivRef, settingsV0,
                    // settingsRevJac};

  SVDT tester(allSettings);

  // Пример использования статического метода compareMatrices с выбранным
  // алгоритмом.

  std::cout << "\nResults have been saved in folder: " << folderName << "\n";

  return 0;
*/
}
