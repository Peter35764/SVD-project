#include "lib/SVD_project.h"

int main(){
    using SVDT = SVD_Project::SVDT;
    std::string folderName = SVD_Project::genNameForBundleFolder();

    std::vector<double> sigmaRatios = {1.01, 1.2, 1.6, 2.1, 8, 30, 50, 100};
    std::vector<std::pair<int, int>> matrixSizes = {{3, 3}, {5, 5}, {10, 10}, {20, 20}, {50, 50}};
    int sampleCount = 20;

    std::vector<SVDT::MetricSettings> metricsSettings = {
        SVDT::MetricSettings(SVDT::ERROR_SIGMA,    0.7, true,  "AVG err. sigma",      true),
        SVDT::MetricSettings(SVDT::ERROR_SIGMA,    0.7, false, "AVG err. sigma",      true),
        SVDT::MetricSettings(SVDT::RECON_ERROR,    0.7, false, "AVG recon error",     true),
        SVDT::MetricSettings(SVDT::RECON_ERROR,    0.7, true,  "AVG recon error",     true),
        SVDT::MetricSettings(SVDT::MAX_DEVIATION,  0.7, false, "AVG max deviation",   true),
        SVDT::MetricSettings(SVDT::MAX_DEVIATION,  0.7, true,  "AVG max deviation",   true)
    };

    SVDT::svd_test_funcSettings settingsNewton{
        folderName + "/newton_polar_table.txt",
        sigmaRatios,
        matrixSizes,
        sampleCount,
        "SVD_Project::Newton_Polar_SVD",
        3,                   
        metricsSettings,
        false
    };

    SVDT::svd_test_funcSettings settingsJacobi{
        folderName + "/reference_JacobiSVD_table.txt",
        sigmaRatios,
        matrixSizes,
        sampleCount,
        "Eigen::JacobiSVD",
        1,
        metricsSettings,
        false
    };


    std::vector<SVDT::svd_test_funcSettings> allSettings = {
        settingsJacobi, settingsNewton
    };

    SVDT tester(allSettings);

    std::cout << "\nResults have been saved in folder: " << folderName << "\n";

}