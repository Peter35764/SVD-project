#ifndef SVD_TEST_CONFIG_H
#define SVD_TEST_CONFIG_H

#include <type_traits>

#include "../SVD_project.h"

namespace SVD_Project {

// Traits для определения, требует ли алгоритм передачи спектра.
// По умолчанию алгоритм не требует передачи спектра.
template <typename SVDClass>
struct requires_sigma : std::false_type {};

template <typename Matrix>
struct requires_sigma<RevJac_SVD<Matrix>> : std::true_type {};

template <typename Matrix>
struct requires_sigma<v0_RevJac_SVD<Matrix>> : std::true_type {};

template <typename Matrix>
struct requires_sigma<TGKInv_SVD<Matrix>> : std::true_type {};

template <typename FloatingPoint, typename MatrixType>
const std::vector<typename SVD_Test<FloatingPoint, MatrixType>::AlgorithmInfo>
    SVD_Test<FloatingPoint, MatrixType>::algorithmsInfo = {
        createAlgorithmInfoEntry<SVD_Project::GivRef_SVD>(
            "SVD_Project::GivRef_SVD"),
        createAlgorithmInfoEntry<SVD_Project::v0_GivRef_SVD>(
            "SVD_Project::v0_GivRef_SVD"),
        createAlgorithmInfoEntry<SVD_Project::v1_GivRef_SVD>(
            "SVD_Project::v1_GivRef_SVD"),
        createAlgorithmInfoEntry<SVD_Project::RevJac_SVD>(
            "SVD_Project::RevJac_SVD"),
        createAlgorithmInfoEntry<SVD_Project::v0_RevJac_SVD>(
            "SVD_Project::v0_RevJac_SVD"),
        createAlgorithmInfoEntry<SVD_Project::NaiveMRRR_SVD>(
            "SVD_Project::NaiveMRRR_SVD"),
        createAlgorithmInfoEntry<SVD_Project::NaiveBidiagSVD>(
            "SVD_Project::NaiveBidiagSVD"),
        createAlgorithmInfoEntry<SVD_Project::TGKInv_SVD>(
            "SVD_Project::TGKInv_SVD"),
        createAlgorithmInfoEntry<Eigen::JacobiSVD>("Eigen::JacobiSVD")};

}  // namespace SVD_Project

#endif  // SVD_TEST_CONFIG_H
