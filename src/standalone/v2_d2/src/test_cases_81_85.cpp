#include "test_common.h" // Включает все необходимое, включая perform_single_precision_run

// Вспомогательная функция-диспетчер для диапазона NumDigits 81-85
// Она будет вызываться из run_precision_study
template<int AlgoID,
         template<typename> class SVDAlgoClass,
         template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_81_85(
    int current_digits_val, // Это значение из цикла в run_precision_study
    const std::pair<int, int>& mat_size_pair,
    float sigma_ratio_for_gen_float,
    int num_runs_for_averaging)
{
    switch (current_digits_val) {
        case 81: return perform_single_precision_run<81, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
        case 82: return perform_single_precision_run<82, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
        case 83: return perform_single_precision_run<83, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
        case 84: return perform_single_precision_run<84, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
        case 85: return perform_single_precision_run<85, AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
        default:
            // Эта ветка не должна вызываться, если current_digits_val действительно в диапазоне 81-85
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cerr << "Error: dispatch_digits_81_85 called with unexpected digits: " << current_digits_val << std::endl;
            }
            PrecisionStudyResultRow error_row;
            error_row.iterations_count = -3; error_row.digits = current_digits_val; /* ... заполнить остальные поля ... */
            return error_row;
    }
}

// Явные инстанциации для каждого AlgoID, который будет использовать этот диапазон.
// Это гарантирует, что код будет сгенерирован в этой единице компиляции.
// Тебе нужно будет повторить это для каждого твоего SVDAlgoClass (AOIR_SVD_1, _4, _5, _6, _8)
// и для SVDGeneratorEngine (SVD_Project::SVDGenerator).

// Для AlgoID = 1
template PrecisionStudyResultRow dispatch_digits_81_85<1, SVD_Project::AOIR_SVD_1, SVD_Project::SVDGenerator>(int, const std::pair<int, int>&, float, int);
// Для AlgoID = 4
template PrecisionStudyResultRow dispatch_digits_81_85<4, SVD_Project::AOIR_SVD_4, SVD_Project::SVDGenerator>(int, const std::pair<int, int>&, float, int);
// Для AlgoID = 5
template PrecisionStudyResultRow dispatch_digits_81_85<5, SVD_Project::AOIR_SVD_5, SVD_Project::SVDGenerator>(int, const std::pair<int, int>&, float, int);
// Для AlgoID = 6
template PrecisionStudyResultRow dispatch_digits_81_85<6, SVD_Project::AOIR_SVD_6, SVD_Project::SVDGenerator>(int, const std::pair<int, int>&, float, int);
// Для AlgoID = 8
template PrecisionStudyResultRow dispatch_digits_81_85<8, SVD_Project::AOIR_SVD_8, SVD_Project::SVDGenerator>(int, const std::pair<int, int>&, float, int);
