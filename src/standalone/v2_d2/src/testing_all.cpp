#include "test_common.h" 

// --- НАЧАЛО: Объявления функций-диспетчеров ---
template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_20_24(int, const std::pair<int, int>&, float, int);
template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_25_29(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_30_35(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_36_40(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_41_45(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_46_50(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_51_55(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_56_60(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_61_65(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_66_70(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_71_75(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_76_80(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_81_85(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_86_90(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_91_95(int, const std::pair<int, int>&, float, int);

template<int AlgoID, template<typename> class SVDAlgoClass, template<typename> class SVDGeneratorEngine>
PrecisionStudyResultRow dispatch_digits_96_100(int, const std::pair<int, int>&, float, int); // Или до 101, если нужно
// --- КОНЕЦ: Объявления функций-диспетчеров ---


// Функция run_precision_study теперь вызывает соответствующий диспетчер
template<int AlgoID,
         template<typename> class SVDAlgoClass,
         template<typename> class SVDGeneratorEngine>
void run_precision_study(
    const std::string& output_csv_filename_base,
    const std::vector<int>& digit_levels_config,
    const std::vector<std::pair<int, int>>& matrix_sizes_config,
    const std::vector<float>& sigma_ratios_config,
    int num_runs_for_averaging)
{
    std::string full_output_csv_filename = output_csv_filename_base + "_algo" + std::to_string(AlgoID) + ".csv";
    std::ofstream csv_file(full_output_csv_filename);

    if (!csv_file.is_open()) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Error: Could not open output CSV file: " << full_output_csv_filename << std::endl;
        return;
    }

    csv_file << "AlgorithmID,Digits,Epsilon,MatrixRows,MatrixCols,SigmaRatioGen,"
             << "Iterations,Time_s,AchievedSigmaRelError,AchievedUOrthoError,AchievedVOrthoError\n";

    for (int current_digits_val : digit_levels_config) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\n---> Testing Algorithm ID: " << AlgoID << " with Digits: " << current_digits_val << std::endl;
        }

        for (const auto& mat_size_pair : matrix_sizes_config) {
            if (AlgoID == 4 && mat_size_pair.first < mat_size_pair.second) {
                 std::lock_guard<std::mutex> lock(cout_mutex);
                 std::cout << "   Skipping Matrix: " << mat_size_pair.first << "x" << mat_size_pair.second << " for Algo " << AlgoID
                           << " (constraint m>=n)" << std::endl;
                 continue;
            }

            for (float sigma_ratio_for_gen_float : sigma_ratios_config) {
                PrecisionStudyResultRow result_row_data;
                bool dispatched = false;

                // Вызов нужного диспетчера на основе current_digits_val
                if (current_digits_val >= 20 && current_digits_val <= 24) {
                    result_row_data = dispatch_digits_20_24<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 25 && current_digits_val <= 29) {
                    result_row_data = dispatch_digits_25_29<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 30 && current_digits_val <= 35) {
                    result_row_data = dispatch_digits_30_35<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 36 && current_digits_val <= 40) {
                    result_row_data = dispatch_digits_36_40<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 41 && current_digits_val <= 45) {
                    result_row_data = dispatch_digits_41_45<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 46 && current_digits_val <= 50) {
                    result_row_data = dispatch_digits_46_50<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 51 && current_digits_val <= 55) {
                    result_row_data = dispatch_digits_51_55<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 56 && current_digits_val <= 60) {
                    result_row_data = dispatch_digits_56_60<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 61 && current_digits_val <= 65) {
                    result_row_data = dispatch_digits_61_65<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 66 && current_digits_val <= 70) {
                    result_row_data = dispatch_digits_66_70<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 71 && current_digits_val <= 75) {
                    result_row_data = dispatch_digits_71_75<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 76 && current_digits_val <= 80) {
                    result_row_data = dispatch_digits_76_80<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 81 && current_digits_val <= 85) {
                    result_row_data = dispatch_digits_81_85<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 86 && current_digits_val <= 90) {
                    result_row_data = dispatch_digits_86_90<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 91 && current_digits_val <= 95) {
                    result_row_data = dispatch_digits_91_95<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                } else if (current_digits_val >= 96 && current_digits_val <= 100) { // или до 101, если это твой верхний предел
                    result_row_data = dispatch_digits_96_100<AlgoID, SVDAlgoClass, SVDGeneratorEngine>(current_digits_val, mat_size_pair, sigma_ratio_for_gen_float, num_runs_for_averaging);
                    dispatched = true;
                }
                // Добавь сюда `else if` для других диапазонов, если ты создал больше файлов

                if (!dispatched) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cerr << "      Skipping Digits: " << current_digits_val << " - not handled by any dispatcher for AlgoID " << AlgoID << "." << std::endl;
                    result_row_data.algorithm_id = AlgoID; result_row_data.digits = current_digits_val; result_row_data.iterations_count = -2;
                    result_row_data.achieved_sigma_relative_error_str = "UnhandledDigits";
                    result_row_data.matrix_rows = mat_size_pair.first; result_row_data.matrix_cols = mat_size_pair.second;
                    result_row_data.sigma_ratio_for_generator_float = sigma_ratio_for_gen_float;
                    result_row_data.time_taken_seconds = 0.0f;
                    result_row_data.epsilon_str = "N/A";
                    result_row_data.achieved_u_ortho_error_str = "UnhandledDigits";
                    result_row_data.achieved_v_ortho_error_str = "UnhandledDigits";
                }

                // Запись в CSV, если тест был выполнен (iterations_count не равен кодам ошибок)
                if (result_row_data.iterations_count != -2 && result_row_data.iterations_count != -3) {
                    csv_file << result_row_data.algorithm_id << ","
                             << result_row_data.digits << ","
                             << "\"" << result_row_data.epsilon_str << "\"" << ","
                             << result_row_data.matrix_rows << ","
                             << result_row_data.matrix_cols << ","
                             << value_to_string_scientific(result_row_data.sigma_ratio_for_generator_float, 6) << ","
                             << result_row_data.iterations_count << ","
                             << std::fixed << std::setprecision(6) << result_row_data.time_taken_seconds << ","
                             << result_row_data.achieved_sigma_relative_error_str << ","
                             << result_row_data.achieved_u_ortho_error_str << ","
                             << result_row_data.achieved_v_ortho_error_str << "\n";
                    csv_file.flush();
                }
            }
        }
    }
    csv_file.close();
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\n---> Precision study for Algorithm ID " << AlgoID << " finished."
                  << " Results saved to: " << full_output_csv_filename << std::endl;
    }
}

int main() {
    std::cout << "SVD Algorithm Precision Study Test Suite" << std::endl;
    std::cout << "Compilation Date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "Using " << THREADS << " threads for OpenMP (if enabled and used by algorithms)." << std::endl;
    std::cout << "============================================================" << std::endl;

    std::vector<int> digit_levels_to_study_config;
    // Настраиваем диапазон NumDigits (например, от 21 до 101, что соответствует exponent 20-100)
    int min_exponent_for_epsilon = 20;
    int max_exponent_for_epsilon = 100; // Убедись, что это покрывает все твои диапазоны

    for (int exponent = min_exponent_for_epsilon; exponent <= max_exponent_for_epsilon; ++exponent) {
        digit_levels_to_study_config.push_back(exponent + 1);
    }

    std::vector<std::pair<int, int>> matrix_sizes_config = {
        {50, 50},
        // {50, 40}, {100,90}, {200,180}, {500,450}, {1000,900} // Раскомментируй для больших тестов
    };

    std::vector<float> sigma_ratios_config = {100.0f /*, 1e3f, 1e6f, 1e9f, 1e12f, 1e15f */};
    int num_runs_for_averaging_config = 1;

    std::cout << "Study Configuration:" << std::endl;
    std::cout << "  Digit levels to test (" << digit_levels_to_study_config.size() << " levels): ";
    if (!digit_levels_to_study_config.empty()) {
        std::cout << digit_levels_to_study_config.front() << "..." << digit_levels_to_study_config.back();
    }
    std::cout << std::endl;

    run_precision_study<1, SVD_Project::AOIR_SVD_1, SVD_Project::SVDGenerator>("precision_study_results", digit_levels_to_study_config, matrix_sizes_config, sigma_ratios_config, num_runs_for_averaging_config);
    run_precision_study<4, SVD_Project::AOIR_SVD_4, SVD_Project::SVDGenerator>("precision_study_results", digit_levels_to_study_config, matrix_sizes_config, sigma_ratios_config, num_runs_for_averaging_config);
    run_precision_study<5, SVD_Project::AOIR_SVD_5, SVD_Project::SVDGenerator>("precision_study_results", digit_levels_to_study_config, matrix_sizes_config, sigma_ratios_config, num_runs_for_averaging_config);
    run_precision_study<6, SVD_Project::AOIR_SVD_6, SVD_Project::SVDGenerator>("precision_study_results", digit_levels_to_study_config, matrix_sizes_config, sigma_ratios_config, num_runs_for_averaging_config);
    run_precision_study<8, SVD_Project::AOIR_SVD_8, SVD_Project::SVDGenerator>("precision_study_results", digit_levels_to_study_config, matrix_sizes_config, sigma_ratios_config, num_runs_for_averaging_config);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "All configured studies finished." << std::endl;
    std::cout << "Check CSV files starting with 'precision_study_results_algoX.csv'." << std::endl;

    return 0;
}
