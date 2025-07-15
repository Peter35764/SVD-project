#include <Eigen/Dense>
#include <algorithm> // Для std::replace, std::min
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>   // Для std::setw, std::left, std::fixed, std::setprecision
#include <mutex>
#include <random>
#include <sstream>   // Для std::ostringstream
#include <string>
#include <vector>
#include <limits>    // Для std::numeric_limits
#include <map>       // Для std::map
#include <array>     // Для std::array

// <<< ДОБАВЛЕНО: Заголовки Boost Multiprecision >>>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp> // Для abs/sqrt

#include "config.h" // обязательно для THREADS (Если используется)

// <<< ИЗМЕНЕНО: Подключаем только нужные алгоритмы >>>
#include "iterative_refinement_1.h"
#include "iterative_refinement_4.h"
#include "iterative_refinement_5.h"
#include "iterative_refinement_6.h"
// #include "iterative_refinement_8.h" // <<< УДАЛЕНО >>>
#include "generate_svd.h"           // <<< ВНИМАНИЕ: Убедитесь, что generate_svd адаптирован! >>>

// <<< ДОБАВЛЕНО: Определяем наш тип float100 >>>
using float100 = boost::multiprecision::cpp_dec_float_100;


// --- ПАРАМЕТРЫ ТЕСТОВ ---
// <<< ИЗМЕНЕНО: Используем float100 и const вместо constexpr >>>
const std::array<float100, 1> sigma_ratio = {float100("100000.0")}; // Используем строки
const std::array<std::pair<int, int>, 13> matrix_size = {{{3, 3},{5, 3}, {5, 5}, {10, 9}, {12, 12}, {20, 19}, {30, 30}, {50, 40}, {75, 75},{100, 90}, {100, 100}, {125, 100}, {125, 125}}}; // int остается
constexpr int matrix_num_for_sample_averaging = 1; // Оставляем 1 для примера
// --- КОНЕЦ ПАРАМЕТРОВ ---

std::mutex cout_mutex; // Глобальный мьютекс для вывода в консоль

// Универсально извлекает вектор сингулярных значений (без изменений)
template<typename Derived>
Eigen::Vector<typename Derived::Scalar, Eigen::Dynamic> extract_singular_values(const Derived& sv) {
    if (sv.rows() == 0 || sv.cols() == 0) {
        return Eigen::Vector<typename Derived::Scalar, Eigen::Dynamic>();
    }
    if (sv.cols() == 1 || sv.rows() == 1) {
        return sv; // уже вектор
    } else {
        // Возвращаем диагональ длины min(rows, cols)
        return sv.diagonal().head(std::min(sv.rows(), sv.cols()));
    }
}

template<typename MatrixType>
class AOIRWrapper_1 {
public:
    AOIRWrapper_1(const MatrixType& A, const std::string& history_filename_base = "", unsigned int flags = Eigen::ComputeFullU | Eigen::ComputeFullV) {
        impl = SVD_Project::AOIR_SVD_1<typename MatrixType::Scalar>(A, history_filename_base);
    }
    auto matrixU() const { return impl.matrixU(); }
    auto matrixV() const { return impl.matrixV(); }
    auto singularValues() const { return extract_singular_values(impl.singularValues()); }
private: SVD_Project::AOIR_SVD_1<typename MatrixType::Scalar> impl;
};

template<typename MatrixType>
class AOIRWrapper_4 {
public:
    AOIRWrapper_4(const MatrixType& A, const std::string& history_filename_base = "", unsigned int flags = Eigen::ComputeFullU | Eigen::ComputeFullV) {
        impl = SVD_Project::AOIR_SVD_4<typename MatrixType::Scalar>(A, history_filename_base);
    }
    auto matrixU() const { return impl.matrixU(); }
    auto matrixV() const { return impl.matrixV(); }
    auto singularValues() const { return extract_singular_values(impl.singularValues()); }
private: SVD_Project::AOIR_SVD_4<typename MatrixType::Scalar> impl;
};

template<typename MatrixType>
class AOIRWrapper_5 {
public:
    AOIRWrapper_5(const MatrixType& A, const std::string& history_filename_base = "", unsigned int flags = Eigen::ComputeFullU | Eigen::ComputeFullV) {
        impl = SVD_Project::AOIR_SVD_5<typename MatrixType::Scalar>(A, history_filename_base);
    }
    auto matrixU() const { return impl.matrixU(); }
    auto matrixV() const { return impl.matrixV(); }
    auto singularValues() const { return extract_singular_values(impl.singularValues()); }
private: SVD_Project::AOIR_SVD_5<typename MatrixType::Scalar> impl;
};

template<typename MatrixType>
class AOIRWrapper_6 {
public:
    AOIRWrapper_6(const MatrixType& A, const std::string& history_filename_base = "", unsigned int flags = Eigen::ComputeFullU | Eigen::ComputeFullV) {
        impl = SVD_Project::AOIR_SVD_6<typename MatrixType::Scalar>(A, history_filename_base);
    }
    auto matrixU() const { return impl.matrixU(); }
    auto matrixV() const { return impl.matrixV(); }
    auto singularValues() const { return extract_singular_values(impl.singularValues()); }
private: SVD_Project::AOIR_SVD_6<typename MatrixType::Scalar> impl;
};

// --- ОСНОВНАЯ ТЕСТОВАЯ ФУНКЦИЯ ---
// Адаптирована для float100 и исключен Алгоритм 8
template<typename T, template <typename> class gen_cl>
void run_all_svd_tests(std::string combined_fileName, // Имя файла для общей таблицы
                       const std::vector<T>& SigmaMaxMinRatiosVec, // Тип Т
                       const std::vector<std::pair<int,int>>& MatSizesVec, // int
                       const int n, // matrix_num_for_sample_averaging
                       std::string individual_times_fileName) // Имя файла для времени
{
    // Лямбда-функции для вывода таблиц (полные версии)
    auto printTable = [](std::ostream& out, const std::vector<std::vector<std::string>>& data){
        if (data.empty()) return;
        std::vector<size_t> widths;
        // Определяем ширину колонок
        for (const auto &row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (widths.size() <= i) widths.push_back(0);
                widths[i] = std::max(widths[i], row[i].size());
            }
        }
        // Печатаем данные
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << std::left << std::setw(widths[i] + 3) << row[i]; // +3 для отступа
            }
            out << "\n";
        }
    };
    auto printCSV = [](std::ostream& out, const std::vector<std::vector<std::string>>& data) {
        for (const auto& row : data) {
            bool first = true;
            for (const auto& cell : row) {
                if (!first) out << ",";
                std::string cellFormatted = cell;
                // Экранируем кавычки
                size_t pos = cellFormatted.find('"');
                while (pos != std::string::npos) {
                    cellFormatted.replace(pos, 1, "\"\"");
                    pos = cellFormatted.find('"', pos + 2);
                }
                // Заключаем в кавычки, если нужно
                if (cellFormatted.find(',') != std::string::npos ||
                    cellFormatted.find('"') != std::string::npos ||
                    cellFormatted.find('\n') != std::string::npos) {
                    cellFormatted = "\"" + cellFormatted + "\"";
                }
                out << cellFormatted;
                first = false;
            }
            out << "\n";
        }
    };
    // Лямбда преобразования числа в строку (тип T)
    auto num2str = [](T value){
        std::ostringstream oss;
        // <<< ИЗМЕНЕНО: Точность для float100 >>>
        const int print_precision = std::numeric_limits<T>::max_digits10;
        oss << std::scientific << std::setprecision(print_precision) << value;
        return oss.str();
    };
    // Лямбда для вывода времени (double)
    auto time2str = [](double value){
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << value; // 6 знаков после запятой для секунд
        return oss.str();
    };

    // Определения типов Eigen (T будет float100)
    using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorDynamic = Eigen::Vector<T, Eigen::Dynamic>;

    // Параметры тестов (интервалы)
    const std::vector<std::pair<T,T>> Intervals = {{T("1.0"), T("1e6")}};

    // Генератор случайных чисел (может быть нужен для SVDGenerator)
    std::random_device rd;
    std::mt19937 gen(rd());

    // --- Структура для хранения результатов одного алгоритма ---
    struct AlgoResult {
        T avg_dev_UUt = T("0.0"), avg_dev_UtU = T("0.0"), avg_dev_VVt = T("0.0"); // Инициализация T("0.0")
        T avg_dev_VtV = T("0.0"), avg_relErr_sigma = T("0.0");
        double cumulative_time = 0.0; // Время в double
    };

    // --- ИЗМЕНЕНО: Структура таблицы результатов (БЕЗ Алгоритма 8) ---
    std::vector<std::vector<std::string>> table;
    table.push_back({
        "Dimension", "SigmaRatio", "SV_Interval",
        "A1_||I-UU'||", "A1_||I-U'U||", "A1_||I-VV'||", "A1_||I-V'V||", "A1_RelSigErr", "A1_Time(s)",
        "A4_||I-UU'||", "A4_||I-U'U||", "A4_||I-VV'||", "A4_||I-V'V||", "A4_RelSigErr", "A4_Time(s)",
        "A5_||I-UU'||", "A5_||I-U'U||", "A5_||I-VV'||", "A5_||I-V'V||", "A5_RelSigErr", "A5_Time(s)",
        "A6_||I-UU'||", "A6_||I-U'U||", "A6_||I-VV'||", "A6_||I-V'V||", "A6_RelSigErr", "A6_Time(s)"
      // Столбцы для Алгоритма 8 удалены
    });


    // Расчет прогресса
    const size_t total_combinations = MatSizesVec.size() * SigmaMaxMinRatiosVec.size() * Intervals.size();
    size_t combinations_done = 0;
    int current_line_for_progress = 1; // Стартовая линия для прогресс-бара

    // Переменные для матриц (тип T)
    MatrixDynamic U_true, S_true_mat, V_true;
    MatrixDynamic U_calc, V_calc; // Временные для каждого алгоритма
    VectorDynamic SV_calc, SV_true; // Временные для каждого алгоритма

    // --- Начало циклов по параметрам ---
    for (const auto& MatSize : MatSizesVec) {
        const int N = MatSize.first; const int M = MatSize.second; int minNM = std::min(N, M);

        // Изменяем размер матриц
        U_true.resize(N, N); S_true_mat.resize(N, M); V_true.resize(M, M); SV_true.resize(minNM);

        for (const auto& SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
            for (const auto& interval : Intervals) {
                // Проверка валидности
                 if (interval.first <= T("0.0") || interval.second <= interval.first || interval.first * SigmaMaxMinRatio > interval.second) {
                     std::lock_guard<std::mutex> lock(cout_mutex);
                     std::cerr << "Warning: Skipping invalid interval/ratio combination for ["
                               << interval.first << ", " << interval.second << "] with ratio " << SigmaMaxMinRatio << "\n";
                     continue;
                 }
                 T upper_bound = interval.second / SigmaMaxMinRatio;
                 if (interval.first > upper_bound) {
                     std::lock_guard<std::mutex> lock(cout_mutex);
                     std::cerr << "Warning: Skipping invalid interval/ratio (lower > upper/ratio) for ["
                               << interval.first << ", " << interval.second << "] with ratio " << SigmaMaxMinRatio << "\n";
                    continue;
                 }

                // Инициализация аккумуляторов ошибок и времени для нужных алгоритмов
                std::map<int, AlgoResult> results; // Ключ - номер алгоритма (1, 4, 5, 6)

                // Определяем целевые sigma_min и sigma_max
                T target_sigma_min = interval.first;
                T target_sigma_max = target_sigma_min * SigmaMaxMinRatio;

                // --- Цикл усреднения ---
                for (int iter = 0; iter < n; ++iter) {

                    // Генерируем ОДНУ матрицу A_gen и ее истинное разложение
                    // <<< ВНИМАНИЕ: Убедитесь, что gen_cl (SVDGenerator) адаптирован под T! >>>
                    gen_cl<T> svd_gen(N, M, target_sigma_min, target_sigma_max);
                    svd_gen.generate();

                    U_true = svd_gen.MatrixU(); S_true_mat = svd_gen.MatrixS(); V_true = svd_gen.MatrixV();
                    SV_true = S_true_mat.diagonal().head(minNM);
                    MatrixDynamic A_gen = (U_true * S_true_mat * V_true.transpose());

                    // Функция для расчета ошибок
                    auto calculate_errors = [&](const MatrixDynamic& U_c, const MatrixDynamic& V_c, const VectorDynamic& SV_c) -> AlgoResult {
                        AlgoResult err_res;
                        MatrixDynamic I_N = MatrixDynamic::Identity(N, N); MatrixDynamic I_M = MatrixDynamic::Identity(M, M);
                        err_res.avg_dev_UUt = (I_N - U_c * U_c.transpose()).norm();
                        err_res.avg_dev_UtU = (I_N - U_c.transpose() * U_c).norm();
                        err_res.avg_dev_VVt = (I_M - V_c * V_c.transpose()).norm();
                        err_res.avg_dev_VtV = (I_M - V_c.transpose() * V_c).norm();

                        T relErr_sigma_sum = T("0.0"); int valid_sigma_count = 0;
                        for (int k = 0; k < minNM; ++k) {
                            T true_sv = (k < SV_true.size()) ? SV_true(k) : T("0.0");
                            T calc_sv = (k < SV_c.size()) ? SV_c(k) : T("0.0");
                             // <<< ИЗМЕНЕНО: Используем boost::abs и литералы T >>>
                            if (boost::multiprecision::abs(true_sv) > std::numeric_limits<T>::epsilon() * T("100.0")) {
                                relErr_sigma_sum += boost::multiprecision::abs(calc_sv - true_sv) / boost::multiprecision::abs(true_sv);
                                valid_sigma_count++;
                            } else if (boost::multiprecision::abs(calc_sv) > std::numeric_limits<T>::epsilon() * T("100.0")) { /* Пропуск */ }
                        }
                        err_res.avg_relErr_sigma = (valid_sigma_count > 0) ? (relErr_sigma_sum / T(valid_sigma_count)) : T("0.0");
                        return err_res;
                    };

                    // Макрос для запуска
                    #define RUN_ALGO(ID, WRAPPER_CLASS) \
                    { \
                        auto t_start = std::chrono::high_resolution_clock::now(); \
                        std::ostringstream history_base_stream; history_base_stream << "log_A" << ID << "_" << N << "x" << M << "_r" << std::fixed << std::setprecision(3) << SigmaMaxMinRatio << "_i" << std::scientific << std::setprecision(3) << interval.first << "_n" << iter; \
                        std::string name = history_base_stream.str(); std::replace(name.begin(), name.end(), '.', '_'); std::replace(name.begin(), name.end(), ',', '_'); std::replace(name.begin(), name.end(), '+', 'p'); std::replace(name.begin(), name.end(), '-', 'm'); \
                        WRAPPER_CLASS<MatrixDynamic> svd_func(A_gen, name, Eigen::ComputeFullU | Eigen::ComputeFullV); \
                        AlgoResult err = calculate_errors(svd_func.matrixU(), svd_func.matrixV(), svd_func.singularValues()); \
                        auto t_end = std::chrono::high_resolution_clock::now(); \
                        results[ID].cumulative_time += std::chrono::duration<double>(t_end - t_start).count(); \
                        results[ID].avg_dev_UUt += err.avg_dev_UUt; results[ID].avg_dev_UtU += err.avg_dev_UtU; results[ID].avg_dev_VVt += err.avg_dev_VVt; results[ID].avg_dev_VtV += err.avg_dev_VtV; results[ID].avg_relErr_sigma += err.avg_relErr_sigma; \
                    }

                    // <<< ИЗМЕНЕНО: Запускаем только нужные алгоритмы >>>
                    RUN_ALGO(1, AOIRWrapper_1);
                    RUN_ALGO(4, AOIRWrapper_4);
                    RUN_ALGO(5, AOIRWrapper_5);
                    RUN_ALGO(6, AOIRWrapper_6);
                    // RUN_ALGO(8, AOIRWrapper_8); // <<< УДАЛЕНО >>>

                    #undef RUN_ALGO

                } // Конец цикла усреднения

                // Усредняем ошибки и ВРЕМЯ
                if (n > 0) {
                    // Используем auto& для изменения значений в map
                    for(auto& pair : results) {
                        pair.second.avg_dev_UUt /= n; pair.second.avg_dev_UtU /= n;
                        pair.second.avg_dev_VVt /= n; pair.second.avg_dev_VtV /= n;
                        pair.second.avg_relErr_sigma /= n;
                        pair.second.cumulative_time /= n; // Усредняем время
                    }
                } else {
                    // Обнуляем, если n=0
                     for(auto& pair : results) { pair.second = AlgoResult(); }
                }

                // Обновление прогресса (логика без изменений)
                combinations_done++;
                double percent = (total_combinations > 0) ? (static_cast<double>(combinations_done) * 100.0 / total_combinations) : 100.0;
                int barWidth = 50; int pos = static_cast<int>(barWidth * percent / 100.0); pos = std::min(pos, barWidth);
                std::ostringstream progressStream; progressStream << "Overall Progress: " << std::fixed << std::setprecision(2) << percent << "% [";
                for (int j = 0; j < barWidth; ++j) { progressStream << (j < pos ? "=" : " "); } progressStream << "]";
                // Вывод прогресс-бара (логика без изменений)
                // Примечание: корректность отображения зависит от терминала
                { std::lock_guard<std::mutex> lock(cout_mutex); std::cout << "\033[" << current_line_for_progress << ";0H" << progressStream.str() << "\033[K" << std::flush; }


                // --- Формирование строки для ОБЩЕЙ таблицы (БЕЗ Алгоритма 8) ---
                std::vector<std::string> row_data;
                row_data.push_back(std::to_string(N) + "x" + std::to_string(M));
                row_data.push_back(num2str(SigmaMaxMinRatio));
                row_data.push_back("[" + num2str(interval.first) + ", " + num2str(interval.second) + "]");

                // Добавляем результаты для каждого алгоритма (1, 4, 5, 6)
                for(int algo_id : {1, 4, 5, 6}) {
                    // Проверяем, есть ли результат для данного ID (на случай, если n=0)
                    if (results.count(algo_id)) {
                        row_data.push_back(num2str(results[algo_id].avg_dev_UUt));
                        row_data.push_back(num2str(results[algo_id].avg_dev_UtU));
                        row_data.push_back(num2str(results[algo_id].avg_dev_VVt));
                        row_data.push_back(num2str(results[algo_id].avg_dev_VtV));
                        row_data.push_back(num2str(results[algo_id].avg_relErr_sigma));
                        row_data.push_back(time2str(results[algo_id].cumulative_time));
                    } else {
                        // Добавляем пустые значения, если результат не был посчитан
                         for(int k=0; k<6; ++k) row_data.push_back("");
                    }
                }
                // Данные для Алгоритма 8 НЕ добавляются

                table.push_back(row_data);

                 // --- Сохранение индивидуального времени (БЕЗ Алгоритма 8) ---
                 std::ofstream individual_time_file(individual_times_fileName, std::ios::app); // Открываем на дозапись
                 if (individual_time_file) {
                     if (combinations_done == 1) { // Пишем заголовок только раз
                         individual_time_file << "Dimension,SigmaRatio,SV_Interval,Algo1_Time(s),Algo4_Time(s),Algo5_Time(s),Algo6_Time(s)\n"; // Убрали Algo8
                     }
                     individual_time_file << std::to_string(N) << "x" << std::to_string(M) << ","
                                          << num2str(SigmaMaxMinRatio) << ","
                                          << "[" << num2str(interval.first) << ";" << num2str(interval.second) << "]," // Используем ;
                                          << time2str(results[1].cumulative_time) << "," // Время для Алг. 1
                                          << time2str(results[4].cumulative_time) << "," // Время для Алг. 4
                                          << time2str(results[5].cumulative_time) << "," // Время для Алг. 5
                                          << time2str(results[6].cumulative_time) << "\n"; // Время для Алг. 6 (Убрали results[8])
                     individual_time_file.close();
                 } else {
                      std::lock_guard<std::mutex> lock(cout_mutex);
                      std::cerr << "Error opening individual time file " << individual_times_fileName << "!\n";
                 }
                 // --- Конец сохранения времени ---

            } // Конец цикла по Intervals
        } // Конец цикла по SigmaMaxMinRatiosVec
    } // Конец цикла по MatSizesVec

    // Запись общей таблицы в файлы txt и csv (логика без изменений)
    std::ofstream file(combined_fileName);
    if (file) { printTable(file, table); file.close(); }
    else { std::lock_guard<std::mutex> lock(cout_mutex); std::cerr << "Error opening file " << combined_fileName << "!\n"; }

    std::string csvFileName = combined_fileName;
    size_t pos_dot = csvFileName.rfind(".txt");
    if (pos_dot != std::string::npos) csvFileName.replace(pos_dot, 4, ".csv"); else csvFileName += ".csv";
    std::ofstream csv_file(csvFileName);
    if (csv_file) { printCSV(csv_file, table); csv_file.close(); }
    else { std::lock_guard<std::mutex> lock(cout_mutex); std::cerr << "Error opening file " << csvFileName << "!\n"; }
};
// --- КОНЕЦ ОСНОВНОЙ ТЕСТОВОЙ ФУНКЦИИ ---


// <<< ИЗМЕНЕНО: Создаем векторы параметров типа float100 >>>
const std::vector<float100> sigma_ratio_vec(sigma_ratio.begin(), sigma_ratio.end());
const std::vector<std::pair<int, int>> matrix_size_vec(matrix_size.begin(), matrix_size.end());


// --- main ---
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "\033[2J\033[H"; // Очистка экрана
    std::cout << "\033[?25l";   // Скрыть курсор

    // Имена файлов для результатов
    std::string combined_table_filename = "combined_svd_results_table_float100.txt"; // Обновленное имя
    std::string individual_times_filename = "individual_algo_times_float100.csv"; // Обновленное имя

    // Очищаем файл времени перед запуском тестов (если он существует)
    std::ofstream clear_time_file(individual_times_filename, std::ios::trunc);
    if (clear_time_file) { clear_time_file.close(); }
    else { std::cerr << "Warning: Could not clear file " << individual_times_filename << std::endl; }


    // <<< ИЗМЕНЕНО: Вызов с типом float100 >>>
    // <<< ВНИМАНИЕ: SVDGenerator должен быть адаптирован под float100! >>>
    run_all_svd_tests<float100, SVDGenerator>(
        combined_table_filename,
        sigma_ratio_vec,
        matrix_size_vec,
        matrix_num_for_sample_averaging,
        individual_times_filename // Передаем имя файла для времени
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGlobal = end - start;

    // Вывод общего времени
    // Позиционирование курсора (может потребоваться +1 или +2)
    std::cout << "\033[" << 2 << ";0H"; // Перемещаем курсор на пару строк вниз от прогресс-бара
    std::cout << "\n\nTotal execution time for all algorithms = "
              << std::fixed << std::setprecision(4) << durationGlobal.count() << " seconds.\n";
    std::cout << "Individual algorithm times saved to: " << individual_times_filename << std::endl;
    std::cout << "Combined results saved to: " << combined_table_filename << " (and .csv)" << std::endl;


    // Запись общего времени (файл для всех тестов)
    std::ofstream timeFile("test_times_combined_float100.txt"); // Обновленное имя
    if (timeFile) {
        timeFile << "Total execution time : " << durationGlobal.count() << " seconds\n";
        timeFile.close(); // Закрываем файл
    } else {
        // Не блокируем мьютекс здесь, так как основной вывод уже завершен
        std::cerr << "Error while creating/opening test_times_combined_float100.txt!\n";
    }


    std::cout << "\033[?25h"; // Показать курсор снова
    return 0; // Успешное завершение
}