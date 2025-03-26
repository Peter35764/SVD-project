//Александр Нам, КМБО-04-20
//Any questions: alexnam16@gmail.com

#include "dqds.h"
#include "generate_svd.h"
#include "reverse_jacobi.h"

#include <Eigen/Dense>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

//Назначение функции - тестирование и оценка точности методов SVD разложения, унаследованных от SVD Base

/*
 Функция возвращает матрицу-таблицу формата:
 | размерность | sigma_max/sigma_min | диап. синг. чисел | 
 | sum_n(norm[I - U.transpose*U])/n | sum_n(norm[I - U*U.transpose])/n |
 | sum_n(norm[I - V.transpose*V])/n | sum_n(norm[I - V*V.transpose])/n | 
 | max(abs((sigma_true_i - sigma_calc_i)/sigma_true_i)) |;
 Размер выборки фиксированного размера матриц определяется задаваемым параметром 'n'.
*/

/*
Функция написана с теми условиями, что:
    1. Класс SVD разложения наследуется от класса Eigen::SVDBase, причем должен существовать конструктор класса,
       который вторым параметром принимает настройку вычислений матриц U и V, т.е. thin или full.
    2. Генерация случайных матриц происходит с помощью SVDGenerator из generate_svd.h
    3. В функцию передаётся std::vector соотношений максимумального и минимального сингулярного числа
    4. В функцию передаётся std::vector<std::pair<int,int>> размеров матриц для исследования
    5. В функцию передаётся int n размер выборки фиксированного размера матриц для подсчёта средних
    6. Функция работает достаточно долго, особенно для матриц больших размеров, поэтому выводится прогресс в процентах
    7. Результат исследования не печатается в консоль, а сохраняется в файл, название выбирается первым параметром
*/

/*
Функция принимает параметрами: 
- fileName: имя текстового файла, куда будет сохранен результат, т.е. таблица
- SigmaMaxMinRatiosVec: вектор сооотношения максимального и минимального сингулярных чисел;
                        нужен т.к. ошибка может сильно отличаться у разных соотношений сингулярных чисел;
- MatSizesVec: вектор размеров матриц для теста;
- n: количество матриц, которые генерируются с одинаковыми параметрами для усреднения выборки и подсчета средних
*/
template<typename T, template<typename> class gen_cl, template<typename> class svd_cl>
void svd_test_func(
    std::string fileName,
    const std::vector<T> &SigmaMaxMinRatiosVec,
    const std::vector<std::pair<int, int>> &MatSizesVec,
    const int n)
{
    // -----------объявление локальных функций и инициализация-------------

    //лямбда-функция печати таблицы, вторым параметром принимает матрицу строк(так реализована таблица)
    auto printTable = [](std::ostream &out, const std::vector<std::vector<std::string>> &data) {
        if (data.empty())
            return;
        // Определение ширины столбцов
        std::vector<size_t> widths;
        for (const auto &row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (widths.size() <= i)
                    widths.push_back(0);
                widths[i] = std::max(widths[i], row[i].size());
            }
        }
        // Вывод данных в поток
        for (const auto &row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << std::left << std::setw(widths[i] + 3) << row[i];
            }
            out << "\n";
        }
    };

    //лямбда-функция преобразования числа в строку
    auto num2str = [](T value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    };

    //нужно для того, чтобы более менее равномерно выводить прогресс работы, больше размер матрицы -> дольше будет вычисляться;
    T generalSum = 0;
    for (const auto &MatSize : MatSizesVec) {
        generalSum += (MatSize.first * MatSize.second);
    }

    using MatrixDynamic = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorDynamic = Eigen::Vector<T, Eigen::Dynamic>;

    //интервалы, в которых будут генерироваться сингулярные числа для теста на малых по модулю сингулярных числах и больших
    const std::vector<std::pair<T, T>> Intervals = {{0, 1}, {1, 100}};

    //сам генератор чисел
    std::random_device rd;                //случайное число - значение сида
    std::default_random_engine gen(rd()); //генерация последовательности псевдослучайных цифр

    //название столбцов таблицы
    std::vector<std::vector<std::string>> table = {
        {"Dimension",
         "Sigma-max/min-ratio",
         "SV interval",
         "AVG ||I-U_t*U||",
         "AVG ||I-U*U_t||",
         "AVG ||I-V_t*V||",
         "AVG ||I-V*V_t||",
         "AVG relative err. sigma"}};

    T ProgressCoeff = n * Intervals.size() * SigmaMaxMinRatiosVec.size() * generalSum
                      / 100.0; //наибольшее значение прогресса
    T progress = 0;            //инициализация аккумулятора прогресса

    MatrixDynamic U_true, S_true, V_true, U_calc, V_calc, V_calc_transpose;
    VectorDynamic SV_calc; //аналог S_true, но не матрица, а вектор сингулярных значений

    // -----------------------
    for (const auto &MatSize : MatSizesVec) {
        const int N = MatSize.first;
        const int M = MatSize.second;
        int minNM = std::min(N, M);

        U_true.resize(N, N);
        U_calc.resize(N, N);
        S_true.resize(N, M);
        SV_calc.resize(minNM);
        V_true.resize(M, M);
        V_calc.resize(M, M);
        V_calc_transpose.resize(M, M);

        for (const auto &SigmaMaxMinRatio : SigmaMaxMinRatiosVec) {
            for (const auto &interval : Intervals) {
                //interval.first и interval.second - l и r границы интервала
                assert(
                    (interval.first < interval.second,
                     "\nError: left boundary >= right boundary\n"));

                //определение промежутка [sigma_min, sigma_max]:
                /*
                    проверка на то, что отношение sigmaMax/sigmaMin <= b/a, где a и b - границы интервала,
                    очевидно, что если неравенство не выполняется, то сингулярных чисел с заданным соотношением
                    и в заданном интервале не существует
                */
                assert(
                    (interval.first * SigmaMaxMinRatio <= interval.second,
                     "\nError: no sigma values exist with such ratio in such interval\n"));

                /*
                  случайным образом генерируем sigma_min в допустимом интервале,
                  заметим, что прошлый assert гарантирует, что a < b/SigmaRatio
                */

                std::uniform_real_distribution<T>
                    distrSigmaMin(interval.first, interval.second / SigmaMaxMinRatio);
                T sigma_min = distrSigmaMin(gen);
                T sigma_max = SigmaMaxMinRatio * sigma_min;

                //заполнение вектора истинных сингулярных чисел из уже определенного диапазона:
                std::uniform_real_distribution<T> distr(sigma_min, sigma_max);
                assert((minNM >= 2, "\nError: no columns or rows allowed\n"));

                //инициализация аккумуляторов выводимых в таблицу значений характеристик
                T avg_dev_UUt = 0, avg_dev_UtU = 0, avg_dev_VVt = 0, avg_dev_VtV = 0,
                  avg_relErr_sigma = 0;

                //цикл для усреднения результатов(выборка из n матриц с одинаковыми параметрами)
                for (int i = 1; i <= n; ++i) {
                    gen_cl<T> svd_gen(N, M, gen, distr, true); //инициализация параметров генерации
                    svd_gen.generate(minNM);                   //генерация

                    //подсчет результатов:
                    U_true = svd_gen.MatrixU();
                    S_true = svd_gen.MatrixS();
                    V_true = svd_gen.MatrixV();
                    svd_cl<MatrixDynamic> svd_func(
                        (U_true * S_true * V_true.transpose()).eval(),
                        Eigen::ComputeFullU | Eigen::ComputeFullV);
                    U_calc = svd_func.matrixU();
                    SV_calc = svd_func.singularValues();
                    V_calc = svd_func.matrixV();
                    avg_dev_UUt += (MatrixDynamic::Identity(N, N) - U_calc * U_calc.transpose())
                                       .squaredNorm()
                                   / n;
                    avg_dev_UtU += (MatrixDynamic::Identity(N, N) - U_calc.transpose() * U_calc)
                                       .squaredNorm()
                                   / n;
                    avg_dev_VVt += (MatrixDynamic::Identity(M, M) - V_calc * V_calc.transpose())
                                       .squaredNorm()
                                   / n;
                    avg_dev_VtV += (MatrixDynamic::Identity(M, M) - V_calc.transpose() * V_calc)
                                       .squaredNorm()
                                   / n;
                    avg_relErr_sigma += (S_true.diagonal() - SV_calc)
                                            .cwiseQuotient(S_true.diagonal())
                                            .cwiseAbs()
                                            .maxCoeff()
                                        / n;

                    //вывод прогресса
                    progress += M * N / ProgressCoeff;
                    std::cout << "\n" + num2str(progress) + "% was done.\n";
                }

                //заполнение одной строки таблицы
                table.emplace_back(std::vector<std::string>{
                    num2str(N) + "x" + num2str(M),
                    num2str(SigmaMaxMinRatio),
                    "[" + num2str(interval.first) + ", " + num2str(interval.second) + "]",
                    num2str(avg_dev_UUt),
                    num2str(avg_dev_UtU),
                    num2str(avg_dev_VVt),
                    num2str(avg_dev_VtV),
                    num2str(avg_relErr_sigma)});
            }
        }
    }

    //непосредственно сохранение таблицы в файл
    std::ofstream file(fileName);
    if (file) {
        printTable(file, table);
        file.close();
    } else {
        std::cerr << "Error while creating/opening file!\n";
    }
};

template<typename _MatrixType>
using RevJac_DQDS_SVD = SVD_Project::RevJac_SVD<DQDS_SVD<_MatrixType>>;

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    //генерируеся таблицу в файле "jacobi_test_table.txt" теста метода Eigen::JacobiSVD
    //с соотношением сингулярных чисел:  1.01, 1.2, 2, 5, 10, 50      ---    6
    //причем каждое соотношение относится к двум интервалам сингулярных чисел:
    //                      маленьких {0,1}, больших {1,100} (это не параметризованно)   ---   2
    //с матрицами размеров: {3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}   ---   6
    //6*2*6 = 72 - всего столько строк будет в таблице
    //размер выборки для усреднения: 20
    svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>(
        "jacobi_test_table.txt",
        {1.01, 1.2, 2, 5, 10, 50},
        {{3, 3}, {5, 5}, {10, 10}, {20, 20}, {50, 50}, {100, 100}},
        20);
    svd_test_func<double, SVDGenerator, DQDS_SVD>(
        "dqds_test_table.txt",
        {1.01, 1.2, 2, 5, 10, 50},
        {{3, 3}, {5, 5}, {10, 10}, {20, 20}, {50, 50}, {100, 100}},
        20);
    svd_test_func<double, SVDGenerator, RevJac_DQDS_SVD>(
        "jacobi_inverse_dqds_test_table.txt", {1.01, 1.2, 2, 5, 10, 50}, {{3, 3}, {5, 5}}, 10);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGlobal = end - start;
    std::cout << "\nFull execution time = " << durationGlobal.count() << " seconds.\n";

    char c;
    std::cin >> c;
    return 0;
}