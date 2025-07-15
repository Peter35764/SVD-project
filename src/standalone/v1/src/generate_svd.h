#ifndef GENERATE_SVD_H
#define GENERATE_SVD_H

#include <Eigen/Dense>
#include <ctime>      
#include <random>       
#include <cassert>
#include <algorithm>    
#include <vector>       
#include <stdexcept>    
#include <iostream>    




template <typename T, int M = Eigen::Dynamic,  int N = Eigen::Dynamic>
class SVDGenerator
{
    private:

    // Используем псевдонимы с T
    using MatrixUType = Eigen::Matrix<T, M, M>;
    using MatrixVType = Eigen::Matrix<T, N, N>;
    using MatrixSType = Eigen::Matrix<T, M, N>;
    using DynamicMatrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
    using SingValVector =  std::vector<T>; // Используем std::vector<T>

    bool generatedFLG = false;
    MatrixUType U;
    MatrixVType V;
    MatrixSType S;
    SingValVector sigmas; 
    T sigma_min_limit;    
    T sigma_max_limit;

    int rows;
    int cols;
    int p; 


    public:

    // Основной конструктор
    SVDGenerator(int rows1, int cols1, T min_sv, T max_sv)
    {
        assert(rows1 > 0);
        assert(cols1 > 0);
        assert(max_sv >= min_sv);
         // <<< ИЗМЕНЕНО: Сравнение с T("0.0") >>>
        assert(min_sv > T("0.0")); // Сингулярные числа должны быть строго положительны

        rows = rows1;
        cols = cols1;
        p = std::min(rows, cols);

        sigma_min_limit = min_sv;
        sigma_max_limit = max_sv;

        // Инициализация матриц нулями
        U = MatrixUType::Zero(rows, rows);
        V = MatrixVType::Zero(cols, cols);
        S = MatrixSType::Zero(rows, cols);
         // <<< ИЗМЕНЕНО: Инициализация вектора T("0.0") >>>
        sigmas.resize(p, T("0.0"));

    }

    // Старый конструктор (оставлен для примера, но помечен как потенциально устаревший)
    SVDGenerator(int rows1, int cols1, const std::default_random_engine& RNG_src, const std::uniform_real_distribution<T>& dist_src, bool includeBoundaries_flag = false)
     : SVDGenerator(rows1, cols1, dist_src.a(), dist_src.b()) // Делегируем основному конструктору
    {
        // Можно добавить предупреждение
        // std::cerr << "Warning: SVDGenerator called with distribution, using Arithmetic mode." << std::endl;
        // RNG_src и includeBoundaries_flag здесь не используются
    }


    // Методы доступа
    MatrixUType MatrixU()
    {
        if (!generatedFLG)
            generate(); // Вызываем generate() без аргументов
        return U;
    }

    MatrixVType MatrixV()
    {
        if (!generatedFLG)
            generate();
        return V;
    }

    MatrixSType MatrixS()
    {
        if (!generatedFLG)
            generate();
        return S;
    }

    // Основной метод генерации
    // nonzeros_ignored больше не нужен
    void generate(int nonzeros_ignored = -1)
    {
        if (generatedFLG) return; // Генерируем только один раз
        generatedFLG = true;

        // <<< ИЗМЕНЕНО: Инициализация вектора T("0.0") >>>
        sigmas.assign(p, T("0.0")); // Очищаем вектор перед заполнением

        // --- Генерация сингулярных чисел: Арифметическая прогрессия (Mode 4) ---
        if (p > 1) {
            // <<< ИЗМЕНЕНО: Используем T(...) для каста и литерала >>>
            T p_minus_1 = T(p - 1); // Преобразуем p-1 в тип T
            T step = (sigma_max_limit - sigma_min_limit) / p_minus_1;
            for (int i = 0; i < p; ++i) {
                T i_T = T(i); // Преобразуем i в тип T
                sigmas[i] = sigma_max_limit - i_T * step;
                // Проверка на отрицательные значения (маловероятно, но все же)
                // <<< ИЗМЕНЕНО: Сравнение с T("0.0") >>>
                if (sigmas[i] < T("0.0")) sigmas[i] = T("0.0");
            }
        } else if (p == 1) {
            // Если только одно сингулярное число, берем максимальное
            sigmas[0] = sigma_max_limit;
        }
        // Сортировка не нужна, значения уже упорядочены по убыванию.
        // ---------------------------------------------------------------------

        // Заполняем диагональ матрицы S
        S.setZero();
        for(int i = 0; i < p; i++) {
            S(i,i) = sigmas[i];
        }


        // --- Генерация U и V через QR ---
        // DynamicMatrix здесь будет использовать тип T
        DynamicMatrix T_1(rows, rows), T_2(cols, cols);
        // MatrixUType Q_1(rows,rows); // HouseholderQR::householderQ() возвращает динамический тип
        // MatrixVType Q_2(cols,cols); // Поэтому используем DynamicMatrix
        DynamicMatrix Q_1, Q_2;

        // <<< ИЗМЕНЕНО: Литералы типа T >>>
        T HI = T("10.0");
        T LO = T("0.1");
        T range = HI - LO;
        T one = T("1.0");
        T two = T("2.0");

        // Используем локальный RNG для U, V
        std::mt19937 local_rng(std::random_device{}());

        // Генерируем случайные матрицы в диапазоне [-1, 1] (тип T)
        T_1 = MatrixUType::Random(rows,rows);
        // Масштабируем в [LO, HI]
        T_1 = (T_1 + MatrixUType::Constant(rows,rows, one)) * range / two;
        T_1 = (T_1 + MatrixUType::Constant(rows, rows, LO));

        // Генерируем случайные матрицы в диапазоне [-1, 1] (тип T)
        T_2 = MatrixVType::Random(cols,cols);
         // Масштабируем в [LO, HI]
        T_2 = (T_2 + MatrixVType::Constant(cols, cols, one)) * range / two;
        T_2 = (T_2 + MatrixVType::Constant(cols, cols, LO));

        // Выполняем QR разложение (Eigen обработает тип T)
        Eigen::HouseholderQR<DynamicMatrix> qr1(T_1);
        Eigen::HouseholderQR<DynamicMatrix> qr2(T_2);
        Q_1 = qr1.householderQ(); // Получаем ортогональную Q1
        Q_2 = qr2.householderQ(); // Получаем ортогональную Q2

        // Сохраняем результаты
        U = Q_1;
        // V = Q2^T, но так как V имеет тип N x N, а Q2 может быть cols x cols,
        // нужно быть осторожным, если M != N в объявлении шаблона.
        // В нашем случае M и N динамические, так что V будет cols x cols.
        V = Q_2.transpose(); // V = Q2^T
        // ---------------------------------------------------------
    }
};

#endif // GENERATE_SVD_H