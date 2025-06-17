/** @file SVD_Test_config.h
 * @brief Содержит конфигурационные настройки и регистрацию алгоритмов для
 * SVD_Test.
 *
 * В этом файле определяются статические структуры и функции,
 * используемые для добавления новых SVD-алгоритмов в тестовый фреймворк
 * и управления их поведением (например, требуется ли передача спектра).
 */
#ifndef SVD_TEST_CONFIG_H
#define SVD_TEST_CONFIG_H

#include <type_traits>

#include "../SVD_project.h"

namespace SVD_Project {

/**
 * @brief Трейт для определения, требует ли конструктор алгоритма SVD передачи
 * истинных сингулярных значений.
 * @tparam SVDClass Класс алгоритма SVD.
 *
 * Некоторые алгоритмы SVD, особенно итерационные методы или методы,
 * использующие уточнение, могут принимать истинные (или эталонные)
 * сингулярные значения в качестве входных данных для своей работы
 * или для определенных режимов.
 *
 * Этот трейт используется в тестовом фреймворке для определения,
 * нужно ли генерировать истинные сингулярные значения и передавать их
 * в конструктор алгоритма при создании его экземпляра для тестирования.
 *
 * По умолчанию (для неспециализированного шаблона) считается, что
 * алгоритм НЕ требует передачи сингулярных значений (`std::false_type`).
 */
template <typename SVDClass>
struct requires_sigma : std::false_type {};

/**
 * @brief Специализация для RevJac_SVD, указывающая, что конструктор этого
 * алгоритма может требовать спектр.
 * @tparam Matrix Тип матрицы.
 */
template <typename Matrix>
struct requires_sigma<RevJac_SVD<Matrix>> : std::true_type {};

/**
 * @brief Специализация для PseudoRevJac_SVD, указывающая, что конструктор этого
 * алгоритма может требовать спектр.
 * @tparam Matrix Тип матрицы.
 */
template <typename Matrix>
struct requires_sigma<PseudoRevJac_SVD<Matrix>> : std::true_type {};

/**
 * @brief Специализация для v0_RevJac_SVD, указывающая, что конструктор этого
 * алгоритма может требовать спектр.
 * @tparam Matrix Тип матрицы.
 */
template <typename Matrix>
struct requires_sigma<v0_RevJac_SVD<Matrix>> : std::true_type {};

/**
 * @brief Специализация для GivRef_SVD, указывающая, что конструктор этого
 * алгоритма может требовать спектр.
 * @tparam Matrix Тип матрицы.
 */
template <typename Matrix>
struct requires_sigma<GivRef_SVD<Matrix>> : std::true_type {};

/**
 * @brief Примечание по добавлению новых алгоритмов с требованием спектра.
 *
 * Если новый алгоритм SVD требует передачи истинных сингулярных значений
 * в свой конструктор, необходимо добавить аналогичную специализацию
 * шаблонного трейта `requires_sigma` для этого алгоритма.
 *
 * @code{.cpp}
 * template <typename Matrix>
 * struct requires_sigma<YourNewSVDAlgorithm<Matrix>> : std::true_type {};
 * @endcode
 *
 * Замените `YourNewSVDAlgorithm` на имя вашего класса алгоритма SVD.
 */

/**
 * @brief Вектор константных структур AlgorithmInfo, содержащий информацию обо
 * всех зарегистрированных алгоритмах.
 *  std::ofstream divergence_output_file("givens_compare_divergence.txt");

 * Этот статический вектор содержит список всех алгоритмов SVD, доступных для
 * тестирования. Каждый элемент вектора создается с помощью статического метода
 * `createAlgorithmInfoEntry`, который связывает шаблонный класс алгоритма SVD с
 * его строковым именем и функциональными объектами для запуска тестов и
 * выполнения разложения.
 *
 * Чтобы добавить новый алгоритм SVD в тестовый фреймворк, выполните следующие
 * шаги:
 * 1. **Реализуйте класс вашего алгоритма SVD.** Ваш класс должен быть шаблонным
 * по типу данных и иметь публичные методы для получения результатов SVD:
 * - `const MatrixDynamic& matrixU() const;` (или аналогичный, возвращающий
 * левые сингулярные векторы)
 * - `const VectorDynamic& singularValues() const;` (или аналогичный,
 * возвращающий сингулярные значения)
 * - `const MatrixDynamic& matrixV() const;` (или аналогичный, возвращающий
 * правые сингулярные векторы) Класс также должен иметь конструктор(ы),
 * принимающий(е) входную матрицу для разложения и опции вычисления (например,
 * `unsigned int computationOptions`). Опционально могут поддерживаться
 * конструкторы с `std::ostream*` для вывода отладки и с вектором истинных
 * сингулярных значений (если алгоритм их использует).
 * 2. Если ваш алгоритм требует передачи истинных сингулярных значений в
 * конструктор, добавьте специализацию трейта `requires_sigma` (см. выше).
 * 3. Добавьте вызов
 * `createAlgorithmInfoEntry<YourNewSVDAlgorithm>("YourNewAlgorithmName")` в
 * список инициализации этого вектора.
 *
 * @code{.cpp}
 * template <typename FloatingPoint, typename MatrixType>
 * const std::vector<typename SVD_Test<FloatingPoint,
 * MatrixType>::AlgorithmInfo> SVD_Test<FloatingPoint,
 * MatrixType>::algorithmsInfo = {
 * // ... существующие алгоритмы ...
 * createAlgorithmInfoEntry<YourNewSVDAlgorithm>("YourNewAlgorithmName"), // <--
 * Добавьте здесь вашу запись
 * // ... возможно, другие алгоритмы ...
 * };
 * @endcode
 *
 * Замените `YourNewSVDAlgorithm` на имя вашего C++ класса алгоритма SVD,
 * а `"YourNewAlgorithmName"` на уникальное строковое имя,
 * которое будет использоваться в настройках тестов
 * (`svd_test_funcSettings::algorithmName`).
 */

template <typename FloatingPoint, typename MatrixType>
const std::vector<typename SVD_Test<FloatingPoint, MatrixType>::AlgorithmInfo>
    SVD_Test<FloatingPoint, MatrixType>::algorithmsInfo = {
        createAlgorithmInfoEntry<SVD_Project::GivRef_SVD>(
            "SVD_Project::GivRef_SVD"),
        createAlgorithmInfoEntry<SVD_Project::v0_GivRef_SVD>(
            "SVD_Project::v0_GivRef_SVD"),
        // createAlgorithmInfoEntry<SVD_Project::v1_GivRef_SVD>(
        //     "SVD_Project::v1_GivRef_SVD"),
        createAlgorithmInfoEntry<SVD_Project::RevJac_SVD>(
            "SVD_Project::RevJac_SVD"),
        createAlgorithmInfoEntry<SVD_Project::v0_RevJac_SVD>(
            "SVD_Project::v0_RevJac_SVD"),
        createAlgorithmInfoEntry<SVD_Project::NaiveMRRR_SVD>(
            "SVD_Project::NaiveMRRR_SVD"),
        // createAlgorithmInfoEntry<SVD_Project::NaiveBidiagSVD>(
        //     "SVD_Project::NaiveBidiagSVD"),
        createAlgorithmInfoEntry<Eigen::JacobiSVD>("Eigen::JacobiSVD")};

}  // namespace SVD_Project

#endif  // SVD_TEST_CONFIG_H
