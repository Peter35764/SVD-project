#include "gui.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QHeaderView>
#include <fstream>
#include <sstream>
#include <QString>
#include <QStringList>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    controlPanel(nullptr), resultWidget(nullptr), resultTable(nullptr),
    tableScrollArea(nullptr), progressBar(nullptr), chartView(nullptr), metricButtonWidget(nullptr) {
    setWindowTitle("SVD Test Application");
    resize(800, 600);

    setupUI();
}

MainWindow::~MainWindow() {
    if(resultWidget) {
        delete resultWidget;
    }
    if(chartView) {
        delete chartView;
    }
    if(metricButtonWidget) {
        delete metricButtonWidget;
    }
}

void MainWindow::setupUI() {
    // Создаем центральный виджет и основной layout
    QWidget *centralWidget = new QWidget(this);
    mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(5, 5, 5, 5);

    // Создаем панель управления и задаем для нее фиксированную вертикальную политику размера
    controlPanel = new QWidget(centralWidget);
    QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setSpacing(5);
    controlLayout->setContentsMargins(5, 5, 5, 5);

    // Элементы выбора алгоритма
    algorithmSelector = new QComboBox(controlPanel);
    algorithmSelector->addItem("Jacobi SVD", QVariant("jacobi"));
    algorithmSelector->addItem("DQDS SVD", QVariant("dqds"));
    startButton = new QPushButton("Run SVD Tests", controlPanel);

    controlLayout->addWidget(algorithmSelector);
    controlLayout->addWidget(startButton);

    // Элемент для выбора параметра визуализации (ось X)
    parameterSelector = new QComboBox(controlPanel);
    parameterSelector->addItem("Размер матрицы", QVariant("matrix_size"));
    parameterSelector->addItem("Соотношение сингулярных значений", QVariant("sv_ratio"));
    parameterSelector->addItem("Интервал", QVariant("interval"));
    controlLayout->addWidget(parameterSelector);

    // Фиксированная высота панели управления
    controlPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // Добавляем панель управления в основной layout
    mainLayout->addWidget(controlPanel, 0);

    // Создаем контейнер для вывода результатов (изначально пустой)
    resultWidget = new QWidget(centralWidget);
    resultLayout = new QVBoxLayout(resultWidget);
    resultLayout->setSpacing(5);
    resultLayout->setContentsMargins(5, 5, 5, 5);
    resultLayout->setAlignment(Qt::AlignTop);

    mainLayout->addWidget(resultWidget, 1);

    setCentralWidget(centralWidget);

    // Подключаем сигнал кнопки запуска тестов
    connect(startButton, &QPushButton::clicked, this, &MainWindow::runTests);
}

void MainWindow::createResultWidget() {
    // Если ранее созданная таблица существует, удаляем её
    if (resultTable) {
        resultLayout->removeWidget(resultTable);
        resultTable->deleteLater();
        resultTable = nullptr;
    }
    if (tableScrollArea) {
        resultLayout->removeWidget(tableScrollArea);
        tableScrollArea->deleteLater();
        tableScrollArea = nullptr;
    }

    // Создаем новую таблицу и помещаем её в QScrollArea
    resultTable = new QTableWidget(resultWidget);
    resultTable->setSortingEnabled(true);
    resultTable->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    tableScrollArea = new QScrollArea(resultWidget);
    tableScrollArea->setWidgetResizable(true);
    tableScrollArea->setWidget(resultTable);

    resultLayout->addWidget(tableScrollArea);
}

void MainWindow::createMetricButtons() {
    // Если ранее созданные кнопки существуют, удаляем их
    if (metricButtonWidget) {
        resultLayout->removeWidget(metricButtonWidget);
        metricButtonWidget->deleteLater();
        metricButtonWidget = nullptr;
    }

    if (!resultTable) return;
    int colCount = resultTable->columnCount();
    // Предполагаем, что первые 3 колонки – параметры, остальные – метрики
    if (colCount <= 3) return;

    metricButtonWidget = new QWidget(resultWidget);
    QHBoxLayout *buttonLayout = new QHBoxLayout(metricButtonWidget);
    buttonLayout->setSpacing(5);
    buttonLayout->setContentsMargins(5, 5, 5, 5);

    // Создаем кнопку для каждой метрики (начиная с колонки 3)
    for (int col = 3; col < colCount; ++col) {
        QString headerText;
        QTableWidgetItem *headerItem = resultTable->horizontalHeaderItem(col);
        if (headerItem)
            headerText = headerItem->text();
        else
            headerText = QString("Метрика %1").arg(col);

        QPushButton *metricButton = new QPushButton(headerText, metricButtonWidget);
        // Явно захватываем this и metricColIndex для соответствия C++20
        int metricColIndex = col;
        connect(metricButton, &QPushButton::clicked, this, [this, metricColIndex]() {
            plotGraphForMetric(metricColIndex);
        });
        buttonLayout->addWidget(metricButton);
    }

    resultLayout->addWidget(metricButtonWidget);
}

void MainWindow::runTests() {
    // Отключаем элементы управления во время выполнения теста
    startButton->setEnabled(false);
    algorithmSelector->setEnabled(false);

    QString selectedAlgo = algorithmSelector->currentData().toString();
    currentResultFile = selectedAlgo.toStdString() + "_test_table.txt";

    // Здесь может вызываться функция тестирования
    // if (selectedAlgo == "jacobi") { ... } else if (selectedAlgo == "dqds") { ... }

    createResultWidget();
    populateTable(currentResultFile);
    adjustTableColumns();

    // Создаем кнопки для метрик после таблицы
    createMetricButtons();

    // Восстанавливаем возможность взаимодействия
    startButton->setEnabled(true);
    algorithmSelector->setEnabled(true);
}

void MainWindow::populateTable(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
        QMessageBox::warning(this, "Error", "Could not open result file!");
        return;
    }

    // Читаем заголовки
    std::string line;
    std::getline(file, line);
    std::vector<std::string> headers;
    std::stringstream ss(line);
    std::string cell;
    while (ss >> cell) {
        headers.push_back(cell);
    }

    // Объединяем "SV" и "interval" в "SV interval"
    int svIndex = -1, intervalIndex = -1;
    for (size_t j = 0; j < headers.size(); ++j) {
        if (headers[j] == "SV") svIndex = j;
        if (headers[j] == "interval") intervalIndex = j;
    }

    std::vector<std::string> finalHeaders;
    for (size_t j = 0; j < headers.size(); ++j) {
        if ((int)j == svIndex && intervalIndex == j + 1) {
            finalHeaders.push_back("SV interval");
            j++; // Пропускаем "interval"
        } else {
            finalHeaders.push_back(headers[j]);
        }
    }

    // Чтение данных
    std::vector<std::vector<tableValue>> tempData;
    std::vector<std::string> dimensionValues;
    while (std::getline(file, line)) {
        std::vector<tableValue> row(finalHeaders.size());
        std::stringstream ssRow(line);
        std::string cell;
        size_t col = 0;
        while (ssRow >> cell && col < finalHeaders.size()) {
            if (col == 0) { // Dimension
                dimensionValues.push_back(cell);
                row[col] = tableValue(); // тип string_
            } else if ((int)col == (svIndex >= 0 ? svIndex : -1) && intervalIndex == svIndex + 1) {
                std::string nextCell;
                if (ssRow >> nextCell) {
                    std::string combined = cell + " " + nextCell;
                    int first, second;
                    if (sscanf(combined.c_str(), "[%d, %d]", &first, &second) == 2) {
                        row[col] = tableValue({first, second});
                    } else {
                        row[col] = tableValue();
                    }
                }
            } else if (col == 1) { // Sigma-max/min-ratio
                try {
                    double value = std::stod(cell);
                    row[col] = tableValue(value);
                } catch (...) {
                    row[col] = tableValue();
                }
            } else { // Остальные столбцы
                try {
                    double value = std::stod(cell);
                    row[col] = tableValue(value);
                } catch (...) {
                    row[col] = tableValue();
                }
            }
            col++;
        }
        tempData.push_back(row);
    }
    file.close();

    if (tempData.empty()) {
        QMessageBox::warning(this, "Error", "No valid data in result file!");
        return;
    }

    // Удаление пустых столбцов
    int colCount = finalHeaders.size();
    std::vector<bool> emptyColumn(colCount, true);
    for (int j = 0; j < colCount; ++j) {
        if (j == 0) {
            emptyColumn[j] = false;
        } else {
            for (size_t i = 0; i < tempData.size(); ++i) {
                if (tempData[i][j].sv != tableValue::string_) {
                    emptyColumn[j] = false;
                    break;
                }
            }
        }
    }

    std::vector<std::string> filteredHeaders;
    std::vector<std::vector<tableValue>> filteredData;
    filteredData.push_back(std::vector<tableValue>()); // Заголовки
    for (int j = 0; j < colCount; ++j) {
        if (!emptyColumn[j]) {
            filteredHeaders.push_back(finalHeaders[j]);
            filteredData[0].push_back(tableValue());
        }
    }
    for (size_t i = 0; i < tempData.size(); ++i) {
        std::vector<tableValue> filteredRow;
        for (int j = 0; j < colCount; ++j) {
            if (!emptyColumn[j]) {
                filteredRow.push_back(tempData[i][j]);
            }
        }
        filteredData.push_back(filteredRow);
    }
    tableData = filteredData;

    // Заполнение таблицы
    resultTable->setRowCount(tempData.size());
    resultTable->setColumnCount(filteredHeaders.size());
    QStringList qHeaders;
    for (const auto &header : filteredHeaders) {
        qHeaders << QString::fromStdString(header);
    }
    resultTable->setHorizontalHeaderLabels(qHeaders);
    resultTable->setSortingEnabled(true);

    for (size_t i = 0; i < tempData.size(); ++i) {
        for (size_t j = 0; j < filteredData[i + 1].size(); ++j) {
            CustomTableWidgetItem* item = nullptr;
            if (j == 0) { // Dimension
                item = new CustomTableWidgetItem(QString::fromStdString(dimensionValues[i]));
            } else if (filteredData[i + 1][j].sv == tableValue::double_) {
                item = new CustomTableWidgetItem(filteredData[i + 1][j].double_decimal);
                item->setText(j == 1 ? QString::number(filteredData[i + 1][j].double_decimal, 'f', 2)
                                     : QString::number(filteredData[i + 1][j].double_decimal, 'e', 5));
            } else if (filteredData[i + 1][j].sv == tableValue::pair__) {
                item = new CustomTableWidgetItem(filteredData[i + 1][j].pair_of_int);
                item->setText(QString("[%1, %2]").arg(filteredData[i + 1][j].pair_of_int.first)
                                  .arg(filteredData[i + 1][j].pair_of_int.second));
            } else {
                item = new CustomTableWidgetItem("Invalid");
            }
            resultTable->setItem(i, j, item);
        }
    }
}

void MainWindow::adjustTableColumns() {
    resultTable->resizeColumnsToContents();
    resultTable->horizontalHeader()->setStretchLastSection(false);
}

void MainWindow::saveTable() {
    QString fileName = QFileDialog::getSaveFileName(this, "Save Table", "", "Text Files (*.txt);;All Files (*)");
    if (fileName.isEmpty()) return;

    std::ofstream file(fileName.toStdString());
    if (!file) {
        QMessageBox::warning(this, "Error", "Could not save file!");
        return;
    }

    for (size_t i = 0; i < tableData.size(); ++i) {
        for (size_t j = 0; j < tableData[i].size(); ++j) {
            if (j == 0) {
                file << (i == 0 ? resultTable->horizontalHeaderItem(j)->text().toStdString()
                                : resultTable->item(i - 1, j)->text().toStdString());
            } else if (tableData[i][j].sv == tableValue::double_) {
                file << (j == 1 ? QString::number(tableData[i][j].double_decimal, 'f', 2).toStdString()
                                : QString::number(tableData[i][j].double_decimal, 'e', 5).toStdString());
            } else if (tableData[i][j].sv == tableValue::pair__) {
                file << "[" << tableData[i][j].pair_of_int.first << ", " << tableData[i][j].pair_of_int.second << "]";
            } else {
                file << "Invalid";
            }
            if (j < tableData[i].size() - 1)
                file << " ";
        }
        file << "\n";
    }
    file.close();

    QMessageBox::information(this, "Success", "Table saved successfully!");
}

void MainWindow::plotGraphForMetric(int metricColIndex) {
    if (!resultTable) {
        QMessageBox::warning(this, "Error", "Нет данных для построения графика!");
        return;
    }
    int rowCount = resultTable->rowCount();
    int totalCols = resultTable->columnCount();
    if (rowCount == 0 || metricColIndex >= totalCols) {
        QMessageBox::warning(this, "Error", "Недостаточно данных для построения графика!");
        return;
    }

    QVector<double> xValues;
    QVector<double> yValues;

    // Определяем выбранный параметр для оси X
    QString selectedParam = parameterSelector->currentData().toString();
    int paramCol = 0;
    if (selectedParam == "matrix_size") {
        paramCol = 0;
    } else if (selectedParam == "sv_ratio") {
        paramCol = 1;
    } else if (selectedParam == "interval") {
        paramCol = 2;
    }

    // Собираем данные из таблицы
    for (int i = 0; i < rowCount; i++) {
        double x = 0;
        // Извлекаем x значение в зависимости от выбранного параметра
        if (selectedParam == "matrix_size") {
            // Извлекаем размер матрицы из первой колонки (например, "100x100")
            QString dimText = resultTable->item(i, 0)->text().trimmed();
            QStringList parts = dimText.split("x", Qt::SkipEmptyParts);
            if (!parts.isEmpty()) {
                x = parts[0].toDouble();
            }
        } else if (selectedParam == "sv_ratio") {
            // Извлекаем значение соотношения сингулярных значений из второй колонки
            QString ratioText = resultTable->item(i, 1)->text().trimmed();
            x = ratioText.toDouble();
        } else if (selectedParam == "interval") {
            // Извлекаем интервал из третьей колонки, ожидается формат "[a, b]"
            QString intervalText = resultTable->item(i, 2)->text().trimmed();
            if (intervalText.startsWith("[") && intervalText.endsWith("]")) {
                intervalText = intervalText.mid(1, intervalText.length() - 2);
                QStringList numbers = intervalText.split(",", Qt::SkipEmptyParts);
                if (numbers.size() >= 2) {
                    double a = numbers[0].trimmed().toDouble();
                    double b = numbers[1].trimmed().toDouble();
                    x = b - a; // Используем разность как представление интервала
                }
            }
        }
        xValues.push_back(x);

        // Извлекаем y значение из выбранной метрики
        QString metricText = resultTable->item(i, metricColIndex)->text().trimmed();
        double y = metricText.toDouble();
        yValues.push_back(y);
    }

    // Построение графика с использованием Qt Charts
    QLineSeries *series = new QLineSeries();
    for (int i = 0; i < xValues.size(); i++) {
        series->append(xValues[i], yValues[i]);
    }

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->createDefaultAxes();

    // Задаем подписи осей
    QString xLabel;
    if (selectedParam == "matrix_size") {
        xLabel = "Размер матрицы";
    } else if (selectedParam == "sv_ratio") {
        xLabel = "Соотношение сингулярных значений";
    } else if (selectedParam == "interval") {
        xLabel = "Интервал (разность)";
    }
    // Устанавливаем подпись для оси X
    chart->axes(Qt::Horizontal).first()->setTitleText(xLabel);
    // Устанавливаем подпись для оси Y (размерность метрики)
    chart->axes(Qt::Vertical).first()->setTitleText("Значение метрики");

    // Устанавливаем заголовок графика как название метрики
    QTableWidgetItem *metricHeaderItem = resultTable->horizontalHeaderItem(metricColIndex);
    QString metricTitle = metricHeaderItem ? metricHeaderItem->text() : QString("Метрика");
    chart->setTitle(QString("Зависимость %1 от %2").arg(metricTitle, xLabel));

    // Если ранее был построен график, удаляем его
    if (chartView) {
        resultLayout->removeWidget(chartView);
        chartView->deleteLater();
    }

    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    resultLayout->addWidget(chartView);
}
