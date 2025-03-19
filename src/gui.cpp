#include "gui.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QHeaderView>
#include <fstream>
#include <qvalueaxis.h>
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
    QWidget *centralWidget = new QWidget(this);
    mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(5);
    mainLayout->setContentsMargins(5, 5, 5, 5);

    controlPanel = new QWidget(centralWidget);
    QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setSpacing(5);
    controlLayout->setContentsMargins(5, 5, 5, 5);

    algorithmSelector = new QComboBox(controlPanel);
    algorithmSelector->addItem("Jacobi SVD", QVariant("jacobi"));
    algorithmSelector->addItem("DQDS SVD", QVariant("dqds"));
    startButton = new QPushButton("Run SVD Tests", controlPanel);

    controlLayout->addWidget(algorithmSelector);
    controlLayout->addWidget(startButton);

    parameterSelector = new QComboBox(controlPanel);
    parameterSelector->addItem("Размер матрицы", QVariant("matrix_size"));
    parameterSelector->addItem("Соотношение сингулярных значений", QVariant("sv_ratio"));
    parameterSelector->addItem("Интервал", QVariant("interval"));
    controlLayout->addWidget(parameterSelector);

    controlPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    mainLayout->addWidget(controlPanel, 0);

    resultWidget = new QWidget(centralWidget);
    resultLayout = new QVBoxLayout(resultWidget);
    resultLayout->setSpacing(5);
    resultLayout->setContentsMargins(5, 5, 5, 5);
    resultLayout->setAlignment(Qt::AlignTop);

    mainLayout->addWidget(resultWidget, 1);

    setCentralWidget(centralWidget);

    connect(startButton, &QPushButton::clicked, this, &MainWindow::runTests);
}

void MainWindow::createResultWidget() {
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

    resultTable = new QTableWidget(resultWidget);
    resultTable->setSortingEnabled(true);
    resultTable->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    tableScrollArea = new QScrollArea(resultWidget);
    tableScrollArea->setWidgetResizable(true);
    tableScrollArea->setWidget(resultTable);

    resultLayout->addWidget(tableScrollArea);
}

void MainWindow::createMetricButtons() {
    if (metricButtonWidget) {
        resultLayout->removeWidget(metricButtonWidget);
        metricButtonWidget->deleteLater();
        metricButtonWidget = nullptr;
    }

    if (!resultTable) return;
    int colCount = resultTable->columnCount();
    if (colCount <= 3) return;

    metricButtonWidget = new QWidget(resultWidget);
    QHBoxLayout *buttonLayout = new QHBoxLayout(metricButtonWidget);
    buttonLayout->setSpacing(5);
    buttonLayout->setContentsMargins(5, 5, 5, 5);

    for (int col = 3; col < colCount; ++col) {
        QString headerText;
        QTableWidgetItem *headerItem = resultTable->horizontalHeaderItem(col);
        if (headerItem)
            headerText = headerItem->text();
        else
            headerText = QString("Метрика %1").arg(col);

        QPushButton *metricButton = new QPushButton(headerText, metricButtonWidget);
        int metricColIndex = col;
        connect(metricButton, &QPushButton::clicked, this, [this, metricColIndex]() {
            plotGraphForMetric(metricColIndex);
        });
        buttonLayout->addWidget(metricButton);
    }

    resultLayout->addWidget(metricButtonWidget);
}

void MainWindow::runTests() {
    startButton->setEnabled(false);
    algorithmSelector->setEnabled(false);

    QString selectedAlgo = algorithmSelector->currentData().toString();
    currentResultFile = selectedAlgo.toStdString() + "_test_table.txt";

    // Параметры для svd_test_func
    //std::vector<double> ratios = {1.01, 1.2, 2, 5, 10, 50};
    //std::vector<std::pair<int,int>> sizes = {{3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}};
    //int n = 20;
    // Здесь может вызываться функция тестирования
    //if (selectedAlgo == "jacobi") {
    //    svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>(currentResultFile, ratios, sizes, n);
    //} else if (selectedAlgo == "dqds") {
    //    svd_test_func<double, SVDGenerator, DQDS_SVD>(currentResultFile, ratios, sizes, n);
    //}

    createResultWidget();
    populateTable(currentResultFile);
    adjustTableColumns();

    createMetricButtons();

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
    // Запрашиваем имя файла для сохранения
    QString fileName = QFileDialog::getSaveFileName(this, "Save Table", "", "Text Files (*.txt);;All Files (*)");
    if (fileName.isEmpty())
        return;

    std::ofstream file(fileName.toStdString());
    if (!file) {
        QMessageBox::warning(this, "Error", "Could not save file!");
        return;
    }

    // Записываем заголовки таблицы
    int colCount = resultTable->columnCount();
    for (int j = 0; j < colCount; ++j) {
        if (resultTable->horizontalHeaderItem(j))
            file << resultTable->horizontalHeaderItem(j)->text().toStdString();
        if (j < colCount - 1)
            file << " ";
    }
    file << "\n";

    // Записываем данные таблицы
    int rowCount = resultTable->rowCount();
    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < colCount; ++j) {
            QString cellText;
            QTableWidgetItem *item = resultTable->item(i, j);
            if (item)
                cellText = item->text();
            file << cellText.toStdString();
            if (j < colCount - 1)
                file << " ";
        }
        file << "\n";
    }
    file.close();

    QMessageBox::information(this, "Success", "Table saved successfully!");
}

void MainWindow::plotGraphForMetric(int metricColIndex) {
    if (tableData.empty() || tableData.size() < 2) {
        QMessageBox::warning(this, "Error", "Нет данных для построения графика!");
        return;
    }

    QVector<QPointF> dataPoints;
    QString selectedParam = parameterSelector->currentData().toString();

    if (selectedParam == "matrix_size") {
        std::vector<std::string> origDimensions;
        std::ifstream file(currentResultFile);
        if (file) {
            std::string line;
            std::getline(file, line);
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                if (ss >> cell) {
                    origDimensions.push_back(cell);
                }
            }
            file.close();
        } else {
            QMessageBox::warning(this, "Error", "Не удалось открыть исходный файл для получения размерности матрицы!");
            return;
        }
        if (origDimensions.size() != tableData.size() - 1) {
            QMessageBox::warning(this, "Error", "Количество строк в исходном файле не соответствует ожидаемому.");
            return;
        }
        for (size_t i = 0; i < origDimensions.size(); ++i) {
            QString dimText = QString::fromStdString(origDimensions[i]);
            QStringList parts = dimText.split("x", Qt::SkipEmptyParts);
            double x = 0.0;
            if (!parts.isEmpty()) {
                x = parts[0].toDouble();
            }
            double y = 0.0;
            if (metricColIndex < int(tableData[i+1].size())) {
                if (tableData[i+1][metricColIndex].sv == tableValue::double_)
                    y = tableData[i+1][metricColIndex].double_decimal;
                else if (tableData[i+1][metricColIndex].sv == tableValue::pair__)
                    y = tableData[i+1][metricColIndex].pair_of_int.first;
            }
            dataPoints.append(QPointF(x, y));
        }
    }
    else if (selectedParam == "sv_ratio") {
        for (size_t i = 1; i < tableData.size(); ++i) {
            double x = tableData[i][1].double_decimal;
            double y = 0.0;
            if (metricColIndex < int(tableData[i].size())) {
                if (tableData[i][metricColIndex].sv == tableValue::double_)
                    y = tableData[i][metricColIndex].double_decimal;
                else if (tableData[i][metricColIndex].sv == tableValue::pair__)
                    y = tableData[i][metricColIndex].pair_of_int.first;
            }
            dataPoints.append(QPointF(x, y));
        }
    }
    else if (selectedParam == "interval") {
        for (size_t i = 1; i < tableData.size(); ++i) {
            int a = tableData[i][2].pair_of_int.first;
            int b = tableData[i][2].pair_of_int.second;
            double x = b - a;
            double y = 0.0;
            if (metricColIndex < int(tableData[i].size())) {
                if (tableData[i][metricColIndex].sv == tableValue::double_)
                    y = tableData[i][metricColIndex].double_decimal;
                else if (tableData[i][metricColIndex].sv == tableValue::pair__)
                    y = tableData[i][metricColIndex].pair_of_int.first;
            }
            dataPoints.append(QPointF(x, y));
        }
    }

    std::sort(dataPoints.begin(), dataPoints.end(), [](const QPointF &a, const QPointF &b) {
        return a.x() < b.x();
    });

    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    for (const QPointF &pt : dataPoints) {
        if (pt.y() < minY) minY = pt.y();
        if (pt.y() > maxY) maxY = pt.y();
    }

    // Для отображения очень маленьких значений используем масштабирование
    double absMax = std::max(std::abs(minY), std::abs(maxY));
    int exponent = absMax > 0 ? static_cast<int>(std::floor(std::log10(absMax))) : 0;
    double scale = std::pow(10.0, -exponent);

    // Масштабированные значения y (будем использовать их для оси и для графика)
    double scaledMin = minY * scale;
    double scaledMax = maxY * scale;
    double scaledRange = scaledMax - scaledMin;

    auto niceNum = [](double range, bool round) -> double {
        double exponent = std::floor(std::log10(range));
        double fraction = range / std::pow(10.0, exponent);
        double niceFraction;
        if (round) {
            if (fraction < 1.5)
                niceFraction = 1;
            else if (fraction < 3)
                niceFraction = 2;
            else if (fraction < 7)
                niceFraction = 5;
            else
                niceFraction = 10;
        } else {
            if (fraction <= 1)
                niceFraction = 1;
            else if (fraction <= 2)
                niceFraction = 2;
            else if (fraction <= 5)
                niceFraction = 5;
            else
                niceFraction = 10;
        }
        return niceFraction * std::pow(10.0, exponent);
    };

    int desiredTicks = 11;
    double niceRange = niceNum(scaledRange, false);
    double tickSpacing = niceNum(niceRange / (desiredTicks - 1), true);
    double niceMin = std::floor(scaledMin / tickSpacing) * tickSpacing;
    double niceMax = std::ceil(scaledMax / tickSpacing) * tickSpacing;
    int actualTickCount = static_cast<int>((niceMax - niceMin) / tickSpacing) + 1;

    QValueAxis *axisX = new QValueAxis;
    QString xLabel;
    if (selectedParam == "matrix_size")
        xLabel = "Размер матрицы";
    else if (selectedParam == "sv_ratio")
        xLabel = "Соотношение сингулярных значений";
    else if (selectedParam == "interval")
        xLabel = "Интервал (разность)";
    axisX->setTitleText(xLabel);

    QValueAxis *axisY = new QValueAxis;
    axisY->setTitleText(QString("Значение метрики (x1e%1)").arg(exponent));
    axisY->setLabelFormat("%.2f");
    axisY->setRange(niceMin, niceMax);
    axisY->setTickCount(actualTickCount);
    axisY->setMinorTickCount(4);

    QLineSeries *series = new QLineSeries();
    for (const QPointF &pt : dataPoints)
        series->append(QPointF(pt.x(), pt.y() * scale));

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisX);
    series->attachAxis(axisY);

    QTableWidgetItem *metricHeaderItem = resultTable->horizontalHeaderItem(metricColIndex);
    QString metricTitle = metricHeaderItem ? metricHeaderItem->text() : "Метрика";
    chart->setTitle(QString("Зависимость %1 от %2").arg(metricTitle, xLabel));

    if (chartView) {
        resultLayout->removeWidget(chartView);
        chartView->deleteLater();
    }
    chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    resultLayout->addWidget(chartView);
}
