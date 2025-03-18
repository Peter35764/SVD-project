#include "gui.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QHeaderView>
#include <fstream>
#include <sstream>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    controlPanel(nullptr), resultWidget(nullptr), resultTable(nullptr),
    tableScrollArea(nullptr), progressBar(nullptr) {
    setWindowTitle("SVD Test Application");
    resize(800, 600);

    setupUI();
}

MainWindow::~MainWindow() {
    if(resultWidget) {
        delete resultWidget;
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

    algorithmSelector = new QComboBox(controlPanel);
    algorithmSelector->addItem("Jacobi SVD", QVariant("jacobi"));
    algorithmSelector->addItem("DQDS SVD", QVariant("dqds"));
    startButton = new QPushButton("Run SVD Tests", controlPanel);

    controlLayout->addWidget(algorithmSelector);
    controlLayout->addWidget(startButton);

    // Задаем фиксированную вертикальную политику для controlPanel, чтобы его высота не изменялась
    controlPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    // Добавляем панель управления в основной layout
    mainLayout->addWidget(controlPanel, 0); // stretch 0

    // Создаем контейнер для вывода результатов (изначально пустой)
    resultWidget = new QWidget(centralWidget);
    resultLayout = new QVBoxLayout(resultWidget);
    resultLayout->setSpacing(5);
    resultLayout->setContentsMargins(5, 5, 5, 5);
    resultLayout->setAlignment(Qt::AlignTop);

    mainLayout->addWidget(resultWidget, 1); // stretch 1, чтобы занимал оставшееся пространство

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

    // Добавляем область с таблицей в layout контейнера результатов и выравниваем по верхней границе
    resultLayout->addWidget(tableScrollArea);
}

void MainWindow::runTests() {
    // Отключаем элементы управления во время выполнения теста
    startButton->setEnabled(false);
    algorithmSelector->setEnabled(false);

    QString selectedAlgo = algorithmSelector->currentData().toString();
    currentResultFile = selectedAlgo.toStdString() + "_test_table.txt";

    // Здесь может вызываться функция тестирования:
    // if (selectedAlgo == "jacobi") { ... } else if (selectedAlgo == "dqds") { ... }

    createResultWidget();
    populateTable(currentResultFile);
    adjustTableColumns();

    // Восстанавливаем возможность взаимодействия с элементами управления
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
