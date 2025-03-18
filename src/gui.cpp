#include "gui.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <fstream>
#include <sstream>

/*
 * 1. Потыкать таблицу(2-й патч)
 * 2. Заменить Qtable на CustomTable
 * 3. Научиться трактовать паршенное значение
 */

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("SVD Test Application");
    resize(400, 200);

    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    algorithmSelector = new QComboBox(this);
    algorithmSelector->addItem("Jacobi SVD", QVariant("jacobi"));
    algorithmSelector->addItem("DQDS SVD", QVariant("dqds"));

    startButton = new QPushButton("Run SVD Tests", this);

    layout->addWidget(algorithmSelector);
    layout->addWidget(startButton);
    layout->addStretch();

    setCentralWidget(centralWidget);

    connect(startButton, &QPushButton::clicked, this, &MainWindow::runTests);

    resultWindow = nullptr;
}

MainWindow::~MainWindow() {
    delete resultWindow;
}

void MainWindow::runTests() {
    startButton->setEnabled(false);
    algorithmSelector->setEnabled(false);

    QString selectedAlgo = algorithmSelector->currentData().toString();
    currentResultFile = selectedAlgo.toStdString() + "_test_table.txt";

    // Параметры для svd_test_func
    std::vector<double> ratios = {1.01, 1.2, 2, 5, 10, 50};
    std::vector<std::pair<int,int>> sizes = {{3,3}, {5,5}, {10,10}, {20,20}, {50,50}, {100,100}};
    int n = 20;
/*
    // Вызов svd_test_func в зависимости от выбранного алгоритма
    if (selectedAlgo == "jacobi") {
        svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>(currentResultFile, ratios, sizes, n);
    } else if (selectedAlgo == "dqds") {
        svd_test_func<double, SVDGenerator, DQDS_SVD>(currentResultFile, ratios, sizes, n);
    }
*/
    createResultWindow();
    populateTable(currentResultFile);

    startButton->setEnabled(true);
    algorithmSelector->setEnabled(true);
}

void MainWindow::createResultWindow() {
    // Получаем центральный виджет. Если он не установлен, создаём его.
    QWidget *central = this->centralWidget();
    if (!central) {
        central = new QWidget(this);
        setCentralWidget(central);
    } else {
        // Очищаем центральный виджет от всех элементов, чтобы избежать конфликтов
        if (central->layout()) {
            QLayoutItem *child;
            while ((child = central->layout()->takeAt(0)) != nullptr) {
                if (child->widget()) {
                    child->widget()->deleteLater();
                }
                delete child;
            }
        } else {
            // Если layout отсутствует, создаём новый
            QVBoxLayout *newLayout = new QVBoxLayout(central);
            central->setLayout(newLayout);
        }
    }

    // Получаем (или создаём) layout центрального виджета
    QVBoxLayout *layout = qobject_cast<QVBoxLayout*>(central->layout());
    if (!layout) {
        layout = new QVBoxLayout(central);
        central->setLayout(layout);
    }

    // Создаём новую таблицу и добавляем её в layout
    resultTable = new QTableWidget(central);
    layout->addWidget(resultTable);

    // QTableWidget имеет встроенные полосы прокрутки

    // Добавляем кнопку сохранения
    QPushButton *saveButton = new QPushButton("Save Table", central);
    layout->addWidget(saveButton);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveTable);
}

void MainWindow::populateTable(const std::string& filename) {
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

    // Читаем данные
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
                row[col] = tableValue();
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

    // Удаляем пустые столбцы
    int colCount = finalHeaders.size();
    std::vector<bool> emptyColumn(colCount, true);
    for (int j = 0; j < colCount; ++j) {
        if (j == 0) { // Dimension всегда заполнен
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

    // Устанавливаем таблицу
    resultTable->setRowCount(tempData.size());
    resultTable->setColumnCount(filteredHeaders.size());
    QStringList qHeaders;
    for (const auto& header : filteredHeaders) {
        qHeaders << QString::fromStdString(header);
    }
    resultTable->setHorizontalHeaderLabels(qHeaders);
    resultTable->setSortingEnabled(true);

    // Заполняем таблицу
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
            if (j == 0) { // Dimension как строка
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
            if (j < tableData[i].size() - 1) file << " ";
        }
        file << "\n";
    }
    file.close();

    QMessageBox::information(this, "Success", "Table saved successfully!");
}

/*
void MainWindow::saveTable() {
    QString fileName = QFileDialog::getSaveFileName(this,
        "Save Table", "", "Text Files (*.txt);;All Files (*)");

    if (fileName.isEmpty()) return;

    std::ofstream file(fileName.toStdString());
    if (!file) {
        QMessageBox::warning(this, "Error", "Could not save file!");
        return;
    }

    for (const auto& row : tableData) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << " ";
        }
        file << "\n";
    }
    file.close();

    QMessageBox::information(this, "Success", "Table saved successfully!");
}*/
