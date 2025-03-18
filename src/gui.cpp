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
    tableData.clear();

    std::ifstream file(filename);
    if (!file) {
        QMessageBox::warning(this, "Error", "Could not open result file!");
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (ss >> cell) {
            row.push_back(cell);
        }
        if (!row.empty()) {
            tableData.push_back(row);
        }
    }
    file.close();

    if (tableData.empty())
        return;

    std::vector<std::string> headers = tableData[0];

    int svIndex = -1, intervalIndex = -1;
    for (size_t j = 0; j < headers.size(); ++j) {
        if (headers[j] == "SV")
            svIndex = j;
        if (headers[j] == "interval")
            intervalIndex = j;
    }

    std::vector<std::vector<std::string>> newTableData;
    std::vector<std::string> newHeaders;

    for (size_t j = 0; j < headers.size(); ++j) {
        if ((int)j == svIndex && intervalIndex == j + 1) {
            newHeaders.push_back("SV interval");
            j++; // пропускаем столбец "interval"
        } else {
            newHeaders.push_back(headers[j]);
        }
    }
    newTableData.push_back(newHeaders);

    for (size_t i = 1; i < tableData.size(); ++i) {
        std::vector<std::string> newRow;
        size_t colLimit = std::min(headers.size(), tableData[i].size());
        for (size_t j = 0; j < colLimit; ++j) {
            if ((int)j == svIndex && intervalIndex == j + 1 && (j + 1) < colLimit) {
                newRow.push_back(tableData[i][j] + " " + tableData[i][j + 1]);
                j++; // пропускаем столбец "interval"
            } else {
                newRow.push_back(tableData[i][j]);
            }
        }
        newTableData.push_back(newRow);
    }

    // Удаление пустых столбцов (проверка всех строк, кроме заголовка)
    int colCount = newHeaders.size();
    std::vector<bool> emptyColumn(colCount, true);
    for (int j = 0; j < colCount; ++j) {
        for (size_t i = 1; i < newTableData.size(); ++i) {
            // Если размер строки меньше, чем ожидается, пропускаем проверку
            if (j < newTableData[i].size() && !newTableData[i][j].empty()) {
                emptyColumn[j] = false;
                break;
            }
        }
    }

    // Формирование итоговой таблицы без пустых столбцов
    std::vector<std::vector<std::string>> finalTableData;
    for (size_t i = 0; i < newTableData.size(); ++i) {
        std::vector<std::string> finalRow;
        for (int j = 0; j < colCount; ++j) {
            if (!emptyColumn[j] && j < newTableData[i].size()) {
                finalRow.push_back(newTableData[i][j]);
            }
        }
        finalTableData.push_back(finalRow);
    }
    tableData = finalTableData;

    // Проверка корректности данных для таблицы
    if (tableData.empty() || tableData[0].empty()) {
        QMessageBox::warning(this, "Error", "No valid data to display!");
        return;
    }

    // Обновляем QTableWidget
    resultTable->setRowCount(tableData.size() - 1);
    resultTable->setColumnCount(tableData[0].size());

    QStringList qHeaders;
    for (const auto &header : tableData[0]) {
        qHeaders << QString::fromStdString(header);
    }
    resultTable->setHorizontalHeaderLabels(qHeaders);

    for (size_t i = 1; i < tableData.size(); ++i) {
        for (size_t j = 0; j < tableData[i].size(); ++j) {
            resultTable->setItem(i - 1, j,
                                 new QTableWidgetItem(QString::fromStdString(tableData[i][j])));
        }
    }
}



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
}
