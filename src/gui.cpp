#include "gui.h"
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <fstream>
#include <sstream>

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

    // Вызов svd_test_func в зависимости от выбранного алгоритма
    if (selectedAlgo == "jacobi") {
        svd_test_func<double, SVDGenerator, Eigen::JacobiSVD>(currentResultFile, ratios, sizes, n);
    } else if (selectedAlgo == "dqds") {
        svd_test_func<double, SVDGenerator, DQDS_SVD>(currentResultFile, ratios, sizes, n);
    }

    createResultWindow();
    populateTable(currentResultFile);
    
    startButton->setEnabled(true);
    algorithmSelector->setEnabled(true);
}

void MainWindow::createResultWindow() {
    delete resultWindow;
    resultWindow = nullptr;

    resultWindow = new QWidget();
    resultWindow->setWindowTitle("Test Results");
    resultWindow->resize(800, 600);

    QVBoxLayout *layout = new QVBoxLayout(resultWindow);
    
    resultTable = new QTableWidget(resultWindow);
    QPushButton *saveButton = new QPushButton("Save Table", resultWindow);
    
    layout->addWidget(resultTable);
    layout->addWidget(saveButton);
    
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveTable);
    
    resultWindow->setLayout(layout);
    resultWindow->show();
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

    resultTable->setRowCount(tableData.size() - 1);
    resultTable->setColumnCount(tableData[0].size());
    
    QStringList headers;
    for (const auto& header : tableData[0]) {
        headers << QString::fromStdString(header);
    }
    resultTable->setHorizontalHeaderLabels(headers);
    
    for (size_t i = 1; i < tableData.size(); ++i) {
        for (size_t j = 0; j < tableData[i].size(); ++j) {
            resultTable->setItem(i-1, j, new QTableWidgetItem(
                QString::fromStdString(tableData[i][j])));
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
