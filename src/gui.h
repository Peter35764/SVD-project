#ifndef GUI_H
#define GUI_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QComboBox>
#include <vector>
#include "testing.h"  // Подключаем новый заголовок

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void runTests();
    void saveTable();

private:
    QPushButton *startButton;
    QComboBox *algorithmSelector;
    QWidget *resultWindow;
    QTableWidget *resultTable;
    std::vector<std::vector<std::string>> tableData;
    std::string currentResultFile;  // Для хранения имени файла результатов

    void createResultWindow();
    void populateTable(const std::string& filename);
};

#endif // GUI_H
