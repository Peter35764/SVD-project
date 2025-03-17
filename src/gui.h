#ifndef GUI_H
#define GUI_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QProgressBar>
#include <vector>
#include "testing.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

    using pair_= std::pair<int,int>;
    struct tableValue{
        enum stored_value{double_, pair__, string_};
        tableValue(): double_decimal(0), pair_of_int({0,0}), sv(string_){};
        tableValue(double tmp): double_decimal(tmp), sv(double_){};
        tableValue(pair_ tmp): pair_of_int(tmp), sv(pair__){};
        double double_decimal;
        pair_ pair_of_int;
        stored_value sv;

    };

    // Кастомный класс QTableWidgetItem
    class CustomTableWidgetItem : public QTableWidgetItem {

    public:
        // Конструкторы
        CustomTableWidgetItem(double tmp) : QTableWidgetItem(), value(tmp){};
        CustomTableWidgetItem(pair_ tmp) : QTableWidgetItem(), value(tmp){}
        CustomTableWidgetItem(const QString &text) : QTableWidgetItem(text), value(){}

        // Методы для установки и получения значения
        void setValue(const tableValue& val) {
            value = val;

        }

        tableValue getValue() const {
            return value;
        }

        // Переопределяем метод clone, чтобы корректно копировать объект
        CustomTableWidgetItem* clone() const override {
            CustomTableWidgetItem* item = new CustomTableWidgetItem(*this);
            item->value = this->value;
            return item;
        }

    private:
        tableValue value;
    };

private slots:
    void runTests();
    void saveTable();

private:
    QPushButton *startButton;
    QComboBox *algorithmSelector;
    QWidget *resultWindow;
    QTableWidget *resultTable;
    QScrollArea *tableScrollArea;
	QProgressBar *progressBar;
    std::vector<std::vector<std::string>> tableData;
    std::string currentResultFile;  

    void setupUI();
    void populateTable(const std::string& filename);
    void createResultWindow();
};

#endif // GUI_H
