#ifndef GUI_H
#define GUI_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QProgressBar>
#include <QString>
#include <QStringList>
#include <vector>
#include "testing.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

    using pair_ = std::pair<int, int>;
    struct tableValue {
        enum stored_value { double_, pair__, string_ };
        tableValue() : double_decimal(0), pair_of_int({0, 0}), sv(string_) {}
        tableValue(double tmp) : double_decimal(tmp), sv(double_) {}
        tableValue(pair_ tmp) : pair_of_int(tmp), sv(pair__) {}
        double double_decimal;
        pair_ pair_of_int;
        stored_value sv;
    };

    // Кастомный класс QTableWidgetItem
    class CustomTableWidgetItem : public QTableWidgetItem {
    public:
        // Конструкторы
        CustomTableWidgetItem(double tmp) : QTableWidgetItem(), value(tmp) {}
        CustomTableWidgetItem(pair_ tmp) : QTableWidgetItem(), value(tmp) {}
        CustomTableWidgetItem(const QString &text) : QTableWidgetItem(text), value() {}

        // Методы для установки и получения значения
        void setValue(const tableValue &val) {
            value = val;
        }
        tableValue getValue() const {
            return value;
        }

        // Переопределяем метод clone для корректного копирования объекта
        CustomTableWidgetItem* clone() const override {
            CustomTableWidgetItem* item = new CustomTableWidgetItem(*this);
            item->value = this->value;
            return item;
        }

        // Переопределяем оператор сравнения для корректной сортировки
        bool operator<(const QTableWidgetItem &other) const override {
            const CustomTableWidgetItem *otherItem = dynamic_cast<const CustomTableWidgetItem*>(&other);
            if (otherItem) {
                // Если оба элемента представляют даблы
                if (value.sv == tableValue::double_ && otherItem->value.sv == tableValue::double_) {
                    return value.double_decimal < otherItem->value.double_decimal;
                }
                // Если оба элемента представляют интервалы (пары)
                else if (value.sv == tableValue::pair__ && otherItem->value.sv == tableValue::pair__) {
                    return value.pair_of_int.first < otherItem->value.pair_of_int.first;
                }
                // Если элемент представляет размерность матрицы, сохранённую как строку вида "NxM"
                else if (value.sv == tableValue::string_) {
                    QString leftText = this->text();
                    QString rightText = other.text();
                    QStringList leftParts = leftText.split("x", Qt::SkipEmptyParts);
                    QStringList rightParts = rightText.split("x", Qt::SkipEmptyParts);
                    if (leftParts.size() == 2 && rightParts.size() == 2) {
                        int leftRows = leftParts[0].toInt();
                        int leftCols = leftParts[1].toInt();
                        int rightRows = rightParts[0].toInt();
                        int rightCols = rightParts[1].toInt();
                        // Сравнение по общему количеству элементов матрицы
                        return (leftRows * leftCols) < (rightRows * rightCols);
                    }
                    return leftText < rightText;
                }
            }
            return QTableWidgetItem::operator<(other);
        }

    private:
        tableValue value;
    };

private slots:
    void runTests();
    void saveTable();

private:
    // Элементы панели управления
    QWidget *controlPanel;
    QComboBox *algorithmSelector;
    QPushButton *startButton;

    // Область для вывода результатов
    QWidget *resultWidget;     QTableWidget *resultTable;
    QScrollArea *tableScrollArea;
    QProgressBar *progressBar;

    std::vector<std::vector<tableValue>> tableData;
    std::string currentResultFile;

    // Layout для центрального виджета
    QVBoxLayout *mainLayout;
    QVBoxLayout *resultLayout;

    // Настройка UI
    void setupUI();
    void populateTable(const std::string &filename);
    void createResultWidget();
    void adjustTableColumns();
};

#endif // GUI_H
