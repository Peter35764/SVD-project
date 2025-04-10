import sys
import os
import csv
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QSplitter, QMessageBox, QHeaderView, QLineEdit, QLabel
)
from PyQt6.QtCore import Qt

'''
TODO Заменить количесво порядков на флоат в научной форме 
сделать дефолт значение 1e-31
Таблтчку формат старое знач -> новое знач
'''

class CsvTableWidget(QTableWidget):
    """Класс для отображения CSV в виде таблицы."""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setRowCount(len(data))
        self.setColumnCount(max(len(row) for row in data) if data else 0)
        self.populateTable()
        self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def populateTable(self):
        for i, row in enumerate(self.data):
            for j, cell in enumerate(row):
                item = QTableWidgetItem(cell)
                self.setItem(i, j, item)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сравнение CSV файлов")
        self.resize(1000, 600)
        self.file1_path = None
        self.file2_path = None
        self.data1 = None
        self.data2 = None
        self.comparison_data = {}  # Для хранения пар (старое, новое)
        
        # Центральный виджет и основной layout
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        # Горизонтальный layout для кнопок и поля ввода порогового значения
        self.button_layout = QHBoxLayout()
        self.btn_file1 = QPushButton("Файл 1")
        self.btn_file2 = QPushButton("Файл 2")
        self.btn_compare = QPushButton("Сравнить")
        self.button_layout.addWidget(self.btn_file1)
        self.button_layout.addWidget(self.btn_file2)
        self.button_layout.addWidget(self.btn_compare)
        
        # Добавляем метку и QLineEdit для порогового значения
        self.label_threshold = QLabel("Порог:")
        self.button_layout.addWidget(self.label_threshold)
        self.le_threshold = QLineEdit()
        # Устанавливаем дефолтное значение "1e-31"
        self.le_threshold.setText("1e-31")
        self.button_layout.addWidget(self.le_threshold)
        
        self.main_layout.addLayout(self.button_layout)
        
        # Виджет для отображения таблиц
        self.table_container = QWidget()
        self.table_layout = QHBoxLayout(self.table_container)
        self.main_layout.addWidget(self.table_container)
        
        self.statusBar().showMessage("Кликните на ячейку для отображения сравнения")
        
        self.btn_file1.clicked.connect(self.load_file1)
        self.btn_file2.clicked.connect(self.load_file2)
        self.btn_compare.clicked.connect(self.compare_files)
        
        self.table1 = None
        self.table2 = None
        self.comparison_table = None
        
        self.setAcceptDrops(True)
    
    def load_csv(self, filepath):
        """Считывание CSV-файла в список списков."""
        data = []
        try:
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append(row)
            return data
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл {filepath}:\n{e}")
            return None

    def display_tables(self):
        """Отображение таблиц: если только Файл 1 загружен – одна таблица, если оба – с помощью QSplitter."""
        for i in reversed(range(self.table_layout.count())):
            widget = self.table_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        if self.data1 and not self.data2:
            self.table1 = CsvTableWidget(self.data1)
            self.table_layout.addWidget(self.table1)
        elif self.data1 and self.data2:
            self.table1 = CsvTableWidget(self.data1)
            self.table2 = CsvTableWidget(self.data2)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self.table1)
            splitter.addWidget(self.table2)
            self.table_layout.addWidget(splitter)

    def load_file1(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл для Файла 1", "", "CSV файлы (*.csv)")
        if filepath:
            self.file1_path = filepath
            self.data1 = self.load_csv(filepath)
            self.display_tables()
    
    def load_file2(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл для Файла 2", "", "CSV файлы (*.csv)")
        if filepath:
            self.file2_path = filepath
            self.data2 = self.load_csv(filepath)
            self.display_tables()
    
    def compare_files(self):
        if not self.data1 or not self.data2:
            QMessageBox.warning(self, "Предупреждение", "Необходимо загрузить оба файла для сравнения!")
            return

        # Перечитываем файлы для учета внешних изменений
        if self.file1_path:
            self.data1 = self.load_csv(self.file1_path)
        if self.file2_path:
            self.data2 = self.load_csv(self.file2_path)

        # Файл 1 считается старым, Файл 2 – новым
        older_data = self.data1
        newer_data = self.data2

        rows = min(len(newer_data), len(older_data))
        cols = min(max(len(r) for r in newer_data), max(len(r) for r in older_data))

        self.comparison_table = QTableWidget(rows, cols)
        self.comparison_table.setHorizontalScrollMode(self.comparison_table.ScrollMode.ScrollPerPixel)
        self.comparison_table.setVerticalScrollMode(self.comparison_table.ScrollMode.ScrollPerPixel)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.comparison_data.clear()

        # Считываем пороговое значение из QLineEdit и преобразуем в float
        try:
            threshold = float(self.le_threshold.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Некорректное пороговое значение!")
            return

        for i in range(rows):
            for j in range(cols):
                new_cell = newer_data[i][j] if j < len(newer_data[i]) else ""
                old_cell = older_data[i][j] if j < len(older_data[i]) else ""
                # Форматируем ячейку как "старое знач -> новое знач"
                display_text = f"{old_cell} -> {new_cell}"
                item = QTableWidgetItem(display_text)
                self.comparison_data[(i, j)] = (old_cell, new_cell)

                try:
                    new_val = float(new_cell)
                    old_val = float(old_cell)
                    
                    # Если значения равны – стандартный фон (белый)
                    if new_val == old_val:
                        item.setBackground(Qt.GlobalColor.white)
                    # Если старое значение 0, деление невозможно – оставляем жёлтым при неравенстве
                    elif old_val == 0:
                        item.setBackground(Qt.GlobalColor.yellow)
                    else:
                        diff = new_val - old_val
                        # Если абсолютная разница меньше порога – жёлтый фон
                        if abs(diff) < threshold:
                            item.setBackground(Qt.GlobalColor.yellow)
                        # Если новое значение больше старого и разница не менее порога – красный фон
                        elif diff >= threshold:
                            item.setBackground(Qt.GlobalColor.red)
                        # Если новое значение меньше старого и разница не менее порога – зелёный фон
                        elif diff <= -threshold:
                            item.setBackground(Qt.GlobalColor.green)
                        else:
                            item.setBackground(Qt.GlobalColor.white)
                except ValueError:
                    item.setBackground(Qt.GlobalColor.white)

                self.comparison_table.setItem(i, j, item)

        self.comparison_table.cellClicked.connect(self.on_cell_clicked)

        for i in reversed(range(self.table_layout.count())):
            widget = self.table_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.table_layout.addWidget(self.comparison_table)

    def on_cell_clicked(self, row, col):
        if (row, col) in self.comparison_data:
            old_val, new_val = self.comparison_data[(row, col)]
            self.statusBar().showMessage(f"{old_val} -> {new_val}")
        else:
            self.statusBar().showMessage("Данных для сравнения нет.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            filepath = url.toLocalFile()
            if filepath.lower().endswith(".csv"):
                if not self.data1:
                    self.file1_path = filepath
                    self.data1 = self.load_csv(filepath)
                    self.display_tables()
                elif not self.data2:
                    self.file2_path = filepath
                    self.data2 = self.load_csv(filepath)
                    self.display_tables()
                else:
                    reply = QMessageBox.question(
                        self, 
                        "Файл уже загружен",
                        "Оба файла уже загружены. Заменить Файл 1?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.file1_path = filepath
                        self.data1 = self.load_csv(filepath)
                        self.display_tables()
        event.acceptProposedAction()

def generate_requirements():
    with open("requirements.txt", "w", encoding="utf-8") as f:
        subprocess.run(["pip", "freeze"], stdout=f)

if __name__ == '__main__':
    # Для генерации файла requirements.txt можно раскомментировать следующую строку:
    # generate_requirements()
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
