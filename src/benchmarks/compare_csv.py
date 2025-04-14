import sys
import os
import csv
import re
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QSplitter,
    QMessageBox, QHeaderView, QLineEdit, QLabel
)
from PyQt6.QtCore import Qt

'''
TODO:
- Преобразовать количество порядков в float в научной форме.
- Установить дефолтное значение 1e-31.
- Формат таблицы:
    • Если значение не изменилось – выводим просто значение.
    • Если изменилось – формат "старое -> новое".
    • Если столбец присутствует только в одном из файлов, добавляем пометку (1) или (2).
'''


class CustomTableWidgetItem(QTableWidgetItem):

    def __init__(self, text, header_name=''):
        super().__init__(text)
        self.header_name = header_name

    def __lt__(self, other):
        if not isinstance(other, CustomTableWidgetItem):
            return super().__lt__(other)
        header_l = self.header_name.lower()
        # Логика для столбца "Dimension" / "размерность"
        if header_l in ["dimension", "размерность"]:
            try:
                parts = self.text().lower().split('x')
                self_val = int(parts[0].strip()) * int(parts[1].strip()) if len(parts) == 2 else 0
            except Exception:
                self_val = 0
            try:
                parts = other.text().lower().split('x')
                other_val = int(parts[0].strip()) * int(parts[1].strip()) if len(parts) == 2 else 0
            except Exception:
                other_val = 0
            return self_val < other_val

        # Логика для столбцов, где заголовок содержит "interval"
        elif "interval" in header_l:
            # Ожидается формат вида "[число, число]", например "[0, 1]"
            pattern_brackets = r'^\[\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\]$'
            match_self = re.match(pattern_brackets, self.text())
            if match_self:
                try:
                    self_val = (float(match_self.group(1)) + float(match_self.group(2))) / 2.0
                except Exception:
                    self_val = 0.0
            else:
                try:
                    self_val = float(self.text().strip())
                except Exception:
                    self_val = 0.0

            match_other = re.match(pattern_brackets, other.text())
            if match_other:
                try:
                    other_val = (float(match_other.group(1)) + float(match_other.group(2))) / 2.0
                except Exception:
                    other_val = 0.0
            else:
                try:
                    other_val = float(other.text().strip())
                except Exception:
                    other_val = 0.0
            return self_val < other_val

        # Логика для остальных столбцов
        else:
            try:
                if "->" in self.text():
                    self_val = float(self.text().split("->")[1].strip())
                else:
                    self_val = float(self.text())
            except Exception:
                self_val = 0
            try:
                if "->" in other.text():
                    other_val = float(other.text().split("->")[1].strip())
                else:
                    other_val = float(other.text())
            except Exception:
                other_val = 0
            return self_val < other_val


class CsvTableWidget(QTableWidget):
    """
    Отображение CSV-файла в виде таблицы.
    Для отображения исходного файла используется его заголовок (первая строка).
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        if data and len(data) > 0:
            self.header = data[0]
            self.setColumnCount(len(self.header))
            self.setRowCount(len(data) - 1)
            self.setHorizontalHeaderLabels(self.header)
        else:
            self.header = []
            self.setColumnCount(0)
            self.setRowCount(0)
        self.populateTable()
        self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setSortingEnabled(True)

    def populateTable(self):
        for i, row in enumerate(self.data[1:]):
            for j, cell in enumerate(row):
                col_header = self.header[j] if j < len(self.header) else ''
                item = CustomTableWidgetItem(cell, col_header)
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
        self.comparison_data = {}  # Для хранения пар: (old_val, new_val) по ячейкам

        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self.button_layout = QHBoxLayout()
        self.btn_file1 = QPushButton("Файл 1")
        self.btn_file2 = QPushButton("Файл 2")
        self.btn_compare = QPushButton("Сравнить")
        self.button_layout.addWidget(self.btn_file1)
        self.button_layout.addWidget(self.btn_file2)
        self.button_layout.addWidget(self.btn_compare)

        self.label_threshold = QLabel("Порог:")
        self.button_layout.addWidget(self.label_threshold)
        self.le_threshold = QLineEdit()
        self.le_threshold.setText("1e-31")
        self.button_layout.addWidget(self.le_threshold)
        self.main_layout.addLayout(self.button_layout)

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
        data = []
        try:
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    data.append(row)
            if data and len(data) > 0:
                return data
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать файл {filepath}:\n{e}")
            return None

    def display_tables(self):
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
        # Перечитываем файлы на случай изменений
        if self.file1_path:
            self.data1 = self.load_csv(self.file1_path)
        if self.file2_path:
            self.data2 = self.load_csv(self.file2_path)
        if not (self.data1 and self.data2 and len(self.data1) > 0 and len(self.data2) > 0):
            QMessageBox.critical(self, "Ошибка", "Файлы пустые или некорректны!")
            return

        # Определяем объединенный набор столбцов (union header)
        old_header = self.data1[0]
        new_header = self.data2[0]
        union_header = list(old_header)
        for col in new_header:
            if col not in union_header:
                union_header.append(col)

        # Определяем максимальное число строк данных
        rows1 = len(self.data1) - 1
        rows2 = len(self.data2) - 1
        max_rows = max(rows1, rows2)

        self.comparison_table = QTableWidget(max_rows, len(union_header))
        self.comparison_table.setHorizontalScrollMode(self.comparison_table.ScrollMode.ScrollPerPixel)
        self.comparison_table.setVerticalScrollMode(self.comparison_table.ScrollMode.ScrollPerPixel)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.comparison_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.comparison_table.setHorizontalHeaderLabels(union_header)

        self.comparison_table.setSortingEnabled(False)
        self.comparison_data.clear()

        try:
            threshold = float(self.le_threshold.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Некорректное пороговое значение!")
            return

        # Для каждой строки данных (начиная с индекса 1) формируем итоговую строку
        for r in range(1, max_rows + 1):
            table_row = r - 1
            for j, col in enumerate(union_header):
                # Если столбец присутствует в старом файле, берем значение, иначе пусто
                if col in old_header:
                    idx_old = old_header.index(col)
                    old_val = self.data1[r][idx_old] if r <= rows1 and idx_old < len(self.data1[r]) else ""
                else:
                    old_val = ""
                # Если столбец присутствует в новом файле, берем значение
                if col in new_header:
                    idx_new = new_header.index(col)
                    new_val = self.data2[r][idx_new] if r <= rows2 and idx_new < len(self.data2[r]) else ""
                else:
                    new_val = ""
                # Формирование текста ячейки:
                # Если оба значения есть:
                if old_val and new_val:
                    if old_val == new_val:
                        display_text = old_val
                    else:
                        display_text = f"{old_val} -> {new_val}"
                # Если значение только из старого файла:
                elif old_val:
                    display_text = f"{old_val} (1)"
                # Если значение только из нового файла:
                elif new_val:
                    display_text = f"{new_val} (2)"
                else:
                    display_text = ""
                item = CustomTableWidgetItem(display_text, col)
                self.comparison_table.setItem(table_row, j, item)
                self.comparison_data[(table_row, j)] = (old_val, new_val)
                # Зададим фоновый цвет (при наличии числовых данных, можно добавить выделение по порогу)
                try:
                    if old_val and new_val:
                        num_old = float(old_val)
                        num_new = float(new_val)
                        if num_old == num_new:
                            item.setBackground(Qt.GlobalColor.white)
                        elif num_old == 0:
                            item.setBackground(Qt.GlobalColor.yellow)
                        else:
                            diff = num_new - num_old
                            if abs(diff) < threshold:
                                item.setBackground(Qt.GlobalColor.yellow)
                            elif diff >= threshold:
                                item.setBackground(Qt.GlobalColor.red)
                            elif diff <= -threshold:
                                item.setBackground(Qt.GlobalColor.green)
                            else:
                                item.setBackground(Qt.GlobalColor.white)
                    else:
                        item.setBackground(Qt.GlobalColor.white)
                except Exception:
                    item.setBackground(Qt.GlobalColor.white)

        self.comparison_table.setSortingEnabled(True)
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
