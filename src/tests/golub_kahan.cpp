#include <iostream>
#include "lib/golub_kahan/golub_kahan.h"

void squareTest() {
  std::cout << "Тест на квадратной матрице:" << "\n";

  Eigen::MatrixXd matrix(5, 5);
  matrix << 1, 2, 0, 7, 4, 4, 3, 3, 4, 1, 6, 7, 3, 6, 2, 6, 8, 7, 6, 7, 4, 4, 1,
      0, 4;

  GolubKahanSVD<Eigen::MatrixXd> svd(matrix,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::MatrixXd sValuesMatrix =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
  auto sValues = svd.singularValues();
  for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols();
       ++i) {
    sValuesMatrix(i, i) = sValues(i);
  }
  std::cout << svd.matrixU() << "\n\n";
  std::cout << sValuesMatrix << "\n\n";
  std::cout << svd.matrixV().transpose() << "\n\n";
  std::cout << svd.matrixU() * sValuesMatrix * svd.matrixV().transpose();

  std::cout << "\n--------------------------\n";
}

void rowsColsTest() {
  std::cout << "Тест на матрице где m > n:" << "\n";

  Eigen::MatrixXd matrix(8, 4);
  matrix << 8, 7, 4, 9, 2, 9, 8, 0, 4, 5, 1, 5, 3, 9, 2, 6, 6, 6, 0, 8, 4, 3, 0,
      8, 8, 7, 1, 3, 3, 2, 4, 0;

  GolubKahanSVD<Eigen::MatrixXd> svd(matrix,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::MatrixXd sValuesMatrix =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
  auto sValues = svd.singularValues();
  for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols();
       ++i) {
    sValuesMatrix(i, i) = sValues(i);
  }
  std::cout << svd.matrixU() << "\n\n";
  std::cout << sValuesMatrix << "\n\n";
  std::cout << svd.matrixV().transpose() << "\n\n";
  std::cout << svd.matrixU() * sValuesMatrix * svd.matrixV().transpose();

  std::cout << "\n--------------------------\n";
}

void colsRowsTest() {
  std::cout << "Тест на матрице где m < n:" << "\n";

  Eigen::MatrixXd matrix(4, 9);
  matrix << 4, 2, 1, 1, 9, 0, 5, 6, 7, 4, 4, 4, 3, 2, 0, 3, 5, 0, 0, 9, 4, 8, 7,
      8, 8, 4, 6, 9, 1, 8, 2, 6, 3, 0, 0, 2;

  GolubKahanSVD<Eigen::MatrixXd> svd(matrix,
                                     Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::MatrixXd sValuesMatrix =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
  auto sValues = svd.singularValues();
  for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols();
       ++i) {
    sValuesMatrix(i, i) = sValues(i);
  }
  std::cout << svd.matrixU() << "\n\n";
  std::cout << sValuesMatrix << "\n\n";
  std::cout << svd.matrixV().transpose() << "\n\n";
  std::cout << svd.matrixU() * sValuesMatrix * svd.matrixV().transpose();

  std::cout << "\n--------------------------\n";
}

void thinTest() {
  std::cout << "Тест на тонкое SVD разложение^" << "\n";

  Eigen::MatrixXd matrix(4, 9);
  matrix << 4, 2, 1, 1, 9, 0, 5, 6, 7, 4, 4, 4, 3, 2, 0, 3, 5, 0, 0, 9, 4, 8, 7,
      8, 8, 4, 6, 9, 1, 8, 2, 6, 3, 0, 0, 2;

  GolubKahanSVD<Eigen::MatrixXd> svd(matrix,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::MatrixXd sValuesMatrix =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.rows());
  auto sValues = svd.singularValues();
  for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols();
       ++i) {
    sValuesMatrix(i, i) = sValues(i);
  }
  std::cout << svd.matrixU() << "\n\n";
  std::cout << sValuesMatrix << "\n\n";
  std::cout << svd.matrixV().transpose() << "\n\n";
  std::cout << svd.matrixU() * sValuesMatrix * svd.matrixV().transpose();

  std::cout << "\n--------------------------\n";
}

int main() {
  squareTest();
  rowsColsTest();
  colsRowsTest();
  thinTest();
}
