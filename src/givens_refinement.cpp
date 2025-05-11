#include <Eigen/Dense>
#include <Eigen/Householder>
#include <iostream>

int main() {
  using namespace Eigen;
  using namespace std;

  MatrixXd A(3, 3);
  A << 1, 3, 3, 10, 5, 6, 7, 8, 9;
  std::cout << "Original matrix A:\n" << A << std::endl;

  // init for Householder bidiag
  int m = A.rows();
  int n = A.cols();
  int minDim = std::min(m, n);

  auto bidiag = internal::UpperBidiagonalization<MatrixXd>(A);
  MatrixXd B = bidiag.bidiagonal();
  MatrixXd U = MatrixXd::Identity(m, m);
  MatrixXd V = MatrixXd::Identity(n, n);
  U.applyOnTheLeft(bidiag.householderU());
  V.applyOnTheLeft(bidiag.householderV());
  MatrixXd B_clean = MatrixXd::Zero(m, n);
  for (int i = 0; i < minDim; ++i) {
    B_clean(i, i) = B(i, i);
    if (i < minDim - 1) B_clean(i, i + 1) = B(i, i + 1);
  }

  std::cout << "\nBidiagonal matrix B:\n" << B_clean << std::endl;

  MatrixXd reconstructed_A = U * B_clean * V.transpose();
  std::cout << "\nReconstruction control sample (components of bidiag must yield orig, duh):\n"
            << reconstructed_A << std::endl;

  std::cout << "\nReconstruction error: " << (A - reconstructed_A).norm()
            << std::endl;

  MatrixXd work_B = B_clean;
  MatrixXd left_givens = MatrixXd::Identity(m, m);
  MatrixXd right_givens = MatrixXd::Identity(n, n);
  for (int iter = 0; iter < 5; ++iter) {
    for (int i = 0; i < minDim - 1; ++i) {
      // Right Givens rotation
      JacobiRotation<double> rotRight;
      rotRight.makeGivens(work_B(i, i), work_B(i, i + 1));
      work_B.applyOnTheRight(i, i + 1, rotRight);
      right_givens.applyOnTheRight(i, i + 1, rotRight);

      // Left Givens rotation
      if (i < minDim - 1) {
        JacobiRotation<double> rotLeft;
        rotLeft.makeGivens(work_B(i, i), work_B(i + 1, i));
        work_B.applyOnTheLeft(i, i + 1, rotLeft.transpose());
        left_givens.applyOnTheLeft(i, i + 1, rotLeft.transpose());
      }
    }
  }

  VectorXd S = work_B.diagonal();
  MatrixXd U_givens = left_givens.transpose();
  MatrixXd V_givens = right_givens;
  MatrixXd U_full = U * U_givens;
  MatrixXd V_full = V * V_givens;
  MatrixXd final_reconstructed_A = U_full * S.asDiagonal() * V_full.transpose();

  std::cout << "\nReconstruction by Givens rotations:\n"
            << final_reconstructed_A << std::endl;
  std::cout << "\nGivens reconstruction error: "
            << (A - final_reconstructed_A).norm() << std::endl;

  std::cout << "\nSingular values: ";
  for (int i = 0; i < minDim; ++i) std::cout << std::abs(S(i)) << " ";
  std::cout << std::endl;

  return 0;
}
