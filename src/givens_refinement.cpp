#include "lib/idea1_givref/givens_refinement.h"

#include <Eigen/Dense>
#include <iostream>

int main() {
  using namespace Eigen;
  using namespace SVD_Project;

  MatrixXd B(3, 3);
  B << 1, 2, 0, 0, 3, 4, 0, 0, 5;

  GivRef_SVD<MatrixXd> svd(B, ComputeFullU | ComputeFullV);

  std::cout << "Singular values:\n" << svd.singularValues() << std::endl;
  std::cout << "U:\n" << svd.matrixU() << std::endl;
  std::cout << "V:\n" << svd.matrixV() << std::endl;

  // Fix the syntax error in the reconstructed line
  MatrixXd reconstructed = svd.matrixU() * svd.singularValues().asDiagonal() *
                           svd.matrixV().transpose();

  std::cout << "Reconstructed matrix:\n" << reconstructed << std::endl;
  std::cout << "Original matrix:\n" << B << std::endl;

  return 0;
}
