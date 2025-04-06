#include "generate_svd.h"

#include <iostream>

int main() {
  std::default_random_engine RNG(42);
  std::uniform_real_distribution<double> dist(0, 2);

  SVDGenerator<double> SVD(4, 4, RNG, dist);
  SVD.generate(2);

  std::cout << SVD.MatrixU() * SVD.MatrixU().transpose();

  return 0;
}
