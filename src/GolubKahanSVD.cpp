#include<iostream>
#include<Eigen/SVD>
#include<Eigen/Jacobi>

// Сделано Черниковым Тимофеем КМБО-03-21
// github: chernikovtimofey

template<typename _MatrixType> class GolubKahanSVD;

// Такая структура присутствует для всех классов Eigen и определяет данные, известные на этапе компиляции
template<typename _MatrixType>
struct Eigen::internal::traits<GolubKahanSVD<_MatrixType>> : public Eigen::internal::traits<_MatrixType> {
public:
    typedef _MatrixType MatrixType;
};

// Класс разложения матрицы в SCD по алгоритму Голаба-Кохана для разложение действительных матриц.
// Левую и правую унитарные матрицы можно получить методами matrixU() и matrixV() соответсвенно
// Вектор с сингулярными значениями с помощью singularValues()
template<typename _MatrixType>
class GolubKahanSVD : public Eigen::SVDBase<GolubKahanSVD<_MatrixType>> {
private:
    typedef Eigen::SVDBase<GolubKahanSVD> Base;
public:
    using Base::rows;
    using Base::cols;
    using Base::computeU;
    using Base::computeV;

    // Тип матрицы на разложение и тип ее элементов
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;

    // Тип левой и правой унитарных матриц, тип вектора с сингулярными значениями
    typedef typename Base::MatrixUType MatrixUType;
    typedef typename Base::MatrixVType MatrixVType;
    typedef typename Base::SingularValuesType SingularValuesType;

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

    GolubKahanSVD() : m_isTranspose(false), m_compU(false), m_compV(false) {}

    GolubKahanSVD(Eigen::Index rows, Eigen::Index cols, unsigned int computationOptions = 0) {
        allocate(rows, cols, computationOptions);
    }

    GolubKahanSVD(const MatrixType& matrix, unsigned int computationOptions = 0) {
        compute(matrix, computationOptions);
    }

    // computationOptions - параметр, определяющий тип разожение (тонкое, полное)
    // Eigen::ComputeFull[U, V] - полное разложение
    // Eigen::ComputeThin[U, V] - тонкое разложение
    GolubKahanSVD& compute(const MatrixType& matrix, unsigned int computationOptions);

    GolubKahanSVD& compute(const MatrixType& matrix) {
        return compute(matrix, this->m_computationOptions);
    }

private:
    void allocate(Eigen::Index rows, Eigen::Index cols, unsigned int computationOptions);
    void algorithm();
    template<typename SubMatrixType> void step(SubMatrixType submatrix);
    template<typename SubMatrixType> Scalar cal_shift(SubMatrixType submatrix);
    template<typename HouseholderU, typename HouseholderV>
    void copyUV(const HouseholderU& householderU, const HouseholderV& householderV, MatrixX& naiveU, MatrixX& naiveV); 

protected:
    MatrixX m_naiveU, m_naiveV;
    MatrixX m_computed;
    bool m_isTranspose, m_compU, m_compV;

    using Base::m_singularValues;
    using Base::m_diagSize;
    using Base::m_computeFullU;
    using Base::m_computeFullV;
    using Base::m_computeThinU;
    using Base::m_computeThinV;
    using Base::m_matrixU;
    using Base::m_matrixV;
    using Base::m_info;
    using Base::m_isInitialized;
    using Base::m_nonzeroSingularValues;
};

// Функция подготавливает объект к непосредственному вычислению
template<typename MatrixType>
void GolubKahanSVD<MatrixType>::allocate(Eigen::Index rows, Eigen::Index cols, unsigned int computationOptions) {
    m_isTranspose = (cols > rows);

    if (Base::allocate(rows, cols, computationOptions)) return;

    if (m_isTranspose) std::swap(rows, cols);

    m_naiveU = MatrixX::Identity(m_diagSize, m_diagSize); 
    m_naiveV = MatrixX::Identity(m_diagSize, m_diagSize);
    m_computed = MatrixX::Zero(m_diagSize, m_diagSize);
    m_compU = computeU();
    m_compV = computeV();
    if (m_isTranspose) std::swap(m_compU, m_compV);
}

// функция вычисления разложения матрицы
template<typename MatrixType>
GolubKahanSVD<MatrixType>& GolubKahanSVD<MatrixType>::compute(const MatrixType& matrix, unsigned int computationOptions) {
    allocate(matrix.rows(), matrix.cols(), computationOptions);

    const Scalar considerZero = std::numeric_limits<Scalar>::min();

    // Верхняя бидиоганализация матрицы
    Eigen::internal::UpperBidiagonalization ubd((m_isTranspose ? matrix.adjoint() : matrix));
    m_computed = ubd.bidiagonal().toDenseMatrix();

    algorithm();

    // Выделение сингулярных значений матрицы  
    const Scalar epsilon = Eigen::NumTraits<Scalar>::epsilon();
    bool allSingularValuesNonZero = true;
    for (Eigen::Index i = 0; i < m_diagSize; ++i) {
        m_singularValues(i) = m_computed(i, i);
        if (std::abs(m_singularValues(i)) < epsilon) {
            m_nonzeroSingularValues = i;
            m_singularValues.tail(m_diagSize - i - 1).setZero();
            allSingularValuesNonZero = false;
            break;
        }
    }
    if (allSingularValuesNonZero) m_nonzeroSingularValues = m_diagSize;

    // Выделение левой и правой унитарных матриц
    if (m_isTranspose) copyUV(ubd.householderV(), ubd.householderU(), m_naiveV, m_naiveU);
    else copyUV(ubd.householderU(), ubd.householderV(), m_naiveU, m_naiveV);

    m_isInitialized = true;
    return *this;
}

// Алгоритм 8.6.2 из книги Matrix computation Golub, Van Loan
template<typename MatrixType>
void GolubKahanSVD<MatrixType>::algorithm() {
    const Scalar epsilon = Eigen::NumTraits<Scalar>::epsilon();

    // Выше элемента диагонали с индексом больше diag_border элемент матрицы равен нулю
    Eigen::Index diag_border = m_diagSize-1;
    // Выше элемента диагонали с индексом больше superdiag_border элемент матрицы не равен нулю
    Eigen::Index superdiag_border;
    while (diag_border > 0) {
        for (;
             diag_border > 0 &&
             std::abs(m_computed(diag_border-1, diag_border)) < epsilon;
             --diag_border) {
            m_computed(diag_border-1, diag_border) = 0;
        }

        for (superdiag_border = diag_border;
             superdiag_border > 0 &&
             std::abs(m_computed(superdiag_border-1, superdiag_border)) >= epsilon;
             --superdiag_border);

        bool has_zero_on_diag = false;
        for (Eigen::Index i = superdiag_border; i <= diag_border && i < m_diagSize-1; ++i) {
            if (std::abs(m_computed(i, i)) < epsilon) {
                m_computed(i, i) = 0;
                m_computed(i, i+1) = 0;
                has_zero_on_diag = true;
            }
        }

        if (!has_zero_on_diag && diag_border - superdiag_border + 1 > 1) {
            auto block = m_computed.block(superdiag_border, superdiag_border, diag_border - superdiag_border + 1, diag_border - superdiag_border + 1);
            step(block);
            // std::cout << m_naiveU.transpose() * m_computed * m_naiveV.transpose() << "\n\n";
        }
    }
    m_naiveU.transposeInPlace();
    m_info = Eigen::Success;
}

// Вычисляет собственные значения угловой 2x2 сабматрицы матрицы sumbatrix^T * submatrix, 
// возвращает значение находящееся ближе всего к элементу из правого нижнего угла этой матрицы
template<typename MatrixType> template<typename SubMatrixType>
typename GolubKahanSVD<MatrixType>::Scalar GolubKahanSVD<MatrixType>::cal_shift(SubMatrixType submatrix) {
    Eigen::Index n = submatrix.rows()-1;

    Scalar t_11, t_12, t_21, t_22;
    if (n == 0) return submatrix(0, 0);
    else if (n == 1) {
        t_11 = submatrix(0, 0) * submatrix(0, 0);
        t_12 = submatrix(0, 1) * submatrix(0, 1);
        t_21 = submatrix(1, 0) * submatrix(0, 1);
        t_22 = submatrix(0, 1) * submatrix(0, 1) + submatrix(1, 1) * submatrix(1, 1);
    }
    else {
        t_11 = submatrix(n-1, n-1) * submatrix(n-1, n-1) + submatrix(n-2, n-1) * submatrix(n-2, n-1);
        t_12 = submatrix(n-1, n-1) * submatrix(n-1, n);
        t_21 = submatrix(n-1, n-1) * submatrix(n-1, n);
        t_22 = submatrix(n, n) * submatrix(n, n) + submatrix(n-1, n) * submatrix(n-1, n);
    }

    auto trace = t_11 + t_22;
    auto det = t_11 * t_22 - t_12 * t_21;

    auto shift1 = (trace + std::sqrt(trace * trace - 4 * det)) / 2;
    auto shift2 = (trace - std::sqrt(trace * trace - 4 * det)) / 2;

    return (std::abs(shift1 - t_22) < std::abs(shift2 - t_22) ? shift1 : shift2);
}

// Шаг алгоритма Голоба-Кохана
template<typename MatrixType> template<typename SubMatrixType>
void GolubKahanSVD<MatrixType>::step(SubMatrixType submatrix) {
    Eigen::JacobiRotation<Scalar> G;
    auto n = submatrix.rows();

    auto shift = cal_shift(submatrix);
    auto y = submatrix(0, 0) * submatrix(0, 0) - shift; 
    auto z = submatrix(0, 0) * submatrix(0, 1);
    G.makeGivens(y, z);
    submatrix.applyOnTheRight(0, 1, G);
    m_naiveV.applyOnTheRight(0, 1, G);
    // std::cout << submatrix << "\n\n";
    for (Eigen::Index k = 0; k < n-2; ++k) {
        y = submatrix(k, k);
        z = submatrix(k+1, k);
        G.makeGivens(y, z);
        submatrix.applyOnTheLeft(k, k+1, G.transpose());
        m_naiveU.applyOnTheLeft(k, k+1, G.transpose());
        // std::cout << submatrix << "\n\n";
        y = submatrix(k, k+1);
        z = submatrix(k, k+2);
        G.makeGivens(y, z);
        submatrix.applyOnTheRight(k+1, k+2, G);
        m_naiveV.applyOnTheRight(k+1, k+2, G);
        // std::cout << "\n\n" << submatrix << "\n\n";
    }
    Eigen::Index k = n-2;
    y = submatrix(k, k);
    z = submatrix(k+1, k);
    G.makeGivens(y, z);
    submatrix.applyOnTheLeft(k, k+1, G.transpose());
    m_naiveU.applyOnTheLeft(k, k+1, G.transpose());
    // std::cout << "\n\n" << submatrix << "\n\n";
}

// Выделяет Левую и правую унитарные матрицы
template<typename MatrixType>
template<typename HouseholderU, typename HouseholderV>
void GolubKahanSVD<MatrixType>::copyUV(const HouseholderU& householderU, 
                                       const HouseholderV& householderV, 
                                       GolubKahanSVD<MatrixType>::MatrixX& naiveU, 
                                       GolubKahanSVD<MatrixType>::MatrixX& naiveV) {
    if (computeU()) {
        Eigen::Index Usize = m_computeThinU ? m_diagSize : householderU.cols();
        m_matrixU = MatrixX::Identity(householderU.cols( ), Usize);
        m_matrixU.topLeftCorner(m_diagSize, m_diagSize) = naiveU;
        m_matrixU.applyOnTheLeft(householderU);
    }
    if (computeV()) {
        Eigen::Index Vcols = m_computeThinV ? m_diagSize : householderV.cols();
        m_matrixV = MatrixX::Identity(householderV.cols(), Vcols);
        m_matrixV.topLeftCorner(m_diagSize, m_diagSize) = naiveV;
        m_matrixV.applyOnTheLeft(householderV);
    }
}

void squareTest() {
    std::cout << "Тест на квадратной матрице:" << "\n";

    Eigen::MatrixXd matrix(5, 5);
    matrix << 1, 2, 0, 7, 4,
              4, 3, 3, 4, 1,
              6, 7, 3, 6, 2,
              6, 8, 7, 6, 7,
              4, 4, 1, 0, 4;

    GolubKahanSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV); 

    Eigen::MatrixXd sValuesMatrix = Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
    auto sValues = svd.singularValues();
    for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols(); ++i) {
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
    matrix << 8, 7, 4, 9,
              2, 9, 8, 0,
              4, 5, 1, 5,
              3, 9, 2, 6,
              6, 6, 0, 8,
              4, 3, 0, 8,
              8, 7, 1, 3,
              3, 2, 4, 0;

    GolubKahanSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV); 

    Eigen::MatrixXd sValuesMatrix = Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
    auto sValues = svd.singularValues();
    for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols(); ++i) {
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
    matrix << 4, 2, 1, 1, 9, 0, 5, 6, 7,
              4, 4, 4, 3, 2, 0, 3, 5, 0,
              0, 9, 4, 8, 7, 8, 8, 4, 6,
              9, 1, 8, 2, 6, 3, 0, 0, 2;

    GolubKahanSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV); 

    Eigen::MatrixXd sValuesMatrix = Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols());
    auto sValues = svd.singularValues();
    for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols(); ++i) {
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
    matrix << 4, 2, 1, 1, 9, 0, 5, 6, 7,
              4, 4, 4, 3, 2, 0, 3, 5, 0,
              0, 9, 4, 8, 7, 8, 8, 4, 6,
              9, 1, 8, 2, 6, 3, 0, 0, 2;

    GolubKahanSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV); 

    Eigen::MatrixXd sValuesMatrix = Eigen::MatrixXd::Identity(matrix.rows(), matrix.rows());
    auto sValues = svd.singularValues();
    for (int i = 0; i < sValues.rows() && i < matrix.rows() && i < matrix.cols(); ++i) {
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
