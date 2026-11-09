#pragma once

/*
 * Ишутин Андрей (Ishutin Andrei)
 * telegram: @looks_amazing
 * Github: Andy-823
 */

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
// не у всех эйген одинаково подключается
// но этот костыль не везде заработает
// выяснилось что при использовании cmake регистр имеет значение
// https://stackoverflow.com/questions/142877/can-the-c-preprocessor-be-used-to-tell-if-a-file-exists
#if __has_include(<Eigen/Core>)
#include <Eigen/Core>
#include <Eigen/SVD>
#elif __has_include(<eigen/core>)
#include <eigen/core>
#include <eigen/svd>
#elif __has_include(<eigen3/Eigen/Core>)
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#endif

typedef struct Params {
  float epsilon = 1e-9;
  float tau = 0.1;
  float sweeps_factor = 1;
} params_t;

template <typename _MatrixType>
class JTS_SVD;

/** По большей части написанное здесь есть копия того, что находится в
 * eigen/svd/JacobiSVD.h Основные изменения касаются compute В других местах
 * изменения были внесены только чтобы написанное заработало
 *
 * В compute применен другой метод, который, однако, так же основан на поворотах
 * Якоби Статья о нем: S. Pal, S. Pathak and S. Rajasekaran, "On Speeding-Up
 * Parallel Jacobi Iterations for SVDs," in 2016 IEEE 18th International
 * Conference on High-Performance Computing and Communications, IEEE 14th
 * International Conference on Smart City, and IEEE 2nd International Conference
 * on Data Science and Systems (HPCC/SmartCity/DSS), Sydney, NSW, 2016 pp. 9-16.
 * doi: 10.1109/HPCC-SmartCity-DSS.2016.0013
 *
 * При этом есть сомнения, что описанное заработает на комплексных значениях
 *
 */

template <typename _MatrixType>
struct Eigen::internal::traits<JTS_SVD<_MatrixType>>
    : Eigen::internal::traits<_MatrixType> {
  typedef _MatrixType MatrixType;
};

template <typename _MatrixType>
class JTS_SVD : public Eigen::SVDBase<JTS_SVD<_MatrixType>> {
  typedef Eigen::SVDBase<JTS_SVD> Base;

 public:
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef
      typename Eigen::NumTraits<typename MatrixType::Scalar>::Real RealScalar;

#define Dynamic Eigen::Dynamic
  enum {

    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    DiagSizeAtCompileTime =
        EIGEN_SIZE_MIN_PREFER_DYNAMIC(RowsAtCompileTime, ColsAtCompileTime),
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    MaxDiagSizeAtCompileTime =
        EIGEN_SIZE_MIN_PREFER_FIXED(MaxRowsAtCompileTime, MaxColsAtCompileTime),
    MatrixOptions = MatrixType::Options
  };
#undef Dynamic

  typedef typename Base::MatrixUType MatrixUType;
  typedef typename Base::MatrixVType MatrixVType;
  typedef typename Base::SingularValuesType SingularValuesType;

  typedef typename Eigen::internal::plain_row_type<MatrixType>::type RowType;
  typedef typename Eigen::internal::plain_col_type<MatrixType>::type ColType;

  /** \brief Default Constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via JTS_SVD::compute(const MatrixType&).
   */
  JTS_SVD() {}

  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem size.
   * \sa JTS_SVD()
   */
  JTS_SVD(Eigen::Index rows, Eigen::Index cols,
          unsigned int computationOptions = 0) {
    allocate(rows, cols, computationOptions);
  }

  /** \brief Constructor performing the decomposition of given matrix.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions optional parameter allowing to specify if you
   * want full or thin U or V unitaries to be computed. By default, none is
   * computed. This is a bit-field, the possible bits are #ComputeFullU,
   * #ComputeThinU, #ComputeFullV, #ComputeThinV.
   * \param params describes how algorithm will work
   *               algo calculates modules of scalar product of rows/cols
   *               then takes params.tau part of greatest of them
   *               if greatest of them is less than threshold SVD is supposed to
   * be got threshold = params.epsilon * matrix.squaredNorm() look at this paper
   * to have better understanding of how it works S. Pal, S. Pathak and S.
   * Rajasekaran, "On Speeding-Up Parallel Jacobi Iterations for SVDs," one this
   * iteration is sweep if amount of sweeps made reaches limit algo stops on
   * current result params.sweeps_factor is used to calculate this limit
   *               sweps_factor equal to one means limit m_diagSize *
   * (m_diagSize - 1) / 2
   *
   * Thin unitaries are only available if your matrix type has a Dynamic number
   * of columns (for example MatrixXf).
   */
  explicit JTS_SVD(const MatrixType& matrix,
                   unsigned int computationOptions = 0,
                   const params_t& params = params_t()) {
    compute(matrix, computationOptions, params);
  }

  /** \brief Method performing the decomposition of given matrix using custom
   * options.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions optional parameter allowing to specify if you
   * want full or thin U or V unitaries to be computed. By default, none is
   * computed. This is a bit-field, the possible bits are #ComputeFullU,
   * #ComputeThinU, #ComputeFullV, #ComputeThinV.
   * \param params describes how algorithm will work
   *               algo calculates modules of scalar product of rows/cols
   *               then takes params.tau part of greatest of them
   *               if greatest of them is less than threshold SVD is supposed to
   * be got threshold = params.epsilon * matrix.squaredNorm() look at this paper
   * to have better understanding of how it works S. Pal, S. Pathak and S.
   * Rajasekaran, "On Speeding-Up Parallel Jacobi Iterations for SVDs," one this
   * iteration is sweep if amount of sweeps made reaches limit algo stops on
   * current result params.sweeps_factor is used to calculate this limit
   *               sweps_factor equal to one means limit m_diagSize *
   * (m_diagSize - 1) / 2
   *
   * Thin unitaries are only available if your matrix type has a Dynamic number
   * of columns (for example MatrixXf).
   */
  JTS_SVD& compute(const MatrixType& matrix, unsigned int computationOptions,
                   const params_t& params);

  /** \brief Method performing the decomposition of given matrix using current
   * options.
   *
   * \param matrix the matrix to decompose
   * \param params describes how algorithm will work
   *               algo calculates modules of scalar product of rows/cols
   *               then takes params.tau part of greatest of them
   *               if greatest of them is less than threshold SVD is supposed to
   * be got threshold = params.epsilon * matrix.squaredNorm() look at this paper
   * to have better understanding of how it works S. Pal, S. Pathak and S.
   * Rajasekaran, "On Speeding-Up Parallel Jacobi Iterations for SVDs," one this
   * iteration is sweep if amount of sweeps made reaches limit algo stops on
   * current result params.sweeps_factor is used to calculate this limit
   *               sweps_factor equal to one means limit m_diagSize *
   * (m_diagSize - 1) / 2 This method uses the current \a computationOptions, as
   * already passed to the constructor or to compute(const MatrixType&, const
   * params_t &, unsigned int).
   */
  JTS_SVD& compute(const MatrixType& matrix, const params_t& params) {
    return compute(matrix, m_computationOptions, params);
  }

  using Base::cols;
  using Base::computeU;
  using Base::computeV;
  using Base::rank;
  using Base::rows;

 private:
  void allocate(Eigen::Index rows, Eigen::Index cols,
                unsigned int computationOptions);
  JTS_SVD& compute_basic(const MatrixType& matrix, const params_t& params);
  JTS_SVD& compute_transposed(const MatrixType& matrix, const params_t& params);

 protected:
  using Base::m_cols;
  using Base::m_computationOptions;
  using Base::m_computeFullU;
  using Base::m_computeFullV;
  using Base::m_computeThinU;
  using Base::m_computeThinV;
  using Base::m_diagSize;
  using Base::m_info;
  using Base::m_isAllocated;
  using Base::m_isInitialized;
  using Base::m_matrixU;
  using Base::m_matrixV;
  using Base::m_nonzeroSingularValues;
  using Base::m_prescribedThreshold;
  using Base::m_rows;
  using Base::m_singularValues;
  using Base::m_usePrescribedThreshold;

  MatrixType m_workMatrix;
};

#include "jacobi.cpp"
