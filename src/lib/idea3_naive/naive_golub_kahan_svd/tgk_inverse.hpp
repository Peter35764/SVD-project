#ifndef TGK_INVERSE_HPP
#define TGK_INVERSE_HPP

#include <cmath>
#include <vector>
#include <iostream>

namespace SVD_Project {
namespace detail {              // всё «служебное» прячем в аноним‐подпространство

/* ---------- решение (T-μI)x=b методом прогонки ---------- */
template <typename Scalar>
std::vector<Scalar> solveTriDiag(const std::vector<Scalar>& off,
                                 Scalar shift,
                                 const std::vector<Scalar>& rhs)
{
  const int N = static_cast<int>(rhs.size());
  std::vector<Scalar> a(N, -shift), l(N), y(N), x(N);

  for (int i = 1; i < N; ++i) {
    Scalar piv = std::abs(a[i-1]) < Scalar(1e-14)
               ? (a[i-1] >= 0 ? Scalar(1e-14) : Scalar(-1e-14))
               : a[i-1];
    l[i] = off[i-1] / piv;
    a[i] -= l[i] * off[i-1];
  }
  y[0] = rhs[0];
  for (int i = 1; i < N; ++i) y[i] = rhs[i] - l[i] * y[i-1];

  auto divSafe = [](Scalar d) {
    return (std::abs(d) < Scalar(1e-14)) ? (d >= 0 ? Scalar(1e-14)
                                                   : Scalar(-1e-14))
                                         : d;
  };

  x[N-1] = y[N-1] / divSafe(a[N-1]);
  for (int i = N-2; i >= 0; --i)
    x[i] = (y[i] - off[i] * x[i+1]) / divSafe(a[i]);

  return x;
}

/* ---------- обратная итерация при точном собственном значении μ ---------- */
template <typename Scalar>
std::vector<Scalar> inverseIter(const std::vector<Scalar>& off,
                                Scalar mu,
                                Scalar tol = Scalar(1e-2),
                                int    maxIt = 1000)
{
  const int N = static_cast<int>(off.size()) + 1;
  std::vector<Scalar> x(N, Scalar(0)), y;
  x[0] = Scalar(1);

  auto norm = [](const std::vector<Scalar>& v){
    Scalar s = 0; for (Scalar vv : v) s += vv*vv; return std::sqrt(s);
  };
  for (Scalar &v : x) v /= norm(x);

  for (int it = 0; it < maxIt; ++it) {
    y = solveTriDiag(off, mu, x);
    for (Scalar &v : y) v /= norm(y);

    Scalar diff = 0;
    for (int i = 0; i < N; ++i) diff += (y[i]-x[i])*(y[i]-x[i]);
    if (std::sqrt(diff) < tol) { x = y; break; }
    x.swap(y);
  }
  return x;
}

/* ---------- МГС для ортонормализации столбцов Eigen-матрицы ---------- */
template <typename Matrix>
void mgs(Matrix& Q)
{
  using Scalar = typename Matrix::Scalar;
  const int N = Q.rows(), M = Q.cols();
  for (int j = 0; j < M; ++j) {
    for (int k = 0; k < j; ++k)
      Q.col(j) -= Q.col(k).dot(Q.col(j))*Q.col(k);
    Scalar nrm = Q.col(j).norm();
    if (nrm > Scalar(1e-12)) Q.col(j) /= nrm;
  }
}

} // namespace detail


/* ----------------------------------------------------------------- */
/* -----------------------  реализация класса  ---------------------- */
/* ----------------------------------------------------------------- */
template <typename M>
TGKInv_SVD<M>::TGKInv_SVD(const M& bidiag,
                          const VectorDynamic& sigma,
                          unsigned /*opts*/)
    : m_matrixU(bidiag.rows(), bidiag.cols()),
      m_matrixV(bidiag.cols(), bidiag.cols()),
      m_sigma(sigma),
      m_B(bidiag)
{}

template <typename M>
TGKInv_SVD<M>& TGKInv_SVD<M>::compute()
{
  return compute(nullptr);
}

template <typename M>
TGKInv_SVD<M>& TGKInv_SVD<M>::compute(std::ostream* dbg)
{
  m_dbg = dbg;
  const Index n = m_B.rows();          // у нас квадратная bidiag B n×n
  const Index N = 2*n;

  /* 1) собираем внедиагонали TGK */
  std::vector<Scalar> off(N-1);
  for (Index i = 0; i < N-1; ++i)
    off[i] = (i%2==0) ? m_B(i/2, i/2)        // α
                      : m_B(i/2, i/2+1);     // β

  /* 2) собственные векторы для ±σ */
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q(N, N);
  Index col = 0;
  for (Index i = 0; i < n; ++i) {
    auto qp = detail::inverseIter(off,  m_sigma(i));
    auto qn = detail::inverseIter(off, -m_sigma(i));
    for (Index r=0;r<N;++r) { Q(r,col)   = qp[r]; }
    for (Index r=0;r<N;++r) { Q(r,col+1) = qn[r]; }
    col += 2;
  }
  /* 3) ортонормируем столбцы */
  detail::mgs(Q);

  /* 4) разрезаем Jordan–Wielandt ⇒ U,V */
  for (Index k = 0; k < n; ++k) {
    m_matrixV.col(k) = Q.col(2*k).segment(0, n*2).seq(0, 2).eval();
    m_matrixU.col(k) = Q.col(2*k).segment(1, n*2).seq(0, 2).eval();
    // seq(0,2) берёт элементы 0,2,4,... / 1,3,5,...
  }
  // финальная нормировка
  detail::mgs(m_matrixU);
  detail::mgs(m_matrixV);

  if (m_dbg) (*m_dbg) << "TGKInv_SVD finished for n=" << n << '\n';
  this->m_isInitialized = true;
  return *this;
}

} // namespace SVD_Project
#endif
