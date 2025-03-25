#ifndef MRRR_H
#define MRRR_H

#include <Eigen/Core>

template<typename T>
class MRRR_SVD{
	private:
		Eigen::MatrixXd U;
		Eigen::MatrixXd S;
		Eigen::MatrixXd V;
	void Set_U(const Eigen::MatrixXd& A);
	void Set_V(const Eigen::MatrixXd& A);
	void Set_S(const Eigen::MatrixXd& A);

	protected:
	MRRR_SVD<T> compute_bsvd(const Eigen::MatrixXd& matrix);

	public:
	MRRR_SVD();
	MRRR_SVD(const Eigen::MatrixXd& A);

	Eigen::MatrixXd matrixV();
	Eigen::MatrixXd matrixU();
	Eigen::MatrixXd singularValues();
};

#include <mrrr.hpp>

#endif // MRRR_H
