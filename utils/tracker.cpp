#include "tracker.hpp"

Eigen::MatrixX3d lo::Tracker::residuals_point_2_point_local(const Eigen::MatrixX3d& src, const Eigen::MatrixX3d& tgt, lo::SE3 T) {
    const Eigen::Matrix3d R = T.so3().matrix();
    const Eigen::RowVector3d t = T.translation().transpose();

    Eigen::MatrixX3d r = src * R.transpose();
    r.rowwise() += t;
    r -= tgt;

    return r;
}

Eigen::MatrixX3d lo::Tracker::residuals_point_2_point_global(const Eigen::MatrixX3d& src, const Eigen::MatrixX3d& tgt) {
    Eigen::MatrixX3d r = (src - tgt);
    return (src - tgt);
}

std::vector<Eigen::Matrix<double, 3, 6>> lo::Tracker::jacobians_point2point_local(const Eigen::MatrixX3d& src, const Sophus::SE3d& T) {
    const Eigen::Matrix3d R = T.so3().matrix();
    const Eigen::RowVector3d t = T.translation().transpose();

    const Eigen::Index N = src.rows();
    std::vector<Eigen::Matrix<double, 3, 6>> J;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Vector3d xi = src.row(i).transpose();  // x_i (3x1)
        const Eigen::Vector3d R_xi = R * xi;                // R * x_i
        Eigen::Matrix<double, 3, 6> Ji;
        Ji.block<3, 3>(0, 0) = I;
        Ji.block<3, 3>(0, 3) = -hat(R_xi);
        // Ji.block<3,3>(0,3) = -Sophus::SO3d::hat(R_xi);
        J.push_back(Ji);
    }

    return J;
}

std::vector<Eigen::Matrix<double, 3, 6>> lo::Tracker::jacobians_point2point_global(const Eigen::MatrixX3d& src) {
    const Eigen::Index N = src.rows();
    std::vector<Eigen::Matrix<double, 3, 6>> J;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Vector3d pwi = src.row(i).transpose();  // (3x1)
        Eigen::Matrix<double, 3, 6> Ji;
        Ji.block<3, 3>(0, 0) = I;
        Ji.block<3, 3>(0, 3) = -hat(pwi);
        // Ji.block<3, 3>(0, 3) = -Sophus::SO3d::hat(pwi);  // [p_w_i]_x
        J.push_back(Ji);
    }

    return J;
}

Eigen::VectorXd lo::Tracker::GM_weights(const Eigen::MatrixX3d& r, double c) {
    // w_i = c^2 / (||r_i||^2 + c^2)^2
    const Eigen::Index N = r.rows();
    Eigen::VectorXd s(N);
    for (Eigen::Index i = 0; i < N; i++) s(i) = r.row(i).squaredNorm();
    const double c2 = c * c;
    Eigen::VectorXd w = c2 * (s.array() + c2).inverse().square().matrix();

    // clamp
    const double eps = std::numeric_limits<double>::min();
    for (Eigen::Index i = 0; i < N; ++i)
        if (w(i) < eps) w(i) = eps;

    return w;
}

void lo::Tracker::buildNormalEq(const std::vector<Eigen::Matrix<double, 3, 6>>& J, const Eigen::MatrixX3d& r, const Eigen::VectorXd* w) {
    this->H.setZero();
    this->b.setZero();

    const Eigen::Index N = r.rows();
    for (Eigen::Index i = 0; i < N; i++) {
        const Eigen::Vector3d ri = r.row(i).transpose();  // 3x1
        const auto& Ji = J[static_cast<size_t>(i)];       // 3x6

        const double wi = (w ? (*w)(i) : 1.0);

        H.noalias() += (Ji.transpose() * wi * Ji);
        b.noalias() += -(Ji.transpose() * wi * ri);
    }
}