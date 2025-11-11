#include "tracker.hpp"

#include "utils.hpp"

Eigen::Matrix3Xd lo::Tracker::residuals_point_2_point_local(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt, lo::SE3 T) {
    const Eigen::Matrix3d R = T.so3().matrix();
    const Eigen::Vector3d t = T.translation();

    Eigen::Matrix3Xd r = (R * src).colwise() + t;  // 3xN
    r -= tgt;                                      // 3xN
    return r;
}

Eigen::Matrix3Xd lo::Tracker::residuals_point_2_point_global(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt) {
    Eigen::Matrix3Xd r = (src - tgt);
    return r;
}

std::vector<Eigen::Matrix<double, 3, 6>> lo::Tracker::jacobians_point2point_local(const Eigen::Matrix3Xd& src, const Sophus::SE3d& T) {
    const Eigen::Matrix3d R = T.so3().matrix();
    const Eigen::Index N = src.cols();
    std::vector<Eigen::Matrix<double, 3, 6>> J;
    J.reserve(static_cast<size_t>(N));
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    for (Eigen::Index i = 0; i < N; i++) {
        const Eigen::Vector3d xi = src.col(i);  // x_i (3x1)
        const Eigen::Vector3d R_xi = R * xi;    // R * x_i
        Eigen::Matrix<double, 3, 6> Ji;
        Ji.block<3, 3>(0, 0) = I;
        Ji.block<3, 3>(0, 3) = -hat(R_xi);
        // Ji.block<3,3>(0,3) = -Sophus::SO3d::hat(R_xi);
        J.push_back(Ji);
    }

    return J;
}

std::vector<Eigen::Matrix<double, 3, 6>> lo::Tracker::jacobians_point2point_global(const Eigen::Matrix3Xd& src) {
    const Eigen::Index N = src.cols();
    std::vector<Eigen::Matrix<double, 3, 6>> J;
    J.reserve(static_cast<size_t>(N));
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Vector3d pwi = src.col(i);  // 3x1
        Eigen::Matrix<double, 3, 6> Ji;
        Ji.block<3, 3>(0, 0) = I;
        Ji.block<3, 3>(0, 3) = -hat(pwi);
        // Ji.block<3, 3>(0, 3) = -Sophus::SO3d::hat(pwi);  // [p_w_i]_x
        J.push_back(Ji);
    }

    return J;
}

// w_i = c^2 / (||r_i||^2 + c^2)^2
Eigen::VectorXd lo::Tracker::GM_weights(const Eigen::Matrix3Xd& r, double c) {
    const Eigen::RowVectorXd s_row = r.colwise().squaredNorm();  // 1xN
    Eigen::VectorXd s = s_row.transpose();                       // N×1

    const double c2 = c * c;
    Eigen::VectorXd w = c2 * (s.array() + c2).inverse().square().matrix();

    // clamp
    const double eps = std::numeric_limits<double>::min();
    for (Eigen::Index i = 0; i < w.size(); ++i)
        if (w(i) < eps) w(i) = eps;

    return w;
}

void lo::Tracker::buildNormalEq(const std::vector<Eigen::Matrix<double, 3, 6>>& J, const Eigen::Matrix3Xd& r, const Eigen::VectorXd* w) {
    this->H.setZero();
    this->b.setZero();

    const Eigen::Index N = r.cols();
    for (Eigen::Index i = 0; i < N; i++) {
        const Eigen::Vector3d ri = r.col(i);         // 3x1
        const auto& Ji = J[static_cast<size_t>(i)];  // 3x6

        const double wi = (w ? (*w)(i) : 1.0);

        H += (Ji.transpose() * wi * Ji);
        b += -(Ji.transpose() * wi * ri);
    }
}

Eigen::Matrix<double, 6, 1> lo::Tracker::gn_point_to_point_step_local(
    const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt, const lo::SE3& T, bool robust, double damping) {
    // 1) Residuals & Jacobians
    Eigen::Matrix3Xd r = residuals_point_2_point_local(src, tgt, T);
    auto J = jacobians_point2point_local(src, T);

    // 2) Robust weights
    Eigen::VectorXd w_vec;
    const Eigen::VectorXd* w_ptr = nullptr;
    if (robust) {
        w_vec = GM_weights(r, GM_kernel_scale);
        w_ptr = &w_vec;
    }

    // 3) Normal equations
    buildNormalEq(J, r, w_ptr);

    // 4) Damping
    if (damping > 0.0) H.diagonal().array() += damping;

    // 5) Solve H dx = b
    auto delta_x = H.ldlt().solve(b);

    return delta_x;
}

Eigen::Matrix<double, 6, 1> lo::Tracker::gn_point_to_point_step_global(const Eigen::Matrix3Xd& src,
                                                                       const Eigen::Matrix3Xd& tgt,
                                                                       bool robust,
                                                                       double damping) {
    // 1) Residuals & Jacobians
    Eigen::Matrix3Xd r = residuals_point_2_point_global(src, tgt);
    auto J = jacobians_point2point_global(src);

    // 2) Robust weights
    Eigen::VectorXd w_vec;
    const Eigen::VectorXd* w_ptr = nullptr;
    if (robust) {
        w_vec = GM_weights(r, GM_kernel_scale);
        w_ptr = &w_vec;
    }

    // 3) Normal equations
    buildNormalEq(J, r, w_ptr);

    // 4) Damping
    if (damping > 0.0) H.diagonal().array() += damping;

    // 5) Solve H dx = b
    auto delta_x = H.ldlt().solve(b);

    return delta_x;
}

static inline void unpack_correspondences(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& corr,
                                          Eigen::Matrix3Xd& src,
                                          Eigen::Matrix3Xd& tgt) {
    const Eigen::Index N = static_cast<Eigen::Index>(corr.size());
    src.resize(3, N);
    tgt.resize(3, N);
    for (Eigen::Index i = 0; i < N; ++i) {
        src.col(i) = corr[static_cast<size_t>(i)].first;   // query point
        tgt.col(i) = corr[static_cast<size_t>(i)].second;  // map point
    }
}

Sophus::SE3d lo::Tracker::Tracking(size_t frame_id, const Eigen::Matrix3Xd& source_points, const Sophus::SE3d& init_pose, bool query_locally) {
    int iter_n = this->config.reg_iter_n;
    float reg_convergence_criterion = this->config.reg_convergence_criterion;

    if (frame_id == 1) {
        this->max_valid_dist = 4.0 * this->config.max_valid_dist;
        this->GM_kernel_scale = 4.0 * this->config.GM_kernel_scale;
    } else {
        this->max_valid_dist = this->config.max_valid_dist;
        this->GM_kernel_scale = this->config.GM_kernel_scale;
    }

    auto cur_points = lo::transformPointCloud(source_points, init_pose);
    auto T_icp_se3 = Sophus::SE3d();  // Local update from ICP

    for (size_t i = 0; i < iter_n; i++) {
        auto correspondences = this->vhm.nearestNeighborSearch(cur_points, this->max_valid_dist, true, query_locally);
        // std::cout << "Number of correspondences is: " << correspondences.size() << std::endl;
        Eigen::Matrix3Xd src, tgt;
        unpack_correspondences(correspondences, src, tgt);

        // One GN step
        Sophus::SE3d::Tangent dx;
        if (query_locally) {
            dx = gn_point_to_point_step_local(src, tgt, T_icp_se3, config.use_robust_kernel);
        } else {
            dx = gn_point_to_point_step_global(src, tgt, config.use_robust_kernel);
        }

        // std::cout << T_icp_se3.matrix() << std::endl << std::endl;
        Sophus::SE3d dT = Sophus::SE3d::exp(dx);                      // increment transform
        cur_points = std::move(transformPointCloud(cur_points, dT));  // apply increment transformation

        T_icp_se3 = dT * T_icp_se3;  // increment transformation update

        if (dx.norm() < reg_convergence_criterion) break;
    }

    // Final pose in world frame: T_final = T_icp * init_pose (same as your T_icp_torch @ init_pose)
    Sophus::SE3d T_final = T_icp_se3 * init_pose;
    return T_final;
}