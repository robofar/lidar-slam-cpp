#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "config.hpp"
#include "vhm.hpp"

namespace lo {
using SE3 = Sophus::SE3d;

class Tracker {
  public:
    Config config;
    const VoxelHashMap& vhm;

    bool reg_local_map;

    float max_valid_dist;
    float GM_kernel_scale;

    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> b;

    // ctors
    Tracker(const Config& cfg, const VoxelHashMap& vhm) : config(cfg), vhm(vhm) { reg_local_map = true; }

    // Static Methods
    static inline Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return S;
    }
    static Eigen::Matrix3Xd residuals_point_2_point_local(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt, lo::SE3 T);
    static Eigen::Matrix3Xd residuals_point_2_point_global(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt);
    static std::vector<Eigen::Matrix<double, 3, 6>> jacobians_point2point_local(const Eigen::Matrix3Xd& src, const Sophus::SE3d& T);
    static std::vector<Eigen::Matrix<double, 3, 6>> jacobians_point2point_global(const Eigen::Matrix3Xd& src);

    static Eigen::VectorXd GM_weights(const Eigen::Matrix3Xd& r, double c);

    // Const Methods
    // Methods
    void buildNormalEq(const std::vector<Eigen::Matrix<double, 3, 6>>& J, const Eigen::Matrix3Xd& r, const Eigen::VectorXd* w = nullptr);

    // One GN step (local)
    Eigen::Matrix<double, 6, 1> gn_point_to_point_step_local(
        const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& tgt, const SE3& T, bool robust, double damping = 0.0);

    // One GN step (global)
    Eigen::Matrix<double, 6, 1> gn_point_to_point_step_global(const Eigen::Matrix3Xd& src,
                                                              const Eigen::Matrix3Xd& tgt,
                                                              bool robust,
                                                              double damping = 0.0);

    Sophus::SE3d Tracking(size_t frame_id, const Eigen::Matrix3Xd& source_points, const Sophus::SE3d& init_pose, bool query_locally = false);
};
}  // namespace lo