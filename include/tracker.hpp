#pragma "once"

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
    Tracker(const Config& cfg, const VoxelHashMap& vhm) : config(cfg), vhm(vhm) {
        reg_local_map = true;

        max_valid_dist = this->config.max_valid_dist;
        GM_kernel_scale = this->config.GM_kernel_scale;
    }

    // Static Methods
    static inline Eigen::Matrix3d hat(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return S;
    }
    static Eigen::MatrixX3d residuals_point_2_point_local(const Eigen::MatrixX3d& src, const Eigen::MatrixX3d& tgt, lo::SE3 T);
    static Eigen::MatrixX3d residuals_point_2_point_global(const Eigen::MatrixX3d& src, const Eigen::MatrixX3d& tgt);
    static std::vector<Eigen::Matrix<double, 3, 6>> jacobians_point2point_local(const Eigen::MatrixX3d& src, const Sophus::SE3d& T);
    static std::vector<Eigen::Matrix<double, 3, 6>> jacobians_point2point_global(const Eigen::MatrixX3d& src);

    static Eigen::VectorXd GM_weights(const Eigen::MatrixX3d& r, double c);

    // Const Methods
    // Methods
    void buildNormalEq(const std::vector<Eigen::Matrix<double, 3, 6>>& J, const Eigen::MatrixX3d& r, const Eigen::VectorXd* w = nullptr);
};
}  // namespace lo