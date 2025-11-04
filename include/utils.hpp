#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <vector>

namespace lo {
inline Eigen::MatrixXd transformPointCloud(const Eigen::MatrixXd& points, const Sophus::SE3d& T_w_l) {
    const Eigen::Index N = points.rows();

    Eigen::MatrixXd homo(N, 4);
    homo.leftCols<3>() = points.leftCols<3>();
    homo.col(3).setOnes();

    const Eigen::Matrix4d T = T_w_l.matrix();
    Eigen::MatrixXd out_h = homo * T.transpose();  // (N,4) homogenous ; or (T * homo.transpose()).transpose()

    return out_h.leftCols<3>();  // (N,3)
}
}  // namespace lo
