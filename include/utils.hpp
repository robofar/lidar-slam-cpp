#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <sophus/se3.hpp>
#include <vector>

namespace lo {
inline Eigen::Matrix3Xd transformPointCloud(const Eigen::Matrix3Xd& points, const Sophus::SE3d& T_w_l) {
    const Eigen::Matrix3d R = T_w_l.so3().matrix();
    const Eigen::Vector3d t = T_w_l.translation();
    return (R * points).colwise() + t;  // vectorized: R*X + t
}

using VoxelKey = std::int64_t;

inline std::int64_t packVoxelIndex(int ix, int iy, int iz) {
    std::uint64_t ux = static_cast<std::uint32_t>(ix);
    std::uint64_t uy = static_cast<std::uint32_t>(iy);
    std::uint64_t uz = static_cast<std::uint32_t>(iz);

    ux &= ((1ull << 21) - 1);
    uy &= ((1ull << 21) - 1);
    uz &= ((1ull << 21) - 1);

    return (ux << 42) | (uy << 21) | uz;
}

inline lo::VoxelKey PointToVoxel(const Eigen::Vector3d& point, double voxel_size) {
    int ix = static_cast<int>(std::floor(point.x() / voxel_size));
    int iy = static_cast<int>(std::floor(point.y() / voxel_size));
    int iz = static_cast<int>(std::floor(point.z() / voxel_size));
    return packVoxelIndex(ix, iy, iz);
}

/*
inline Eigen::MatrixXd transformPointCloud(const Eigen::MatrixXd& points, const Sophus::SE3d& T_w_l) {
    const Eigen::Index N = points.rows();

    Eigen::MatrixXd homo(N, 4);
    homo.leftCols<3>() = points.leftCols<3>();
    homo.col(3).setOnes();

    const Eigen::Matrix4d T = T_w_l.matrix();
    Eigen::MatrixXd out_h = homo * T.transpose();  // (N,4) homogenous ; or (T * homo.transpose()).transpose()

    return out_h.leftCols<3>();  // (N,3)
}
*/
}  // namespace lo
