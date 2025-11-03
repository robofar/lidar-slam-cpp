#include "mapper.hpp"

void lo::Mapper::determineMappingPoses(size_t frame_id) {
    if (this->config.pgo_on) {
        this->used_poses.assign(this->dataset.pgo_poses.begin(), this->dataset.pgo_poses.begin() + static_cast<int>(frame_id + 1));
    } else if (this->config.track_on) {
        this->used_poses.assign(this->dataset.odom_poses.begin(), this->dataset.odom_poses.begin() + static_cast<int>(frame_id + 1));
    } else if (this->dataset.gt_pose_provided) {
        this->used_poses.assign(this->dataset.gt_poses.begin(), this->dataset.gt_poses.begin() + static_cast<int>(frame_id + 1));
    }
}

void lo::Mapper::processFrame(size_t frame_id, Eigen::MatrixXd point_cloud, SE3 cur_pose) {
    const Eigen::RowVector3d t = cur_pose.translation().transpose();
    const Eigen::Matrix3d R = cur_pose.so3().matrix();

    auto xyz_columns = point_cloud.leftCols<3>();  // Block reference of point_cloud (not a copy)
    xyz_columns.noalias() = xyz_columns * R.transpose();
    xyz_columns.rowwise() += t;

    this->map.Update(frame_id, point_cloud);
    this->map.resetLocalMap(frame_id, t);
    this->determineMappingPoses(frame_id);
}