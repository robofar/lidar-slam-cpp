#include "mapper.hpp"

#include "utils.hpp"

void lo::Mapper::determineMappingPoses(size_t frame_id) {
    if (this->config.pgo_on) {
        this->used_poses.assign(this->dataset.pgo_poses.begin(), this->dataset.pgo_poses.begin() + static_cast<int>(frame_id + 1));
    } else if (this->config.track_on) {
        this->used_poses.assign(this->dataset.odom_poses.begin(), this->dataset.odom_poses.begin() + static_cast<int>(frame_id + 1));
    } else if (this->dataset.gt_pose_provided) {
        this->used_poses.assign(this->dataset.gt_poses.begin(), this->dataset.gt_poses.begin() + static_cast<int>(frame_id + 1));
    }
}

void lo::Mapper::processFrame(size_t frame_id, const Eigen::Matrix3Xd& point_cloud, const SE3& cur_pose) {
    auto transformed_point_cloud = transformPointCloud(point_cloud, cur_pose);

    this->map.Update(frame_id, transformed_point_cloud);
    this->map.resetLocalMap(frame_id, cur_pose.translation().transpose());
    this->determineMappingPoses(frame_id);
}