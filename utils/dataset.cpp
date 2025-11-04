#include "dataset.hpp"

#include <algorithm>  // std::sort
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

lo::Dataset::Dataset(const Config& config, const KITTIOdometryDataset& loader)
    : cfg(config), loader(loader) {  // cfg(config) is shallow copy (changes made in this->cfg does not reflect in config)

    silence = cfg.silence;
    processed_frame = 0;

    // cfg.end_frame will change, but not config.end_frame (shallow copy)
    cfg.end_frame = std::min(static_cast<int>(this->loader.size()), cfg.end_frame);  // if there is less number of scans in folder than in config file
    total_pc_count = (cfg.end_frame - cfg.begin_frame) / cfg.step_frame;

    this->gt_poses = std::move(getGtPosesAsSE3());
    if (this->gt_poses.size() > 0) {
        this->gt_pose_provided = true;
    } else {
        this->gt_pose_provided = false;
    }

    // if (cfg.track_on) odom_poses.resize(total_pc_count, Eigen::Matrix4d::Identity());
    if (cfg.track_on) odom_poses.resize(total_pc_count, SE3());
    if (cfg.pgo_on) pgo_poses.resize(total_pc_count, SE3());

    travel_dist = std::vector<double>(total_pc_count, 0.0);  // or resize

    // cur_pose = SE3();                  //  Default constructor called when declaring attribute ; = Eigen::Matrix4d::Identity()
    // last_pose = SE3();
    // last_odom_transformation = SE3();
}

void lo::Dataset::readFrameWithLoader(size_t frame_id) {
    auto frame_data = this->loader.get(frame_id);

    this->cur_point_cloud = frame_data.points;
    if (this->cfg.deskew) this->cur_point_ts = frame_data.point_ts;
}

// Initial pose guess for ICP
void lo::Dataset::initialPoseGuess(size_t frame_id) {
    if (frame_id == 0) {
        if (this->gt_pose_provided)
            this->cur_pose_guess = this->gt_poses[frame_id];
        else
            this->cur_pose_guess = SE3();
    } else if (frame_id > 0) {
        if (!(this->cfg.track_on) && this->gt_pose_provided) {  // mapping (no tracking + gt_poses provided)
            this->cur_pose_guess = this->gt_poses[frame_id];
        } else {
            if (this->cfg.uniform_motion_on)  // uniform motion model
                this->cur_pose_guess = (this->last_pose * this->last_odom_transformation);
            else  // no motion prior
                this->cur_pose_guess = this->last_pose;

            // Case: No tracking + no gt_poses provided -> this will never happened here, bcs it is handled in main
        }
    }
}

void lo::Dataset::Deskew() {
    if (!this->cfg.deskew || this->cur_point_ts.size() == 0) return;
    const auto& [min, max] = std::minmax_element(this->cur_point_ts.begin(), this->cur_point_ts.end());
    const double min_time = *min;
    const double max_time = *max;
    const auto normalize = [&](const double t) { return (t - min_time) / (max_time - min_time); };  // 0.0 - 1.0
    const auto& omega = this->last_odom_transformation.log();                                       // 6x1

    const Eigen::Index N = this->cur_point_cloud.rows();
    for (Eigen::Index i = 0; i < N; i++) {
        const double stamp = normalize(this->cur_point_ts[i]);
        const Sophus::SE3d T = Sophus::SE3d::exp((stamp - 1.0) * omega);
        Eigen::Vector3d p = this->cur_point_cloud.row(i).head<3>().transpose();  // 3x1
        const Eigen::Vector3d q = T * p;
        cur_point_cloud.row(i).head<3>() = q.transpose();  // in-place change
    }
}

std::vector<size_t> lo::Dataset::Crop(const Eigen::MatrixXd& pcd, float min_range, float max_range) {
    std::vector<size_t> keep_indices;

    const Eigen::Index N = pcd.rows();
    Eigen::MatrixXd cropped_frame;
    for (Eigen::Index i = 0; i < N; i++) {
        double r = pcd.row(i).head<3>().norm();
        if (r < max_range && r > min_range) keep_indices.push_back(i);
    }

    return keep_indices;
}

lo::Voxel lo::Dataset::PointToVoxel(const Eigen::Vector3d& point, const double voxel_size) {
    return lo::Voxel{static_cast<int>(std::floor(point.x() / voxel_size)), static_cast<int>(std::floor(point.y() / voxel_size)),
                     static_cast<int>(std::floor(point.z() / voxel_size))};
}

std::vector<size_t> lo::Dataset::VoxelDownSamplePointCloud(const Eigen::MatrixXd& pcd, float voxel_size) {
    std::map<lo::Voxel, int> voxel_grid;  // keep first index per voxel

    std::vector<size_t> downsampled_indices;
    std::vector<size_t> indices(pcd.rows());
    // numeric library ; like range in python -> fills vector from 0 to n-1, then you can iterate over
    // indices, since there is no row-kinda-iterator in Eigen
    std::iota(indices.begin(), indices.end(), 0);
    // for (auto row : pcd.rowwise()) std::cout << row << std::endl;
    std::for_each(indices.cbegin(), indices.cend(), [&](int i) {
        auto row = pcd.row(i);
        const auto voxel = lo::Dataset::PointToVoxel(Eigen::Vector3d(row.x(), row.y(), row.z()), voxel_size);
        const auto status = voxel_grid.insert({voxel, i});
        if (status.second) downsampled_indices.emplace_back(i);
    });

    return downsampled_indices;
}

void lo::Dataset::preprocessFrame(size_t frame_id) {
    this->readFrameWithLoader(frame_id);

    // 1. Deskew
    // this->Deskew();

    // 2. Crop pcd for mapping
    /*
    auto valid_indices_crop = lo::Dataset::Crop(this->cur_point_cloud, this->cfg.min_range, this->cfg.max_range);
    Eigen::MatrixXd pcd_cropped(valid_indices_crop.size(), this->cur_point_cloud.cols());
    for (size_t i = 0; i < valid_indices_crop.size(); i++) {
        pcd_cropped.row(i) = this->cur_point_cloud.row(valid_indices_crop[i]);
    }
    this->cur_point_cloud = std::move(pcd_cropped);
    */

    // 3. Voxel Downsample pcd for mapping
    std::vector<size_t> points_idx_mapping = VoxelDownSamplePointCloud(this->cur_point_cloud, this->cfg.vox_down_m);
    Eigen::MatrixXd pcd_mapping(points_idx_mapping.size(), this->cur_point_cloud.cols());
    for (size_t i = 0; i < points_idx_mapping.size(); i++) {
        pcd_mapping.row(i) = this->cur_point_cloud.row(points_idx_mapping[i]);
    }
    this->cur_point_cloud = std::move(pcd_mapping);

    // 3. Voxel Downsample pcd for registration
    std::vector<size_t> points_idx_registration = VoxelDownSamplePointCloud(this->cur_point_cloud, this->cfg.source_vox_down_m);
    this->cur_source_points = Eigen::MatrixXd(points_idx_registration.size(), this->cur_point_cloud.cols());
    for (size_t i = 0; i < points_idx_registration.size(); i++) {
        cur_source_points.row(i) = this->cur_point_cloud.row(points_idx_registration[i]);
    }
}

void lo::Dataset::UpdatePoses(size_t frame_id, SE3 cur_pose) {
    this->cur_pose = std::move(cur_pose);

    if (frame_id == 0)
        this->last_odom_transformation = SE3();
    else {
        this->last_odom_transformation = this->last_pose.inverse() * this->cur_pose;
        auto cur_frame_travel_dist = this->last_odom_transformation.translation().norm();
        this->travel_dist.at(frame_id) = this->travel_dist.at(frame_id - 1) + cur_frame_travel_dist;
    }

    this->last_pose = this->cur_pose;  // for next iteration

    if (this->cfg.track_on) {
        lo::SE3 cur_odom_pose;
        if (frame_id == 0) {
            cur_odom_pose = this->cur_pose;
        } else {
            cur_odom_pose = this->odom_poses.at(frame_id - 1) * this->last_odom_transformation;
        }
        this->odom_poses.at(frame_id) = cur_odom_pose;
    }

    if (this->cfg.pgo_on) {  // initialize the pgo pose
        this->pgo_poses.at(frame_id) = this->cur_pose;
    }

    // deskew pcd for mapping -> but I already deskewed full scan in beginning so I dont need to do it again here
}

void lo::Dataset::UpdatePosesAfterPGO(const std::vector<lo::SE3>& pgo_poses) {
    std::vector<size_t> indices(pgo_poses.size());
    std::iota(indices.begin(), indices.end(), 0);  // np.arange
    std::for_each(indices.begin(), indices.end(), [&](size_t idx) -> void {
        this->pgo_poses[idx] = pgo_poses[idx];  // update pgo pose
    });
    this->cur_pose = pgo_poses.back();  // update cur_pose such that it is corrected, and we continue with correct pose
    this->last_pose = this->cur_pose;   // update for next frame
}

// Private
std::vector<Sophus::SE3d> lo::Dataset::getGtPosesAsSE3() const {
    std::vector<Sophus::SE3d> gt_se3;
    if (this->loader.gt_poses.empty()) {
        std::cout << "[Dataset] Warning: No GT poses available to convert.\n";
        return gt_se3;
    }

    gt_se3.reserve((this->cfg.end_frame - this->cfg.begin_frame) / this->cfg.step_frame + 1);
    /*
    // Load all poses
    for (const auto& T : this->loader.gt_poses) {
        Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
        Eigen::Vector3d t = T.topRightCorner<3, 1>();
        gt_se3.emplace_back(Sophus::SO3d(R), t);
    }
    */

    // Load poses only in certain range
    for (size_t i = this->cfg.begin_frame; i < this->cfg.end_frame; i += this->cfg.step_frame) {
        const auto& T = this->loader.gt_poses[i];
        Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
        Eigen::Vector3d t = T.topRightCorner<3, 1>();
        // R is slightly non-orthogonal, and Sophus is very strict about it, and it would raise error if we try to create SO3 from R
        // Therefore convert this slightly non-orthogonal R into Quaterion, which is normalized, and this slightly non-orthogonal R produces same q
        // anyways
        gt_se3.emplace_back(Sophus::SO3d(Eigen::Quaterniond(R)), t);
    }

    return gt_se3;
}
