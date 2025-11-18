#include "dataset.hpp"

#include <algorithm>  // std::sort
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
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
    this->cur_reflectance = frame_data.reflectance;
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
    const auto& [min_it, max_it] = std::minmax_element(this->cur_point_ts.begin(), this->cur_point_ts.end());
    const double min_time = *min_it;
    const double max_time = *max_it;
    const auto normalize = [&](const double t) { return (t - min_time) / (max_time - min_time); };  // 0..1
    const Eigen::Matrix<double, 6, 1> omega = this->last_odom_transformation.log();

    const Eigen::Index N = this->cur_point_cloud.cols();  // 3×N → iterate cols
    for (Eigen::Index i = 0; i < N; ++i) {
        const double stamp = normalize(this->cur_point_ts[i]);
        const Sophus::SE3d T = Sophus::SE3d::exp((stamp - 1.0) * omega);
        Eigen::Vector3d p = this->cur_point_cloud.col(i);
        this->cur_point_cloud.col(i) = T * p;  // in-place
    }
}

std::vector<size_t> lo::Dataset::Crop(const Eigen::Matrix3Xd& pcd, float min_range, float max_range) {
    const double min2 = double(min_range) * min_range;
    const double max2 = double(max_range) * max_range;

    std::vector<size_t> keep;
    keep.reserve(static_cast<size_t>(pcd.cols()));

    // Fast simple for-loop; compilers auto-vectorize colwise squares well at -O3 -march=native
    for (Eigen::Index i = 0; i < pcd.cols(); ++i) {
        const double r2 = pcd.col(i).squaredNorm();
        if (r2 > min2 && r2 < max2) keep.push_back(static_cast<size_t>(i));
    }
    return keep;
}

std::vector<size_t> lo::Dataset::VoxelDownSamplePointCloud(const Eigen::Matrix3Xd& pcd, float voxel_size) {
    std::unordered_set<lo::VoxelKey> voxel_grid;  // only uniqueness matters
    std::vector<size_t> downsampled_indices;

    const Eigen::Index N = pcd.cols();
    voxel_grid.reserve(static_cast<size_t>(N));           // avoid many rehashes
    downsampled_indices.reserve(static_cast<size_t>(N));  // worst case

    for (Eigen::Index i = 0; i < N; ++i) {
        const Eigen::Vector3d p = pcd.col(i);
        const lo::VoxelKey voxel = lo::PointToVoxel(p, voxel_size);  // or Dataset::PointToVoxel if same logic

        auto [it, inserted] = voxel_grid.insert(voxel);
        if (inserted) {
            downsampled_indices.emplace_back(static_cast<size_t>(i));
        }
    }

    return downsampled_indices;
}

void lo::Dataset::preprocessFrame(size_t frame_id) {
    this->readFrameWithLoader(frame_id);

    // 1. Deskew
    if (this->cfg.deskew) this->Deskew();

    // 2. Crop pcd for mapping
    auto keep = lo::Dataset::Crop(this->cur_point_cloud, this->cfg.min_range, this->cfg.max_range);
    Eigen::Matrix3Xd pcd_cropped(3, keep.size());
    for (size_t k = 0; k < keep.size(); ++k) {
        pcd_cropped.col(k) = this->cur_point_cloud.col(static_cast<Eigen::Index>(keep[k]));
    }
    this->cur_point_cloud = std::move(pcd_cropped);

    // 3. Voxel Downsample pcd for mapping
    std::vector<size_t> idx_map = VoxelDownSamplePointCloud(this->cur_point_cloud, this->cfg.vox_down_m);
    Eigen::Matrix3Xd pcd_mapping(3, idx_map.size());
    for (size_t k = 0; k < idx_map.size(); ++k) {
        pcd_mapping.col(static_cast<Eigen::Index>(k)) = this->cur_point_cloud.col(static_cast<Eigen::Index>(idx_map[k]));
    }
    this->cur_point_cloud = std::move(pcd_mapping);

    // 3. Voxel Downsample pcd for registration
    std::vector<size_t> idx_reg = VoxelDownSamplePointCloud(this->cur_point_cloud, this->cfg.source_vox_down_m);
    this->cur_source_points = Eigen::Matrix3Xd(3, idx_reg.size());
    for (size_t k = 0; k < idx_reg.size(); ++k) {
        this->cur_source_points.col(static_cast<Eigen::Index>(k)) = this->cur_point_cloud.col(static_cast<Eigen::Index>(idx_reg[k]));
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

    // deskew pcd for mapping -> I already deskewed full scan in beginning so I dont need to do it again here
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
