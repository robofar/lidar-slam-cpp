#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <string>

#include "config.hpp"
#include "kitti.hpp"

namespace lo {

using SE3 = Sophus::SE3d;
// using Voxel = Eigen::Vector3i;   // < not defined
using Voxel = std::array<int, 3>;  // < defined

class Dataset {
  public:
    // Attributes
    lo::Config cfg;
    KITTIOdometryDataset loader;
    bool silence;
    bool gt_pose_provided;
    size_t processed_frame;
    size_t total_pc_count;

    std::vector<SE3> odom_poses;  // Eigen::Matrix4d
    std::vector<SE3> pgo_poses;
    std::vector<SE3> gt_poses;

    std::vector<double> travel_dist;

    Eigen::MatrixXd cur_point_cloud;
    Eigen::VectorXd cur_point_ts;
    SE3 cur_pose_guess;  // Default constructor called (R=I, t=0) ; Eigen::Matrix4d
    SE3 cur_pose;
    SE3 last_pose;
    SE3 last_odom_transformation;

    Eigen::MatrixXd cur_source_points;
    Eigen::MatrixXd cur_source_colors;
    Eigen::MatrixXd cur_source_normals;

    explicit Dataset(const Config&, const KITTIOdometryDataset&);

    void readFrameWithLoader(size_t frame_id);
    void initialPoseGuess(size_t frame_id);
    void preprocessFrame(size_t frame_id);  // deskewing (TODO) + voxel downsampling

    static Voxel PointToVoxel(const Eigen::Vector3d& point, const double voxel_size);
    static std::vector<size_t> VoxelDownSamplePointCloud(const Eigen::MatrixXd& pcd, float voxel_size);
    void Deskew();
    static std::vector<size_t> Crop(const Eigen::MatrixXd& pcd, float min_range, float max_range);

    void UpdatePoses(size_t frame_id, SE3 cur_pose);
    void UpdatePosesAfterPGO(const std::vector<SE3>& pgo_poses);  // pgo_poses includes all poses so far (including current frame pose) (i.e.
                                                                  // pgo_poses.size() = (frame_id + 1)

  private:
    std::vector<Sophus::SE3d> getGtPosesAsSE3() const;
};

}  // namespace lo