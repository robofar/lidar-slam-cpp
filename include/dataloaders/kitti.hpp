#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#include "config.hpp"

namespace lo {

class FrameData {
  public:
    Eigen::MatrixXd points;    // (N,4) ; std::vector<Eigen::Vector3d> ; std::vector<Eigen::Vector4d>
    Eigen::VectorXd point_ts;  // (N,) ; std::vector<double>
    FrameData(const Eigen::MatrixXd& points, Eigen::VectorXd point_ts) : points(points), point_ts(point_ts) {}
};

struct CalibData {
    // P_rect_* (3x4)
    Eigen::Matrix<double, 3, 4> P_rect_00, P_rect_10, P_rect_20, P_rect_30;

    // T_cam*_velo (4x4)
    Eigen::Matrix4d T_cam0_velo, T_cam1_velo, T_cam2_velo, T_cam3_velo;

    // intrinsics
    Eigen::Matrix3d K_cam0, K_cam1, K_cam2, K_cam3;

    // baselines
    double b_gray = 0.0;
    double b_rgb = 0.0;
};

class KITTIOdometryDataset {
  public:
    lo::Config config;
    std::string data_dir;
    std::string sequence_id;
    std::string kitti_sequence_dir;
    std::string velodyne_dir;
    std::vector<std::string> scan_files;

    // calibration & poses
    CalibData calib;
    std::vector<Eigen::Matrix4d> gt_poses;  // optional; empty if not available

    // ctors
    KITTIOdometryDataset(const Config& cfg);

    // static
    static Eigen::MatrixXd readPointCloud(const std::string& scan_file);
    static Eigen::VectorXd computePointTimestamps(const Eigen::MatrixXd& points);

    // const
    size_t size() const;
    FrameData get(size_t idx) const;

    // methods

  private:
    std::unordered_map<std::string, std::vector<double>> calib_raw;
    static std::unordered_map<std::string, std::vector<double>> readCalibFile(const std::string& file_path);
    static CalibData loadCalib(const std::unordered_map<std::string, std::vector<double>>& C);

    std::vector<Eigen::Matrix4d> loadPosesCam(const std::string& poses_file) const;
    std::vector<Eigen::Matrix4d> convertCamPosesToLidar(const std::vector<Eigen::Matrix4d>& poses_cam) const;
};
}  // namespace lo