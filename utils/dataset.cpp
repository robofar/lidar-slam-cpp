#include "dataset.hpp"

#include <algorithm>  // std::sort
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

lo::Dataset::Dataset(const Config& config) : cfg(config) {  // cfg(config) is shallow copy (changes made in this->cfg does not reflect in config)
    std::string velodyne_dir = static_cast<std::string>(cfg.data_path) + "sequences/" + cfg.data_loader_seq + "/" + "velodyne/";
    if (!std::filesystem::exists(velodyne_dir)) {
        throw std::runtime_error("Velodyne folder not found: " + velodyne_dir);
    }

    for (const auto& entry : std::filesystem::directory_iterator(velodyne_dir)) {
        if (entry.path().extension() == ".bin") scan_files.push_back(entry.path().string());
    }

    std::sort(scan_files.begin(), scan_files.end());
    std::cout << "[KITTI Dataset] Loaded " << scan_files.size() << " scans from " << velodyne_dir << std::endl;

    config.TestFcn();
    cfg.TestFcn();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    silence = cfg.silence;
    gt_pose_provided = false;
    processed_frame = 0;

    // cfg.end_frame will change, but not config.end_frame (shallow copy)
    cfg.end_frame = std::min(static_cast<int>(scan_files.size()), cfg.end_frame);  // if there is less number of scans in folder than in config file
    total_pc_count = (cfg.end_frame - cfg.begin_frame) / cfg.step_frame;

    if (cfg.track_on) odom_poses.resize(total_pc_count, Eigen::Matrix4d::Identity());
    travel_dist = std::vector<double>(total_pc_count, 0.0);  // or resize

    // cur_point_cloud -> None
    // cur_point_ts -> None
    // cur_pose_guess -> None
    // cur_pose -> None
    // cur_source_points -> None
    // cur_source_colors -> None
    // cur_source_normals -> None

    last_pose_ref = Eigen::Matrix4d::Identity();
    last_odom_transformation = Eigen::Matrix4d::Identity();
    cur_pose_ref = Eigen::Matrix4d::Identity();
}

size_t lo::Dataset::size() const {
    return scan_files.size();
}

Eigen::MatrixXd lo::Dataset::readPointCloud(const std::string& scan_file) {
    std::ifstream ifs(scan_file, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Failed to open file: " + scan_file);

    std::vector<float> buffer((std::istreambuf_iterator<char>(ifs)), {});
    size_t num_floats = buffer.size() / sizeof(float);
    size_t num_points = num_floats / 4;

    ifs.clear();
    ifs.seekg(0, std::ios::beg);
    std::vector<float> data(4 * num_points);
    ifs.read(reinterpret_cast<char*>(data.data()), 4 * num_points * sizeof(float));

    Eigen::MatrixXd points(num_points, 4);
    for (size_t i = 0; i < num_points; ++i) {
        points(i, 0) = static_cast<double>(data[i * 4 + 0]);
        points(i, 1) = static_cast<double>(data[i * 4 + 1]);
        points(i, 2) = static_cast<double>(data[i * 4 + 2]);
        points(i, 3) = static_cast<double>(data[i * 4 + 3]);  // reflectance
    }

    return points;
}

Eigen::VectorXd lo::Dataset::computePointTimestamps(const Eigen::MatrixXd& points) {
    size_t N = points.rows();
    Eigen::VectorXd ts(N);

    for (size_t i = 0; i < N; ++i) {
        double x = points(i, 0);
        double y = points(i, 1);
        double yaw = -std::atan2(y, x);
        ts(i) = 0.5 * (yaw / M_PI + 1.0);
    }

    return ts;
}

lo::FrameData lo::Dataset::get(size_t idx) const {
    if (idx >= scan_files.size()) throw std::out_of_range("Index out of range.");

    Eigen::MatrixXd points = readPointCloud(scan_files[idx]);
    Eigen::VectorXd point_ts = computePointTimestamps(points);

    // return {points, point_ts};
    return lo::FrameData(points, point_ts);
}