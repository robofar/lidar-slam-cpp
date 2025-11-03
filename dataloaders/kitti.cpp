#include "kitti.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

using RowMajor3x4 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

lo::KITTIOdometryDataset::KITTIOdometryDataset(const Config& cfg) : config(cfg) {
    this->data_dir = this->config.data_path;
    this->sequence_id = this->config.data_loader_seq;
    this->kitti_sequence_dir = this->data_dir + std::string("sequences/") + this->sequence_id + std::string("/");
    this->velodyne_dir = this->kitti_sequence_dir + "velodyne/";

    if (!std::filesystem::exists(this->velodyne_dir)) {
        throw std::runtime_error("Velodyne folder not found: " + this->velodyne_dir);
    }

    for (const auto& entry : std::filesystem::directory_iterator(this->velodyne_dir)) {
        if (entry.path().extension() == ".bin") this->scan_files.push_back(entry.path().string());
    }

    std::sort(scan_files.begin(), scan_files.end());
    std::cout << "[KITTI Dataset] Loaded " << this->scan_files.size() << " scans from " << this->velodyne_dir << std::endl;

    const std::string calib_file = this->kitti_sequence_dir + "calib.txt";
    if (!std::filesystem::exists(calib_file)) {
        throw std::runtime_error("Calibration file not found: " + calib_file);
    }
    this->calib_raw = readCalibFile(calib_file);
    this->calib = loadCalib(this->calib_raw);

    // ---- GT poses (only for sequences < 11) ----
    int seq_int = std::stoi(this->sequence_id);
    if (seq_int < 11) {
        std::string poses_file = this->data_dir + "poses/" + this->sequence_id + ".txt";
        if (!std::filesystem::exists(poses_file)) {
            poses_file = this->kitti_sequence_dir + "poses.txt";  // fallback
        }
        if (std::filesystem::exists(poses_file)) {
            auto poses_cam = loadPosesCam(poses_file);
            this->gt_poses = convertCamPosesToLidar(poses_cam);
            std::cout << "[KITTI Dataset] Loaded " << this->gt_poses.size() << " GT poses (LiDAR frame)\n";
        } else {
            std::cout << "[KITTI Dataset] No GT poses found for sequence " << this->sequence_id << "\n";
        }
    }
}

size_t lo::KITTIOdometryDataset::size() const {
    return scan_files.size();
}

Eigen::MatrixXd lo::KITTIOdometryDataset::readPointCloud(const std::string& scan_file) {
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

Eigen::VectorXd lo::KITTIOdometryDataset::computePointTimestamps(const Eigen::MatrixXd& points) {
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

lo::FrameData lo::KITTIOdometryDataset::get(size_t idx) const {
    if (idx >= scan_files.size()) throw std::out_of_range("Index out of range.");

    Eigen::MatrixXd points = readPointCloud(scan_files[idx]);
    Eigen::VectorXd point_ts;
    if (this->config.deskew) point_ts = computePointTimestamps(points);

    // return {points, point_ts};
    return lo::FrameData(points, point_ts);
}

std::unordered_map<std::string, std::vector<double>> lo::KITTIOdometryDataset::readCalibFile(const std::string& file_path) {
    std::unordered_map<std::string, std::vector<double>> out;
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) throw std::runtime_error("Failed to open calib: " + file_path);

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string key_with_colon;
        iss >> key_with_colon;  // e.g. "P0:" or "Tr:"
        if (key_with_colon == "calib_time:") continue;

        std::string key = key_with_colon;
        if (!key.empty() && key.back() == ':') key.pop_back();

        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        out[key] = std::move(vals);
    }
    return out;
}

static Eigen::Matrix<double, 3, 4> map3x4_or_throw(const std::unordered_map<std::string, std::vector<double>>& C, const std::string& k) {
    const auto it = C.find(k);
    if (it == C.end() || it->second.size() != 12) throw std::runtime_error("Missing/invalid calib entry: " + k);
    Eigen::Matrix<double, 3, 4> M;
    M = Eigen::Map<const RowMajor3x4>(it->second.data());
    return M;
}

lo::CalibData lo::KITTIOdometryDataset::loadCalib(const std::unordered_map<std::string, std::vector<double>>& C) {
    CalibData d;
    d.P_rect_00 = map3x4_or_throw(C, "P0");
    d.P_rect_10 = map3x4_or_throw(C, "P1");
    d.P_rect_20 = map3x4_or_throw(C, "P2");
    d.P_rect_30 = map3x4_or_throw(C, "P3");

    // rectified extrinsics T1,T2,T3 (pykitti trick)
    Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
    T1(0, 3) = d.P_rect_10(0, 3) / d.P_rect_10(0, 0);
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2(0, 3) = d.P_rect_20(0, 3) / d.P_rect_20(0, 0);
    Eigen::Matrix4d T3 = Eigen::Matrix4d::Identity();
    T3(0, 3) = d.P_rect_30(0, 3) / d.P_rect_30(0, 0);

    // T_cam0_velo from Tr (3x4)
    const auto itTr = C.find("Tr");
    if (itTr == C.end() || itTr->second.size() != 12) throw std::runtime_error("Missing/invalid 'Tr' in calib");
    d.T_cam0_velo.setIdentity();
    d.T_cam0_velo.topLeftCorner<3, 4>() = Eigen::Map<const RowMajor3x4>(itTr->second.data());

    d.T_cam1_velo = T1 * d.T_cam0_velo;
    d.T_cam2_velo = T2 * d.T_cam0_velo;
    d.T_cam3_velo = T3 * d.T_cam0_velo;

    // intrinsics
    d.K_cam0 = d.P_rect_00.topLeftCorner<3, 3>();
    d.K_cam1 = d.P_rect_10.topLeftCorner<3, 3>();
    d.K_cam2 = d.P_rect_20.topLeftCorner<3, 3>();
    d.K_cam3 = d.P_rect_30.topLeftCorner<3, 3>();

    // baselines in velodyne frame
    const Eigen::Vector4d o(0, 0, 0, 1);
    const Eigen::Vector4d v0 = d.T_cam0_velo.inverse() * o;
    const Eigen::Vector4d v1 = d.T_cam1_velo.inverse() * o;
    const Eigen::Vector4d v2 = d.T_cam2_velo.inverse() * o;
    const Eigen::Vector4d v3 = d.T_cam3_velo.inverse() * o;

    d.b_gray = (v1.head<3>() - v0.head<3>()).norm();
    d.b_rgb = (v3.head<3>() - v2.head<3>()).norm();

    return d;
}

// poses loader: each line 12 numbers (row-major 3x4), expand to 4x4
std::vector<Eigen::Matrix4d> lo::KITTIOdometryDataset::loadPosesCam(const std::string& poses_file) const {
    std::ifstream ifs(poses_file);
    if (!ifs.is_open()) throw std::runtime_error("Failed to open poses: " + poses_file);

    std::vector<Eigen::Matrix4d> out;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) iss >> T(r, c);
        out.push_back(T);
    }
    return out;
}

// convert camera poses to LiDAR frame: Tr^{-1} * P * Tr
std::vector<Eigen::Matrix4d> lo::KITTIOdometryDataset::convertCamPosesToLidar(const std::vector<Eigen::Matrix4d>& poses_cam) const {
    const auto itTr = this->calib_raw.find("Tr");
    if (itTr == this->calib_raw.end() || itTr->second.size() != 12) throw std::runtime_error("Missing/invalid 'Tr' in calib_raw");
    Eigen::Matrix4d Tr = Eigen::Matrix4d::Identity();
    Tr.topLeftCorner<3, 4>() = Eigen::Map<const RowMajor3x4>(itTr->second.data());
    const Eigen::Matrix4d Tr_inv = Tr.inverse();

    std::vector<Eigen::Matrix4d> out;
    out.reserve(poses_cam.size());
    for (const auto& P : poses_cam) out.push_back(Tr_inv * P * Tr);
    return out;
}
