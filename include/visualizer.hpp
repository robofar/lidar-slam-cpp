#pragma "once"
#include <Eigen/Core>
#include <rerun.hpp>
#include <rerun/demo_utils.hpp>
#include <sophus/se3.hpp>

#include "config.hpp"

namespace lo {
class Visualizer {
  public:
    // Create a new `RecordingStream` which sends data over gRPC to the viewer process
    const rerun::RecordingStream recording_stream;
    const Config& config;

    // ctor
    explicit Visualizer(const Config& config);

    // -------- Poses as frames (axes). Helper: log_axes --------
    void log_world_frame(const Sophus::SE3d& T_world) const;
    void log_current_odometry_frame(const Sophus::SE3d& T_odom) const;
    void log_current_slam_frame(const Sophus::SE3d& T_slam) const;
    void log_current_gt_frame(const Sophus::SE3d& T_gt) const;

    // -------- Positions as points --------
    void log_odometry_positions(int frame_id, const std::vector<Sophus::SE3d>& odom_poses) const;  // Eigen::Matrix4d
    void log_slam_positions(int frame_id, const std::vector<Sophus::SE3d>& slam_poses) const;
    void log_gt_positions(int frame_id, const std::vector<Sophus::SE3d>& gt_poses) const;

    // -------- Trajectory as line. Helper: log_trajectory_lines --------
    void log_odom_trajectory(const int frame_id, const std::vector<Sophus::SE3d>& odom_poses) const;
    void log_slam_trajectory(const int frame_id, const std::vector<Sophus::SE3d>& slam_poses) const;
    void log_gt_trajectory(const int frame_id, const std::vector<Sophus::SE3d>& gt_poses) const;

    // -------- Scans / maps (point clouds) --------
    // Accept either Nx3 (XYZ) or Nx4 (XYZI/RGB—ignores extra columns)
    void log_current_scan(const Eigen::MatrixXd& scan_xyz,
                          float radii = 0.1f) const;  // Eigen::MatrixXd accepts shape (N,K) ; or maybe use Eigen::MatrixX3d (N,3)
    void log_current_local_map(const std::vector<Eigen::Vector3d>& local_map_xyz, float radii = 0.1f) const;
    void log_global_map(const std::vector<Eigen::Vector3d>& global_map_xyz, float radii = 0.1f) const;

  private:  // --- helpers ---
    static rerun::datatypes::Vec3D Vec3(const Eigen::Vector3d& v);
    static rerun::components::Color colorRGB(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);

    // Convert Eigen 3x3 to 3 column vectors for Transform3D
    static void eigen_R_to_columns(const Eigen::Matrix3d& R, rerun::datatypes::Vec3D* out_cols);

    // Log a triad of arrows for axes at current entity transform
    void log_axes(const std::string& path, float length, const std::array<std::array<uint8_t, 3>, 3>& rgb_axes) const;

    // Log a point cloud (Nx3 or Nx4)
    void log_pointcloud_xyz(const std::string& path, const Eigen::MatrixXd& pts, const std::array<uint8_t, 3>* rgb, float radii) const;
    void log_pointcloud_xyz(const std::string& path, const std::vector<Eigen::Vector3d>& pts, const std::array<uint8_t, 3>* rgb, float radii) const;

    // Log trajectory as line segments between consecutive positions
    void log_trajectory_lines(const std::string& path,
                              const int frame_id,
                              const std::vector<Sophus::SE3d>& poses,
                              const std::array<uint8_t, 3>& rgb) const;
};
}  // namespace lo