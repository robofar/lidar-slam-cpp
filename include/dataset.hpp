#include <Eigen/Core>
#include <string>

#include "config.hpp"

namespace lo {

class FrameData {
  public:
    Eigen::MatrixXd points;    // (N,4)
    Eigen::VectorXd point_ts;  // (N,)
    FrameData(const Eigen::MatrixXd& points, Eigen::VectorXd point_ts) : points(points), point_ts(point_ts) {}
};

class Dataset {
  private:
    std::vector<std::string> scan_files;
    static Eigen::MatrixXd readPointCloud(const std::string& scan_file);
    static Eigen::VectorXd computePointTimestamps(const Eigen::MatrixXd& points);

  public:
    // Attributes
    lo::Config cfg;
    bool silence;
    bool gt_pose_provided;
    size_t processed_frame;
    size_t total_pc_count;

    std::vector<Eigen::Matrix4d> odom_poses;
    std::vector<Eigen::Matrix4d> pgo_poses;

    std::vector<double> travel_dist;

    Eigen::MatrixXd cur_point_cloud;
    Eigen::VectorXd cur_point_ts;
    Eigen::Matrix4d cur_pose_guess;
    Eigen::Matrix4d cur_pose;

    Eigen::MatrixXd cur_source_points;
    Eigen::MatrixXd cur_source_colors;
    Eigen::MatrixXd cur_source_normals;

    Eigen::Matrix4d last_pose_ref;
    Eigen::Matrix4d last_odom_transformation;
    Eigen::Matrix4d cur_pose_ref;

    explicit Dataset(const Config&);
    size_t size() const;
    FrameData get(size_t idx) const;
};

}  // namespace lo