#include "visualizer.hpp"

lo::Visualizer::Visualizer(const Config& config) : config(config), recording_stream(rerun::RecordingStream("LOVE-ICP")) {
    if (this->config.rerun_viz_on) {
        recording_stream.spawn().exit_on_failure();
        // recording_stream.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);  // Set an up-axis
    }
}

// ---------------- poses as frames ----------------
void lo::Visualizer::log_world_frame(const Sophus::SE3d& T_world) const {
    const Eigen::Matrix4d T_world_Eigen = T_world.matrix();
    const Eigen::Matrix3d R = T_world_Eigen.topLeftCorner<3, 3>();   // or = T_world.so3().matrix();
    const Eigen::Vector3d t = T_world_Eigen.topRightCorner<3, 1>();  // or = T_world.translation();

    rerun::datatypes::Vec3D cols[3];
    eigen_R_to_columns(R, cols);

    recording_stream.log("poses/world",
                         rerun::archetypes::Transform3D(rerun::components::Translation3D(Vec3(t)), rerun::components::TransformMat3x3(cols)));

    // RGB axes
    const std::array<std::array<uint8_t, 3>, 3> colors = {{
        {255, 0, 0},  // X red
        {0, 255, 0},  // Y green
        {0, 0, 255}   // Z blue
    }};

    log_axes("poses/world/axes", static_cast<float>(config.world_axes_length), colors);
}

void lo::Visualizer::log_current_odometry_frame(const Sophus::SE3d& T_odom) const {
    const Eigen::Matrix4d T_odom_Eigen = T_odom.matrix();
    const Eigen::Matrix3d R = T_odom_Eigen.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T_odom_Eigen.topRightCorner<3, 1>();

    rerun::datatypes::Vec3D cols[3];
    eigen_R_to_columns(R, cols);

    recording_stream.log("poses/odometry", rerun::archetypes::Transform3D(rerun::components::Translation3D(Vec3(t)), cols));

    const std::array<std::array<uint8_t, 3>, 3> colors = {{{180, 80, 255}, {0, 255, 255}, {255, 0, 255}}};

    log_axes("poses/odometry/axes", static_cast<float>(config.current_axes_length), colors);
}

void lo::Visualizer::log_current_gt_frame(const Sophus::SE3d& T_gt) const {
    const Eigen::Matrix4d T_gt_Eigen = T_gt.matrix();
    const Eigen::Matrix3d R = T_gt_Eigen.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T_gt_Eigen.topRightCorner<3, 1>();

    rerun::datatypes::Vec3D cols[3];
    eigen_R_to_columns(R, cols);

    recording_stream.log("poses/gt", rerun::archetypes::Transform3D(rerun::components::Translation3D(Vec3(t)), cols));

    const std::array<std::array<uint8_t, 3>, 3> colors = {{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}};

    log_axes("poses/gt/axes", static_cast<float>(config.current_axes_length), colors);
}

void lo::Visualizer::log_current_slam_frame(const Sophus::SE3d& T_slam) const {
    const Eigen::Matrix4d T_slam_Eigen = T_slam.matrix();
    const Eigen::Matrix3d R = T_slam_Eigen.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T_slam_Eigen.topRightCorner<3, 1>();

    rerun::datatypes::Vec3D cols[3];
    eigen_R_to_columns(R, cols);

    recording_stream.log("poses/slam", rerun::archetypes::Transform3D(rerun::components::Translation3D(Vec3(t)), cols));

    const std::array<std::array<uint8_t, 3>, 3> colors = {{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}};

    log_axes("poses/slam/axes", static_cast<float>(config.current_axes_length), colors);
}

// Positions as points
void lo::Visualizer::log_odometry_positions(int frame_id, const std::vector<Sophus::SE3d>& odom_poses) const {
    std::vector<rerun::components::Position3D> pts;
    for (int i = 0; i < frame_id + 1; i++) {
        Eigen::Vector3d position = odom_poses.at(i).translation();
        pts.emplace_back(Vec3(position));
    }

    auto arche = rerun::archetypes::Points3D(std::move(pts))
                     .with_colors(std::vector<rerun::components::Color>{colorRGB(255, 0, 0)})
                     .with_radii(std::vector<rerun::components::Radius>{rerun::components::Radius(0.2f)});

    recording_stream.log("positions/odometry", std::move(arche));
}

void lo::Visualizer::log_slam_positions(int frame_id, const std::vector<Sophus::SE3d>& slam_poses) const {
    std::vector<rerun::components::Position3D> pts;
    for (int i = 0; i < frame_id + 1; i++) {
        Eigen::Vector3d position = slam_poses.at(i).translation();
        pts.emplace_back(Vec3(position));
    }

    auto arche = rerun::archetypes::Points3D(std::move(pts))
                     .with_colors(std::vector<rerun::components::Color>{colorRGB(0, 255, 0)})
                     .with_radii(std::vector<rerun::components::Radius>{rerun::components::Radius(0.2f)});

    recording_stream.log("positions/slam", std::move(arche));
}

void lo::Visualizer::log_gt_positions(int frame_id, const std::vector<Sophus::SE3d>& gt_poses) const {
    std::vector<rerun::components::Position3D> pts;
    for (int i = 0; i < frame_id + 1; i++) {
        Eigen::Vector3d position = gt_poses.at(i).translation();
        pts.emplace_back(Vec3(position));
    }

    auto arche = rerun::archetypes::Points3D(std::move(pts))
                     .with_colors(std::vector<rerun::components::Color>{colorRGB(0, 0, 255)})
                     .with_radii(std::vector<rerun::components::Radius>{rerun::components::Radius(0.2f)});

    recording_stream.log("positions/gt", std::move(arche));
}

// ---------------- trajectories ----------------
void lo::Visualizer::log_odom_trajectory(const int frame_id, const std::vector<Sophus::SE3d>& odom_poses) const {
    // Python used green for odom trajectory
    log_trajectory_lines("trajectory/odometry", frame_id, odom_poses, {0, 255, 0});
}

void lo::Visualizer::log_slam_trajectory(const int frame_id, const std::vector<Sophus::SE3d>& slam_poses) const {
    log_trajectory_lines("trajectory/slam", frame_id, slam_poses, {0, 0, 255});
}

void lo::Visualizer::log_gt_trajectory(const int frame_id, const std::vector<Sophus::SE3d>& gt_poses) const {
    log_trajectory_lines("trajectory/gt", frame_id, gt_poses, {255, 0, 0});
}

// ---------------- helpers ----------------
rerun::datatypes::Vec3D lo::Visualizer::Vec3(const Eigen::Vector3d& v) {
    return rerun::datatypes::Vec3D{static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z())};
}

rerun::components::Color lo::Visualizer::colorRGB(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return rerun::components::Color(r, g, b, a);  // RGBA
}

void lo::Visualizer::eigen_R_to_columns(const Eigen::Matrix3d& R, rerun::datatypes::Vec3D* out_cols) {
    for (int i = 0; i < 3; ++i) out_cols[i] = Vec3(R.col(i));
}

void lo::Visualizer::log_axes(const std::string& path, float length, const std::array<std::array<uint8_t, 3>, 3>& rgb_axes) const {
    // 3 arrows from origin along +X, +Y, +Z
    std::vector<rerun::components::Position3D> origins(3, rerun::components::Position3D(0.f, 0.f, 0.f));

    std::vector<rerun::components::Vector3D> vectors;
    vectors.reserve(3);
    vectors.emplace_back(length, 0.f, 0.f);  // X
    vectors.emplace_back(0.f, length, 0.f);  // Y
    vectors.emplace_back(0.f, 0.f, length);  // Z

    std::vector<rerun::components::Color> colors;
    colors.reserve(3);
    colors.emplace_back(rerun::components::Color(rgb_axes[0][0], rgb_axes[0][1], rgb_axes[0][2]));
    colors.emplace_back(rerun::components::Color(rgb_axes[1][0], rgb_axes[1][1], rgb_axes[1][2]));
    colors.emplace_back(rerun::components::Color(rgb_axes[2][0], rgb_axes[2][1], rgb_axes[2][2]));

    auto arrows = rerun::archetypes::Arrows3D().with_origins(std::move(origins)).with_vectors(std::move(vectors)).with_colors(std::move(colors));

    recording_stream.log(path.c_str(), std::move(arrows));
}

void lo::Visualizer::log_trajectory_lines(const std::string& path,
                                          const int frame_id,
                                          const std::vector<Sophus::SE3d>& poses,
                                          const std::array<uint8_t, 3>& rgb) const {
    if (frame_id < 2) {
        return;
    }

    // Build a list of line strips, each with 2 points [prev, cur]
    std::vector<rerun::components::LineStrip3D> strips;
    // strips.reserve(poses.size() - 1);

    for (size_t i = 1; i < frame_id + 1; i++) {
        const Eigen::Vector3d p0 =
            poses[i - 1].matrix().block<3, 1>(0, 3);  // Sophus::SE3d.matrix() == Eigen::Matrix4d ; or just poses[i - 1].translation()
        const Eigen::Vector3d p1 = poses[i].matrix().block<3, 1>(0, 3);

        std::vector<rerun::components::Position3D> points;
        points.emplace_back(Vec3(p0));
        points.emplace_back(Vec3(p1));
        strips.emplace_back(std::move(points));
    }

    auto arche =
        rerun::archetypes::LineStrips3D(std::move(strips)).with_colors(std::vector<rerun::components::Color>{colorRGB(rgb[0], rgb[1], rgb[2])});

    recording_stream.log(path.c_str(), std::move(arche));
}