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
