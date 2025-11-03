#pragma once

#include <array>
#include <filesystem>
#include <string>

namespace lo {

class Config {
  public:
    ////////////////////// settings
    std::string name = "lidar_slam";  // experiment name
    std::string run_name = name;      // may include timestamp later
    bool use_dataloader = true;
    std::string data_loader_name;
    std::string data_loader_seq;
    std::string experiment_name;

    std::string run_path;
    std::filesystem::path output_root = "./experiments";
    std::filesystem::path data_path = "./data/kitti_example/";
    std::filesystem::path pc_path;
    std::filesystem::path pose_path;
    std::filesystem::path calib_path;

    bool first_frame_ref = true;  // if false, use world as reference
    int begin_frame = 0;
    int end_frame = 1001;
    int step_frame = 1;
    int stop_frame_thre = 20;

    bool silence = true;

    bool kitti_correction_on = true;
    float correction_deg = 0.0;

    bool deskew = true;
    std::string lidar_type_guess = "velodyne";  // hesai

    bool color_map_on = false;
    int color_channel = 0;
    bool color_on = false;

    ////////////////////// preprocess
    float min_range = 3.0;
    float max_range = 60.0;
    bool adaptive_range_on = false;

    float min_z = -5.0;
    float max_z = 80.0;

    float voxel_size_m = 1.0;
    int num_points_per_voxel = 20;

    bool rand_downsample = false;
    float rand_down_r = 1.0;
    float vox_down_m = 0.5 * voxel_size_m;
    float source_vox_down_m = 1.5 * voxel_size_m;

    ////////////////////// hash map data structure
    long int buffer_size = 1000000;

    // Correspondence search
    int num_nei_cells = 1;
    float search_alpha = 1.0;

    // Local Map
    bool temporal_local_map_on = true;
    bool local_map_travel_dist = true;
    float local_map_radius = 50.0;
    float local_map_travel_dist_ratio = 1.0;
    int diff_ts_local = 50;
    bool use_mid_ts = false;

    ////////////////////// mapping
    int local_map_reset_freq = 1;

    ////////////////////// tracking
    bool track_on = false;
    bool uniform_motion_on = true;
    float max_valid_dist = 0.6;
    bool use_robust_kernel = true;
    bool GM_kernel_scale_adaptive = false;
    float GM_kernel_scale = max_valid_dist / 3.0;

    int reg_iter_n = 500;
    float reg_convergence_criterion = 0.0001;

    bool query_locally = true;

    ////////////////////// ReRun Visualizer
    bool rerun_viz_on = true;

    double world_axes_length = 5.0;
    double current_axes_length = 4.0;
    double point_radius = 0.02;

    // RGB colors as {R, G, B}
    std::array<int, 3> odometry_trajectory_color = {255, 165, 0};  // orange
    std::array<int, 3> slam_trajectory_color = {255, 0, 0};        // red
    std::array<int, 3> gt_trajectory_color = {0, 0, 255};          // blue

    ////////////////////// PGO
    bool pgo_on = false;

    // Methods
    void ReadFromYaml(const std::filesystem::path& yaml_file);
    void ConstFuncTest() const;
};

}  // namespace lo