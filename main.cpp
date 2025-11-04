#include <algorithm>
#include <iostream>

#include "dataset.hpp"  // includes config itself
#include "kitti.hpp"
#include "mapper.hpp"
#include "tracker.hpp"
#include "utils.hpp"
#include "vhm.hpp"
#include "visualizer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/config.yaml\n";
        return 1;
    }

    try {
        std::string config_path = argv[1];
        lo::Config cfg;
        cfg.ReadFromYaml(config_path);

        lo::KITTIOdometryDataset kitti(cfg);
        lo::Dataset dataset(cfg, kitti);
        lo::VoxelHashMap vhm(cfg);
        lo::Mapper mapper(cfg, dataset, vhm);
        lo::Tracker tracker(cfg, vhm);
        lo::Visualizer visualizer(cfg);  // if cfg.rerun_vis_on=false it will not be instantiated

        for (size_t frame_id = 0; frame_id < dataset.total_pc_count; frame_id++) {
            std::cout << "Frame: " << frame_id << std::endl;

            // I. Load data ; Preprocess Pm and Pr ; Guess initial pose (constant velocity model)
            if (cfg.use_dataloader)
                dataset.readFrameWithLoader(frame_id);
            else
                throw std::runtime_error("For now, only dataloaders are supported");
            dataset.preprocessFrame(frame_id);
            dataset.initialPoseGuess(frame_id);

            // II. Odometry
            if (frame_id > 0) {
                if (cfg.track_on) {
                    throw std::runtime_error("Tracking mode is not yet implemented...");
                } else {  // incremental mapping with gt pose
                    if (dataset.gt_pose_provided) {
                        std::cout << "Mapping..." << std::endl;
                        dataset.UpdatePoses(frame_id, dataset.cur_pose_guess);
                    } else {
                        throw std::runtime_error("You are using the mapping mode, but no pose is provided.");
                    }
                }
            } else {
                dataset.UpdatePoses(frame_id, dataset.cur_pose_guess);
            }

            vhm.travel_dist = std::vector<double>(frame_id + 1);
            std::transform(dataset.travel_dist.begin(), dataset.travel_dist.begin() + static_cast<int>(frame_id + 1), vhm.travel_dist.begin(),
                           [](double dist) { return dist; });

            // III. PGO
            if (cfg.pgo_on) throw std::runtime_error("PGO not yet integrated...");

            // IV. Mapping
            // Update global map ; Reset local map ; Determine used poses for mapping
            mapper.processFrame(frame_id, dataset.cur_point_cloud, dataset.cur_pose);

            if (cfg.rerun_viz_on) {
                if (frame_id == 0) visualizer.log_world_frame(dataset.gt_poses.at(frame_id));
                if (!dataset.odom_poses.empty()) {
                    visualizer.log_current_odometry_frame(dataset.odom_poses.at(frame_id));
                    visualizer.log_odometry_positions(frame_id, dataset.odom_poses);
                    visualizer.log_odom_trajectory(frame_id, dataset.odom_poses);
                }
                if (!dataset.gt_poses.empty()) {
                    visualizer.log_current_gt_frame(dataset.gt_poses.at(frame_id));
                    visualizer.log_gt_positions(frame_id, dataset.gt_poses);
                    visualizer.log_gt_trajectory(frame_id, dataset.gt_poses);
                }
                // auto cur_point_cloud_visualizer = std::move(lo::transformPointCloud(dataset.cur_point_cloud, dataset.cur_pose));
                // visualizer.log_current_scan(cur_point_cloud_visualizer);
                visualizer.log_current_local_map(vhm.getFlattenedLocalMap());
                // visualizer.log_global_map(vhm.getFlattenedGlobalMap());
            }

            std::cout << "---------------" << std::endl;
        }

        // V. Evaluation
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
