#include <iostream>

#include "dataset.hpp"  // includes config itself

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/config.yaml\n";
        return 1;
    }

    try {
        std::string config_path = argv[1];
        lo::Config cfg;
        cfg.FromFile(config_path);
        std::cout << "Frames: [" << cfg.begin_frame << ", " << cfg.end_frame << "]. Step: " << cfg.step_frame << "\n";
        std::cout << "Deskew: " << cfg.deskew << std::endl;
        std::cout << "Min range: " << cfg.min_range << "[m]. Max range: " << cfg.max_range << "[m]." << std::endl;
        std::cout << "Random downsample: " << cfg.rand_downsample << std::endl;
        std::cout << "Data path: " << cfg.data_path << std::endl;

        lo::Dataset dataset(cfg);
        lo::FrameData frame_data = dataset.get(0);
        std::cout << frame_data.points.innerSize() << " " << frame_data.points.outerSize() << " " << frame_data.point_ts.size() << std::endl
                  << std::endl;

        std::cout << dataset.last_odom_transformation << std::endl << std::endl;
        std::cout << dataset.odom_poses.size() << " " << dataset.travel_dist.size() << std::endl;

    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
