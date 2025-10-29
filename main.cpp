#include <iostream>

#include "config.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/config.yaml\n";
    return 1;
  }

  try {
    std::string config_path = argv[1];
    lo::Config cfg;
    cfg.FromFile(config_path);
    std::cout << "Frames: [" << cfg.begin_frame << ", " << cfg.end_frame
              << "]. Step: " << cfg.step_frame << "\n";
    std::cout << "Deskew: " << cfg.deskew << std::endl;
    std::cout << "Min range: " << cfg.min_range
              << "[m]. Max range: " << cfg.max_range << "[m]." << std::endl;
    std::cout << "Random downsample: " << cfg.rand_downsample << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Failed to load config: " << e.what() << "\n";
    return 2;
  }
  
  return 0;
}
