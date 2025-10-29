#include "dataset.hpp"

#include <algorithm>  // std::sort
#include <filesystem>
#include <iostream>
#include <stdexcept>

lo::Dataset::Dataset(const std::string& data_path, const std::string& sequence) {
    std::string velodyne_dir = data_path + "sequences/" + sequence + "/" + "velodyne/";
    if (!std::filesystem::exists(velodyne_dir)) {
        throw std::runtime_error("Velodyne folder not found: " + velodyne_dir);
    }

    for (const auto& entry : std::filesystem::directory_iterator(velodyne_dir)) {
        if (entry.path().extension() == ".bin") scan_files.push_back(entry.path().string());
    }

    std::sort(scan_files.begin(), scan_files.end());
    std::cout << "[KITTI Dataset] Loaded " << scan_files.size() << " scans from " << velodyne_dir << std::endl;
}

long int lo::Dataset::size() const {
    return scan_files.size();
}