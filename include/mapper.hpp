#pragma once

#include "config.hpp"
#include "dataset.hpp"
#include "vhm.hpp"

namespace lo {
class Mapper {
  public:
    const lo::Config& config;
    lo::Dataset& dataset;
    lo::VoxelHashMap& map;

    std::vector<SE3> used_poses;

    // Constructors
    Mapper(const Config& cfg, Dataset& dataset, VoxelHashMap& vhm) : config(cfg), dataset(dataset), map(vhm) {}

    // Static Methods

    // Const Methods

    // Methods
    void determineMappingPoses(size_t frame_id);
    void processFrame(size_t frame_id, Eigen::MatrixXd point_cloud, SE3 cur_pose);
};
}  // namespace lo