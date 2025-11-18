#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "utils.hpp"

namespace lo {

// using VoxelKey = std::array<int, 3>;  // does not have defined has, but has defined operators <,>,==,!=,...

struct Array3iHasher {
    std::size_t operator()(const std::array<int, 3>& a) const {
        std::size_t h = 0;

        for (auto e : a) {
            h ^= std::hash<int>{}(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }

        return h;
    }
};

class PointData {
  public:
    Eigen::Vector3d xyz;
    int ts_create;
    int ts_update;

    PointData(const Eigen::Vector3d& xyz, int frame_id) : xyz(xyz), ts_create(frame_id), ts_update(frame_id) {}
};

// using VoxelMap = std::unordered_map<VoxelKey, std::vector<PointData>, Array3iHasher>;
using VoxelMap = std::unordered_map<VoxelKey, std::vector<PointData>>;  // std::hash<int64_t> is used automatically

struct NeighborOffset {
    int dx, dy, dz;
};

class VoxelHashMap {
  private:
  public:
    Config config;

    float resolution;
    int k_per_voxel;

    // Global map
    VoxelMap vhm;        // empty initially
    VoxelMap local_vhm;  // empty initially

    // Local Map setting
    bool temporal_local_map_on;
    bool local_map_travel_dist;
    float local_map_radius;
    float diff_travel_dist_local;
    int diff_ts_local;

    std::vector<double> travel_dist;  // set in dataset, but updated here from main, for purposes of setting local map (empty initially)

    // Search Neighborhood
    // std::vector<VoxelKey> neighbor_dx;
    std::vector<NeighborOffset> neighbor_dx;

    // Constructors
    VoxelHashMap(const Config& cfg);

    // Methods
    bool isEmpty() { return (this->vhm.size() == 0); }
    size_t count() { return this->vhm.size(); }

    void Update(size_t frame_id, const Eigen::Matrix3Xd& points);
    void resetLocalMap(size_t frame_id, const Eigen::Vector3d& sensor_position);
    void setSearchNeighborhood(int num_nei_cells = 1, float search_alpha = 1.0);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> nearestNeighborSearch(const Eigen::Matrix3Xd& query_points,
                                                                                   float max_valid_dist,
                                                                                   bool use_neighb_voxels = true,
                                                                                   bool search_in_local_map = false) const;

    // For ReRun visualization
    inline std::vector<Eigen::Vector3d> getFlattenedGlobalMap() {
        size_t total = 0;
        for (const auto& kv : this->vhm) total += kv.second.size();
        std::cout << "Total number of points in global map: " << total << std::endl;

        std::vector<Eigen::Vector3d> out;
        out.reserve(total);
        for (const auto& kv : this->vhm) {
            for (const auto& pd : kv.second) {
                out.push_back(pd.xyz);
            }
        }
        return out;
    }

    // For ReRun visualization
    inline std::vector<Eigen::Vector3d> getFlattenedLocalMap() {
        size_t total = 0;
        for (const auto& kv : this->local_vhm) total += kv.second.size();
        std::cout << "Total number of points in local map: " << total << std::endl;

        std::vector<Eigen::Vector3d> out;
        out.reserve(total);
        for (const auto& kv : this->local_vhm) {
            for (const auto& pd : kv.second) {
                out.push_back(pd.xyz);
            }
        }
        return out;
    }
};
}  // namespace lo