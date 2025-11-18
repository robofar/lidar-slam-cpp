#include "vhm.hpp"

#include <omp.h>

#include <iostream>
#include <limits>

lo::VoxelHashMap::VoxelHashMap(const Config& cfg) : config(cfg) {
    this->resolution = cfg.voxel_size_m;
    this->k_per_voxel = cfg.num_points_per_voxel;

    this->temporal_local_map_on = cfg.temporal_local_map_on;
    this->local_map_travel_dist = cfg.local_map_travel_dist;
    this->local_map_radius = cfg.local_map_radius;
    this->diff_travel_dist_local = (cfg.local_map_radius * cfg.local_map_travel_dist_ratio);
    this->diff_ts_local = cfg.diff_ts_local;

    this->setSearchNeighborhood();
}

void lo::VoxelHashMap::Update(size_t frame_id, const Eigen::Matrix3Xd& points) {
    const Eigen::Index N = points.cols();
    for (Eigen::Index i = 0; i < N; i++) {
        Eigen::Vector3d p = points.col(i);
        VoxelKey vk = PointToVoxel(p, this->resolution);

        auto& bucket = this->vhm[vk];
        if (bucket.size() < static_cast<size_t>(this->k_per_voxel)) {
            bucket.emplace_back(p, static_cast<int>(frame_id));
        }
    }

    /*
    auto it = this->vhm.begin();
    while (it != this->vhm.end()) {
        std::cout << "Entry with key: " << it->first[0] << " has this number of elements: " << it->second.size() << std::endl;
        it++;
    }
    */
}

void lo::VoxelHashMap::resetLocalMap(size_t frame_id, const Eigen::Vector3d& sensor_position) {
    this->local_vhm.clear();
    const double radius2 = this->local_map_radius * this->local_map_radius;
    for (auto it = this->vhm.begin(); it != this->vhm.end(); it++) {  // for every voxel
        std::vector<PointData> kept_voxel_points;
        for (const auto& pt : it->second) {  // for every point in a voxel
            // temporal mask
            bool time_ok = true;
            if (this->temporal_local_map_on) {
                if (this->local_map_travel_dist) {
                    const double d = std::abs((this->travel_dist)[frame_id] - (this->travel_dist)[pt.ts_create]);
                    time_ok = (d < this->diff_travel_dist_local);
                } else {
                    const int delta_t = std::abs(static_cast<int>(frame_id) - pt.ts_create);
                    time_ok = (delta_t < this->diff_ts_local);
                }
            }

            if (!time_ok) continue;

            // radius mask
            const double d2 = (pt.xyz - sensor_position).squaredNorm();
            if (d2 >= radius2) continue;

            kept_voxel_points.push_back(pt);
        }

        if (!kept_voxel_points.empty()) {
            this->local_vhm.emplace(it->first, std::move(kept_voxel_points));
        }
    }
}

void lo::VoxelHashMap::setSearchNeighborhood(int num_nei_cells, float search_alpha) {
    const double radius2 = std::pow(num_nei_cells + search_alpha, 2);
    neighbor_dx.clear();
    neighbor_dx.reserve((2 * num_nei_cells + 1) * (2 * num_nei_cells + 1) * (2 * num_nei_cells + 1));

    for (int x = -num_nei_cells; x <= num_nei_cells; x++) {
        for (int y = -num_nei_cells; y <= num_nei_cells; y++) {
            for (int z = -num_nei_cells; z <= num_nei_cells; z++) {
                const double dist2 = static_cast<double>(x * x + y * y + z * z);
                if (dist2 < radius2) {
                    neighbor_dx.push_back({x, y, z});
                }
            }
        }
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lo::VoxelHashMap::nearestNeighborSearch(const Eigen::Matrix3Xd& query_points,
                                                                                                 float max_valid_dist,
                                                                                                 bool use_neighb_voxels,
                                                                                                 bool search_in_local_map) const {
    const Eigen::Index N = query_points.cols();
    const double max_valid_dist2 = static_cast<double>(max_valid_dist) * max_valid_dist;

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> data_association_pairs;
    data_association_pairs.reserve(static_cast<size_t>(N));

    const VoxelMap& map = search_in_local_map ? local_vhm : vhm;

    for (Eigen::Index i = 0; i < N; i++) {
        const Eigen::Vector3d q_pt = query_points.col(i);

        // Base voxel index in integer form
        int ix = static_cast<int>(std::floor(q_pt.x() / resolution));
        int iy = static_cast<int>(std::floor(q_pt.y() / resolution));
        int iz = static_cast<int>(std::floor(q_pt.z() / resolution));
        const VoxelKey base_key = packVoxelIndex(ix, iy, iz);

        double best_d2 = std::numeric_limits<double>::infinity();
        const PointData* closest_pt = nullptr;

        auto inspect_voxel = [&](VoxelKey key) {
            auto it = map.find(key);
            if (it == map.end()) return;

            const auto& bucket = it->second;
            for (size_t j = 0; j < bucket.size(); ++j) {
                const double dist2 = (bucket[j].xyz - q_pt).squaredNorm();
                if (dist2 < best_d2) {
                    best_d2 = dist2;
                    closest_pt = &bucket[j];
                }
            }
        };

        if (use_neighb_voxels) {
            for (const auto& d : neighbor_dx) {
                VoxelKey k = packVoxelIndex(ix + d.dx, iy + d.dy, iz + d.dz);
                inspect_voxel(k);
            }
        } else {
            inspect_voxel(base_key);
        }

        if (closest_pt && best_d2 <= max_valid_dist2) {
            data_association_pairs.emplace_back(q_pt, closest_pt->xyz);
        }
    }

    return data_association_pairs;
}
