#include "vhm.hpp"

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
}

void lo::VoxelHashMap::Update(size_t frame_id, const Eigen::MatrixXd& points) {
    const Eigen::Index N = points.rows();
    for (Eigen::Index i = 0; i < N; i++) {
        Eigen::Vector3d p = points.row(i).head<3>().transpose();  // 3x1
        lo::VoxelKey vk = lo::VoxelHashMap::PointToVoxel(p, this->resolution);
        if (this->vhm[vk].size() < this->k_per_voxel) {
            this->vhm[vk].push_back(lo::PointData(p, frame_id));
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

    for (int x = -num_nei_cells; x <= num_nei_cells; x++) {
        for (int y = -num_nei_cells; y <= num_nei_cells; y++) {
            for (int z = -num_nei_cells; z <= num_nei_cells; z++) {
                const double dist2 = static_cast<double>(x * x + y * y + z * z);
                if (dist2 < radius2) {
                    this->neighbor_dx.emplace_back(VoxelKey{x, y, z});
                }
            }
        }
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> lo::VoxelHashMap::nearestNeighborSearch(const Eigen::MatrixXd& query_points,
                                                                                                 float max_valid_dist,
                                                                                                 bool use_neighb_voxels) {
    const Eigen::Index N = query_points.rows();
    const double max_valid_dist2 = max_valid_dist * max_valid_dist;

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> data_association_pairs;

    for (Eigen::Index i = 0; i < N; i++) {
        Eigen::Vector3d q_pt = query_points.row(i).head<3>().transpose();
        auto voxel = lo::VoxelHashMap::PointToVoxel(q_pt, this->resolution);

        double best_d2 = std::numeric_limits<double>::infinity();  // captured by reference in lambda
        const PointData* closest_pt = nullptr;  // has to be ptr rather than index, cuz of the neighboring voxels, so its easier to just remember
                                                // addres of closest point than some indices. No need for smart pointers here

        auto inspect_voxel = [&](const VoxelKey& key) {
            auto it = vhm.find(key);
            if (it == vhm.end()) return;  // out of the map
            const auto& bucket = it->second;
            for (size_t i = 0; i < bucket.size(); i++) {
                const double dist2 = (bucket.at(i).xyz - q_pt).squaredNorm();
                if (dist2 < best_d2) {
                    best_d2 = dist2;
                    closest_pt = &bucket.at(i);
                }
            }
        };

        if (use_neighb_voxels) {
            for (const VoxelKey& d : neighbor_dx) {
                VoxelKey g{voxel[0] + d[0], voxel[1] + d[1], voxel[2] + d[2]};
                inspect_voxel(g);
            }
        } else {
            inspect_voxel(voxel);
        }

        if (closest_pt && best_d2 <= max_valid_dist2) {
            data_association_pairs.emplace_back(q_pt, closest_pt->xyz);
        }
    }

    return data_association_pairs;
}