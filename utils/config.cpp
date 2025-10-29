#include "config.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

void lo::Config::FromFile(const std::filesystem::path &yaml_file) {
  
  if (!std::filesystem::exists(yaml_file)) {
    std::cout << "Provided config YAML file does not exist... Default parameters are used..." << std::endl;
  } else {
    try {
        YAML::Node root = YAML::LoadFile(yaml_file.string());
        // lo::Config cfg;

        // Helper lambdas
        auto has = [](const YAML::Node &n, const char *k) -> bool {
          return n[k] && !n[k].IsNull();
        };
        auto get_or_bool = [&](const YAML::Node& n, const char* k, bool d) -> bool {
          return has(n,k) ? n[k].as<bool>() : d;
        };
        auto get_or_int = [&](const YAML::Node& n, const char* k, int d) -> int {
          return has(n,k) ? n[k].as<int>() : d;
        };
        auto get_or_str = [&](const YAML::Node &n, const char *k, const std::string& d) -> std::string {
          return has(n,k) ? n[k].as<std::string>() : d;
        };
        auto get_or_float = [&](const YAML::Node& n, const char* k, float d) -> float {
          return has(n,k) ? n[k].as<float>() : d;
        };
        auto get_or_path = [&](const YAML::Node& n, const char* k, const std::filesystem::path& d) -> std::filesystem::path {
          return has(n,k) ? std::filesystem::path(n[k].as<std::string>()) : d;
        };


        if (root["setting"]) {
          const YAML::Node s = root["setting"];

          this->name            = get_or_str(s, "name", this->name);
          this->run_name        = this->name;
          this->use_dataloader  = get_or_bool(s, "use_kiss_icp_dataloader", this->use_dataloader);
          this->data_loader_name= get_or_str(s, "data_loader_name", this->data_loader_name);
          this->data_loader_seq = get_or_str(s, "data_loader_seq", this->data_loader_seq);
          this->experiment_name = this->name + std::string("_") + this->data_loader_name + std::string("_") + this->data_loader_seq;

          this->output_root     = get_or_path(s, "output_root", this->output_root);
          this->data_path       = get_or_path(s, "data_path",   this->data_path);
          this->pose_path       = get_or_path(s, "pose_path",   this->pose_path);
          this->calib_path      = get_or_path(s, "calib_path",  this->calib_path);

          this->first_frame_ref = get_or_bool(s, "first_frame_ref", this->first_frame_ref);
          this->begin_frame     = get_or_int (s, "begin_frame",     this->begin_frame);
          this->end_frame       = get_or_int (s, "end_frame",       this->end_frame);
          this->step_frame      = get_or_int (s, "step_frame",      this->step_frame);
          this->stop_frame_thre =
              get_or_int(s, "stop_frame_thre", this->stop_frame_thre);

          this->kitti_correction_on = get_or_bool(s, "kitti_correction_on", this->kitti_correction_on);
          if (this->kitti_correction_on)
            this->correction_deg =
                get_or_float(s, "correction_deg", this->correction_deg);

          this->deskew = get_or_bool(s, "deskew", this->deskew);

          std::cout << "setting done" << std::endl;
          
        }

        if (root["process"]) {
            const YAML::Node s = root["process"];
            this->min_range = get_or_float(s, "min_range_m", this->min_range);
            this->max_range = get_or_float(s, "max_range_m", this->max_range);
            this->adaptive_range_on =
                get_or_bool(s, "adaptive_range_on", this->adaptive_range_on);

            this->min_z = get_or_float(s, "min_z_m", this->min_z);
            this->max_z = get_or_float(s, "max_z_m", this->max_z);

            this->rand_downsample = get_or_bool(s, "rand_downsample", this->rand_downsample);
            if (this->rand_downsample)
              this->rand_down_r =
                  get_or_float(s, "rand_down_r", this->rand_down_r);
            else
              this->vox_down_m =
                  get_or_float(s, "vox_down_m", this->vox_down_m);


            std::cout << "process done" << std::endl;

            
        }
    }
    catch (const std::exception& e) {
        throw; // just pass it to main
    }
  }

  
}