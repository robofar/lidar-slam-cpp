#include <Eigen/Core>
#include <string>

namespace lo {

class FrameData {
  public:
    Eigen::MatrixXd points;    // (N,4)
    Eigen::VectorXd point_ts;  // (N,)
};

class Dataset {
  public:
    explicit Dataset(const std::string&, const std::string&);

    long int size() const;
    FrameData get(long int idx) const;

  private:
    std::vector<std::string> scan_files;
    static Eigen::MatrixXd readPointCloud(const std::string& scan_file);
    static Eigen::VectorXd computePointTimestamps(const Eigen::MatrixXd& points);
};

}  // namespace lo