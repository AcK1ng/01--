#pragma once
#include <json/json.h>
#include <butil/status.h>
#include "common/utils.h"
#include <butil/memory/singleton.h>

namespace ors {

struct ProfileConfig {
  std::string profile_data_path = "data.profile";
};

class LatencyEstimator {
public:
    LatencyEstimator() {
      profile_data_path_ = "./data.profile";
    }
    ~LatencyEstimator() {}
    butil::Status Init(const ProfileConfig& config);
    void AddOpLatency(OpKey key, std::string op_name,
                      int32_t device_id, int64_t latency_us);
    void AddProfilingData(const OpKey& key,
                          std::string name,
                          int32_t device_id,
                          uint64_t latency_us,
                          bool is_memory_intensive);
    OpProfileData GetProfiledData(const OpKey& key);
    Json::Value ProfileToJson();
    std::unordered_map<OpKey, OpProfileData, OpHash> JsonToProfile(Json::Value& root);
    butil::Status DumpProfile();
    void ClearProfileData() {profile_database_.clear();}
private:
    std::string profile_data_path_;

    // The contents of the file at `profile_data_path_`.
    // We keep this separately from `profile_database_`, since we cannot
    // immediately put `profile_data_path_`'s contents into `profile_database_`
    // because the model name --> int mapping is not available at init time.
    Json::Value profile_database_json_;

    std::unordered_map<OpKey, OpProfileData, OpHash> profile_database_;
};  


}