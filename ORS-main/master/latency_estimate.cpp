#include "master/latency_estimate.h"
#include "common/json_util.h"

namespace ors {

butil::Status LatencyEstimator::Init(const ProfileConfig& config) {
    profile_data_path_ = config.profile_data_path;
    profile_database_json_ = LoadFromFile(config.profile_data_path);
    profile_database_ = JsonToProfile(profile_database_json_);
    return butil::Status::OK();
}

void LatencyEstimator::AddOpLatency(OpKey key, std::string op_name,
                                    int32_t device_id, int64_t latency_us) {
    // 如果 key 不存在，则创建新的项
    auto& profile = profile_database_[key];
    // 更新 device 对应的延迟
    profile.op_name = op_name;
    profile.latency_profile[device_id] = latency_us;
}

void LatencyEstimator::AddProfilingData(const OpKey& key,
                                        std::string op_name,
                                        int32_t device_id,
                                        uint64_t latency_us,
                                        bool is_memory_intensive)
{
    // 获取或创建 Key 对应的 ProfileData
    auto& profile = profile_database_[key];
    // 更新op_name名字
    profile.op_name = op_name;
    // 更新 memory intensive 属性
    profile.is_memory_intensive = (is_memory_intensive && 
        profile.is_memory_intensive);

    // 更新 device 的 latency（单位: us）
    profile.latency_profile[device_id] = latency_us;
}

OpProfileData LatencyEstimator::GetProfiledData(const OpKey& key) {
    return profile_database_[key];
}

butil::Status LatencyEstimator::DumpProfile() {
    return WriteToFile(ProfileToJson(), profile_data_path_);
}

std::unordered_map<OpKey, OpProfileData, OpHash> LatencyEstimator::JsonToProfile(Json::Value& root) {
    std::unordered_map<OpKey, OpProfileData, OpHash> profile_map;

    for (const auto& item : root) {
        if (!item.isObject() || !item.isMember("model_id") || !item.isMember("op_id"))
            continue;

        // 构造 key
        OpKey key;
        key.model_id = item["model_id"].asString();
        key.op_id = item["op_id"].asUInt64();

        // 构造 value
        OpProfileData value;
        value.op_name = item.get("op_name", "").asString();
        value.is_memory_intensive = item.get("is_memory_intensive", false).asBool();
        value.expected_device = item.get("expected_device", 0).asUInt();

        // 解析 device_latency
        if (item.isMember("device_latency") && item["device_latency"].isObject()) {
            const auto& dev_latency = item["device_latency"];
            for (const auto& dev_id_str : dev_latency.getMemberNames()) {
                uint32_t device_id = static_cast<uint32_t>(std::stoi(dev_id_str));
                uint64_t latency_us = dev_latency[dev_id_str].asUInt64(); // JSON 中已经是 us
                value.latency_profile[device_id] = latency_us;
            }
        }

        profile_map[key] = value;
    }

    return profile_map;
}

Json::Value LatencyEstimator::ProfileToJson() {
    Json::Value root(Json::arrayValue); // 最外层数组
    for (auto& pair : profile_database_) {
        const OpKey& key = pair.first;
        const OpProfileData& data = pair.second;

        Json::Value item(Json::objectValue);
        item["model_id"] = key.model_id;
        item["op_id"] = Json::UInt64(key.op_id);
        
        // operator name
        item["op_name"] = data.op_name;
        // is_memory_intensive
        item["is_memory_intensive"] = data.is_memory_intensive;
        // expected_device
        item["expected_device"] = data.expected_device;

        // device_latency
        Json::Value dev_latency(Json::objectValue);
        for (const auto& dev_kv : data.latency_profile) {
            dev_latency[std::to_string(dev_kv.first)] = Json::UInt64(dev_kv.second);
        }
        item["device_latency"] = dev_latency;
        // 加入数组
        root.append(item);
    }
    return root;
}


}