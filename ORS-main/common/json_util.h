#pragma once
#include <json/json.h>
#include <butil/status.h>

namespace ors {

Json::Value LoadFromFile(std::string file_path);

butil::Status WriteToFile(const Json::Value& json_object, std::string file_path);

bool Validate(const Json::Value& root, std::vector<std::string> required);

template <typename T>
bool AssignIfValid(T& lhs, const Json::Value& value, const char* key) {
    if (!value[key].isNull()) {
        lhs = value[key].as<T>();
        return true;
    } else {
        return false;
    }
}

}