#include "common/json_util.h"
#include <sys/stat.h>
#include <fstream>
#include <glog/logging.h>

namespace ors {

inline bool FileExists(const std::string& name) {
    struct stat buffer;
    return stat(name.c_str(), &buffer) == 0;
}   

inline bool IsEmpty(std::ifstream& ifs) {
    return ifs.peek() == std::ifstream::traits_type::eof();
}

Json::Value LoadFromFile(std::string file_path) {
    if (!FileExists(file_path)) {
        LOG(WARNING) << "There is no such file " << file_path.c_str();
        return {};
    }
    std::ifstream in(file_path, std::ifstream::binary);
    if (IsEmpty(in)) {
        LOG(WARNING) << "File " << file_path << "is empty";
    }

    Json::Value json_object;
    in >> json_object;
    return json_object;
}

butil::Status WriteToFile(const Json::Value& json_object,
                          std::string file_path) {
    std::ofstream out_file(file_path, std::ios::out);
    if (!out_file.is_open()) {
        return butil::Status(EIO, "Cannot save profiled results");
    }
    out_file << json_object;
    return butil::Status::OK();
}

bool Validate(const Json::Value& root, std::vector<std::string> required) {
    if (root.isNull()) {
        LOG(ERROR) << "Please validate the json config file";
        return false;
    }
    for (auto key : required) {
        if (root[key].isNull()) {
            LOG(ERROR) << "Please check if the argument " << key
                       << " is given in the config file";
            return false;
        }
    }
    return true;
}


}