#include "common/utils.h"

namespace ors {

std::string uuid_gen() {
    std::string uuid_str = butil::GenerateGUID();
    std::transform(uuid_str.begin(), uuid_str.end(), uuid_str.begin(), ::tolower);
    return std::move(uuid_str);
}

}