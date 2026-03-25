#include "TransmitManager.h"

std::shared_ptr<SimpleTransmitManager>
Get_SimpleTransmitManager() {
    return std::make_shared<SimpleTransmitManager>();
}

std::shared_ptr<SimpleTransmitManager>
Get_TransmitManager_FromJson(const nlohmann::json &j) {
    std::string type = j.at("type").get<std::string>();
    if (type == "SimpleTransmitManager") {
        auto manager = Get_SimpleTransmitManager();
        manager->FromJson(j);
        return manager;
    } else
        assert(0);
    return nullptr;
}