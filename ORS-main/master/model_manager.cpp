#include "master/model_manager.h"

namespace ors {

void ModelManager::add_model(scoped_refptr<Model> model) {
    std::unique_lock<std::mutex> lck(_mutex);
    _model_map[model->Id()] = model;
}

scoped_refptr<Model> ModelManager::get_model(const std::string& model_id) {
    std::unique_lock<std::mutex> lck(_mutex);
    auto it = _model_map.find(model_id);
    if (it != _model_map.end()) {
        return it->second;
    }
    return nullptr;
}

}