#pragma once
#include <map>
#include <mutex>
#include <string>
#include "master/model.h"
#include <butil/memory/singleton.h>

namespace ors {

class ModelManager {
friend struct DefaultSingletonTraits<ModelManager>;
public:
    static ModelManager* GetInstance() {
        return Singleton<ModelManager>::get();
    }

    void add_model(scoped_refptr<Model> model);

    scoped_refptr<Model> get_model(const std::string& model_id);
private:
    ModelManager() {}
    ~ModelManager() {}
    
    std::map<std::string, scoped_refptr<Model>> _model_map;
    std::mutex _mutex;
};

#define g_model_manager ModelManager::GetInstance()

} // namespace ors