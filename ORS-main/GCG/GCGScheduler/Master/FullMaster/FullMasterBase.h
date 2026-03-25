#ifndef FULL_MASTER_BASE_H
#define FULL_MASTER_BASE_H

#include "Base.h"
#include "PerfPredictor.h"

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <algorithm>
#include <list>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <random>

class AliveIDs {
public:
    AliveIDs() : current_id(0) {}

    size_t allocateID() {
        size_t newID = generateNewID();
        alive_ids.insert(newID);
        return newID;
    }

    bool contains(size_t id) {
        return this->alive_ids.contains(id);
    }

    void releaseID(size_t id) {
        if (alive_ids.find(id) == alive_ids.end()) {
            throw std::invalid_argument("ID not found or already released");
        }
        alive_ids.erase(id);
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(AliveIDs,
        alive_ids,
        current_id);

private:
    std::unordered_set<size_t> alive_ids;
    size_t current_id;

    size_t generateNewID() {
        while (true) {
            if (UINT_MAX < current_id) {
                current_id = 0;
                throw std::runtime_error("ID pool exhausted");
            }
            size_t newID = current_id++;
            if (alive_ids.find(newID) == alive_ids.end()) {
                return newID;
            }
        }
    }
};


#endif