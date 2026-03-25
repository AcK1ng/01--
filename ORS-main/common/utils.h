#pragma once
#include <string>
#include <algorithm>
#include <butil/guid.h>
#include <fstream>

namespace ors {

std::string uuid_gen();

template <typename DataType>
class DataProcessor
{
public:
    DataProcessor() { data_.reserve(kReserveSize); }
    ~DataProcessor() = default;

    void Add(const DataType& data)
    {
        sorted_ = false;
        sum_ += data;
        data_.push_back(data);
    }

    void Clear()
    {
        sorted_ = false;
        sum_ = 0;
        data_.clear();
    }

    void Organize()
    {
        if (sorted_) return;
        std::sort(data_.begin(), data_.end());
        sorted_ = true;
    }

    size_t Cnt() const
    {
        return data_.size();
    }

    DataType Sum() const
    {
        return sum_;
    }

    DataType Avg() const
    {
        if (data_.empty()) return 0;
        return sum_ / data_.size();
    }

    DataType Percentile(double p)
    {
        if (data_.empty()) return 0;

        this->Organize();
        size_t idx = p * data_.size();
        CHECK(idx <= data_.size()) << "Invalid percentile value: " << p;
        return data_[idx];
    }

    void SaveCDF(const std::string &file)
    {
        this->Organize();
        std::ofstream ofs(file);
        for (size_t i = 0; i < data_.size(); ++i) {
            ofs << i / (double)data_.size() << " " << data_[i] << '\n';
        }
        ofs.close();
    }

    static const size_t kReserveSize = 1024;

private:
    bool sorted_ = false;
    DataType sum_ = 0;
    std::vector<DataType> data_;
};

struct OpKey {
    std::string model_id;
    uint64_t op_id;

    bool operator==(const OpKey& key) const {
        return ((model_id == key.model_id) && 
                (op_id == key.op_id));
    }

    bool operator<(const OpKey& key) const {
        if (model_id != key.model_id)
            return model_id < key.model_id;
        return op_id < key.op_id;
    }
};

struct OpProfileData {
    std::string op_name;
    bool is_memory_intensive = true;
    int32_t expected_device;
    // device_id------------->latency
    std::unordered_map<int32_t, uint64_t> latency_profile;
};

struct OpHash {
    std::size_t operator()(const OpKey& p) const {
        size_t h = std::hash<std::string>()(p.model_id);
        h ^= std::hash<int64_t>()(p.op_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;  
    }
};


}