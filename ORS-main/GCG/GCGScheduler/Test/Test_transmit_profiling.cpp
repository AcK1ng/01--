#include "Base.h"

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <algorithm>
#include <list>
#include <float.h>

#include <torch/jit.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/constants.h>
#include <stdexcept>  
#include <ctime>
#include <random>

using AcceleratorModel = std::string;
using Time_Ns = int;

class SimpleModelForTransmit {
public:
    SimpleModelForTransmit(int warmup_step, double fix_factor)
        : warmup_step(warmup_step), fix_factor(fix_factor), times() {}
 
    void train(NBytes nbytes, Time_Ns time) {
        if (this->warmup_step > 0) {
            --this->warmup_step;
            return;
        }
        if (this->times.find(nbytes) == this->times.end()) {
            this->times[nbytes] = time;
        } else {
            Time_Ns old_time = this->times[nbytes];
            Time_Ns new_time = static_cast<Time_Ns>(this->fix_factor * old_time + (1 - this->fix_factor) * time);
            this->times[nbytes] = new_time;
        }
    }
 
    Time_Ns predict(NBytes nbytes) {
        if(this->times.find(nbytes) == this->times.end())
            return -1;
        return this->times[nbytes];
    }
 
    bool unknown_bw() {
        return this->times.empty();
    }

    double bw()  const {
        if (this->times.empty()) 
            return -1;
        double total = 0;
        for (auto pair : this->times) 
            total += static_cast<double>(pair.first) / pair.second;
        return total / this->times.size();
    }
 
    bool operator<=(std::shared_ptr<SimpleModelForTransmit> other) {
        return this->bw() < other->bw();
    }
 
private:
    int warmup_step;
    double fix_factor;
    std::unordered_map<NBytes, Time_Ns> times;
};

class TransmitPerformancePredictor {
public:
    void sign_in(Rank new_rank) {
        assert(this->ranks.find(new_rank) == this->ranks.end());
        for (const Rank& old_rank : this->ranks) {
            Rank small_rank = std::min(old_rank, new_rank);
            Rank big_rank = std::max(old_rank, new_rank);
            this->pair_models[std::make_tuple(small_rank, big_rank)] = new_model_for_pair();
        }
        this->ranks.insert(new_rank);
    }
 
    void sign_out(Rank old_rank) {
        assert(this->ranks.find(old_rank) != this->ranks.end());
        std::vector<RankPair> pair_to_delete;
        for(auto pair_to_model: this->pair_models) {
            RankPair rank_pair = pair_to_model.first;
            if (old_rank == std::get<0>(rank_pair) || old_rank == std::get<1>(rank_pair))
                pair_to_delete.push_back(rank_pair);
        }
        for(auto pair: pair_to_delete)
            this->pair_models.erase(pair);
        this->ranks.erase(old_rank);
    }
 
    void train(Rank one_peer, Rank another, const std::vector<std::tuple<NBytes, Time_Ns>>& pairs) {
        if (one_peer == another) 
            return;
        Rank small_rank = std::min(one_peer, another);
        Rank big_rank = std::max(one_peer, another);
        std::shared_ptr<SimpleModelForTransmit> model = this->pair_models[std::make_tuple(small_rank, big_rank)];
        this->model_train(model, pairs);
    }
 
    std::vector<Time_Ns> predict(Rank one_peer, Rank another, const std::vector<NBytes>& nbytes) {
        if (one_peer == another) 
            return std::vector<Time_Ns>(nbytes.size(), 0);
 
        Rank small_rank = std::min(one_peer, another);
        Rank big_rank = std::max(one_peer, another);
        std::shared_ptr<SimpleModelForTransmit> model = this->pair_models[std::make_tuple(small_rank, big_rank)];
        std::vector<Time_Ns> predictions;
        for (NBytes n : nbytes) {
            predictions.push_back(n == -1 ? 0 : this->model_predict(model, n));
        }
        return predictions;
    }
 
    std::vector<std::pair<Rank, double>> adjacent_peers(Rank one_peer) {
        std::vector<std::pair<Rank, double>> rank_to_bw;
        for (auto another : this->ranks) {
            if (another == one_peer) 
                continue;
            Rank small_rank = std::min(one_peer, another);
            Rank big_rank = std::max(one_peer, another);
            std::shared_ptr<SimpleModelForTransmit> model = this->pair_models[std::make_tuple(small_rank, big_rank)];
            rank_to_bw.push_back(std::make_pair(another, model->bw()));
        }
        rank_to_bw.push_back({one_peer, std::numeric_limits<double>::max()});

        // rank_to_bw.push_back(std::make_pair<one_peer, std::numeric_limits<double>::max()>);
        std::sort(rank_to_bw.begin(), rank_to_bw.end(), [](const auto& a, const auto& b) {
            return a.second > b.second; // 降序排序
        });
        return rank_to_bw;
    } 
 
protected:
    virtual std::shared_ptr<SimpleModelForTransmit> new_model_for_pair() {assert(0);};
    virtual void model_train(std::shared_ptr<SimpleModelForTransmit> model, const std::vector<std::tuple<NBytes, Time_Ns>>& pairs) {assert(0);};
    virtual Time_Ns model_predict(std::shared_ptr<SimpleModelForTransmit> model, NBytes nbytes) {assert(0);};
 
private:
    std::unordered_set<Rank> ranks;
    // TODO: 修改数据结构
    std::unordered_map<RankPair, std::shared_ptr<SimpleModelForTransmit>> pair_models;
};



class SimpleTransmitPerformancePredictor: public TransmitPerformancePredictor {
public:
    std::shared_ptr<SimpleModelForTransmit> new_model_for_pair() override {
        return std::make_shared<SimpleModelForTransmit>(3, 0.2);   
    }
    void model_train(std::shared_ptr<SimpleModelForTransmit> model, const std::vector<std::tuple<NBytes, Time_Ns>>& pairs) override {
        for(auto pair: pairs) {
            NBytes nbytes = std::get<0>(pair);
            Time_Ns time_ns = std::get<1>(pair);
            model->train(nbytes, time_ns);
        }
    }
    Time_Ns model_predict(std::shared_ptr<SimpleModelForTransmit> model, NBytes nbytes) override {
        return model->predict(nbytes);
    }
};
int main()
{
    std::shared_ptr<SimpleTransmitPerformancePredictor> transmit_predict = std::make_shared<SimpleTransmitPerformancePredictor>();
    std::shared_ptr<SimpleModelForTransmit> model = transmit_predict->new_model_for_pair();
    std::vector<std::tuple<NBytes, Time_Ns>> data;
    data.emplace_back(64, 1);
    for (size_t i = 0; i < 10; i++)
        transmit_predict->model_train(model, data);
    auto res = transmit_predict->model_predict(model, 64);
    std::cout << "Time_Ns:" << res << std::endl;
}