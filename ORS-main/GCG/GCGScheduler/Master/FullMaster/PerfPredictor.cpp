#include "PerfPredictor.h"


class SimpleTaskPerformancePredictor: public TaskPerformancePredictor {
public:
    SimpleTaskPerformancePredictor(int warmup_step, double fix_factor)
        : warmup_step(warmup_step), fix_factor(fix_factor) {}

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SimpleTaskPerformancePredictor, warmup_step, fix_factor)
    virtual nlohmann::json ToJson() const override {
        nlohmann::json j;
        to_json(j, *this);
        nlohmann::json parent = this->TaskPerformancePredictor::ToJson();
        j.insert(parent.begin(), parent.end());
        j["type"] = "SimpleTaskPerformancePredictor";
        return j;
    }
    virtual void FromJson(const nlohmann::json &j) {
        from_json(j, *this);
        this->TaskPerformancePredictor::FromJson(j);
    }
private:
    class SimpleModelForTask: public OperatorPerformanceModel {
    public:
        SimpleModelForTask(int warmup_step, double fix_factor)
            : warmup_step(warmup_step), fix_factor(fix_factor), time(-1) {}
     
        virtual void train(Duration time) override {
            if (this->warmup_step > 0) {
                this->warmup_step--;
                return;
            }

            if (this->time == -1) 
                this->time = time; 
            else 
                this->time = this->fix_factor * time + (1 - this->fix_factor) * this->time;
        }

        virtual Duration predict() override {
            if (this->time == -1)
                return 1000000000 /* 1s */;
            return this->time;
        }

    private:
        int warmup_step;
        double fix_factor;
        Duration time;

    public:
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(SimpleModelForTask, warmup_step, fix_factor, time)
        virtual nlohmann::json ToJson() const override {
            nlohmann::json j;
            to_json(j, *this);
            return j;
        }
    };

private:
    int warmup_step;
    double fix_factor;
    virtual std::shared_ptr<OperatorPerformanceModel> __new_model() override {
        return std::make_shared<SimpleModelForTask>(this->warmup_step, this->fix_factor);   
    }
    
    virtual std::shared_ptr<OperatorPerformanceModel>
    __get_model__from_json(const nlohmann::json& j) override {
        auto model = std::make_shared<SimpleModelForTask>(this->warmup_step, this->fix_factor);
        j.get_to(*model);
        return model;
    }
};

std::shared_ptr<TaskPerformancePredictor>
GetSimpleTaskPerformancePredictor(int warmup_step, double fix_factor) {
    return std::make_shared<SimpleTaskPerformancePredictor>(warmup_step, fix_factor);
}



std::shared_ptr<TaskPerformancePredictor>
GetTaskPerformancePredictorFromJson(const nlohmann::json& j) {
    std::string type = j.at("type").get<std::string>();
    if (type == "SimpleTaskPerformancePredictor") {
        auto predictor = std::make_shared<SimpleTaskPerformancePredictor>(0, 0.5);
        predictor->FromJson(j);
        return predictor;
    } else
        assert(0);
    return nullptr;
}




class LinearTransmitPerformancePredictor: public TransmitPerformancePredictor {
public:
    LinearTransmitPerformancePredictor(int warmup_step, double lr)
        : warmup_step(warmup_step), lr(lr) { }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(LinearTransmitPerformancePredictor, warmup_step, lr)
    virtual nlohmann::json ToJson() const override {
        nlohmann::json j;
        to_json(j, *this);
        nlohmann::json parent = this->TransmitPerformancePredictor::ToJson();
        j.insert(parent.begin(), parent.end());
        j["type"] = "LinearTransmitPerformancePredictor";
        return j;
    }
    virtual void FromJson(const nlohmann::json& j) override {
        from_json(j, *this);
        this->TransmitPerformancePredictor::FromJson(j);
    }
private:
    class LinearModelForTransmit: public TransmitrPerformanceModel {
    public:
        LinearModelForTransmit(int warmup_step, double lr)
            : warmup_step(warmup_step), lr(lr), _bw(1), c(0), constant(0) { }

        virtual void
        train(NBytes nbyte, Duration time) override {
            double MB = nbyte / (double)(1 << 20);
            double ms = time / (double)(1e6);

            if (this->warmup_step > 0) {
                if (MB < 64)
                    return;
                --this->warmup_step;
                this->_bw = MB / ms;
                return;
            }

            if (MB < 1) {
                this->constant = this->constant * (1 - this->lr) + ms * this->lr;
            } else if (64 <= MB) {
                // 回归模型 t(n) = n / _bw + c
                // 损失函数：(t(n) - t)^2 -> 0
                // 梯度：grad__bw = - 2 * (n / _bw^2) (t(n) - t)
                //         grad_c = 2 * (t(n) - t)
                double a = MB / this->_bw + this->c - ms;
                double grad_bw = - 2 * (MB / (this->_bw * this->_bw)) * a;
                double grad_c = 2 * a;

                this->_bw -= this->lr * grad_bw;
                this->c -= this->lr * grad_c;
            }

        }

        virtual Duration
        predict(NBytes nbytes) override {
            double MB = nbytes / (double)(1 << 20);
            if (MB < 1)
                return (Duration)(this->constant * 1e6);
            return (Duration)((MB / this->_bw + this->c) * 1e6);
        }

        virtual bool
        operator<(std::shared_ptr<TransmitrPerformanceModel> other_) override {
            auto other = std::static_pointer_cast<struct LinearModelForTransmit>(other_);
            return this->_bw < other->_bw;
        }

        virtual double
        bw() override {
            return this->_bw;
        }
        
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(LinearModelForTransmit, warmup_step, lr, _bw, c, constant)
        virtual nlohmann::json ToJson() const override {
            nlohmann::json j;
            to_json(j, *this);
            return j;
        }

    private:
        int warmup_step;
        double lr;
        double _bw /* MB per ms */, c; // 当64 <= MB，t(n) = n / _bw + c
        double constant/* ms */; // 当 MB < 1，t(n) = constant
    };
private:
    int warmup_step;
    double lr;
    virtual std::shared_ptr<TransmitrPerformanceModel>
    __new_model() override {
        return std::make_shared<LinearModelForTransmit>(this->warmup_step, this->lr);   
    }

    virtual std::shared_ptr<TransmitrPerformanceModel>
    __get_model__from_json(const nlohmann::json& j) {
        auto model = std::make_shared<LinearModelForTransmit>(this->warmup_step, this->lr);
        j.get_to(*model);
        return model;
    }
};

std::shared_ptr<TransmitPerformancePredictor>
GetLinearTransmitPerformancePredictor(int warmup_step, double lr) {
    return std::make_shared<LinearTransmitPerformancePredictor>(warmup_step, lr);
}


std::shared_ptr<TransmitPerformancePredictor>
GetTransmitPerformancePredictorFromJson(const nlohmann::json &j) {
    std::string type = j.at("type").get<std::string>();
    if (type == "LinearTransmitPerformancePredictor") {
        auto transmit_predictor = GetLinearTransmitPerformancePredictor(3, 0.01 /* dummy */);
        transmit_predictor->FromJson(j);
        return transmit_predictor;
    } else
        assert(0);
    return nullptr;
}