#include "Base.h"
#include "Master/Master.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <sstream>

#define CPPHTTPLIB_NO_EXCEPTIONS
#include <httplib.h>

#include <json.hpp>

using namespace httplib;

std::string dump_headers(const Headers &headers) {
    std::string s;
    char buf[BUFSIZ];

    for (auto it = headers.begin(); it != headers.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
        s += buf;
    }

    return s;
}

std::string log(const Request &req, const Response &res) {
    std::string s;
    char buf[BUFSIZ];

    s += "================================\n";

    snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
            req.version.c_str(), req.path.c_str());
    s += buf;

    std::string query;
    for (auto it = req.params.begin(); it != req.params.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%c%s=%s",
                (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
                x.second.c_str());
        query += buf;
    }
    snprintf(buf, sizeof(buf), "%s\n", query.c_str());
    s += buf;

    s += dump_headers(req.headers);

    s += "--------------------------------\n";

    snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
    s += buf;
    s += dump_headers(res.headers);
    s += "\n";

    if (!res.body.empty()) { s += res.body; }

    s += "\n";

    return s;
}

class HTTPedMaster;
static void
http_thread(HTTPedMaster *m, int port);

class HTTPedMaster final: public Master {
public:
    HTTPedMaster(std::shared_ptr<Master> m, int port):
        master_impl(m), port(port) { }
    ~HTTPedMaster() { this->master_impl = nullptr; }

    virtual void DEBUG() override {
        this->master_impl->DEBUG();
    }
        
    virtual void SetCluster(std::shared_ptr<Cluster> p) override {
        this->master_impl->SetCluster(p);
    }
    
    virtual void Clear() override {
        this->master_impl->Clear();
        this->master_impl = nullptr;
    }

    virtual void StartUp() override {
        if (this->master_impl)
            this->master_impl->StartUp();
        // 当master_impl为nullptr时，为自举模式，生成模拟器用

        std::thread t(http_thread, this, this->port);
        t.detach();
    }

    virtual void SendAccEvent(Rank from,
            std::optional<IssuingID> issuing_id,
            Timestamp acc_timestamp,
            MasterEventEnum e,
            std::shared_ptr<MasterEventParamPayload> param) override {
        this->master_impl->SendAccEvent(from, issuing_id, acc_timestamp, e, param);
    }

    virtual void SendWatchdogEvent(MasterEventEnum e,
            std::shared_ptr<MasterEventParamPayload> param) override {
        this->master_impl->SendWatchdogEvent(e, param);
    }

    virtual TaskID SubmitGraph(std::string root_graph,
            std::unordered_map<std::string, std::string> sub_graphs,
            std::unordered_map<int, std::string> symbol__to__symexpr,
            bool all_links_to_successor = false) override {
        return this->master_impl->SubmitGraph(root_graph, sub_graphs, symbol__to__symexpr, all_links_to_successor);
    }
    virtual void DropTask(TaskID task_id) override {
        this->master_impl->DropTask(task_id);
    }
    virtual std::vector<Future> RunTask(TaskID task_id,
                                        std::vector<Future> inputs,
                                        std::optional<std::vector<Rank>> manual_assignment,
                                        std::vector<size_t> debug_output_i) override {
        return this->master_impl->RunTask(task_id, inputs, manual_assignment, debug_output_i);
    }
    virtual Future UploadTensor_ToCluster(std::vector<char> f, std::vector<long int> shape, int dtype) override {
        return this->master_impl->UploadTensor_ToCluster(f, shape, dtype);
    }
    virtual void DropFuture(Future fut) override {
        this->master_impl->DropFuture(fut);
    }
    virtual nlohmann::json ExportMasterStatus() override {
        return this->master_impl->ExportMasterStatus();
    }

    void ConstructFullClusterSimulator(nlohmann::json master_status_j) {
        assert(this->master_impl == nullptr);
        Timestamp s = RealTimeNow();
        this->master_impl = Master::GetMasterFromJson(master_status_j);
        this->simulator = this->master_impl->GetFullClusterSimulator(this->master_impl);
        auto callback_for_action_done = [this](
            Rank r, std::shared_ptr<const AccActionSpec> a,
            Timestamp end, Duration duration) {
            this->nr_ops_simulated++;
            std::ostringstream oss;
            oss << std::setprecision(2)
                << "Timestamp[" << end << "] "
                << "Rank[" << r << "] "
                << "Action Done "
                << "Duration[" << duration << "] "
                << *a;
            std::cout << oss.str() << std::endl;
        };
        this->simulator->__register_action_done_callback(callback_for_action_done);
        Timestamp e = RealTimeNow();
        std::cout << "Simulator Construction Complete, "
                  << "spending " << ((double)(e - s) / 1000000) << "ms "
                  << std::endl;
    }

    void SimulatorAdvanceTime(Timestamp t) {
        this->nr_ops_simulated = 0;
        Timestamp s = RealTimeNow();
        this->simulator->AdvanceTime(t);
        Timestamp e = RealTimeNow();
        std::cout << "Simulator AdvanceTime[" << t << "] ns "
                  << this->nr_ops_simulated << " ops go beyound, "
                  << "spending " << ((double)(e - s) / 1000000) << "ms "
                  << std::endl;
    }

    void ClearSimulator() {
        this->master_impl = nullptr;
        this->simulator = nullptr;
    }

private:
    std::shared_ptr<Master> master_impl;
    int port;

    std::shared_ptr<Simulator> simulator;
    size_t nr_ops_simulated;
};


static void
http_thread(HTTPedMaster *m, int port) {
    Server svr;
    if (!svr.is_valid()) {
        printf("server has an error...\n");
        exit(-1);
    }

    svr.Post("/DEBUG", [&](const Request &req, Response &res) {
        m->DEBUG();
        });

    svr.Post("/submitgraph", [&](const Request &req, Response &res) {
            auto recv_json = nlohmann::json::parse(req.body);
            int task_id = m->SubmitGraph(
                        recv_json["graph"].get<std::string>(),
                        recv_json["submods"].get<std::unordered_map<std::string, std::string>>(),
                        recv_json["symbol__to__symexpr"].get<std::unordered_map<int, std::string>>(),
                        recv_json["all_links_to_successor"].get<bool>());
            nlohmann::json ret = task_id;
            res.set_content(ret.dump(), "text/plain");
            });

    svr.Post("/droptask", [&](const Request &req, Response &res) {
            auto recv_json = nlohmann::json::parse(req.body);
            m->DropTask(recv_json["task_id"].get<TaskID>());
            });

    svr.Post("/runtask", [&](const Request &req, Response &res) {
            auto recv_json = nlohmann::json::parse(req.body);
            nlohmann::json outputs;
            if (recv_json.contains("manual_assignment")) {
                outputs = m->RunTask(
                        recv_json["task_id"].get<TaskID>(),
                        recv_json["inputs"].get<std::vector<Future>>(),
                        recv_json["manual_assignment"].get<std::vector<Rank>>(),
                        recv_json["debug_output_i"].get<std::vector<size_t>>());
            } else {
                outputs = m->RunTask(
                        recv_json["task_id"].get<TaskID>(),
                        recv_json["inputs"].get<std::vector<Future>>(),
                        std::nullopt,
                        recv_json["debug_output_i"].get<std::vector<size_t>>());
            }
            res.set_content(outputs.dump(), "text/plain");
            });

    svr.Post("/uploadtensor", [&](const Request &req, Response &res) {
            auto recv_json = nlohmann::json::parse(req.body);
            nlohmann::json output = m->UploadTensor_ToCluster(recv_json["serialized_data"].get<std::vector<char>>(),
                                                              recv_json["shape"].get<std::vector<long int>>(),
                                                              recv_json["dtype"].get<int>());
            res.set_content(output.dump(), "text/plain");
            });

    svr.Post("/dropfuture", [&](const Request &req, Response &res) {
            auto recv_json = nlohmann::json::parse(req.body);
            m->DropFuture(recv_json["future"].get<Future>());
            });

    svr.Post("/masterstatus", [&](const Request &req, Response &res) {
            nlohmann::json status_json = m->ExportMasterStatus();
            res.set_content(status_json.dump(2), "text/plain");
            });

    svr.Post("/constructfullclustersimulator", [&](const Request &req, Response &res) {
            auto master_status_j = nlohmann::json::parse(req.body);
            m->ConstructFullClusterSimulator(master_status_j);
            });

    svr.Post("/simulatoradvancetime", [&](const Request &req, Response &res) {
            auto timestamp_j = nlohmann::json::parse(req.body);
            auto timestamp = timestamp_j.get<Timestamp>();
            m->SimulatorAdvanceTime(timestamp);
            });
            
    svr.Post("/clearsimulator", [&](const Request &req, Response &res) {
            m->ClearSimulator();
            });

// #ifndef NDEBUG
//     svr.set_logger([](const Request &req, const Response &res) {
//             printf("%s", log(req, res).c_str());
//             });
// #endif

    svr.set_exception_handler([](const auto& req, auto& res, std::exception_ptr ep) {
        auto fmt = "<h1>Error 500</h1><p>%s</p>";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(ep);
        } catch (std::exception &e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) { // See the following NOTE
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/html");
        res.status = StatusCode::InternalServerError_500;
    });

    svr.listen("localhost", port);
    exit(-1);
}


std::shared_ptr<Master>
WrapHTTPedMaster(std::shared_ptr<Master> m, int port) {
    return std::make_shared<HTTPedMaster>(m, port);
}
