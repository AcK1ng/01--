#include "Cluster/Cluster.h"
#include "Master/Master.h"
#include <thread>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class ClusterWrapper {
public:
    ClusterWrapper(std::shared_ptr<Cluster> cluster,
             std::shared_ptr<Master> master):
        cluster_sys(std::make_shared<ClusterSys>(cluster, master)), master(master) {
        cluster_sys->StartUp();
    }

    void DEBUG() {
        this->master->DEBUG();
    }

    TaskID SubmitGraph(std::string root_graph,
            std::unordered_map<std::string, std::string> sub_graphs,
            std::unordered_map<int, std::string> symbol__to__symexpr,
            bool all_links_to_successor = false) {
        return this->master->SubmitGraph(root_graph, sub_graphs, symbol__to__symexpr, all_links_to_successor);
    }
    void DropTask(TaskID task_id) {
        this->master->DropTask(task_id);
    }
    std::vector<Future> RunTask(TaskID task_id,
                                        std::vector<Future> inputs,
                                        std::optional<std::vector<Rank>> manual_assignment,
                                        std::vector<size_t> debug_output_i) {
        return this->master->RunTask(task_id, inputs, manual_assignment, debug_output_i);
    }
    Future UploadTensor_ToCluster(std::vector<char> f, std::vector<long int> shape, int dtype) {
        return this->master->UploadTensor_ToCluster(f, shape, dtype);
    }
    void DropFuture(Future fut) {
        this->master->DropFuture(fut);
    }
    std::string ExportMasterStatus() {
        return this->master->ExportMasterStatus().dump(2);
    }
public:
    std::shared_ptr<Master> master;
private:
    std::shared_ptr<ClusterSys> cluster_sys;
};

PYBIND11_MODULE(PythonGCG, m) {
    m.def("GetNativeCPUCluster", &GetNativeCPUCluster);
    m.def("GetNativeCANNCPUCluster", &GetNativeCANNCPUCluster);
    m.def("GetNativeCUDACPUCluster", &GetNativeCUDACPUCluster);
    m.def("GetSimpleFullMaster", &GetSimpleFullMaster);

    py::class_<Master, std::shared_ptr<Master>>(m, "Master");
    py::class_<Cluster, std::shared_ptr<Cluster>>(m, "Cluster");
    py::class_<ClusterWrapper, std::shared_ptr<ClusterWrapper>>(m, "ClusterWrapper")
        .def(py::init<std::shared_ptr<Cluster>, std::shared_ptr<Master>>())
        .def("DEBUG", &ClusterWrapper::DEBUG)
        .def("SubmitGraph", &ClusterWrapper::SubmitGraph)
        .def("DropTask", &ClusterWrapper::DropTask)
        .def("RunTask", &ClusterWrapper::RunTask)
        .def("UploadTensor_ToCluster", [](ClusterWrapper &self, py::buffer buf, std::vector<long int> shape, int dtype) {
            py::buffer_info info = buf.request();
            if (info.ndim != 1) {
                throw std::runtime_error("Buffer must be one-dimensional");
            }

            auto* data_ptr = static_cast<char*>(info.ptr);
            size_t length = info.size * info.itemsize;
            std::vector<char> f(data_ptr, data_ptr + length);
            return self.master->UploadTensor_ToCluster(f, shape, dtype);
        })
        .def("DropFuture", &ClusterWrapper::DropFuture)
        .def("ExportMasterStatus", &ClusterWrapper::ExportMasterStatus);
}
