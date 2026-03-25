#include <torch/csrc/distributed/c10d/TCPStore.hpp>
//#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#define USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

static auto
init_TorchDist(std::string master_ip, size_t world_size, size_t rank) {
    std::cout << "I'm rank[" << rank << "] "
              << "MasterIP[" << master_ip << "] "
              << "WorldSize[" << world_size << "] "
              << std::endl;
    uint16_t port = 29500;
    bool isServer = (rank == 0);
    std::chrono::milliseconds timeout = std::chrono::milliseconds(10000);
    std::optional<std::size_t> numWorkers = world_size;
    bool waitWorkers = false;
    bool multiTenant = false;
    std::optional<int> masterListenFd = c10::nullopt;
    bool useLibUV = false;
    c10d::TCPStoreOptions opts{port,
        isServer,
        numWorkers,
        waitWorkers,
        timeout,
        multiTenant,
        masterListenFd,
        useLibUV};

    auto store = c10::make_intrusive<::c10d::TCPStore>(master_ip, opts);
    auto pg = std::make_shared<c10d::ProcessGroupNCCL>(store, rank, world_size);
    return pg;
}

int
main () {
    std::string master_ip = std::string("10.26.42.225");
    size_t world_size = 3;
    size_t rank = 2;
    auto pg = init_TorchDist(master_ip, world_size, rank);



    std::cout << "halo world_size " << world_size << " rank: " << rank << std::endl;


    while(1)
        std::this_thread::sleep_for(std::chrono::seconds(5));

    return 0;
}
