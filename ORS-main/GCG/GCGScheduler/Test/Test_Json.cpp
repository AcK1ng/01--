#include "Base.h"
#include <json.hpp>
#include <iostream>

struct myRunATenOP {
    std::vector<int> list;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(myRunATenOP, list);
};

int main () {
    struct myRunATenOP run_atenop1;
    run_atenop1.list.reserve(4);
    for (size_t i = 0; i < 4; i++)
        run_atenop1.list.push_back(i);
    nlohmann::json json1 = run_atenop1;
    std::cout << json1.dump() << std::endl;


    struct myRunATenOP run_atenop2;
    run_atenop2.list.reserve(4);
    for (size_t i = 0; i < 4; i++)
        run_atenop2.list.push_back(i);
    run_atenop2.list[2] = 54343;

    nlohmann::json json2 = run_atenop2;
    std::cout << json2.dump() << std::endl;


    std::string jsoned = "{\"node__to_transmit\":22,\"recv_host\":0,\"recv_rank\":1,\"recv_resource\":\"cuda\",\"send_host\":0,\"send_rank\":0,\"send_resource\":\"cuda\",\"transmit_id\":0}";
    auto j = nlohmann::json::parse(jsoned);
    auto transmit_info = j.get<struct TransmitInfo>();

    nlohmann::json json333 = transmit_info;

    std::cout << json333.dump() << std::endl;
    return 0;
}
