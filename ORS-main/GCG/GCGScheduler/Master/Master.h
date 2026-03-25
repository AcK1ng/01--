#ifndef MASTER_H
#define MASTER_H

#include "Base.h"

#include <memory>

extern std::shared_ptr<Master>
GetHelloWorldMaster();

extern std::shared_ptr<Master>
GetSingleAccGraphMaster();

extern std::shared_ptr<Master>
GetTransmitTestMaster();

extern std::shared_ptr<Master>
GetMiniGCGMaster();

extern std::shared_ptr<Master>
GetSimpleFullMaster();

extern std::shared_ptr<Master>
GetSimpleFullMasterFromJson(const nlohmann::json &);

extern std::shared_ptr<Master>
WrapHTTPedMaster(std::shared_ptr<Master>, int port);

#endif
