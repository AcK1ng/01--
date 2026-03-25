#include "Base.h"

#include <iostream>

class HelloWorldMaster final: public Master {
public:
  virtual void StartUp() override {
      std::cout << "Hello World Master starts!" << std::endl;
  }

  virtual void SendWatchdogEvent(MasterEventEnum event,
          std::shared_ptr<MasterEventParamPayload> param_) override {
    if (event == WatchdogSignAccIn) {
      auto param = std::static_pointer_cast<struct WatchdogSignAccIn>(param_);
      auto op_param = std::make_shared<struct HelloWorld>();
      op_param->i = param->rank;

      this->ActionStart();
      this->IssueAction_ToCluster(param->rank, std::nullopt, HelloWorld, op_param,
              {}, std::nullopt, std::nullopt,
              0, 0, true, std::nullopt, false);
      this->ActionCommit();
    }
  }
};

std::shared_ptr<Master>
GetHelloWorldMaster() {
  return std::make_shared<HelloWorldMaster>();
}

