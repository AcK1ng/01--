#include "Cluster/OOQueue/OOQueue.h"
#include <iostream>
#include <vector>




auto cmp (const std::shared_ptr<Action>& l, const std::shared_ptr<Action>& r) {
  return *(l.get()) < *(r.get());
};
int
main () {
  my_priority_queue<std::shared_ptr<Action>> priority_queue1(cmp);
  AccActionSpec spec;

  spec.acc_action_base.priority = 0;
  spec.acc_action_base.master_hint_sequence = 1;
  spec.acc_action_base.action_commit_id = 45;
  auto a = std::make_shared<Action>(std::nullopt, spec, nullptr, nullptr);
  priority_queue1.push(a);

  spec.acc_action_base.master_hint_sequence = 3;
  spec.acc_action_base.action_commit_id = 45;
  a = std::make_shared<Action>(std::nullopt, spec, nullptr, nullptr);
  priority_queue1.push(a);

  for (; !priority_queue1.empty(); priority_queue1.pop()) {
    auto action = priority_queue1.top();
    std::cout << *action << std::endl;
  }
  return 0;
}