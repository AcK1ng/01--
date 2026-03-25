#ifndef GCG_H
#define GCG_H

#include "Base.h"
#include "PerfPredictor.h"
#include "Master/FullMaster/FullMasterBase.h"

#include <map>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <algorithm>
#include <list>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <random>
#include <forward_list>
#include <functional>
#include <iterator>
#include <json.hpp>

// forward declare
class RankManager;

enum GCGNode_Type {
    Node_For_Future = 0,
    Node_For_Task,
    Node_For_Op,
    Nr_Node_Type,
};

NLOHMANN_JSON_SERIALIZE_ENUM(GCGNode_Type, {
  {Node_For_Future, "Future"},
  {Node_For_Task, "Task"},
  {Node_For_Op, "Op"},
})

#define EMPTY_NODE_ID 0

struct NodePayload {
    ///////////////////////////////////////////////////
    bool debug;

    // Below for GCG
    NodeID ref;
    size_t reffed_cnt;
    bool lost_inputs;

    // 插入时间，用于调度算法
    Timestamp inserting_timestamp;

    bool is_scheduled;
    bool done;
    Timestamp done_timestamp;

    bool is_checkpoint;
    // 变成checkpoint的时间，OOM的时候可能用来排除
    Timestamp checkpoint_timestamp;

    enum GCGNode_Type node_type;

    ///////////////////////////////////////////////
    // Flags for Op
    Rank assigned_to; // 算子发射到哪里
    bool is_linked_to_successor; // 指这个算子要和其后继捆绑调度，通常用于const或者empty算子。总之单独调度不值当，而且会出现临时上下文分配的位置和计算位置不相等的麻烦
    bool is_linked_to_predecessor; // 指这个算子要和其前驱捆绑调度，通常用于tupleindex事实上不占用计算开销，但如果tupleindex的位置和前驱不同，还得非心思传输，何必呢？
    bool non_compute_op; // 指这个算子不占用计算资源，empty或TupleIndex，所以就不用训练预测器

    std::unordered_map<NodeID, Rank> checkpoint_src; // 如果input是checkpoint，这个input应该从哪里取
    size_t execution_sequence;
    IssuingID issuing_id;
    Timestamp issuing_timestamp;


    TaskID task_id;
    TaskNodeID task_node_id;

    bool shape_unknown;
    // NodePayload的拷贝构造函数就用默认的了；所以如果要动shape，尤其是要搞符号推理，这个shape要单独拷贝一份
    // 但是动shape的场合只是在unscheduled时，unscheduled node不会参与深拷贝，所以这个就不用搞深拷贝
    std::shared_ptr<VariableDescriptor> shape;

    bool is_constant;

    // for non constant
    bool has_tensor_payload;
    std::string aten_op_name;
    std::string target;

    // for constant or tensor payload
    struct MyConstantPayload const_payload;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(NodePayload,
        debug,
        ref, reffed_cnt, lost_inputs,
        inserting_timestamp, is_scheduled, done, done_timestamp,
        is_checkpoint, checkpoint_timestamp,
        node_type,
        assigned_to, is_linked_to_successor, is_linked_to_predecessor, non_compute_op,
        checkpoint_src, execution_sequence, issuing_id, issuing_timestamp,
        task_id, task_node_id, shape,
        is_constant,
        has_tensor_payload, aten_op_name, target,
        const_payload
    );
};


struct Node {
    NodeID node_id;
    std::vector<NodeID> inputs;
    std::unordered_set<NodeID> uses;
    struct Node *prev, *next;

    struct NodePayload _;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Node,
        node_id,
        inputs,
        uses,
        _);
};

class Graph {
public:
    Graph() {
        NodeID empty_node_id = this->node_id_pool.allocateID();
        assert(empty_node_id == EMPTY_NODE_ID);
    }

    Graph(const Graph &_g) {
        this->inputs = _g.inputs;
        this->outputs = _g.outputs;
        this->node_id_pool = _g.node_id_pool;
        this->size = _g.size;

        this->begin_node = nullptr;
        this->end_node = nullptr;
        for (auto &old_node: _g) {
            auto *new_node = new struct Node(old_node);
            this->node_id__to__node[old_node.node_id] = new_node;
            if (this->begin_node == nullptr) {
                this->begin_node = new_node;
                this->end_node = new_node;
                new_node->prev = nullptr;
                new_node->next = nullptr;
            } else {
                new_node->prev = this->end_node;
                this->end_node->next = new_node;
                new_node->next = nullptr;
                this->end_node = new_node;
            }
        }
    }

    Graph(Graph &&_g) {
        // 移动构造函数，先把_g的数据移过来
        this->inputs = _g.inputs;
        this->outputs = _g.outputs;
        this->node_id_pool = _g.node_id_pool;
        this->node_id__to__node = _g.node_id__to__node;
        this->begin_node = _g.begin_node;
        this->end_node = _g.end_node;
        this->size = _g.size;
        
        // 然后清空_g
        _g.node_id__to__node = {};
    }

    ~Graph() {
        for (auto pair: this->node_id__to__node) {
            NodeID node_id = pair.first;
            struct Node *node = pair.second;
            delete node;
        }
    }

    friend void
    to_json(nlohmann::json& j, const Graph& t) {
        j["inputs"] = t.inputs;
        j["outputs"] = t.outputs;
        j["node_id_pool"] = t.node_id_pool;
        j["size"] = t.size;
        j["nodes"] = nlohmann::json::array();
        for (auto &node: t)
            j["nodes"].push_back(node);
    }
    
    friend void
    from_json(const nlohmann::json& j, Graph& t) {
        j.at("inputs").get_to(t.inputs);
        j.at("outputs").get_to(t.outputs);
        j.at("node_id_pool").get_to(t.node_id_pool);
        j.at("size").get_to(t.size);
        
        t.begin_node = nullptr;
        t.end_node = nullptr;
        for (auto &node_j: j.at("nodes")) {
            struct Node t_node;
            node_j.get_to(t_node);
            struct Node *node = new struct Node(t_node);
            t.node_id__to__node[t_node.node_id] = node;
            if (t.begin_node == nullptr) {
                t.begin_node = node;
                t.end_node = node;
                node->prev = nullptr;
                node->next = nullptr;
            } else {
                node->prev = t.end_node;
                t.end_node->next = node;
                node->next = nullptr;
                t.end_node = node;
            }
        }
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg) const {
        dumped_gcg->node_id_pool = this->node_id_pool;
        dumped_gcg->size = nodes_will_be_reserved.size();
        
        dumped_gcg->begin_node = nullptr;
        dumped_gcg->end_node = nullptr;
        for (NodeID node_id: nodes_will_be_reserved) {
            auto *old_node = this->__get_node(node_id);
            auto *new_node = new struct Node(*old_node);
            dumped_gcg->node_id__to__node[node_id] = new_node;
            if (dumped_gcg->begin_node == nullptr) {
                dumped_gcg->begin_node = new_node;
                dumped_gcg->end_node = new_node;
                new_node->prev = nullptr;
                new_node->next = nullptr;
            } else {
                new_node->prev = dumped_gcg->end_node;
                dumped_gcg->end_node->next = new_node;
                new_node->next = nullptr;
                dumped_gcg->end_node = new_node;
            }
        }
        return dumped_gcg;
    }

    std::vector<NodeID> inputs, outputs;

private:
    AliveIDs node_id_pool;
    std::unordered_map<NodeID, struct Node *> node_id__to__node;
    struct Node *begin_node = nullptr, *end_node = nullptr;
    size_t size = 0;

public:
    template<bool IsConst>
    class Iterator {
    public:
        using T = std::conditional_t<IsConst, const struct Node, struct Node>;
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;
    public:
        Iterator(pointer node) : node(node) {}
        reference operator*() { return *node; }
        Iterator & operator++() { this->node = node->next; return *this; }
        Iterator & operator--() { this->node = node->prev; return *this; }
        bool operator==(const Iterator &other) const { return this->node == other.node; }
        bool operator!=(const Iterator &other) const { return this->node != other.node; }
    private:
        pointer node;
    };
    Iterator<0> begin() { return Iterator<0>(this->begin_node); }
    Iterator<0> end() { return Iterator<0>(nullptr); }
    Iterator<1> begin() const { return Iterator<1>(this->begin_node); }
    Iterator<1> end() const { return Iterator<1>(nullptr); }
    Iterator<1> cbegin() { return Iterator<1>(this->begin_node); }
    Iterator<1> cend() { return Iterator<1>(nullptr); }

public:
    inline size_t
    __size() const {
        // 注意，这里的size包含了input的长度
        return this->size;
    }

    inline bool
    __contains_node(NodeID node_id) const {
        return this->node_id__to__node.contains(node_id);
    }

    inline struct Node *
    __get_node(NodeID node_id) {
        return this->node_id__to__node.at(node_id);
    }
    inline const struct Node *
    __get_node(NodeID node_id) const {
        return this->node_id__to__node.at(node_id);
    }

    inline struct Node *
    __add_node(std::vector<NodeID> inputs) {
        NodeID node_id = this->node_id_pool.allocateID();
        struct Node *node = new struct Node;
        node->node_id = node_id;
        node->inputs = inputs;
        for (auto input_id: inputs) {
            struct Node *input_node = this->__get_node(input_id);
            input_node->uses.insert(node_id);
        }

        node->next = nullptr;
        if (this->size == 0) {
            node->prev = nullptr;
            this->begin_node = node;
            this->end_node = node;
        } else {
            node->prev = this->end_node;
            this->end_node->next = node;
        }
        this->end_node = node;

        this->node_id__to__node[node_id] = node;
        this->size++;

        return node;
    }


private:
    inline void
    __unlink_all_inputs__of_node(struct Node *node) {
        for (auto input_id: node->inputs) {
            struct Node *input_node = this->__get_node(input_id);
            input_node->uses.erase(node->node_id);
        }
    }

public:
    inline void
    __remove_all_inputs__of_node(NodeID node_id) {
        struct Node *node = this->__get_node(node_id);
        this->__unlink_all_inputs__of_node(node);
        node->inputs.clear();
    }

    inline void
    __del_node(NodeID node_id) {
        struct Node *node = this->__get_node(node_id);

        if (this->begin_node == node && this->end_node == node) {
            /* this->size == 1 */
            this->begin_node = nullptr;
            this->end_node = nullptr;
        } else if (this->begin_node == node && this->end_node != node) {
            this->begin_node = node->next;
            this->begin_node->prev = nullptr;
        } else if (this->begin_node != node && this->end_node == node) {
            this->end_node = node->prev;
            this->end_node->next = nullptr;
        } else if (this->begin_node != node && this->end_node != node) {
            node->prev->next = node->next;
            node->next->prev = node->prev;
        }

        this->size--;

        assert(node->uses.empty());
        this->__unlink_all_inputs__of_node(node);
        this->node_id_pool.releaseID(node->node_id);
        delete node;
        this->node_id__to__node.erase(node_id);
    }

    template<bool Reverse> inline void
    BFS(const std::unordered_set<NodeID> &start_nodes,
        std::function<bool (struct Node *)> visit,
        std::function<bool ()> stop_condition = []() { return false; }) {
        std::unordered_set<NodeID> visited;
        std::queue<NodeID> bfs_queue;
        for (auto node_id: start_nodes)
            bfs_queue.push(node_id);
        while (!bfs_queue.empty()) {
            auto node_id = bfs_queue.front();
            bfs_queue.pop();

            if (visited.contains(node_id))
                continue;

            struct Node *node = this->__get_node(node_id);
            bool stop_in_the_path = visit(node);
            visited.insert(node_id);

            if (stop_condition())
                return;

            if (!stop_in_the_path) {
                if constexpr (Reverse) {
                    for (auto input_id: node->inputs)
                        if (!visited.contains(input_id))
                            bfs_queue.push(input_id);
                } else {
                    for (auto use_id: node->uses)
                        if (!visited.contains(use_id))
                            bfs_queue.push(use_id);
                }
            }
        }
    }
};


class GCGBase: public Graph {
public:
    void
    __register_timer(std::function<Timestamp ()> timer) {
        this->Now = timer;
    }

    void
    __register_callback_for_node_insertion(
        enum GCGNode_Type node_type,
        std::function<void (struct Node *)> callback) {
        this->callbacks__for_node_insertion[node_type].push_back(callback);
    }

    void
    __register_callback_for_execute_state_transfer(
        enum GCGNode_Type node_type,
        std::function<void (struct Node *,
                            bool old_scheduled, bool old_done,
                            bool new_scheduled, bool new_done
                            )> callback) {
        this->callbacks__for_node_execute_state_trasfer[node_type].push_back(callback);
    }
    
    void
    __register_callback_for_storage_state_transfer(
        enum GCGNode_Type node_type,
        std::function<void (struct Node *,
                            bool old_checkpoint, bool new_checkpoint
                            )> callback) {
        this->callbacks__for_node_storage_state_trasfer[node_type].push_back(callback);
    }
    
    void
    __register_callback_for_input_lost(
        enum GCGNode_Type node_type,
        std::function<void (struct Node *)> callback) {
        this->callbacks__for_input_lost[node_type].push_back(callback);
    }

    void
    __register_callback_for_node_deletion(
        enum GCGNode_Type node_type,
        std::function<void (struct Node *)> callback) {
        this->callbacks__for_node_deletion[node_type].push_back(callback);
    }

public:
    GCGBase() {
        this->__register_timer(RealTimeNow);
    }

    GCGBase(const GCGBase &_old): Graph(_old) {
        this->__register_timer(RealTimeNow);
    }

    GCGBase(const nlohmann::json& j): Graph(j) {
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg) const override {
        auto ret = std::static_pointer_cast<GCGBase>(dumped_gcg);
        std::unordered_set<NodeID> done_inputs;
        for (NodeID node_id: nodes_will_be_reserved) {
            auto *node = this->__get_node(node_id);
            for (NodeID input_id: node->inputs) {
                auto *input = this->__get_node(input_id);
                if (input->_.done)
                    done_inputs.insert(input_id);
            }
        }
        nodes_will_be_reserved.merge(done_inputs);
        this->Graph::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, ret);
        return ret;
    }

    friend void
    to_json(nlohmann::json& j, const GCGBase& t) {
        nlohmann::to_json(j, static_cast<const Graph &>(t));
    }

protected:
    std::function<Timestamp ()> Now;

    std::vector<
        std::function<void (struct Node *)>> callbacks__for_node_insertion[Nr_Node_Type];
    std::vector<
        std::function<void (struct Node *,
                            bool old_scheduled, bool old_done,
                            bool new_scheduled, bool new_done
                            )>> callbacks__for_node_execute_state_trasfer[Nr_Node_Type];
    std::vector<
        std::function<void (struct Node *,
                            bool old_checkpoint, bool new_checkpoint
                            )>> callbacks__for_node_storage_state_trasfer[Nr_Node_Type];
    std::vector<
        std::function<void (struct Node *)>> callbacks__for_node_deletion[Nr_Node_Type];
    std::vector<
        std::function<void (struct Node *)>> callbacks__for_input_lost[Nr_Node_Type];

    inline struct Node *
    __insert_unscheduled_undone_noncheckpoint_node(std::vector<NodeID> inputs,
                                     enum GCGNode_Type node_type,
                                     NodeID ref = EMPTY_NODE_ID) {
        auto node = this->__add_node(inputs);
        if (ref != EMPTY_NODE_ID) {
            auto reffed_node = this->__get_node(ref);
            reffed_node->_.reffed_cnt++;
        }
        node->_.debug = false;
        node->_.ref = ref;
        node->_.reffed_cnt = 0;
        node->_.lost_inputs = false;
        node->_.inserting_timestamp = this->Now();
        node->_.is_scheduled = false;
        node->_.done = false;
        node->_.done_timestamp = 0;
        node->_.is_checkpoint = false;
        node->_.checkpoint_timestamp = 0;
        node->_.node_type = node_type;

        if (node_type != Node_For_Op) {
            node->_.assigned_to = 0;
            node->_.is_linked_to_successor = false;
            node->_.is_linked_to_predecessor = false;
            node->_.non_compute_op = true;
            node->_.execution_sequence = 0;
            node->_.issuing_id = 0;
            node->_.issuing_timestamp = 0;
            node->_.task_id = 0;
            node->_.task_node_id = 0;
            node->_.shape_unknown = true;
            node->_.is_constant = false;
            node->_.has_tensor_payload = false;
        }

        return node;
    }

    inline void
    __post_insert_node(struct Node *node) {
        // 在insert之后可能改变node的状态，
        // 因此insert node之后，先改变node中的flag，然后调用回调函数
        for (auto &callback: this->callbacks__for_node_insertion[node->_.node_type])
            callback(node);
    }

    inline void
    __mark_node_unscheduled(NodeID node_id) {
        assert(this->__contains_node(node_id));
        struct Node *node = this->__get_node(node_id);
        assert(node->_.is_scheduled);

        for (auto &callback: this->callbacks__for_node_execute_state_trasfer[node->_.node_type])
            callback(node,
                     true, node->_.done, /* 这里可能是从Done->Unschedule，容错情况 */
                     false, false);
        node->_.is_scheduled = false;
        node->_.done = false;
    }

    inline void
    __mark_node_scheduled_undone(NodeID node_id, IssuingID issuing_id) {
        assert(this->__contains_node(node_id));
        struct Node *node = this->__get_node(node_id);
        assert(node->_.is_scheduled == false);
        assert(node->_.done == false);

        node->_.issuing_id = issuing_id;
        node->_.issuing_timestamp = this->Now();
        for (auto &callback: this->callbacks__for_node_execute_state_trasfer[node->_.node_type])
            callback(node,
                     false, false,
                     true, false);
        node->_.is_scheduled = true;
        node->_.done = false;
    }

    inline void
    __mark_node_checkpoint(NodeID node_id) {
        assert(this->__contains_node(node_id));
        struct Node *node = this->__get_node(node_id);
        assert(node->_.is_checkpoint == false);
        for (auto &callback: this->callbacks__for_node_storage_state_trasfer[node->_.node_type])
            callback(node, node->_.is_checkpoint, true);
        node->_.is_checkpoint = true;
        node->_.checkpoint_timestamp = this->Now();
    }

    inline void
    __mark_node_non_checkpoint(NodeID node_id) {
        assert(this->__contains_node(node_id));
        struct Node *node = this->__get_node(node_id);
        assert(node->_.is_checkpoint == true);
        for (auto &callback: this->callbacks__for_node_storage_state_trasfer[node->_.node_type])
            callback(node, node->_.is_checkpoint, false);
        node->_.is_checkpoint = false;
    }

    inline void
    __delete_useless_node(std::shared_ptr<std::unordered_set<NodeID>> nodes_maybe_deleted) {

        // including nodes that may be effected as unused
        auto nodes_for_next_iteration = std::make_shared<std::unordered_set<NodeID>>();

        while (0 < nodes_maybe_deleted->size()) {
            // std::cout << "nodes_maybe_deleted[";
            // for (auto node_id: *nodes_maybe_deleted)
            //     std::cout << node_id << ",";
            // std::cout << "]" << std::endl;

            for (auto node_maybe_deleted: *nodes_maybe_deleted) {
                assert(this->__contains_node(node_maybe_deleted));
                auto node = this->__get_node(node_maybe_deleted);

                if (!node->_.done)
                    continue; // do not delete the un-done node

                // Here, the node is done

                if (!node->uses.empty() || node->_.reffed_cnt != 0)
                    continue; // do not delete the used node or "ref != 0"

                // Here, the node is "unused and done", let's delete it

                nodes_for_next_iteration->erase(node_maybe_deleted);
                for (auto input_id: node->inputs)
                    nodes_for_next_iteration->insert(input_id);
                if (node->_.ref != EMPTY_NODE_ID) {
                    auto ref_node = this->__get_node(node->_.ref);
                    ref_node->_.reffed_cnt--;
                    nodes_for_next_iteration->insert(node->_.ref);
                    node->_.ref = EMPTY_NODE_ID;
                }

                for (auto &callback: this->callbacks__for_node_deletion[node->_.node_type])
                    callback(node);

                this->__del_node(node_maybe_deleted);
            }
            nodes_maybe_deleted = nodes_for_next_iteration;
            nodes_for_next_iteration = std::make_shared<std::unordered_set<NodeID>>();
        }
    }

    inline void
    __disconnect_predecesor(NodeID node_id) {
        // 取消前驱
        struct Node *node = this->__get_node(node_id);

        // including nodes that may be deleted
        auto nodes_maybe_deleted = std::make_shared<std::unordered_set<NodeID>>();

        for (auto input_id: node->inputs)
            nodes_maybe_deleted->insert(input_id);

        for (auto &callback: this->callbacks__for_input_lost[node->_.node_type])
            callback(node) ;

        this->__remove_all_inputs__of_node(node_id);
        if (node->_.ref != EMPTY_NODE_ID) {
            auto ref_node = this->__get_node(node->_.ref);
            ref_node->_.reffed_cnt--;
            nodes_maybe_deleted->insert(node->_.ref);
            node->_.ref = EMPTY_NODE_ID;
        }
        node->_.lost_inputs = true;

        this->__delete_useless_node(nodes_maybe_deleted);
    }

    inline void
    __mark_node_done(NodeID done_node_id) {
        struct Node *done_node = this->__get_node(done_node_id);

        // including nodes that may be deleted
        auto nodes_maybe_deleted = std::make_shared<std::unordered_set<NodeID>>();

        if (done_node->_.done == false) {
            for (auto &callback: this->callbacks__for_node_execute_state_trasfer[done_node->_.node_type])
                callback(done_node,
                         done_node->_.is_scheduled, done_node->_.done, true, true);
            // 这里done_node->_.is_scheduled可能是false
            // 比如constant最开始加入GCG就是Unscheduled，但是我们直接给他搞成Done，没有Scheduled步骤
            done_node->_.is_scheduled = true;
            done_node->_.done = true;
            done_node->_.done_timestamp = this->Now();
    
            nodes_maybe_deleted->insert(done_node_id);
        }

        // Then delete "unused and done" nodes
        this->__delete_useless_node(nodes_maybe_deleted);
    }
};

class GCG_Op_Vertex_Cut_Manager: public GCGBase {
public:
    GCG_Op_Vertex_Cut_Manager() {
        this->__activate_this_component();
    }

    GCG_Op_Vertex_Cut_Manager(const GCG_Op_Vertex_Cut_Manager &_old): GCGBase(_old) {
        this->scheduling_vertex_cutting = _old.scheduling_vertex_cutting;
        this->source_nodes = _old.source_nodes;
        this->unscheduling_vertex_cutting = _old.unscheduling_vertex_cutting;
        this->__activate_this_component();
    }

    GCG_Op_Vertex_Cut_Manager(const nlohmann::json& j): GCGBase(j) {
        j.at("scheduling_vertex_cutting").get_to(this->scheduling_vertex_cutting);
        j.at("source_nodes").get_to(this->source_nodes);
        j.at("unscheduling_vertex_cutting").get_to(this->unscheduling_vertex_cutting);
        this->__activate_this_component();
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg) const override {
        auto ret = std::static_pointer_cast<GCG_Op_Vertex_Cut_Manager>(dumped_gcg);
        this->GCGBase::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, dumped_gcg);
        return dumped_gcg;
    }

    friend void
    to_json(nlohmann::json& j, const GCG_Op_Vertex_Cut_Manager& t) {
        nlohmann::to_json(j, static_cast<const GCGBase &>(t));
        j["scheduling_vertex_cutting"] = t.scheduling_vertex_cutting;
        j["source_nodes"] = t.source_nodes;
        j["unscheduling_vertex_cutting"] = t.unscheduling_vertex_cutting;
    }

private:
    void __activate_this_component() {
        auto may_become_scheduling_vertex_cutting =
            [this](struct Node *node,
                   // 下面的parent指的是，node的父节点将要改变scheduled和done状态。所以scheduled和done状态要看will_be_scheduled和will_be_done
                   NodeID parent = EMPTY_NODE_ID, bool will_be_scheduled = false, bool will_be_done = false) {
            if (!(node->_.is_scheduled == true && node->_.done == false))
                return;
            // node is Scheduled
            bool is_scheduling_vertex_cutting = true;
            for (auto input_id: node->inputs) {
                auto *input = this->__get_node(input_id);
                if ((input_id != parent && input->_.done == false)
                    || (input_id == parent && will_be_done == false)) {
                    // input is Unscheduled or Scheduled
                    is_scheduling_vertex_cutting = false;
                    break;
                }
            }
            if (is_scheduling_vertex_cutting)
                this->scheduling_vertex_cutting.insert(node->node_id);
        };

        auto may_become_unscheduling_vertex_cutting =
            [this](struct Node *node,
                   // 下面的parent指的是，node的父节点将要改变scheduled和done状态。所以scheduled和done状态要看will_be_scheduled和will_be_done
                   NodeID parent = EMPTY_NODE_ID, bool will_be_scheduled = false, bool will_be_done = false
                ) {
            if (!(node->_.is_scheduled == false))
                return;
            // node is Unsheduled
            bool is_unscheduling_vertex_cutting = true;
            for (auto input_id: node->inputs) {
                auto *input = this->__get_node(input_id);
                if ((input_id != parent && input->_.is_scheduled == false)
                    || (input_id == parent && will_be_scheduled == false)) {
                    // input is Unscheduled
                    is_unscheduling_vertex_cutting = false;
                    break;
                }
            }
            if (is_unscheduling_vertex_cutting) {
                this->unscheduling_vertex_cutting.insert(node->node_id);
            }
        };

        auto op_insertion =
            [may_become_scheduling_vertex_cutting,
             may_become_unscheduling_vertex_cutting,
             this] (struct Node *node) {
            assert(node->_.is_scheduled == false);
            if (node->inputs.empty())
                this->source_nodes.insert(node->node_id);
            may_become_unscheduling_vertex_cutting(node);
        };
        this->__register_callback_for_node_insertion(Node_For_Op, op_insertion);

        auto op_execute_state_transfer =
            [may_become_scheduling_vertex_cutting,
             may_become_unscheduling_vertex_cutting,
             this] (struct Node *node,
                    bool old_scheduled, bool old_done,
                    bool new_scheduled, bool new_done) {
            NodeID node_id = node->node_id;
            if (old_scheduled == false && new_scheduled == true && new_done == false) {
                // Unscheduled -> Scheduled

                this->unscheduling_vertex_cutting.erase(node_id);

                may_become_scheduling_vertex_cutting(node);

                // 该node的状态变化，可能让后继变为unscheduling点割
                for (auto use: node->uses) {
                    auto *user = this->__get_node(use);
                    may_become_unscheduling_vertex_cutting(user, node_id, new_scheduled, new_done);
                }
            } else if (old_scheduled == false && new_scheduled == true && new_done == true) {
                // Unscheduled -> Done
                this->unscheduling_vertex_cutting.erase(node_id);

                // 如果后继是unscheduled，该后继不会改变点割状态
                // 如果后继是scheduled，该后继可能成为scheduled点割
                for (auto use: node->uses) {
                    auto *user = this->__get_node(use);
                    may_become_scheduling_vertex_cutting(user, node_id, new_scheduled, new_done);
                }
            } else if (old_scheduled == true && old_done == false && new_scheduled == false) {
                // Scheduled -> Unscheduled
                this->scheduling_vertex_cutting.erase(node_id);
                may_become_unscheduling_vertex_cutting(node);

                // 该node的状态变化，可能让后继不是unscheduling点割
                for (auto use: node->uses)
                    this->unscheduling_vertex_cutting.erase(use);
            } else if (old_scheduled == true && old_done == true && new_scheduled == false) {
                // Done -> Unscheduled
                may_become_unscheduling_vertex_cutting(node);

                // 该node的状态变化，可能让后继不是unscheduling点割、也不是scheduling点割
                for (auto use: node->uses) {
                    this->scheduling_vertex_cutting.erase(use);
                    this->unscheduling_vertex_cutting.erase(use);
                }
            } else if (old_done == false && new_done == true) {
                // Scheduled -> Done
                this->scheduling_vertex_cutting.erase(node_id);

                // 如果后继是unscheduled，该后继不会改变点割状态
                // 如果后继是scheduled，该后继可能成为scheduled点割
                for (auto use: node->uses) {
                    auto *user = this->__get_node(use);
                    may_become_scheduling_vertex_cutting(user, node_id, new_scheduled, new_done);
                }
            }
        };
        this->__register_callback_for_execute_state_transfer(Node_For_Op, op_execute_state_transfer);

        auto op_deletion =
            [this] (struct Node *node) {
            this->source_nodes.erase(node->node_id);
            this->scheduling_vertex_cutting.erase(node->node_id);
            this->unscheduling_vertex_cutting.erase(node->node_id);
        };
        this->__register_callback_for_node_deletion(Node_For_Op, op_deletion);

        auto input_lost_op =
            [may_become_scheduling_vertex_cutting,
             may_become_unscheduling_vertex_cutting,
             this] (struct Node *node) {
            this->source_nodes.insert(node->node_id);
            may_become_unscheduling_vertex_cutting(node);
            may_become_scheduling_vertex_cutting(node);
        };
        this->__register_callback_for_input_lost(Node_For_Op, input_lost_op);
    }

public:
    // 这个模拟器会用，所以放在public
    std::unordered_set<NodeID> scheduling_vertex_cutting; // scheduled节点。要么前驱都是done；要么没前驱。

protected:
    std::unordered_set<NodeID> source_nodes; // 入度为0的node，指的是DAG的起点

    // 这个用来做调度
    std::unordered_set<NodeID> unscheduling_vertex_cutting; // unscheduled节点。要么前驱都是done或scheduled；要么没前驱。
};

class TaskManager {
public:
    std::unordered_map<TaskID,
                       std::tuple<std::shared_ptr<Graph>, // root torchscript
                                  std::unordered_map<std::string, std::string>, // target -> submod torchscript
                                  std::unordered_map<int, std::string> // symbol -> symexpr
                                  >> tasks;

    void
    add_task(TaskID task_id,
             std::shared_ptr<Graph> root_graph,
             std::unordered_map<std::string, std::string> sub_graphs,
             std::unordered_map<int, std::string> symbol__to__symexpr) {
        this->tasks[task_id] = {root_graph, sub_graphs, symbol__to__symexpr};
    }
    void
    drop_task(TaskID task_id) {
        this->tasks.erase(task_id);
    }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(TaskManager, tasks);
};

class GCG_Adding_TaskManagement: public GCG_Op_Vertex_Cut_Manager {
public:
    // 下面这个tasks放在public，因为发射算子的时候需要这个；但是发射算子的时候不应该改
    std::shared_ptr<TaskManager> task_manager;
protected:
    std::shared_ptr<TaskPerformancePredictor> task_predictor;
    TaskID dummy_task_id; // 空的task_id，主要为了Uploaded_Tensor占位用的

private:
    bool train_task_predictor;

    void __activate_this_component() {
        auto callback_for_task_node_deletion = [this] (struct Node *node) {
            TaskID task_id = node->node_id;
            this->task_manager->drop_task(task_id);
            this->task_predictor->drop_task(task_id);
        };
        this->__register_callback_for_node_deletion(Node_For_Task, callback_for_task_node_deletion);
    }

public:
    GCG_Adding_TaskManagement() {
        this->task_manager = std::make_shared<TaskManager>();
        this->task_predictor = GetSimpleTaskPerformancePredictor();
        auto *dummy_task_node = this->__insert_unscheduled_undone_noncheckpoint_node({}, Node_For_Task);
        this->__post_insert_node(dummy_task_node);
        this->dummy_task_id = dummy_task_node->node_id;
        this->task_predictor->new_task(this->dummy_task_id, 1);
        this->__activate_this_component();
    }

    GCG_Adding_TaskManagement(const GCG_Adding_TaskManagement &_old, bool need_task_manager): GCG_Op_Vertex_Cut_Manager(_old) {
        if (need_task_manager) {
            this->task_manager = std::make_shared<TaskManager>(*(_old.task_manager.get()));
            // this->task_predictor = _old.task_predictor;
            assert(0);
            this->SetTrainTaskPredictor(true);
            this->__activate_this_component();
        } else {
            this->task_manager = nullptr;
            this->task_predictor = nullptr;
            this->SetTrainTaskPredictor(false);
        }
        this->train_task_predictor = _old.train_task_predictor;
        this->dummy_task_id = _old.dummy_task_id;
    }

    GCG_Adding_TaskManagement(const nlohmann::json& j): GCG_Op_Vertex_Cut_Manager(j) {
        j.at("task_manager").get_to(this->task_manager);
        j.at("train_task_predictor").get_to(this->train_task_predictor);
        this->task_predictor = GetTaskPerformancePredictorFromJson(j.at("task_predictor"));
        j.at("dummy_task_id").get_to(this->dummy_task_id);
        this->__activate_this_component();
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg) const override {
        auto ret = std::static_pointer_cast<GCG_Adding_TaskManagement>(dumped_gcg);
        ret->SetTrainTaskPredictor(false);
        ret->dummy_task_id = this->dummy_task_id;
        this->GCG_Op_Vertex_Cut_Manager::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, dumped_gcg);
        return dumped_gcg;
    }

    friend void
    to_json(nlohmann::json& j, const GCG_Adding_TaskManagement& t) {
        nlohmann::to_json(j, static_cast<const GCG_Op_Vertex_Cut_Manager &>(t));
        j["task_manager"] = t.task_manager;
        j["train_task_predictor"] = t.train_task_predictor;
        j["task_predictor"] = t.task_predictor;
        j["dummy_task_id"] = t.dummy_task_id;
    }

public:
    static Graph
    GetGraphFrom(std::string ts_graph_str);

    inline TaskID
    SubmitGraph(const std::string &root_graph,
                const std::unordered_map<std::string, std::string> &sub_graphs,
                const std::unordered_map<int, std::string> &symbol__to__symexpr,
                bool all_links_to_successor = false) {
        auto *task_node = this->__insert_unscheduled_undone_noncheckpoint_node({}, Node_For_Task);
        this->__post_insert_node(task_node);

        TaskID task_id = task_node->node_id;

        auto graph = std::make_shared<Graph>(GetGraphFrom(root_graph));

        TaskNodeID task_node_id = 0;
        for (auto &node: *graph) {
            node._.assigned_to = 0;
            if (all_links_to_successor) {
                node._.is_linked_to_successor = true;
                node._.is_linked_to_predecessor = false;
                node._.non_compute_op = true;
            } else {
                if (node._.is_constant) {
                    node._.is_linked_to_successor = true;
                    node._.is_linked_to_predecessor = false;
                    node._.non_compute_op = true;
                } else if (node._.aten_op_name == "prim::TupleIndex") {
                    node._.is_linked_to_successor = false;
                    node._.is_linked_to_predecessor = true;
                    node._.non_compute_op = true;
                } else {
                    node._.is_linked_to_successor = false;
                    node._.is_linked_to_predecessor = false;
                    node._.non_compute_op = false;
                }
            }
            node._.execution_sequence = 0;
            node._.issuing_id = 0;
            node._.issuing_timestamp = 0;
            node._.task_id = task_id;
            node._.task_node_id = task_node_id++;
            node._.shape_unknown = false;
        }

        this->task_manager->add_task(task_id, graph, sub_graphs, symbol__to__symexpr);
        this->task_predictor->new_task(task_id, task_node_id);
        return task_id;
    }

    void
    SetTrainTaskPredictor(bool train) {
        this->train_task_predictor = train;
    }

    inline void
    DropTask(TaskID task_id) {
        this->__mark_node_done(task_id);
    }

    Duration
    PredictTask(TaskID task_id, TaskNodeID task_node_id, AccModel model) {
        return this->task_predictor->predict(task_id, task_node_id, model);
    }
    
protected:
    inline void
    TrainTaskPredictor(TaskID task_id, TaskNodeID task_node_id, AccModel model, Duration duration) {
        if (this->train_task_predictor)
           this->task_predictor->train(task_id, task_node_id, model, duration);
    }
};

class GCG_Adding_CheckpointManagement: public GCG_Adding_TaskManagement {
private:
    std::unordered_map<Rank, std::unordered_set<NodeID>> rank_checkpoints; // 这里的是，每个rank上都有哪些checkpoint

    std::vector<
        std::function<void (NodeID, Rank, bool add)>> callbacks__for__checkpoint_placement_change;

    // checkpoint的所在位置，调度算法会需要，所以这两个放在protected
    // checkpoint_prelocations 包含 checkpoint_locations
public:
    // 这个属性放在public，在发射算子的时候需要这个信息；但是发射算子的时候不应该改
    std::unordered_map<NodeID, std::unordered_set<Rank>> checkpoint_prelocations; // 这里的checkpoint是，决定在哪里放checkpoint
    
    // 这个属性放在public，生成模拟器的时候需要这个信息
    std::unordered_map<NodeID, std::unordered_set<Rank>> checkpoint_locations; // 这里的checkpoint是，收到汇报的确切checkpoint位置
protected:
    std::unordered_set<NodeID> persistent_checkpoints;

public:
    // 这个函数放在public，在调度的时候需要设定把checkpoint放在哪里
    inline void
    __place_checkpoint_to_rank(NodeID node_id, Rank rank) {
        assert(this->checkpoint_prelocations.contains(node_id));
        for (auto &callback: this->callbacks__for__checkpoint_placement_change)
            callback(node_id, rank, true);
        this->rank_checkpoints.at(rank).insert(node_id);
        this->checkpoint_prelocations.at(node_id).insert(rank);
    }

protected:
    inline void
    __withdraw_checkpoint_from_rank(NodeID node_id, Rank rank) {
        assert(this->checkpoint_prelocations.contains(node_id));
        for (auto &callback: this->callbacks__for__checkpoint_placement_change)
            callback(node_id, rank, false);
        this->rank_checkpoints.at(rank).erase(node_id);
        this->checkpoint_prelocations.at(node_id).erase(rank);
        this->checkpoint_locations.at(node_id).erase(rank);
    }

    inline bool
    __is_checkpoint__ready_in_cluster(NodeID node_id) {
        return this->checkpoint_locations.at(node_id).empty() == false;
    }

    inline void
    SettleOpAsCheckpoint(NodeID node_id) {
        this->__mark_node_checkpoint(node_id);
    }
    
    inline void
    DiscardCheckpoint(NodeID node_id) {
        auto *node = this->__get_node(node_id);

        // 我们规定future一定要是checkpoint，所以future不能取消checkpoint
        assert(!this->__is_op_future(node_id));

        // 决定持久化的checkpoint因为已经取消了前缀，所以不能重新恢复数据
        // 所以不能从checkpoint中撤回去
        assert(!node->_.lost_inputs);

        this->__mark_node_non_checkpoint(node_id);
    }

    inline bool
    __is_op_future(NodeID node_id) {
        auto *node = this->__get_node(node_id);
        if (node->_.is_checkpoint && node->_.node_type == Node_For_Op && 0 < node->_.reffed_cnt)
            return true;
        return false;
    }

    virtual void
    AccSignIn(Rank rank) {
        this->rank_checkpoints[rank] = {};
    }

    virtual void
    AccSignOut(Rank rank) {
        for (auto node_id: this->rank_checkpoints.at(rank))
            this->__withdraw_checkpoint_from_rank(node_id, rank);
        this->rank_checkpoints.erase(rank);
    }

private:
    void
    __activate_this_component() {
        auto __register_checkpoint = [this](NodeID node_id) {
            this->checkpoint_prelocations[node_id] = {};
            this->checkpoint_locations[node_id] = {};
        };

        auto __discard_checkpoint = [this](NodeID node_id) {
            // 这里先把set拷贝一份，因为下面__withdraw_checkpoint_from_rank会修改preloc，所以不能直接迭代preloc
            const auto locs = this->checkpoint_prelocations.at(node_id);
            for (auto rank: locs)
                this->__withdraw_checkpoint_from_rank(node_id, rank);
            this->checkpoint_prelocations.erase(node_id);
            this->checkpoint_locations.erase(node_id);
            this->persistent_checkpoints.erase(node_id);
        };

        auto callback_op_insertion =
            [__register_checkpoint, __discard_checkpoint](struct Node *node) {
            if (node->_.is_checkpoint)
                __register_checkpoint(node->node_id);
        };
        auto callback_op_storage_state_transfer =
            [__register_checkpoint, __discard_checkpoint](struct Node *node, bool old_checkpoint, bool new_checkpoint) {
            if (old_checkpoint == true && new_checkpoint == false)
                __discard_checkpoint(node->node_id);
            else if (old_checkpoint == false && new_checkpoint == true)
                __register_checkpoint(node->node_id);
        };
        auto callback_op_deletion =
            [__register_checkpoint, __discard_checkpoint](struct Node *node) {
            if (node->_.is_checkpoint)
                __discard_checkpoint(node->node_id);
        };

        
        auto callback_op_mark_done = [this](struct Node *node,
                                            bool old_scheduled, bool old_done,
                                            bool new_scheduled, bool new_done) {
            if (!old_done && new_done) {
                if (!node->_.is_constant && node->_.has_tensor_payload) {
                    node->_.has_tensor_payload = false;
                    node->_.const_payload.serialized_data = nullptr;
                }
            }
        };

        this->__register_callback_for_node_insertion(Node_For_Op, callback_op_insertion);
        this->__register_callback_for_storage_state_transfer(Node_For_Op, callback_op_storage_state_transfer);
        this->__register_callback_for_execute_state_transfer(Node_For_Op, callback_op_mark_done);
        this->__register_callback_for_node_deletion(Node_For_Op, callback_op_deletion);
    }

public:
    GCG_Adding_CheckpointManagement() {
        this->__activate_this_component();
    }

    GCG_Adding_CheckpointManagement(
        const GCG_Adding_CheckpointManagement &_old,
        bool need_task_manager): GCG_Adding_TaskManagement(_old, need_task_manager) {
        this->rank_checkpoints = _old.rank_checkpoints;
        this->checkpoint_prelocations = _old.checkpoint_prelocations;
        this->checkpoint_locations = _old.checkpoint_locations;
        this->persistent_checkpoints = _old.persistent_checkpoints;
        this->__activate_this_component();
    }

    GCG_Adding_CheckpointManagement(const nlohmann::json& j): GCG_Adding_TaskManagement(j) {
        j.at("rank_checkpoints").get_to(this->rank_checkpoints);
        j.at("checkpoint_prelocations").get_to(this->checkpoint_prelocations);
        j.at("checkpoint_locations").get_to(this->checkpoint_locations);
        j.at("persistent_checkpoints").get_to(this->persistent_checkpoints);
        this->__activate_this_component();
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg) const override {
        auto ret = std::static_pointer_cast<GCG_Adding_CheckpointManagement>(dumped_gcg);

        ret->rank_checkpoints = this->rank_checkpoints;
        ret->checkpoint_prelocations = this->checkpoint_prelocations;
        ret->checkpoint_locations = this->checkpoint_locations;
        ret->persistent_checkpoints = this->persistent_checkpoints;
        ret->__activate_this_component();

        for (auto &p: this->checkpoint_prelocations) {
            NodeID ckpt_id = p.first;
            nodes_will_be_reserved.insert(ckpt_id);
        }

        this->GCG_Adding_TaskManagement::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, dumped_gcg);
        return ret;
    }

    friend void
    to_json(nlohmann::json& j, const GCG_Adding_CheckpointManagement& t) {
        nlohmann::to_json(j, static_cast<const GCG_Adding_TaskManagement &>(t));
        j["rank_checkpoints"] = t.rank_checkpoints;
        j["checkpoint_prelocations"] = t.checkpoint_prelocations;
        j["checkpoint_locations"] = t.checkpoint_locations;
        j["persistent_checkpoints"] = t.persistent_checkpoints;
    }

    inline void
    __register_callback__for__checkpoint_placement_change(
        std::function<void (NodeID, Rank, bool add)> callback) {
        this->callbacks__for__checkpoint_placement_change.push_back(callback);
    }

private:
    void
    __init_cached_symbol_value__for_shape_propagation(
        std::vector<NodeID> inputs,
        std::shared_ptr<Graph> root,
        std::unordered_map<std::string, size_t> &cached_symbol_id_value,
        std::unordered_map<int, size_t> &cached_symbol_value,
        const std::unordered_map<int, std::string> &symbol__to__symexpr);

    void
    __propagate_shape__from_cached_symbol_value(
        std::shared_ptr<VariableDescriptor> incomplete_shape,
        const std::unordered_map<std::string, size_t> &cached_symbol_id_value,
        std::unordered_map<int, size_t> &cached_symbol_value,
        const std::unordered_map<int, std::string> &symbol__to__symexpr);

public:
    /* 删除冗余的checkpoint，应对OOM */
    void
    Remove_Redundant_Checkpoints(Rank rank) {
        assert(0);
    }

    inline Future
    InsertUnscheduledConstantTensor(std::vector<char> serialized_data,
                                    std::shared_ptr<VariableDescriptor> shape) {
        auto const_payload = MyConstantPayload(serialized_data);
        if (!const_payload.is_tensor)
            return EMPTY_NODE_ID;

        auto *node = this->__insert_unscheduled_undone_noncheckpoint_node({}, Node_For_Op);
        auto node_id = node->node_id;

        node->_.assigned_to = 0;
        node->_.is_linked_to_successor = true;
        node->_.is_linked_to_predecessor = false;
        node->_.non_compute_op = true;
        node->_.execution_sequence = 0;
        node->_.task_id = dummy_task_id;
        node->_.task_node_id = 0;
        node->_.shape_unknown = false;
        node->_.shape = shape;
        node->_.is_constant = false;
        node->_.has_tensor_payload = true;
        node->_.const_payload = const_payload;

        this->__post_insert_node(node);

        auto *fut_node = this->__insert_unscheduled_undone_noncheckpoint_node({}, Node_For_Future, node_id);
        this->__post_insert_node(fut_node);

        Future ret = fut_node->node_id;

#ifndef NDEBUG
        nlohmann::json jsoned_node = *node;
        jsoned_node.at("_").erase("shape");
        jsoned_node.at("_").erase("const_payload");
        std::cout << "InsertOP: " << jsoned_node.dump() << std::endl;
#endif

        this->__mark_node_checkpoint(node_id);
        return ret;
    }

    /* 追加Unscheduled Op，可能改动调度边界 */
    inline std::vector<Future>
    InsertUnscheduledTask(TaskID task_id,
                          const std::vector<Future> &input_futures,
                          std::optional<std::vector<Rank>> &manual_assignment,
                          const std::vector<size_t> &debug_output_i) {
        std::vector<NodeID> inputs;
        for (auto fut: input_futures)
            inputs.push_back(this->__get_node(fut)->_.ref);

        auto __insert_task =
            [&, this] (std::vector<NodeID> graph_inputs, TaskID task_id) -> std::vector<NodeID> {

            auto &[_sub_graph, _, symbol__to__symexpr] = this->task_manager->tasks.at(task_id);

            // for shape propagation
            std::unordered_map<std::string, size_t> cached_symbol_id_value;
            std::unordered_map<int, size_t> cached_symbol_value; 

            this->__init_cached_symbol_value__for_shape_propagation(
                graph_inputs, _sub_graph,
                cached_symbol_id_value, cached_symbol_value,
                symbol__to__symexpr);

            const Graph &sub_graph = *_sub_graph;
            assert(graph_inputs.size() == sub_graph.inputs.size());

            std::unordered_map<NodeID, NodeID> sub_graph_node_id__to__node_id;

            for (size_t i = 0; i < graph_inputs.size(); i++)
                sub_graph_node_id__to__node_id[sub_graph.inputs[i]] = graph_inputs[i];

#ifndef NDEBUG
            std::vector<NodeID> new_nodes_ids;
#endif
            std::unordered_set<NodeID> constants_in_GCG;
            std::unordered_set<NodeID> sub_graph_input_ids;
            sub_graph_input_ids.insert(sub_graph.inputs.begin(), sub_graph.inputs.end());
            std::vector<NodeID> t;

            for (auto &sub_graph_node: sub_graph) {
                if (sub_graph_input_ids.contains(sub_graph_node.node_id))
                    continue;
                t.clear();
                for (auto input_id: sub_graph_node.inputs) {
                    NodeID node_id_in_GCG = sub_graph_node_id__to__node_id.at(input_id);
                    auto *input_in_GCG = this->__get_node(node_id_in_GCG);
                    t.push_back(node_id_in_GCG);
                }

                auto new_node = this->__insert_unscheduled_undone_noncheckpoint_node(t, Node_For_Op, task_id);
                new_node->_.assigned_to = 0;
                new_node->_.is_linked_to_successor = sub_graph_node._.is_linked_to_successor;
                new_node->_.is_linked_to_predecessor = sub_graph_node._.is_linked_to_predecessor;
                new_node->_.non_compute_op = sub_graph_node._.non_compute_op;
                new_node->_.execution_sequence = new_node->node_id;
                new_node->_.task_id = sub_graph_node._.task_id;
                new_node->_.task_node_id = sub_graph_node._.task_node_id;
                new_node->_.is_constant = sub_graph_node._.is_constant;
                new_node->_.has_tensor_payload = sub_graph_node._.has_tensor_payload;
                new_node->_.aten_op_name = sub_graph_node._.aten_op_name;
                new_node->_.target = sub_graph_node._.target;
                new_node->_.const_payload = sub_graph_node._.const_payload;
                if (!new_node->_.is_constant) {
                    assert(sub_graph_node._.shape);
                    new_node->_.shape = std::make_shared<VariableDescriptor>(*(sub_graph_node._.shape));
                    this->__propagate_shape__from_cached_symbol_value(
                        new_node->_.shape, cached_symbol_id_value, cached_symbol_value,
                        symbol__to__symexpr);
                }

#ifndef NDEBUG
                new_nodes_ids.push_back(new_node->node_id);
                nlohmann::json jsoned_node = *new_node;
                // std::cout << "InsertOP: " << jsoned_node.dump() << std::endl;
#endif

                if (new_node->_.is_constant) {
                    // 这里的意思是，constant直接随着发射信息带过去，不上传到cluster
                    constants_in_GCG.insert(new_node->node_id);
                }
                sub_graph_node_id__to__node_id[sub_graph_node.node_id] = new_node->node_id;

                this->__post_insert_node(new_node);
            }

            
            if (manual_assignment.has_value()) {
                auto _manual_assignment = manual_assignment.value();
                size_t i = 0;
                for (auto &sub_graph_node: sub_graph) {
                    if (sub_graph_input_ids.contains(sub_graph_node.node_id))
                        continue;
                    NodeID node_id = sub_graph_node_id__to__node_id.at(sub_graph_node.node_id);
                    auto *node = this->__get_node(node_id);
                    node->_.assigned_to = _manual_assignment[i++];
                }
            }

            for (auto constant_node_id: constants_in_GCG) {
                this->__disconnect_predecesor(constant_node_id);
                this->__mark_node_done(constant_node_id);
            }

            std::vector<NodeID> outputs;
            for (auto sub_graph_output_id: sub_graph.outputs) {
                outputs.push_back(sub_graph_node_id__to__node_id.at(sub_graph_output_id));
            }

#ifndef NDEBUG
            for (auto node_id: new_nodes_ids) {
                auto *node = this->__get_node(node_id);
                nlohmann::json jsoned_node = *node;
                // std::cout << "RunOP: " << jsoned_node.dump() << std::endl;

            }
#endif

            return outputs;
        };

        std::vector<NodeID> outputs = __insert_task(inputs, task_id);
        for (size_t output_idx: debug_output_i) {
            struct Node *output = this->__get_node(outputs[output_idx]);
            output->_.debug = true;
        }

        std::vector<Future> output_futs;
        for (auto output_id: outputs) {
            this->__mark_node_checkpoint(output_id);
            auto *out_fut =
                this->__insert_unscheduled_undone_noncheckpoint_node({}, Node_For_Future, output_id);
            this->__post_insert_node(out_fut);
            output_futs.push_back(out_fut->node_id);
        }

        return output_futs;
    }

    inline bool
    ReportCheckpointArrived(Rank rank, NodeID node_id, bool is_rank_reliable) {
        if (this->checkpoint_prelocations.contains(node_id)
            && this->checkpoint_prelocations.at(node_id).contains(rank)) {
            this->checkpoint_locations.at(node_id).insert(rank);

            if (is_rank_reliable) {
                this->persistent_checkpoints.insert(node_id);

                // persistent算子的结果已经不会丢失，因此我们去掉该算子的所有前驱
                this->__disconnect_predecesor(node_id);
            }
            return true;
        }
        return false;
    }

    inline void
    DropFuture(Future fut) {
        this->__mark_node_done(fut);
    }
};

class GCG_Adding_OpManagement: public GCG_Adding_CheckpointManagement {
public:
    // 这个属性模拟器要用
    std::unordered_map<Rank, std::pair<NodeID, Timestamp>> last_done_node__for_each_rank;
private:
    std::unordered_map<IssuingID, size_t> issuing_id__undone_refcnt;
    std::unordered_map<IssuingID, std::unordered_set<NodeID>> issuing_id__to__node_ids;
    std::unordered_map<IssuingID, std::unordered_set<Rank>> issuing_id__to__ranks;
    std::unordered_map<Rank, std::unordered_set<IssuingID>> rank__to__issuing_ids;

    std::vector<
        std::function<void (IssuingID issuing_id)>> callbacks__for_issuing_id_discard;

    void
    __activate_this_component() {
        auto issuing_id_defcnt = [this](struct Node *node,
                                        bool old_scheduled, bool old_done,
                                        bool new_scheduled, bool new_done) {
            if (old_scheduled == true && old_done == false && new_done == true) {
                // Scheduled -> Done

                // Unscheduled -> Done的不考虑在内！

                auto issuing_id = node->_.issuing_id;
                (this->issuing_id__undone_refcnt.at(issuing_id))--;
                if (this->issuing_id__undone_refcnt.at(issuing_id) == 0)
                    this->__discard_issuing_id(issuing_id);
            }
        };
        this->__register_callback_for_execute_state_transfer(Node_For_Op, issuing_id_defcnt);
    }

public:
    GCG_Adding_OpManagement() {
        this->__activate_this_component();
    }

    GCG_Adding_OpManagement(
        const GCG_Adding_OpManagement &_old,
        bool need_task_manager): GCG_Adding_CheckpointManagement(_old, need_task_manager) {
        this->issuing_id__undone_refcnt = _old.issuing_id__undone_refcnt;
        this->issuing_id__to__node_ids = _old.issuing_id__to__node_ids;
        this->issuing_id__to__ranks = _old.issuing_id__to__ranks;
        this->rank__to__issuing_ids = _old.rank__to__issuing_ids;
        this->last_done_node__for_each_rank = _old.last_done_node__for_each_rank;
        this->__activate_this_component();
    }

    GCG_Adding_OpManagement(const nlohmann::json& j): GCG_Adding_CheckpointManagement(j) {
        j.at("issuing_id__undone_refcnt").get_to(this->issuing_id__undone_refcnt);
        j.at("issuing_id__to__node_ids").get_to(this->issuing_id__to__node_ids);
        j.at("issuing_id__to__ranks").get_to(this->issuing_id__to__ranks);
        j.at("rank__to__issuing_ids").get_to(this->rank__to__issuing_ids);
        j.at("last_done_node__for_each_rank").get_to(this->last_done_node__for_each_rank);
        this->__activate_this_component();
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg) const override {
        auto ret = std::static_pointer_cast<GCG_Adding_OpManagement>(dumped_gcg);

        ret->last_done_node__for_each_rank = this->last_done_node__for_each_rank;

        this->GCG_Adding_CheckpointManagement::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, dumped_gcg);
        return ret;
    }

    virtual std::shared_ptr<GCG_Adding_OpManagement>
    _clone() const {
        return std::make_shared<GCG_Adding_OpManagement>(*this, false);
    }

    virtual nlohmann::json
    ToJson() const {
        nlohmann::json j;
        nlohmann::to_json(j, static_cast<const GCG_Adding_CheckpointManagement &>(*this));
        j["issuing_id__undone_refcnt"] = this->issuing_id__undone_refcnt;
        j["issuing_id__to__node_ids"] = this->issuing_id__to__node_ids;
        j["issuing_id__to__ranks"] = this->issuing_id__to__ranks;
        j["rank__to__issuing_ids"] = this->rank__to__issuing_ids;
        j["last_done_node__for_each_rank"] = this->last_done_node__for_each_rank;
        return j;
    }

    friend void
    to_json(nlohmann::json& j, const GCG_Adding_OpManagement& t) {
        j = t.ToJson();
    }

public:
    virtual size_t get_nr_unscheduled() const {assert(0);}
    virtual size_t get_nr_scheduled() const {assert(0);}

private:
    inline void
    __register_issuing_id(IssuingID issuing_id,
                          const std::unordered_set<NodeID> &node_ids) {
        assert(!this->issuing_id__undone_refcnt.contains(issuing_id));
        this->issuing_id__undone_refcnt[issuing_id] = node_ids.size();
        this->issuing_id__to__node_ids[issuing_id] = node_ids;

        this->issuing_id__to__ranks[issuing_id] = {};
        for (auto node_id: node_ids) {
            Rank rank = this->__get_node(node_id)->_.assigned_to;
            this->rank__to__issuing_ids.at(rank).insert(issuing_id);
            this->issuing_id__to__ranks.at(issuing_id).insert(rank);
        }
    }

    inline void
    __discard_issuing_id(IssuingID issuing_id) {
        assert(this->issuing_id__undone_refcnt.at(issuing_id) == 0);
        for (auto &callback: this->callbacks__for_issuing_id_discard)
            callback(issuing_id);
        for (auto rank: this->issuing_id__to__ranks.at(issuing_id))
            this->rank__to__issuing_ids.at(rank).erase(issuing_id);
        this->issuing_id__undone_refcnt.erase(issuing_id);
        this->issuing_id__to__node_ids.erase(issuing_id);
        this->issuing_id__to__ranks.erase(issuing_id);
    }

    inline void
    __withdraw_issuing_ids__for_fault_tolerance(std::unordered_set<IssuingID> withdrawn_issuing_ids) {
        // std::unordered_set<NodeID> withdrawn_nodes;
        // for (auto issuing_id: withdrawn_issuing_ids)
        //     for (auto node_id: this->issuing_id__to__node_ids[issuing_id])
        //         withdrawn_nodes.insert(node_id);

        // std::unordered_set<NodeID> new_scheduling_vertex_cut;
        // auto finding_scheduling_vertex_cut = [&, this](struct Node *node) {
        //     if (withdrawn_issuing_ids.contains(node->_.issuing_id))
        //         return true;

        //     bool all_inputs_are_checkpoints = true;
        //     for (auto input_id: node->inputs) {
        //         auto input_node = this->__get_node(input_id);
        //         if (input_node->_.is_checkpoint
        //             && this->__is_checkpoint__ready_in_cluster(input_id)) {
        //         } else {
        //             all_inputs_are_checkpoints = false;
        //             break;
        //         }
        //     }
        //     this->__report_maybe_scheduling_vertex_cut(node->node_id, all_inputs_are_checkpoints);
        //     if (exist_input__as_checkpoint) {
        //         new_scheduling_vertex_cut.insert(node->node_id);
        //         return false;
        //     }
        //     return true;
        // };
        // this->BFS<true>(withdrawn_nodes, finding_scheduling_vertex_cut);

        // auto withdraw_related_ops = [&, this](struct Node *node) {
        //     if (!new_scheduling_vertex_cut.contains(node->node_id))
        //         this->__discard_scheduling_vertex_cut_node(node->node_id);
        //     if (node->_.is_checkpoint && !this->__is_op_future(node->node_id))
        //         this->__mark_node_non_checkpoint(node->node_id);
        //     if (node->_.is_scheduled)
        //         this->__mark_node_unscheduled(node->node_id);
        //     return true;
        // };
        // this->BFS<false>(new_scheduling_vertex_cut, withdraw_related_ops);
    }

public:
    void __register_callback_for_issuing_id_discard(
        std::function<void (IssuingID issuing_id)> callback) {
        this->callbacks__for_issuing_id_discard.push_back(callback);
    }

    virtual void
    AccSignIn(Rank rank) {
        this->rank__to__issuing_ids[rank] = {};
        this->last_done_node__for_each_rank[rank] = {EMPTY_NODE_ID, 0};
        this->GCG_Adding_CheckpointManagement::AccSignIn(rank);
    }

    /* AccSignOut用于容错，加速器签退、或者加速器重置，会产生Unscheduled算子 */
    virtual void
    AccSignOut(Rank rank) {
        std::unordered_set<IssuingID> related_issuing_ids;
        for (auto issuing_id: this->rank__to__issuing_ids.at(rank))
            related_issuing_ids.insert(issuing_id);
        this->__withdraw_issuing_ids__for_fault_tolerance(related_issuing_ids);
        for (auto issuing_id: related_issuing_ids) {
            this->issuing_id__to__ranks.at(issuing_id).erase(rank);
        }
        this->last_done_node__for_each_rank.erase(rank);
        this->GCG_Adding_CheckpointManagement::AccSignOut(rank);
    }

    /* OpFaultTolerance用于容错，单个算子/传输执行失败，会产生Unscheduled算子 */
    void
    OpFaultTolerance(NodeID node_id) { 
        IssuingID issuing_id = this->__get_node(node_id)->_.issuing_id;
        this->__withdraw_issuing_ids__for_fault_tolerance({issuing_id});
    }

    /* 这个函数从点割出发选择一部分算子拿来调度 */
    virtual std::pair<std::vector<NodeID>, std::unordered_set<NodeID>>
    _Step_1_SelectOpsToSchedule() const { assert(0); return {{}, {}};}

    void
    _Step_2_UpdateVertexCut(const std::vector<NodeID> &ops_to_be_scheduled,
                            const std::unordered_set<NodeID> &ops) {
        // 然后找出来该组算子后第一层未调度算子，成为新的调度点割
        for (auto node_id: ops) {
            auto *node = this->__get_node(node_id);

            bool exist_unscheduled_successor = false;
            for (auto use: node->uses) {
                if (!ops.contains(use)) {
                    exist_unscheduled_successor = true;
                    break;
                }
            }

            // 如果存在未调度后继，那么就把这个op变成checkpoint，以便后继能用上
            if (exist_unscheduled_successor && !node->_.is_checkpoint)
                this->SettleOpAsCheckpoint(node_id);

            // 我们保证，每个Unscheduling点割的所有Done和Scheduled前驱，要么是checkpoint、要么是constant
        }
    }

    virtual std::unordered_map<NodeID, Rank>
    _Step_3_Op_Scheduling(
        std::function<Duration (AccModel, TaskID, TaskNodeID)> task_predictor,
        std::function<std::pair<Duration, NBytes> (Rank, Rank, std::shared_ptr<VariableDescriptor>)> transmit_predictor,
        std::shared_ptr<GCG_Adding_OpManagement> GCG,
        std::shared_ptr<RankManager> rank_manager,
        std::function<std::shared_ptr<Simulator> (void)> get_simulator,
        const std::vector<NodeID> &nodes_to_schedule
    ) {return {};}

    void
    _Step_4_InjectAssigned_For_OPs(NodeID node_id, Rank rank) {
        auto *node = this->__get_node(node_id);
        node->_.assigned_to = rank;
    }

    void
    _Step_5_PostIssuing(const std::vector<NodeID> &ops_to_be_scheduled,
                        const std::unordered_set<NodeID> &ops,
                        IssuingID new_issuling_id) {
        // 然后逆拓扑序，将算子变为Scheduled undone，发射算子
        for (auto it = ops_to_be_scheduled.rbegin(); it != ops_to_be_scheduled.rend(); it++) {
            NodeID node_id = *it;
            this->__mark_node_scheduled_undone(node_id, new_issuling_id);
        }

        this->__register_issuing_id(new_issuling_id, ops);
    }

    inline void
    MarkOpDone(NodeID node_id, AccModel acc_model, Duration running_time) {
        auto *node = this->__get_node(node_id);
        assert(node->_.node_type == Node_For_Op);

        // 这里先处理和node有关的信息，mark_node_done之后可能就把node给删了，就不太好了

        this->last_done_node__for_each_rank.at(node->_.assigned_to) = {node_id, this->Now()};

        if (node->_.non_compute_op == false)
            this->TrainTaskPredictor(node->_.task_id, node->_.task_node_id, acc_model, running_time);

        this->__mark_node_done(node_id);
    }
};


class Simple_Full_GCG: public GCG_Adding_OpManagement {
private:
    // 所有的Node_For_Op都可归为三类
    // 这个用来搞超薄调度用，针对non_compute_op == false的算子
    size_t nr_unscheduled; // non_compute_op == false && is_scheduled == false && done == false
    size_t nr_scheduled; // non_compute_op == false && is_scheduled == true && done == false
    size_t nr_done; // non_compute_op == false，不管is_scheduled，但是 done == true

    void
    __activate_this_component() {
        auto callback_insert_unscheduled_op = [this](struct Node *node) {
            if (node->_.non_compute_op == false) {
                this->nr_unscheduled++;
            }
        };
        this->__register_callback_for_node_insertion(Node_For_Op, callback_insert_unscheduled_op);

        auto callback_scheduling_op = [this](struct Node *node,
                                        bool old_scheduled, bool old_done,
                                        bool new_scheduled, bool new_done) {
            if (node->_.non_compute_op == false) {
                if (old_scheduled == false && old_done == false && new_done == true) {
                    // Unscheduled -> Done
                    this->nr_unscheduled--;
                    this->nr_done++;
                } else if (old_scheduled == false && old_done == false && new_scheduled == true && new_done == false) {
                    // Unscheduled -> Scheduled
                    this->nr_unscheduled--;
                    this->nr_scheduled++;
                } else if (old_scheduled == true && old_done == false && new_done == true) {
                    // Scheduled -> Done
                    this->nr_scheduled--;
                    this->nr_done++;
                } else if (old_done == true && new_scheduled == false && new_done == false) {
                    // Done -> Unscheduled
                    this->nr_done--;
                    this->nr_unscheduled++;
                } else
                    assert(0);
            }
        };
        this->__register_callback_for_execute_state_transfer(Node_For_Op, callback_scheduling_op);

        auto callback_delete_op = [this](struct Node *node) {
            if (node->_.non_compute_op == false) {
                this->nr_done--;
            }
        };
        this->__register_callback_for_node_deletion(Node_For_Op, callback_delete_op);
    }
public:
    virtual size_t
    get_nr_unscheduled() const {
        return this->nr_unscheduled;
    }

    virtual size_t
    get_nr_scheduled() const {
        return this->nr_scheduled;
    }

public:
    Simple_Full_GCG() {
        this->nr_unscheduled = 0;
        this->nr_scheduled = 0;
        this->nr_done = 0;
        this->__activate_this_component();
    }

    Simple_Full_GCG(
        const Simple_Full_GCG &_old,
        bool need_task_manager): GCG_Adding_OpManagement(_old, need_task_manager) {
        this->nr_unscheduled = _old.nr_unscheduled;
        this->nr_scheduled = _old.nr_scheduled;
        this->nr_done = _old.nr_done;
        this->__activate_this_component();
    }

    Simple_Full_GCG(const nlohmann::json& j): GCG_Adding_OpManagement(j) {
        j.at("nr_unscheduled").get_to(this->nr_unscheduled);
        j.at("nr_scheduled").get_to(this->nr_scheduled);
        j.at("nr_done").get_to(this->nr_done);
        this->__activate_this_component();
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg = nullptr) const override {
        auto ret = std::make_shared<Simple_Full_GCG>();
        this->GCG_Adding_OpManagement::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, ret);
        return ret;
    }

    virtual std::shared_ptr<GCG_Adding_OpManagement>
    _clone() const override {
        return std::make_shared<Simple_Full_GCG>(*this, false);
    }

    virtual nlohmann::json
    ToJson() const override {
        nlohmann::json j;
        nlohmann::json parent = this->GCG_Adding_OpManagement::ToJson();
        j["type"] = "Simple_Full_GCG";
        j["nr_unscheduled"] = this->nr_unscheduled;
        j["nr_scheduled"] = this->nr_scheduled;
        j["nr_done"] = this->nr_done;
        j.insert(parent.begin(), parent.end());
        return j;
    }

    virtual std::pair<std::vector<NodeID>, std::unordered_set<NodeID>>
    _Step_1_SelectOpsToSchedule() const override {
        size_t nr_ops_to_schedule = 250;
        size_t scheduled_ops__uplimit = 250;

#ifndef STATIC_SCHEDULE
        if (scheduled_ops__uplimit < this->nr_scheduled)
            return {};
#endif

        if (this->unscheduling_vertex_cutting.empty())
            return {};

        std::vector<NodeID> ops_to_be_scheduled;
        std::unordered_set<NodeID> ops;

        NodeID start = *(this->unscheduling_vertex_cutting.begin());
        std::queue<NodeID> bfs_like_queue;
        bfs_like_queue.push(start);

        // Assert 1: 加入 bfs_like_queue 的有可能是ready op
        // Assert 2: 在 ops 中的不可能在 bfs_like_queue
        // Assert 3: 加入 ops 的一定是 ready op

        // bfs_like_queue有可能重复出队，这个不可避免；
        // 因为一个算子A ready之后搞了一堆后继加入bfs_like_queue，
        // 然后另一个算子B ready之后搞了一堆后继加入bfs_like_queue
        // A和B可能有相同后继
        // 但是一定要保证 ops_to_be_scheduled 不能重复入队

        while(!bfs_like_queue.empty() && ops_to_be_scheduled.size() < nr_ops_to_schedule) {
            NodeID cur_op_id = bfs_like_queue.front();
            auto *cur_op = this->__get_node(cur_op_id);
            bfs_like_queue.pop();

            auto can_get_value = [&](NodeID node_id) -> bool {
                if (ops.contains(node_id))
                    return true;
                auto *input = this->__get_node(node_id);
                if (input->_.is_checkpoint && !this->checkpoint_prelocations.at(node_id).empty())
                    return true;
                if (input->_.is_constant)
                    return true;
                return false;
            };

            auto can_be_scheduled = [&](NodeID node_id) -> bool {
                auto *node = this->__get_node(node_id);
                for (auto input_id: node->inputs) {
                    if (can_get_value(input_id)) {
                    } else {
                        return false;
                    }
                }
                return true;
            };

            if (!ops.contains(cur_op_id) && can_be_scheduled(cur_op_id)) {
                ops_to_be_scheduled.push_back(cur_op_id);
                ops.insert(cur_op_id);
                for (auto use_id: cur_op->uses) {
                    if (!ops.contains(use_id))
                        bfs_like_queue.push(use_id);
                }
            }
        }

        return {ops_to_be_scheduled, ops};
    }

    virtual std::unordered_map<NodeID, Rank>
    _Step_3_Op_Scheduling(
        std::function<Duration (AccModel, TaskID, TaskNodeID)> _task_predictor,
        std::function<std::pair<Duration, NBytes> (Rank, Rank, std::shared_ptr<VariableDescriptor>)> _transmit_predictor,
        std::shared_ptr<GCG_Adding_OpManagement> example_GCG,
        std::shared_ptr<RankManager> rank_manager,
        std::function<std::shared_ptr<Simulator> (void)> get_simulator,
        const std::vector<NodeID> &nodes_to_schedule
    );
};


class Full_GCG: public GCG_Adding_OpManagement {
private:
    float lookahead_factor = 3.0; // 前瞻性
    int simulation_operators_per_ms = 50; // 每毫秒模拟的算子数量

    float avg_op_duration_ms = 15; // 平均每个算子执行多长时间

    int scheduling_horizon_ms = 100; // 要调度多长时间的算子

    size_t nr_ranks;


private:
    // 所有的Node_For_Op都可归为三类
    // 这个用来搞超薄调度用，针对non_compute_op == false的算子
    size_t nr_unscheduled; // non_compute_op == false && is_scheduled == false && done == false
    size_t nr_scheduled; // non_compute_op == false && is_scheduled == true && done == false
    size_t nr_done; // non_compute_op == false，不管is_scheduled，但是 done == true

    void
    __activate_this_component() {
        auto callback_insert_unscheduled_op = [this](struct Node *node) {
            if (node->_.non_compute_op == false) {
                this->nr_unscheduled++;
            }
        };
        this->__register_callback_for_node_insertion(Node_For_Op, callback_insert_unscheduled_op);

        auto callback_scheduling_op = [this](struct Node *node,
                                        bool old_scheduled, bool old_done,
                                        bool new_scheduled, bool new_done) {
            if (node->_.non_compute_op == false) {
                if (old_scheduled == false && old_done == false && new_done == true) {
                    // Unscheduled -> Done
                    this->nr_unscheduled--;
                    this->nr_done++;
                } else if (old_scheduled == false && old_done == false && new_scheduled == true && new_done == false) {
                    // Unscheduled -> Scheduled
                    this->nr_unscheduled--;
                    this->nr_scheduled++;
                } else if (old_scheduled == true && old_done == false && new_done == true) {
                    // Scheduled -> Done
                    this->nr_scheduled--;
                    this->nr_done++;
                } else if (old_done == true && new_scheduled == false && new_done == false) {
                    // Done -> Unscheduled
                    this->nr_done--;
                    this->nr_unscheduled++;
                } else
                    assert(0);
            }
        };
        this->__register_callback_for_execute_state_transfer(Node_For_Op, callback_scheduling_op);

        auto callback_delete_op = [this](struct Node *node) {
            if (node->_.non_compute_op == false) {
                this->nr_done--;
            }
        };
        this->__register_callback_for_node_deletion(Node_For_Op, callback_delete_op);
    }
public:
    virtual size_t
    get_nr_unscheduled() const {
        return this->nr_unscheduled;
    }

    virtual size_t
    get_nr_scheduled() const {
        return this->nr_scheduled;
    }

public:
    Full_GCG() {
        this->nr_ranks = 0;
        this->nr_unscheduled = 0;
        this->nr_scheduled = 0;
        this->nr_done = 0;
        this->__activate_this_component();
    }

    Full_GCG(
        const Full_GCG &_old,
        bool need_task_manager): GCG_Adding_OpManagement(_old, need_task_manager) {
        this->nr_ranks = _old.nr_ranks;
        this->nr_unscheduled = _old.nr_unscheduled;
        this->nr_scheduled = _old.nr_scheduled;
        this->nr_done = _old.nr_done;
        this->__activate_this_component();
    }

    Full_GCG(const nlohmann::json& j): GCG_Adding_OpManagement(j) {
        j.at("lookahead_factor").get_to(this->lookahead_factor);
        j.at("simulation_operators_per_ms").get_to(this->simulation_operators_per_ms);
        j.at("avg_op_duration_ms").get_to(this->avg_op_duration_ms);
        j.at("scheduling_horizon_ms").get_to(this->scheduling_horizon_ms);
        j.at("nr_ranks").get_to(this->nr_ranks);
        j.at("nr_unscheduled").get_to(this->nr_unscheduled);
        j.at("nr_scheduled").get_to(this->nr_scheduled);
        j.at("nr_done").get_to(this->nr_done);
        this->__activate_this_component();
    }

    virtual std::shared_ptr<Graph>
    _dump_for_scheduling_time_simulation(std::unordered_set<NodeID> &nodes_will_be_reserved,
                                         std::shared_ptr<Graph> dumped_gcg = nullptr) const override {
        auto ret = std::make_shared<Simple_Full_GCG>();
        this->GCG_Adding_OpManagement::_dump_for_scheduling_time_simulation(nodes_will_be_reserved, ret);
        return ret;
    }

    virtual std::shared_ptr<GCG_Adding_OpManagement>
    _clone() const override {
        return std::make_shared<Full_GCG>(*this, false);
    }

    virtual nlohmann::json
    ToJson() const override {
        nlohmann::json j;
        nlohmann::json parent = this->GCG_Adding_OpManagement::ToJson();
        j["type"] = "Full_GCG";
        j["lookahead_factor"] = this->lookahead_factor;
        j["simulation_operators_per_ms"] = this->simulation_operators_per_ms;
        j["avg_op_duration_ms"] = this->avg_op_duration_ms;
        j["scheduling_horizon_ms"] = this->scheduling_horizon_ms;
        j["nr_ranks"] = this->nr_ranks;
        j["nr_unscheduled"] = this->nr_unscheduled;
        j["nr_scheduled"] = this->nr_scheduled;
        j["nr_done"] = this->nr_done;
        j.insert(parent.begin(), parent.end());
        return j;
    }

    virtual void
    AccSignIn(Rank rank) {
        this->nr_ranks++;
        this->GCG_Adding_OpManagement::AccSignIn(rank);
    }

    virtual void
    AccSignOut(Rank rank) {
        this->nr_ranks--;
        this->GCG_Adding_OpManagement::AccSignOut(rank);
    }

    inline void
    bfs_uses(NodeID node_id,
             std::function<bool (NodeID)> visit,
             std::function<bool ()> stop_condition = []() {return false;}) const {
        std::unordered_set<NodeID> visited;
        std::queue<NodeID> bfs_queue;
        bfs_queue.push(node_id);

        while(!bfs_queue.empty()) {
            NodeID cur_node_id = bfs_queue.front();
            bool stop_in_the_path = visit(cur_node_id);
            visited.insert(cur_node_id);
            if (stop_condition())
                return;
            if (!stop_in_the_path) {
                auto *cur_node = this->__get_node(cur_node_id);
                for (auto use_id: cur_node->uses)
                    if (!visited.contains(use_id))
                        bfs_queue.push(use_id);
            }
            bfs_queue.pop();
        }
    }

    inline void
    bfs_inputs(NodeID node_id,
               std::function<bool (NodeID)> visit,
               std::function<bool ()> stop_condition = []() {return false;}) const {
        std::unordered_set<NodeID> visited;
        std::queue<NodeID> bfs_queue;
        bfs_queue.push(node_id);

        while(!bfs_queue.empty()) {
            NodeID cur_node_id = bfs_queue.front();
            bool stop_in_the_path = visit(cur_node_id);
            visited.insert(cur_node_id);
            if (stop_condition())
                return;
            if (!stop_in_the_path) {
                auto *cur_node = this->__get_node(cur_node_id);
                for (auto input_id: cur_node->inputs)
                    if (!visited.contains(input_id))
                        bfs_queue.push(input_id);
            }
            bfs_queue.pop();
        }
    }

    inline void
    dfs_uses_1(NodeID node_id,
               std::unordered_set<NodeID> &visited,
               std::function<bool (NodeID)> visit,
               std::function<bool ()> stop_condition) const {
        auto *node = this->__get_node(node_id);
        bool stop_in_the_path = visit(node_id);
        visited.insert(node_id);
        if (stop_condition())
            return;
        if (!stop_in_the_path) {
            for (auto use_id: node->uses)
                this->dfs_uses_1(use_id, visited, visit, stop_condition);
        }
    }

    inline void
    dfs_uses(NodeID node_id,
             std::function<bool (NodeID)> visit,
             std::function<bool ()> stop_condition = []() {return false;}) const {
        std::unordered_set<NodeID> visited;
        this->dfs_uses_1(node_id, visited, visit, stop_condition);
    }

    virtual std::pair<std::vector<NodeID>, std::unordered_set<NodeID>>
    _Step_1_SelectOpsToSchedule() const override {
        size_t nr_ops_to_schedule = this->scheduling_horizon_ms / this->avg_op_duration_ms;
        size_t scheduled_ops__uplimit = (this->scheduling_horizon_ms * this->lookahead_factor) / this->avg_op_duration_ms;

#ifndef STATIC_SCHEDULE
        if (scheduled_ops__uplimit < this->nr_scheduled)
            return {};
#endif

        std::vector<NodeID> ops_to_be_scheduled;
        std::unordered_set<NodeID> ops;
        size_t nr_compute_ops = 0;

    #ifndef NDEBUG
        std::cout << "[DEBUG SelectOpsToSchedule] horizon_ms=" << this->scheduling_horizon_ms
              << " avg_op_ms=" << this->avg_op_duration_ms
              << " lookahead_factor=" << this->lookahead_factor
              << " nr_ops_to_schedule=" << nr_ops_to_schedule
              << " scheduled_uplimit=" << scheduled_ops__uplimit
              << " nr_scheduled=" << this->nr_scheduled
              << " unscheduling_vertex_cutting_size=" << this->unscheduling_vertex_cutting.size()
              << std::endl;
    #endif

        auto can_get_value = [&](NodeID node_id) -> bool {
            if (ops.contains(node_id))
                return true;
            auto *node = this->__get_node(node_id);
            if (node->_.is_checkpoint && !this->checkpoint_prelocations.at(node_id).empty())
                return true;
            if (node->_.is_constant)
                return true;
            return false;
        };

        auto can_be_scheduled = [&](NodeID node_id) -> bool {
            auto *node = this->__get_node(node_id);
            for (auto input_id: node->inputs) {
                if (can_get_value(input_id)) {
                } else {
                    return false;
                }
            }
            return true;
        };

        for (auto it = this->unscheduling_vertex_cutting.cbegin(); it != this->unscheduling_vertex_cutting.cend(); ++it) {

            
            NodeID start = *it;
#ifndef NDEBUG
            std::cout << "[DEBUG SelectOpsToSchedule] start vertex_cut node=" << start << std::endl;
#endif
            std::queue<NodeID> bfs_like_queue;
            bfs_like_queue.push(start);

            auto add_to_schedule = [&](NodeID node_id) {
                const struct Node *op = this->__get_node(node_id);
                if (!ops.contains(node_id)) {
                    ops_to_be_scheduled.push_back(node_id);
                    ops.insert(node_id);
                    if (op->_.non_compute_op == false)
                        nr_compute_ops++;
#ifndef NDEBUG
                    std::cout << "[DEBUG SelectOpsToSchedule] add_to_schedule node=" << node_id
                              << " non_compute=" << op->_.non_compute_op
                              << " is_checkpoint=" << op->_.is_checkpoint
                              << " is_constant=" << op->_.is_constant
                              << " assigned_to=" << op->_.assigned_to
                              << " inputs_count=" << op->inputs.size()
                              << " uses_count=" << op->uses.size()
                              << std::endl;
#endif
                    for (auto use_id: op->uses) {
                        if (!ops.contains(use_id))
                            bfs_like_queue.push(use_id);
                    }
                }
            };

            // Assert 1: 加入 bfs_like_queue 的有可能是ready op
            // Assert 2: 在 ops 中的不可能在 bfs_like_queue
            // Assert 3: 加入 ops 的一定是 ready op

            // bfs_like_queue有可能重复出队，这个不可避免；
            // 因为一个算子A ready之后搞了一堆后继加入bfs_like_queue，
            // 然后另一个算子B ready之后搞了一堆后继加入bfs_like_queue
            // A和B可能有相同后继
            // 但是一定要保证 ops_to_be_scheduled 不能重复入队

            while(!bfs_like_queue.empty() && nr_compute_ops < nr_ops_to_schedule) {
                NodeID cur_op_id = bfs_like_queue.front();
                const struct Node *cur_op = this->__get_node(cur_op_id);
                bfs_like_queue.pop();

                if (cur_op->_.non_compute_op == false && ops.contains(cur_op_id)) {
                    continue;
                }

#ifndef NDEBUG
                std::cout << "[DEBUG SelectOpsToSchedule] pop cur_op_id=" << cur_op_id
                          << " non_compute=" << cur_op->_.non_compute_op
                          << " is_scheduled=" << cur_op->_.is_scheduled
                          << " done=" << cur_op->_.done
                          << " inputs_count=" << cur_op->inputs.size()
                          << " uses_count=" << cur_op->uses.size()
                          << std::endl;
#endif

                std::vector<NodeID> batch_scheduling_ops;
                bool find_one_compute_op = false;
                auto is_a_ready_compute_op = [&](NodeID node_id) -> bool {
                    const struct Node *node = this->__get_node(node_id);

#ifndef NDEBUG
                    std::cout << "[Search compute OP] is Node[" << node_id << "] a compute op? ";
#endif

                    if (node->_.non_compute_op == true) {
#ifndef NDEBUG
                        std::cout << "No." << std::endl;
#endif
                        return false;
                    }

                    if (can_get_value(node_id)) {
#ifndef NDEBUG
                        std::cout << "Yes. But it is already decided to be scheduled." << std::endl;
#endif
                        return true; // 只要是compute op，都在这条path上停止
                    }

                    batch_scheduling_ops.clear();

                    bool ready = true;
                    this->bfs_inputs(node_id,
                                     [&](NodeID input_id) -> bool {
                                         auto *input = this->__get_node(input_id);
                                         if (can_get_value(input_id)) {
                                             return true; // 如果这个输入是ready，这个节点就不继续遍历了
                                         } else if (input->_.non_compute_op == true && input->_.is_linked_to_successor == true) {
                                             batch_scheduling_ops.push_back(input_id);
                                             return false; // 跳过linked to successor
                                         } else if (input_id == node_id) {
                                             batch_scheduling_ops.push_back(input_id);
                                             return false; // 跳过compute_op节点本身
                                         } else {
                                             ready = false;
                                             return true; // 在这里的意思就是说，这个compute op不能拿去调度，因为有输入不ready
                                         }
                                     },
                                     [&]() -> bool {return ready == false;}); // 如果 ready == false，则立刻返回，没必要接着遍历了
                    if (ready)
                        find_one_compute_op = true;
#ifndef NDEBUG
                    std::cout << "Yes. It is " << (ready ? "ready." : "not ready.") << std::endl;
#endif
                    return true; // 只要是compute op，都在这条path上停止
                };

                this->dfs_uses(cur_op_id,
                               is_a_ready_compute_op,
                               [&]() -> bool {return find_one_compute_op;}); // 先遍历 link_to_successor 的节点树，触及真正的compute op节点


                if (!find_one_compute_op)
                    continue;

                for (auto it = batch_scheduling_ops.rbegin(); it != batch_scheduling_ops.rend(); ++it) {
                    NodeID scheduling_op_id = *it;
                    add_to_schedule(scheduling_op_id);
                }

                NodeID compute_op_id = batch_scheduling_ops[0];
                this->bfs_uses(compute_op_id,
                               [&](NodeID node_id) -> bool {
                                   auto *node = this->__get_node(node_id);
                                   if (node_id == compute_op_id)
                                        return false; // compute_op节点本身继续遍历

                                   if (node->_.non_compute_op == true && node->_.is_linked_to_predecessor == true
                                       && can_be_scheduled(node_id)) {
                                       add_to_schedule(node_id);
                                       return false;
                                   }
                                   return true; // 如果不是 compute_op 辐射出的link_to_predecessor，那就别遍历了
                               }); // 再遍历 compute_op 辐射出的link_to_predecessor节点树
            }

            if (nr_ops_to_schedule <= nr_compute_ops)
                break;
        }
        
#ifndef NDEBUG
        std::cout << "[DEBUG SelectOpsToSchedule] ops_to_be_scheduled: ";
        for (auto id: ops_to_be_scheduled)
            std::cout << id << ",";
        std::cout << std::endl;
#endif

        return {ops_to_be_scheduled, ops};
    }
};

extern std::shared_ptr<GCG_Adding_OpManagement>
Get_FullGCG();

extern std::shared_ptr<GCG_Adding_OpManagement>
Get_SimpleFullGCG();

extern std::shared_ptr<GCG_Adding_OpManagement>
Get_GCG_FromJson(const nlohmann::json &j);

#endif