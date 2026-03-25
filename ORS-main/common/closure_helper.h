#pragma once
#include <google/protobuf/stubs/callback.h>
#include <butil/status.h>
#include <bthread/countdown_event.h>

namespace ors {

class Closure : public google::protobuf::Closure {
public:
    butil::Status& status() { return _st; }
    const butil::Status& status() const { return _st; }

private:
    butil::Status _st;
};


template<bool DELETE_AFTER_FINISH = true, uint32_t SMALL_COUNT = 4>
class ClosureGroup {
public:
    class SubClosure : public Closure {
    public:
        SubClosure(ClosureGroup* g) : _g(g) {}

        void Run() override {
            _g->finish_one();
        }

    private:
        ClosureGroup* _g;
    };

    ClosureGroup(size_t count, Closure* done)
        : _remain_count(count), _count(count), _done(done) {
        for (size_t i = 0; i < std::min(size_t(SMALL_COUNT), _count); ++i) {
            new (_static_done_mem + i * sizeof(SubClosure)) SubClosure(this);
        }
        if (_count > SMALL_COUNT) {
            _extra_done_mem = (char*) malloc((_count - SMALL_COUNT) * sizeof(SubClosure));
            for (size_t i = 0; i < _count - SMALL_COUNT; ++i) {
                new (_extra_done_mem + i * sizeof(SubClosure)) SubClosure(this);
            }
        }
    }

    ~ClosureGroup() {
        for (size_t i = 0; i < _count; ++i) {
            sub_done(i)->~SubClosure();
        }
        if (_count > SMALL_COUNT) {
            free(_extra_done_mem);
        }
    }

    SubClosure* sub_done(size_t i) {
        CHECK_LT(i, _count);
        if (i < SMALL_COUNT) {
            return reinterpret_cast<SubClosure*>(_static_done_mem) + i;
        } else {
            return reinterpret_cast<SubClosure*>(_extra_done_mem) + i - SMALL_COUNT;
        }
    }

    size_t count() const {
        return _count;
    }

    size_t remain_count() const {
        return _remain_count.load(std::memory_order_acquire);
    }

private:
     void finish_one() {
        if (_remain_count.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);

            // Merge status from sub closures
            for (size_t i = 0; i < _count; ++i) {
                if (!sub_done(i)->status().ok()) {
                    _done->status() = sub_done(i)->status();
                    break;
                }
            }

            _done->Run();

            if (DELETE_AFTER_FINISH) {
                delete this;
            }
        }
    }

    std::atomic<size_t> _remain_count = 0;
    size_t _count = 0;
    Closure* _done = nullptr;
    char _static_done_mem[SMALL_COUNT * sizeof(SubClosure)];
    char* _extra_done_mem = nullptr;
};

class SynchronizedClosure : public Closure {
public:
    SynchronizedClosure() : _event(1), _num_signal(1) {}

    SynchronizedClosure(int num_signal) : _event(num_signal), _num_signal(num_signal) {
    }

    void Run() override {
        _event.signal();
    }

    void wait() {
        _event.wait();
    }

    void add_count() {
        _event.add_count();
    }

    void reset() {
        status().reset();
        _event.reset(_num_signal);
    }

    void reset(int num_signal) {
        _num_signal = num_signal;
        reset();
    }

    int timed_wait(int timeout_ms) {
        return _event.timed_wait(butil::milliseconds_from_now(timeout_ms));
    }

private:
    bthread::CountdownEvent _event;
    int _num_signal = 1;
};

template <typename L>
class ClosureWithLambda : public Closure {
public:
    ClosureWithLambda(L&& l) : _l(l) {}
    void Run() override {
        _l(status());
        delete this;
    }
private:
    L _l;
};

template<typename L>
Closure* create_closure(L&& l) {
    return new ClosureWithLambda<L>(std::forward<L>(l));
};

}
