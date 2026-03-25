#include <unordered_map>
#include <cstddef>
typedef int Rank;

struct WorkerPair {
    Rank s, r;
    WorkerPair(Rank s, Rank r) : s(s), r(r) {}

    bool operator== (const struct WorkerPair &p) const {
        return this->s == p.s && this->r == p.r;
    }

    bool operator< (const struct WorkerPair &p) const {
        if (this->s != p.s)
            return this->s < p.s;
        return this->r < p.r;
    }
};

template<>
struct std::hash<WorkerPair> {
    size_t operator()(const WorkerPair& k) const {
        return std::hash<Rank>{}(k.s) ^ std::hash<Rank>{}(k.r);
    }
};

int main () {
    std::unordered_map<WorkerPair, int> map;
    return 0;
}
