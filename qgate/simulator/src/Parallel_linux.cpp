#include "Parallel.h"
#include "Types.h"

using namespace qgate;

#include <sched.h>
#include <unistd.h>


int Parallel::nDefaultWorkers_ = -1;
bool Parallel::dynamicNWorkers_ = true;

int Parallel::getDefaultNWorkers() {
    if (nDefaultWorkers_ != -1)
        return nDefaultWorkers_;

    const char *env = getenv("QGATE_DYNAMIC_NUM_WORKERS");
    if (env != NULL) {
        int v = atoi(env);
        Parallel::dynamicNWorkers_ = (v != 0);
    }

    env = getenv("QGATE_NUM_WORKERS");
    if (env != NULL) {
        int nWorkers = atoi(env);
        if (0 < nWorkers) {
            nDefaultWorkers_ = nWorkers;
            return nDefaultWorkers_;
        }
        qgate::log("Igonring QGATE_NUM_WORKERS=%s.\n", env);
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    nDefaultWorkers_ = CPU_COUNT(&cpuset);
    return nDefaultWorkers_;
}

int Parallel::getNWorkers() const {
    return nWorkers_;
}

int Parallel::getNWorkers(QstateSize nLoops) const {
    if (Parallel::dynamicNWorkers_) {
        if (parallelThreshold_ < nLoops)
            return getNWorkers();
        return 1;
    }
    return getNWorkers();
}
