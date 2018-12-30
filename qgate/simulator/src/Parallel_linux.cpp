#include "Parallel.h"

using namespace qgate;

#include <sched.h>

int Parallel::getDefaultNumThreads() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    return CPU_COUNT(&cpuset);
}

