#ifdef _MSC_VER
#  define WIN32_LEAN_AND_MEAN
#  define _CRT_SECURE_NO_WARNINGS
#  include <windows.h>
#endif

#ifdef __linux__
#  include <sched.h>
#  include <unistd.h>
#endif

#include "Parallel.h"
#include "Types.h"

using namespace qgate;


int Parallel::nDefaultWorkers_ = -1;
bool Parallel::dynamicNWorkers_ = true;

int Parallel::getDefaultNWorkers() {
    if (nDefaultWorkers_ != -1)
        return nDefaultWorkers_;

    const char *env = getenv("QGATE_DYNAMIC_NUM_WORKERS");
    if (env != NULL)
        Parallel::dynamicNWorkers_ = (env[0] != '0');

    env = getenv("QGATE_NUM_WORKERS");
    if (env != NULL) {
        char *endptr = NULL;
        int nWorkers = strtol(env, &endptr, 10);
        if ((env != endptr) || (0 < nWorkers)) {
            nDefaultWorkers_ = nWorkers;
            return nDefaultWorkers_;
        }
        qgate::log("Igonring QGATE_NUM_WORKERS=%s.\n", env);
    }

#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    nDefaultWorkers_ = CPU_COUNT(&cpuset);
#endif

#ifdef _MSC_VER
    HANDLE h = GetCurrentProcess();
    DWORD_PTR processAffinityMask, systemAffinityMask;
    GetProcessAffinityMask(h, &processAffinityMask, &systemAffinityMask);
    nDefaultWorkers_ = (int)__popcnt64(processAffinityMask);
#endif
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
