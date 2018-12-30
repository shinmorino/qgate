#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "Parallel.h"

using namespace qgate;

int Parallel::getDefaultNumThreads() {
    HANDLE h = GetCurrentProcess();
    DWORD_PTR processAffinityMask, systemAffinityMask;
    GetProcessAffinityMask(h, &processAffinityMask, &systemAffinityMask);
    return (int)__popcnt64(processAffinityMask);
}
