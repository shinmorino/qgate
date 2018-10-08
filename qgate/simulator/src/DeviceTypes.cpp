#include "DeviceTypes.h"
#include <stdio.h>

void cuda_runtime::
_throwError(cudaError_t status, const char *file, unsigned long line, const char *expr) {
    char msg[512];
    const char *errName = cudaGetErrorName(status);
    snprintf(msg, sizeof(msg), "%s(%d), %s.", errName, (int)status, expr);
    runtime::_throwError(file, line, msg);
}
