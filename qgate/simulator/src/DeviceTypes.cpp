#include "DeviceTypes.h"
#include <stdio.h>

void qgate_cuda::
_throwError(cudaError_t status, const char *file, unsigned long line, const char *expr) {
    char msg[512];
    const char *errName = cudaGetErrorName(status);
    snprintf(msg, sizeof(msg), "%s(%d), %s.", errName, (int)status, expr);
    qgate::_throwError(file, line, msg);
}
