#include "Misc.h"
#include <string.h>
#include "Parallel.h"

void qgate::fillZeros(void *_array, QstateSize byteSize) {
    char *array = static_cast<char*>(_array);
    auto setZeroFunc = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
        memset(&array[spanBegin], 0, spanEnd - spanBegin);
    };
    Parallel().distribute(setZeroFunc, 0LL, (QstateSize)byteSize);
}
