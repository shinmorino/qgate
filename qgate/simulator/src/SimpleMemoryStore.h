#pragma once

#include "Types.h"

namespace qgate_cuda {

class SimpleMemoryStore {
public:

    SimpleMemoryStore(void *pv, size_t capacity) :
            mem_(static_cast<char*>(pv)), capacity_(capacity) { }

    template<class V>
    V *allocate(size_t size) {
        throwErrorIf(capacity<V>() < size, "Requested size too large.");
        V *v = reinterpret_cast<V*>(mem_);
        size_t consumed = sizeof(V) * size;
        mem_ += consumed;
        capacity_ -= consumed;
        return v;
    }

    template<class V>
    size_t capacity() const {
        return capacity_ / sizeof(V);
    }
    
private:    
    char *mem_;
    size_t capacity_;
};

}

