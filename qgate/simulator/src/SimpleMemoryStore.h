#pragma once

#include "Types.h"

namespace qgate_cuda {

class SimpleMemoryStore {
public:

    SimpleMemoryStore() { 
        mem_ = NULL;
        capacity_ = 0;
        reset();
    }

    void set(void *pv, size_t capacity) {
        mem_ = static_cast<char*>(pv);
        capacity_ = capacity;
        reset();
    }

    void reset() {
        free_ = mem_;
        remaining_ = capacity_;
    }

    template<class V>
    V *allocate(size_t size) {
        throwErrorIf(remaining<V>() < size, "Requested size too large.");
        V *allocated = reinterpret_cast<V*>(free_);
        size_t consumed = sizeof(V) * size;
        free_ += consumed;
        remaining_ -= consumed;
        return allocated;
    }

    template<class V>
    size_t remaining() const {
        return remaining_ / sizeof(V);
    }
    
private:    
    char *mem_;
    size_t capacity_;
    char *free_;
    size_t remaining_;

    SimpleMemoryStore(const SimpleMemoryStore &);
};

}

