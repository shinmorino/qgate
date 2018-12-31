#pragma once

#include "Types.h"
#include <thread>
#include <functional>
#include <algorithm>

namespace qgate {

struct Parallel {

    template<class Iterator, class C>
    void distribute(Iterator begin, Iterator end, const C &functor) {
        throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");
        if (parallelThreshold_ < end - begin) {
            int nWorkers = nThreads_ + 1;
            Iterator span = (end - begin + nWorkers - 1) / nWorkers;
            span = ((span + spanBase_ - 1) / spanBase_) * spanBase_;
            auto distributed = [=](int threadIdx) {
                Iterator spanBegin = std::min(begin + span * threadIdx, end);
                Iterator spanEnd = std::min(begin + span * (threadIdx + 1), end);
                functor(threadIdx, spanBegin, spanEnd);
            };
            run(distributed);
        }
        else {
            functor(0, begin, end);
        }
    }

    template<class C>
    void for_each(QstateIdx begin, QstateIdx end, const C &functor) {
        auto forloop = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx) {
                functor(idx);
            }
        };
        distribute(begin, end, forloop);
    }

    template<class real, class C>
    real sum(QstateIdx begin, QstateIdx end, const C &functor) {
        throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");
        int nWorkers = nThreads_ + 1;

        real *partialSum = new real[nWorkers];
        auto forloop = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            real v = real(0.);
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx) {
                v += functor(idx);
            }
            partialSum[threadIdx] = v;
        };
        distribute(begin, end, forloop);

        real sum = real(0.);
        for (QstateIdx idx = 0; idx < nWorkers; ++idx) {
            sum += partialSum[idx];
        }
        delete[] partialSum;
 
        return sum;
    }


    Parallel() {
        threads_ = NULL;
        spanBase_ = 16;
        parallelThreshold_ = 1 << 16; /* not checked.  May not be optimal.  */
        int nThreads = getDefaultNumThreads();
        
        initialize(nThreads);
    }

    ~Parallel() {
        finalize();
    }
    
    void initialize(int nWorkers) {
        nThreads_ = nWorkers - 1;
        if (0 < nThreads_)
            threads_ = (std::thread*)malloc(sizeof(std::thread) * nThreads_);
    }
    
    void finalize() {
        if (threads_ != NULL)
            free(threads_);
        threads_ = NULL;
    }
    
    template<class F>
    void run(F &f, int nWorkers = -1) {
        functor_ = f;
        if (nWorkers == -1)
            nWorkers = nThreads_ + 1;
        if (1 < nWorkers) {
            for (int idx = 0; idx < nThreads_; ++idx) {
                auto threadFunc = [this, idx] { Parallel::threadEntry(this, idx + 1); };
                new (&threads_[idx]) std::thread(threadFunc);
            }
            
            /* run the 0-th worker in main thread. */
            functor_(0);
            
            for (int idx = 0; idx < nThreads_; ++idx) {
                threads_[idx].join();
                threads_[idx].~thread();
            }
        }
        else {
            /* nWorkers == 1 */
            functor_(0);
        }
    }

    static int getDefaultNumThreads();
    
private:

    static
    void threadEntry(Parallel *_this, int threadIdx) {
        _this->functor_(threadIdx);
    }

    int spanBase_;
    int parallelThreshold_;

    std::thread *threads_;
    int nThreads_;
    std::function<void(int)> functor_;
};

}
