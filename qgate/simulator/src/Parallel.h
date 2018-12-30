#pragma once

#include "Types.h"
#include <thread>
#include <functional>

namespace qgate {

struct Parallel {

    template<class C>
    void for_each(QstateIdx begin, QstateIdx end, const C &functor) {
        throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");
        if ((1LL << 16) < end - begin) {
            int nWorkers = nThreads_ + 1;
            auto dispatched = [=](int threadIdx) {
                QstateSize span = (end - begin + nWorkers - 1) / nWorkers;
                span = ((span + 15) / 16) * 16;
                QstateIdx thisBegin = std::min(begin + span * threadIdx, end);
                QstateIdx thisEnd = std::min(begin + span * (threadIdx + 1), end);
                for (QstateIdx idx = thisBegin; idx < thisEnd; ++idx) {
                    functor(idx);
                }
            };
            run(dispatched);
        }
        else {
            for (QstateIdx idx = begin; idx < end; ++idx) {
                functor(idx);
            }
        }
    }

    template<class real, class C>
    real sum(QstateIdx begin, QstateIdx end, const C &functor) {
        throwErrorIf(0x40000000LL < end, "end < 0x40000000LL");
        
        real sum = real(0.);
        if ((1LL << 16) < end - begin) {
            int nWorkers = nThreads_ + 1;
            real *partialSum = new real[nWorkers];
            auto dispatched = [=](int threadIdx) {
                QstateSize span = (end - begin + nWorkers - 1) / nWorkers;
                span = ((span + 15) / 16) * 16;
                QstateIdx thisBegin = std::min(begin + span * threadIdx, end);
                QstateIdx thisEnd = std::min(begin + span * (threadIdx + 1), end);
                real v = real(0.);
                for (QstateIdx idx = thisBegin; idx < thisEnd; ++idx) {
                    v += functor(idx);
                }
                partialSum[threadIdx] = v;
            };
            run(dispatched);
            for (QstateIdx idx = 0; idx < nWorkers; ++idx) {
                sum += partialSum[idx];
            }
            delete [] partialSum;
        }
        else {
            for (QstateIdx idx = begin; idx < end; ++idx) {
                sum += functor(idx);
            }
        }
        
        return sum;
    }


    Parallel() {
        threads_ = NULL;
        int nThreads = getDefaultNumThreads();
        initialize(nThreads);
    }

    ~Parallel() {
        finalize();
    }
    
    void initialize(int nWorkers) {
        nThreads_ = nWorkers - 1;
        if (0 < nThreads_) {
            threads_ = (std::thread*)malloc(sizeof(std::thread) * nThreads_);
        }
    }
    
    void finalize() {
        if (nThreads_ != 0)
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
                new (&threads_[idx]) std::thread([this, idx]{
                            Parallel::threadEntry(this, idx + 1); });
            }
            for (int idx = 0; idx < nThreads_; ++idx)
                threads_[idx].detach();
            
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

    std::thread *threads_;
    int nThreads_;
    std::function<void(int)> functor_;

};

}
