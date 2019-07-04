#pragma once

#include "Types.h"
#include <thread>
#include <functional>
#include <algorithm>

namespace qgate {

struct Parallel {

    int getNWorkers(QstateSize nLoops) const {
        if (parallelThreshold_ < nLoops)
            return getDefaultNumThreads();
        return 1;
    }

    template<class Iterator, class C>
    void distribute(Iterator begin, Iterator end, const C &functor) {

        if ((parallelThreshold_ < end - begin) && (1 < nWorkers_)) {
            Iterator span = (end - begin + nWorkers_ - 1) / nWorkers_;
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

        real *partialSum = new real[nWorkers_]();
        auto forloop = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            real v = real(0.);
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx) {
                v += functor(idx);
            }
            partialSum[threadIdx] = v;
        };
        distribute(begin, end, forloop);

        real sum = real(0.);
        for (QstateIdx idx = 0; idx < nWorkers_; ++idx) {
            sum += partialSum[idx];
        }
        delete[] partialSum;
 
        return sum;
    }


    explicit Parallel(int nWorkers = -1, int spanBase = 16, int parallelThreashold = 1 << 16) {
        nWorkers_ = nWorkers;
        if (nWorkers_ == -1)
            nWorkers_ = getDefaultNumThreads();
        spanBase_ = spanBase;
        parallelThreshold_ = parallelThreashold;
    }
    
    template<class F>
    void run(F &f, int nWorkers = -1) {
        functor_ = f;
        if (nWorkers == -1)
            nWorkers = nWorkers_;
        if (1 < nWorkers) {
            int nThreads = nWorkers - 1;
            std::thread *threads = (std::thread*)malloc(sizeof(std::thread) * nThreads);

            for (int idx = 0; idx < nThreads; ++idx) {
                auto threadFunc = [this, idx] { Parallel::threadEntry(this, idx + 1); };
                new (&threads[idx]) std::thread(threadFunc);
            }
            
            /* run the 0-th worker in main thread. */
            functor_(0);
            
            for (int idx = 0; idx < nThreads; ++idx) {
                threads[idx].join();
                threads[idx].~thread();
            }
            free(threads);
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

    int nWorkers_;
    int spanBase_;
    int parallelThreshold_;

    std::function<void(int)> functor_;
};

}
