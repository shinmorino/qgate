#pragma once

#include "Types.h"
#include <thread>
#include <functional>
#include <algorithm>

namespace qgate {

struct Parallel {

    template<class Iterator, class C>
    void distribute(const C &functor, Iterator begin, Iterator end, Iterator spanBase = 16) {
        int nWorkers = getNWorkers(end - begin);
        Iterator span = (end - begin + nWorkers - 1) / nWorkers;
        span = ((span + spanBase - 1) / spanBase) * spanBase;
        auto distributed = [=](int threadIdx) {
            Iterator spanBegin = std::min(begin + span * threadIdx, end);
            Iterator spanEnd = std::min(begin + span * (threadIdx + 1), end);
            functor(threadIdx, spanBegin, spanEnd);
        };
        run(distributed, nWorkers);
    }

    template<class C>
    void for_each(const C &functor, QstateIdx begin, QstateIdx end, QstateIdx spanBase = 16) {
        auto forloop = [=](int threadIdx, QstateIdx spanBegin, QstateIdx spanEnd) {
            for (QstateIdx idx = spanBegin; idx < spanEnd; ++idx) {
                functor(idx);
            }
        };
        distribute(forloop, begin, end, spanBase);
    }

    template<class real, class C>
    real sum(const C &functor, QstateIdx begin, QstateIdx end) {
        int nWorkers = getNWorkers(end - begin);
        real *partialSum = new real[nWorkers]();
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
    
    template<class F>
    void run(F &f) {
        int nWorkers = getNWorkers();
        run(f, nWorkers);
    }

    template<class F>
    void run(F &f, int nWorkers) {
        functor_ = f;
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

    explicit Parallel(int nWorkers = -1) {
        nWorkers_ = nWorkers;
        if (nWorkers_ == -1)
            nWorkers_ = Parallel::getDefaultNWorkers();
        parallelThreshold_ = 1LL << 16;
    }

    int getNWorkers() const;

    int getNWorkers(QstateSize nLoops) const;

    static int getDefaultNWorkers();
    
private:

    static
    void threadEntry(Parallel *_this, int threadIdx) {
        _this->functor_(threadIdx);
    }

    int nWorkers_;
    QstateSize parallelThreshold_;

    static int nDefaultWorkers_;
    static bool dynamicNWorkers_;

    std::function<void(int)> functor_;
};

}
