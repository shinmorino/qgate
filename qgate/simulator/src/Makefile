.SUFFIXES: .cpp .o .h .c

include incpath

TARGETS= ../glue.so ../cpuext.so ../cudaext.so
glue_so_OBJS=glue.o GateMatrix.o Parallel_linux.o Misc.o Types.o
cpuext_OBJS=CPUQubitStates.o CPUQubitProcessor.o CPUSamplingPool.o CPUQubitsStatesGetter.o BitPermTable.o Parallel_linux.o Types.o
cpuext_so_OBJS=$(cpuext_OBJS) cpuext.o

cudaext_OBJS=CUDAGlobals.o CUDAQubitStates.o CUDAQubitProcessor.o TransferringRunner.o CUDAQubitsStatesGetter.o CUDADevice.o DeviceProcPrimitives.o DeviceGetStates.o DeviceProbArrayCalculator.o MultiDeviceMemoryStore.o DeviceTypes.o ProcessorRelocator.o CPUSamplingPool.o BitPermTable.o Parallel_linux.o Types.o
cudaext_so_OBJS=cudaext.o $(cudaext_OBJS)

test_gate_matrix_OBJS=GateMatrix.o GateMatrixTest.o

CXX=g++
CC=gcc
CFLAGS=-g -O2 -Wall -ggdb -Wno-format-security -fPIC -pthread
#CFLAGS=-g -Wall -ggdb -Wno-format-security -fPIC -pthread -D_DEBUG
CXXFLAGS=$(CFLAGS) -std=c++11
LDFLAGS=

NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=--expt-extended-lambda -std=c++11 -lineinfo
NVCCARCH=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70

INCLUDE+=-I/usr/local/cuda/include


all: $(TARGETS)

check: all test_gate_matrix

cuda_obj : DeviceProcPrimitives.o DeviceGetStates.o DeviceProbArrayCalculator.o

../glue.so: $(glue_so_OBJS) Types.h Interfaces.h
	$(CXX) -shared $(CXXFLAGS) $(glue_so_OBJS) $(LDFLAGS)  -o $@

../cpuext.so: $(cpuext_so_OBJS) Types.h Interfaces.h pyglue.h Parallel.h CPUQubitProcessor.h
	$(CXX) -shared $(CXXFLAGS) $(cpuext_so_OBJS) $(LDFLAGS)  -o $@

../cudaext.so: $(cudaext_so_OBJS) Types.h CUDAQubitStates.h CUDAQubitProcessor.h pyglue.h DeviceTypes.h DeviceParallel.h
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler "$(CXXFLAGS)" $(cudaext_so_OBJS) $(LDFLAGS) -lcudart -o $@

test_gate_matrix: $(test_gate_matrix_OBJS)
	$(CXX) $(CXXFLAGS) $(test_gate_matrix_OBJS) $(LDFLAGS) -o $@

%.o: %.cu $<
	$(NVCC) -c $(INCLUDE) $(NVCCFLAGS) $(NVCCARCH) -Xcompiler "$(CXXFLAGS)" $< -o $@

%.o: %.cpp $<
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c $<
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) test_gate_matrix $(glue_so_OBJS) $(cpuext_so_OBJS) $(cudaext_so_OBJS)
