.SUFFIXES: .cpp .o .h .c

include incpath

TARGETS= ../glue.so ../cpuext.so ../cudaext.so
glue_so_OBJS=glue.o Types.o
cpuext_so_OBJS=cpuext.o CPUQubitStates.o CPUQubitProcessor.o Types.o
cudaext_so_OBJS=cudaext.o CUDAQubitStates.o CUDAQubitProcessor.o DeviceTypes.o CUDADevice.o Types.o

CXX=g++
CC=gcc
CFLAGS=-g -O2 -Wall -ggdb -Wno-format-security -fPIC -fopenmp
#CFLAGS=-g -Wall -ggdb -Wno-format-security -fPIC
CXXFLAGS=$(CFLAGS) -std=c++11
LDFLAGS=

NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=--expt-extended-lambda -std=c++11
NVCCARCH=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70

INCLUDE+=-I/usr/local/cuda/include


all: $(TARGETS)

cuda_obj : CUDAQubitStates.o CUDAQubitProcessor.o

../glue.so: $(glue_so_OBJS) Types.h Interfaces.h
	$(CXX) -shared $(CXXFLAGS) $(glue_so_OBJS) $(LDFLAGS)  -o $@

../cpuext.so: $(cpuext_so_OBJS) Types.h Interfaces.h pyglue.h parallel.h CPUQubitProcessor.h
	$(CXX) -shared $(CXXFLAGS) $(cpuext_so_OBJS) $(LDFLAGS)  -o $@

../cudaext.so: $(cudaext_so_OBJS) Types.h CUDAQubitStates.h CUDAQubitProcessor.h pyglue.h DeviceTypes.h DeviceParallel.h
	$(NVCC) -shared -Xcompiler "$(CXXFLAGS)" $(cudaext_so_OBJS) $(LDFLAGS) -lcudart -o $@

%.o: %.cu
	$(NVCC) -c $(INCLUDE) $(NVCCFLAGS) $(NVCCARCH) -Xcompiler "$(CXXFLAGS)" $< -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(glue_so_OBJS) $(cpuext_so_OBJS) $(cudaext_so_OBJS)
