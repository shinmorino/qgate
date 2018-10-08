.SUFFIXES: .cpp .o .h .c

include incpath

TARGETS= ../cpuext.so ../cudaext.so
cpuext_so_OBJS=CPURuntime.o cpuext.o Types.o
cudaext_so_OBJS=CUDARuntime.o DeviceSum.o DeviceTypes.o cudaext.o Types.o

CXX=g++
CC=gcc
CFLAGS=-g -O2 -Wall -ggdb -Wno-format-security -fPIC #-fopenmp
#CFLAGS=-g -Wall -ggdb -Wno-format-security -fPIC
CXXFLAGS=$(CFLAGS) -std=c++11
LDFLAGS=

NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=--expt-extended-lambda -std=c++11
INCLUDE+=-I/usr/local/cuda/include


all: $(TARGETS)

../cpuext.so: $(cpuext_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $(cpuext_so_OBJS) $(LDFLAGS)  -o $@

../cudaext.so: $(cudaext_so_OBJS)
	$(NVCC) -shared -Xcompiler "$(CXXFLAGS)" $(cudaext_so_OBJS) $(LDFLAGS) -lcudart -o $@

%.o: %.cu
	$(NVCC) -c $(INCLUDE) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" $< -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cpuext_so_OBJS) $(cudaext_so_OBJS)
