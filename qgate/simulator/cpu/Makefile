.SUFFIXES: .cpp .o .h .c

include incpath

TARGETS= ../cpuext.so
cpuext_so_OBJS=CPUKernel.o cpuext.o

CXX=g++
CC=gcc
CFLAGS=-g -O2 -Wall -ggdb -Wno-format-security -fopenmp -fPIC
CXXFLAGS=$(CFLAGS) -std=c++11
LDFLAGS=

all: $(TARGETS)

../cpuext.so: $(cpuext_so_OBJS)
	$(CXX) -shared $(CXXFLAGS) $(cpuext_so_OBJS) $(LDFLAGS)  -o $@

%.o: %.cpp 
	$(CXX) -c $(INCLUDE) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@


.PHONY:

clean:
	rm -f $(TARGETS) $(cpuext_so_OBJS)
