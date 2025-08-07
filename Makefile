CC = gcc
CXX = g++
LDLIBS = -lpng -lm -Xcompiler -fopenmp
CFLAGS = -lm
NVFLAGS  := -std=c++11 -Xptxas="-v" -arch=sm_61
VECOPT = -fopt-info-vec-all -march=native
# CFLAGS += -pthread
# CXXFLAGS = $(CFLAGS)
TARGETS = main main_multi

all: $(TARGETS)

.PHONY: main
main: main.cu
	nvcc $(NVFLAGS) $(LDLIBS) -o $@ $?

.PHONY: main_multi
main_multi: main_multi.cu
	nvcc $(NVFLAGS) $(LDLIBS) -o $@ $?

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
