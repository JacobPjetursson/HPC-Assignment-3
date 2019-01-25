SRCS	= poisson.cu init.cu datatools.cu iterator.cu
OBJS	= $(SRCS:.cu=.o)
TARGET	= poisson

OPT	= -Ofast -march=native -fno-trapping-math -funroll-loops
PIC	=
XOPTS = -Xptxas=-v
ARCH  = -arch=sm_70
OMP   = -fopenmp

CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PIC) $(OMP)" $(ARCH) $(XOPTS)

CUDA_PATH ?= /appl/cuda/10.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

SOFLAGS =
XLIBS	=

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f $(TARGET) $(OBJS)