# CudaMat: A Matlab toolbox for using GPU computing in matlab by Rainer Heintzmann
# www.nanoimaging.de/CudaMat
# Define ins:qtallation location for CUDA and compilation flags compatible
# with the CUDA include files.
#
# ----------------------------------------
# How to install under Linux and Mac:
# Change the paths in this Makefile and type: make all
# alternatively under Matlab:
# mex cuda_cuda.c cudaArith.obj -I/usr/local/cuda/include/ -LC:/usr/local/cuda/lib/ -lcublas -lcufft -lcuda -lcudart
#
# See if the installation was successful: 
# applemantest(1);
#
# -------------------------------------
#  How to install under Windows:
# - install cuda SDK
# add the path of the (visual studio) cl.exe comiler into PATH (windows -> home, or right click computer)
# to compile in Matlab:
# cd c:\Pro'gram Files'\dip\CudaMat\
# system('nvcc --compile cudaArith.cu')
# 
# mex -setup
# mex cuda_cuda.c cudaArith.obj -Ic:\CUDA\include\ -LC:\CUDA\lib -lcublas -lcufft -lcuda -lcudart
#
# See if the installation was successful: 
# applemantest(1);

VERSION = 1_0_01
CUDAHOME    = /usr/local/cuda
INCLUDEDIR  = -I$(CUDAHOME)/include
INCLUDELIB  = -L$(CUDAHOME)/lib -L. -lcudaArith -lcufft -lcudart -lcublas -Wl,-rpath,$(CUDAHOME)/lib 
CFLAGS      =  -Wall -fPIC -D_GNU_SOURCE -pthread -fexceptions
COPTIMFLAGS = -Wall -O3 -funroll-loops -msse2

# Define installation location for MATLAB.
export MATLAB = /usr/local/matlab
#export MATLAB = /Applications/MATLAB_R2007b
MEX           = $(MATLAB)/bin/mex
MEXEXT        = .$(shell $(MATLAB)/bin/mexext)

# nvmex is a modified mex script that knows how to handle CUDA .cu files.
NVMEX = ./nvmex

# List the mex files to be built.  The .mex extension will be replaced with the
# appropriate extension for this installation of MATLAB, e.g. .mexglx or
# .mexa64.
MEXFILES = cuda_cuda.mex

all: libcudaArith.so $(MEXFILES:.mex=$(MEXEXT)) 

clean:
	rm -f $(MEXFILES:.mex=$(MEXEXT))

.SUFFIXES: .cu .cu_o .mexglx .mexa64 .mexmaci

.c.mexglx: libcudaArith.so cudaArith.h
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexglx: libcudaArith.so cudaArith.h
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)

.c.mexa64: libcudaArith.so cudaArith.h
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB) 

.cu.mexa64: libcudaArith.so cudaArith.h
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)

.c.mexmaci: libcudaArith.so cudaArith.h
	$(MEX) CFLAGS='$(CFLAGS)' COPTIMFLAGS='$(COPTIMFLAGS)' $< \
        $(INCLUDEDIR) $(INCLUDELIB)

.cu.mexmaci: libcudaArith.so cudaArith.h
	$(NVMEX) -f nvopts.sh $< $(INCLUDEDIR) $(INCLUDELIB)
# cudaArith: cudaArith.cu cadaArith.h
# 	nvcc -O3  -o cudaArith cudaArith.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft -lcudart 
libcudaArith.so: cudaArith.cu cudaArith.h
	nvcc cudaArith.cu -O3 --compiler-options '-fPIC' -shared -o libcudaArith.so -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft -lcudart 
