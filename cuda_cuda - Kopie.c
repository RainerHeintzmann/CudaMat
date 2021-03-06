/************************* CudaMat ******************************************
 *   Copyright (C) 2008-2009 by Rainer Heintzmann                          *
 *   heintzmann@gmail.com                                                  *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; Version 2 of the License.               *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************
Compile with:
Windows:
mex cuda_cuda.c cudaArith.obj -Ic:\\CUDA\include\ -Lc:\\CUDA\lib64\ -lcublas -lcufft -lcudart

Windows 64 bit:
No Cula:
mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" -lcublas -lcufft -lcudart
Cula:
mex cuda_cuda.c cudaArith.obj "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" "-LC:\Program Files\CULA\R14\lib64" -lcublas -lcufft -lcudart -lcula_core -lcula_lapack

Linux:
mex -Dnocula cuda_cuda.c cudaArith.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart -v
or Linux including CULA support:
mex cuda_cuda.c cudaArith.o -I/usr/local/cuda/include -I/usr/local/cula/include -L/usr/local/cuda/lib64 -L/usr/local/cula/lib64 -lcublas -lcufft -lcudart -lcula 

 */
/* 
        This is a mex file to perform a number of operations using cuda on graphic cards.
 *      The sytax is always cuda_cuda('operation',arg1, arg2) whereas arg2 can be empty.
        'alloc' convert a matlab single into a cuda object which is stored on the graphics card. Returns integer reference to cuda object.
 */


#include "mex.h"
#include "cufft.h"
#include "cuda_runtime.h"
#include "cublas.h"
#define externC
#include "cudaArith.h"
#include "matrix.h"
#include "stdio.h"
#include "string.h"

#include "cufft.h"

#ifndef NOCULA
// #include "culadevice.h"    // Only for old Cula releases
#include "cula.h"   // Later releases such as R14
#include "cula_device.h"   // Later releases such as R14
#endif

// #define DEBUG

// #define UseHeap            // if defined the heap allocation and recycling is used. Otherwise normal allocation and free is used
#define AllocBlas          // use the cublasAlloc and free functions instead of cudaMalloc and cudaFree

#define MAX_ARRAYS 65738   // maximum number of arrays simultaneously on card

#ifdef UseHeap
#define MAX_HEAP 5        // size of the heap of arrays to recycle
#endif

#define CHECK_CUDAREF(p)     {if ((((double) (int) p) != p) || (p < 0) || (p >= MAX_ARRAYS)) \
          mexErrMsgTxt("cuda: Reference must be an integer between 0 and max_array\n");}

#define CHECK_CUDASIZES(ref1,ref2)     {if (cuda_array_dim[ref1] != cuda_array_dim[ref2]) mexErrMsgTxt("cuda: Arrays have different dimensionalities\n"); \
         int d; for (d=0;d< cuda_array_dim[ref1]) if (cuda_array_size[ref1][d] != cuda_array_size[ref2][d])\
          mexErrMsgTxt("cuda: Array sizes must be equal in all dimensions\n");}

#ifdef DEBUG
#define Dbg_printf(arg) printf(arg)
#define Dbg_printf2(arg1,arg2) printf(arg1,arg2)
#define Dbg_printf3(arg1,arg2,arg3) printf(arg1,arg2,arg3)
#define Dbg_printf4(arg1,arg2,arg3,arg4) printf(arg1,arg2,arg3,arg4)
#define Dbg_printf5(arg1,arg2,arg3,arg4,arg5) printf(arg1,arg2,arg3,arg4,arg5)
#else
#define Dbg_printf(arg) 
#define Dbg_printf2(arg1,arg2) 
#define Dbg_printf3(arg1,arg2,arg3) 
#define Dbg_printf4(arg1,arg2,arg3,arg4) 
#define Dbg_printf5(arg1,arg2,arg3,arg4,arg5) 
#endif

//  ----------------- Macros of code snippets, defining common ways of calling Cuda ---------
#define CallCUDA_BinaryFkt(FktName,AllocFkt)                                         \
    const char *ret=0;                                                                        \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "needs three arguments\n");               \
    if (isComplexType(getCudaRefNum(prhs[1])) && isComplexType(getCudaRefNum(prhs[2]))) {   \
        Dbg_printf("cuda: complex array " #FktName " complex array\n");                     \
        ret=CUDAcarr_##FktName##_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); } \
    else if (isComplexType(getCudaRefNum(prhs[1]))) {                                       \
        Dbg_printf("cuda: complex array " #FktName " float array\n");                       \
        ret=CUDAcarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); } \
    else if (isComplexType(getCudaRefNum(prhs[2]))) {                                       \
        Dbg_printf("cuda: float array " #FktName " complex array\n");                       \
        ret=CUDAarr_##FktName##_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[2]),getTotalSizeFromRef(prhs[1])); }\
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \

//  ----------------- for calling with array and constant ---------
#define CallCUDA_UnaryFktConst(FktName,AllocFkt)                                            \
    const char *ret=0;                                                                      \
    int ref;                                                                                \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "_alpha needs three arguments\n");        \
    ref=getCudaRefNum(prhs[1]);                                                         \
    if (mxIsComplex(prhs[2])) {                                                             \
        double  myreal = mxGetScalar(prhs[2]);                                              \
        double  myimag = * ((double *) (mxGetPi(prhs[2])));                                 \
        if (isComplexType(ref)) {                                        \
            Dbg_printf("cuda: complex array " #FktName " complex-const\n");                 \
            ret=CUDAcarr_##FktName##_const(getCudaRef(prhs[1]),(float) myreal,(float) myimag,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            int tmp=cuda_array_type[ref]; float * narr=AllocFkt(prhs[1]);cuda_array_type[ref]=tmp; cuda_array_type[ref]=scomplex;   \
            Dbg_printf("cuda: float array " #FktName " complex-const\n");                   \
            ret=CUDAarr_##FktName##_Cconst(getCudaRef(prhs[1]),(float) myreal,(float) myimag,narr,getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[2]);                                                \
        if (isComplexType(ref)) {                                        \
            Dbg_printf("cuda: complex array " #FktName " real-const\n");                    \
            ret=CUDAcarr_##FktName##_const(getCudaRef(prhs[1]),(float) alpha,0.0,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1]));    \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName " real-const\n");                      \
            ret=CUDAarr_##FktName##_const(getCudaRef(prhs[1]),(float) alpha,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } 


//  ----------------- for calling with array and constant but in Reverse order---------
#define CallCUDA_UnaryFktConstR(FktName,AllocFkt)                                           \
    const char *ret=0;                                                                      \
    int ref;                                                                                \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "_alpha needs three arguments\n");        \
    ref=getCudaRefNum(prhs[1]);                                                         \
    if (mxIsComplex(prhs[2])) {                                                             \
        double  myreal = mxGetScalar(prhs[2]);                                              \
        double  myimag = * ((double *) (mxGetPi(prhs[2])));                                 \
        if (isComplexType(ref)) {                                                           \
            Dbg_printf("cuda: complex array " #FktName " complex-const\n");                 \
            ret=CUDAconst_##FktName##_carr(getCudaRef(prhs[1]),(float) myreal,(float) myimag,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            int tmp=cuda_array_type[ref]; float * narr=AllocFkt(prhs[1]);cuda_array_type[ref]=tmp;  cuda_array_type[ref]=scomplex;  \
            Dbg_printf("cuda: float array " #FktName " complex-const\n");                   \
            ret=CUDACconst_##FktName##_arr(getCudaRef(prhs[1]),(float) myreal,(float) myimag,narr,getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[2]);                                                \
        if (isComplexType(ref)) {                                                           \
            Dbg_printf("cuda: complex array " #FktName " real-const\n");                    \
            ret=CUDAconst_##FktName##_carr(getCudaRef(prhs[1]),(float) alpha,0.0,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1]));    \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName " real-const\n");                      \
            ret=CUDAconst_##FktName##_arr(getCudaRef(prhs[1]),(float) alpha,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } 

//  --------- The ones below are FOR REAL-VALUED Functions only --- (e.g. comparison operations) ----------
#define CallCUDA_BinaryHRealFkt(FktName,AllocFkt)                                         \
    const char *ret=0;                                                                        \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "needs three arguments\n");               \
    if (isComplexType(getCudaRefNum(prhs[1])) && isComplexType(getCudaRefNum(prhs[2]))) {   \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real argument data.\n");}  \
    else if (isComplexType(getCudaRefNum(prhs[1]))) {                                       \
        Dbg_printf("cuda: complex array " #FktName " float array\n");                       \
        ret=CUDAcarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); } \
    else if (isComplexType(getCudaRefNum(prhs[2]))) {                                       \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real argument data.\n");}  \
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \

//  --------- The ones below are FOR REAL-VALUED Functions only --- (e.g. comparison operations) ----------
#define CallCUDA_BinaryRealFkt(FktName,AllocFkt)                                         \
    const char *ret=0;                                                                      \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "needs three arguments\n");               \
    if (isComplexType(getCudaRefNum(prhs[1])) || isComplexType(getCudaRefNum(prhs[2])))     \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real valued data.\n");  \
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  


//  ----------------- for calling with array and constant ---------
#define CallCUDA_UnaryRealFktConst(FktName,AllocFkt)                                                 \
    const char *ret=0;                                                                      \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "_alpha needs three arguments\n");        \
    if (mxIsComplex(prhs[2])) {                                                             \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real valued data.\n");   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[2]);                                                \
        if (isComplexType(getCudaRefNum(prhs[1]))) {                                        \
          mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real valued data.\n"); \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName " real-const\n");                      \
            ret=CUDAarr_##FktName##_const(getCudaRef(prhs[1]),(float) alpha,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
        }                                                                                   \
    }                                                                                       \
   if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");} 


//  ----------------- for calling with array and constant but in Reverse order   (e.g. for alpha / array) ---------
#define CallCUDA_UnaryRealFktConstR(FktName,AllocFkt)                                       \
    const char *ret=0;                                                                      \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "_alpha needs three arguments\n");        \
    if (mxIsComplex(prhs[2])) {                                                             \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real valued data.\n");   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[2]);                                                \
        if (isComplexType(getCudaRefNum(prhs[1]))) {                                        \
          mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real valued data.\n"); \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName " real-const\n");                      \
            ret=CUDAconst_##FktName##_arr(getCudaRef(prhs[1]),(float) alpha,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");} \
        }                                                                                   \
    } 

//  ----------------- Unary function for real valued data only ------AllocCommand determines whether the result type is that same, complex or real ----------
#define CallCUDA_UnaryRealFkt(FktName,AllocCommand)                                             \
    const char *ret=0;                                                                      \
    if (nrhs != 2) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");              \
        if (isComplexType(getCudaRefNum(prhs[1]))) {                                        \
            mexErrMsgTxt("cuda error " #FktName ": Tried to apply to complex valued data."); \
            ret="Error " #FktName ": Tried to apply to complex valued data.";    \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName "\n");                                 \
            ret=CUDA##FktName##_arr(getCudaRef(prhs[1]),AllocCommand(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");} \
    } 

//  ----------------- Unary function  ------AllocCommand determines whether the result type is that same, complex or real ----------
#define CallCUDA_UnaryFkt(FktName,AllocCommand)                                             \
    const char *ret=0;                                                                      \
    if (nrhs != 2) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");              \
        if (isComplexType(getCudaRefNum(prhs[1]))) {                                        \
            Dbg_printf("cuda: complex array " #FktName "\n");                               \
            ret=CUDA##FktName##_carr(getCudaRef(prhs[1]),AllocCommand(prhs[1]),getTotalSizeFromRef(prhs[1]));    \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName "\n");                                 \
            ret=CUDA##FktName##_arr(getCudaRef(prhs[1]),AllocCommand(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");} \
    } 



static const char * ERROR_NAMES[]={
    "SUCCESS",
    "INVALID_PLAN",
    "ALLOC_FAILED",
    "INVALID_TYPE",
    "INVALID_VALUE",
    "INTERNAL_ERROR",
    "EXEC_FAILED",
    "SETUP_FAILED",
//    "SHOWDOWN_FAILED",
    "INVALID_SIZE",
    "" // needed to stop the search
};

enum CUDA_TYPE {
    single,
    int16,
//    fftSingle,
    fftHalfSComplex,
    scomplex
};

static const char * CUDA_TYPE_NAMES[]={
    "single",
    "int16",
 //   "fftSingle",
    "fftHalfSComplex",
    "scomplex",
    "" // needed to stop the search
};

static int CUDA_TYPE_SIZE[]={
    sizeof(float),
    2,
//    sizeof(float),
    sizeof(cufftComplex),
    sizeof(cufftComplex)
};

static int CUDA_MATLAB_CLASS[]={
    mxSINGLE_CLASS,
    mxINT16_CLASS,
//    mxSINGLE_CLASS,
    mxSINGLE_CLASS,
    mxSINGLE_CLASS
};


static float * cuda_arrays[MAX_ARRAYS];  // static array of cuda arrays
static cufftHandle cuda_FFTplan[MAX_ARRAYS];  // stores fft plans (when needed)
static int cuda_array_dim[MAX_ARRAYS];  // type tags see CUDA_TYPE definitions above
static int * cuda_array_size[MAX_ARRAYS];  // dynamically allocated 
static int cuda_array_origFTsize[MAX_ARRAYS];  // for storing the original data size, when doing FFTs
static float cuda_array_FTscale[MAX_ARRAYS];  // to do the maginitude correction
static int cuda_array_type[MAX_ARRAYS];  // type tags see CUDA_TYPE definitions above

static int free_array=0;   // next number of array to fill
static int cuda_curr_arrays=0;
static int cuda_initialized=0;
// static int sizes[100];
// static int dims=0,totalsize=0;
static float * pOne=0, * pZero=0;

static int ignoreDelete=0;  // Needed to avoid delete (or copyiing the whole array) after subassign
static int ignoreRef=-1;  // Needed to avoid delete (or copyiing the whole array) after subassign
static void * fastMem=0, * fastMemI=0;
static const int fastMemSize=1024; // number of byte for fast transfer

#ifdef UseHeap
static void * mem_heap[MAX_HEAP];   // memory which  can be reused if size matches
static int memsize_heap[MAX_HEAP];  // sizes in bytes
static int mem_heap_pos=0;
static int mem_heap_allocated=0;
#endif

/**************************************************************************/

/* MATLAB stores complex numbers in separate arrays for the real and
   imaginary parts.  The following functions take the data in
   this format and pack it into a complex work array, or
   unpack it, respectively.  
   We are using cufftComplex defined in cufft.h to  handle complex on Windows and Linux

*/

#ifndef NOCULA
void checkCULAStatus(char * text, culaStatus status)
{
    if(!status)
        return;

    if(status == culaArgumentError)
        printf("%s: Invalid value for parameter %d\n", text,culaGetErrorInfo());
    else if(status == culaDataError)
        printf("%s: Data error (%d)\n", text,culaGetErrorInfo());
    else if(status == culaBlasError)
        printf("%s: Blas error (%d)\n", text,culaGetErrorInfo());
    else if(status == culaRuntimeError)
        printf("%s: Runtime error (%d)\n", text,culaGetErrorInfo());
    else
        printf("%s: %s\n", text,culaGetStatusString(status));

    mexErrMsgTxt("CULA Error: Bailing out\n");
    // culaShutdown();
}
#endif


void checkCudaError(char * text, cudaError_t err)
{
    if(!err)
        return;

    printf("%s: %s\n", text, cudaGetErrorString(err));

    mexErrMsgTxt("CULA Error: Bailing out\n");
}


// returns array size in 3D coordinates (expanding with ones)
void get3DSize(int ref, int * mysize) {
    int d;
    for (d=0;d<3;d++)
        if (d< cuda_array_dim[ref])
            mysize[d]=cuda_array_size[ref][d];
        else
            mysize[d]=1;
}
    

 int getCudaRefNum(const mxArray * arg) {
    double cudaref;
    if (! mxIsDouble(arg))
      mexErrMsgTxt("cuda: Obtaining reference number. Number must be a double");
    
    cudaref = mxGetScalar(arg);

    /* printf("cuda: get ref %g, next free array %d\n",cudaref,free_array); */

    CHECK_CUDAREF(cudaref);
    if (cuda_array_size[(int) cudaref] == 0)
         mexErrMsgTxt("cuda: Trying to access non-existing cuda reference.");
        
    return (int) cudaref;
 }

bool isComplexType(int ref) {
    return (cuda_array_type[ref] >= fftHalfSComplex);
}

int getTotalSize(int dim,const int * sizevec) {
    int totalsize=1,d;
    for (d=0;d<dim;d++)
        totalsize *= sizevec[d];
    Dbg_printf2("Totalsize = %d\n",totalsize);

    return totalsize;
}

int getTotalSizeFromRefNum(int pos) {
    return getTotalSize(cuda_array_dim[pos],cuda_array_size[pos]);
}

int getTotalSizeFromRef(const mxArray * arg) {
    return getTotalSizeFromRefNum(getCudaRefNum(arg));
}

int getTotalFloatSizeFromRef(const mxArray * arg) {   // sizes in floating point numbers
    int ref=getCudaRefNum(arg);
    if (isComplexType(ref))  // this is a complex datatyp
        return getTotalSizeFromRefNum(ref)*2;
    else
        return getTotalSizeFromRefNum(ref);
}

float * MemAlloc(int mysize) {   // returns an array from the heap or a fresh array
#ifdef UseHeap
    int p;
    for (p=0;p<MAX_HEAP;p++)
        if (mysize == memsize_heap[p])
        {
            memsize_heap[p]=0;
            Dbg_printf2("Array allocated from heap %d\n",p);    
            float * tmp=(float *) mem_heap[p];
            mem_heap[p]=0;
            mem_heap_allocated--;
            return tmp;
        }
    Dbg_printf2("No matching size in heap\n",mysize);    
#endif
    float * p_cuda_data; float ** pp_cuda_data= & p_cuda_data;
    int custatus;
#ifdef AllocBlas
    custatus = cublasAlloc((mysize+3)/sizeof(float), sizeof(float), (void**) pp_cuda_data);
    if (custatus != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("cuda: Device memory allocation error (on card)\n");
        return 0;
    }
#else
    custatus=cudaMalloc((void **) pp_cuda_data, mysize);
    if (custatus!=cudaSuccess) { printf("cuda Malloc: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
#endif

    Dbg_printf2("Allocated %d bytes\n",mysize);    
    return p_cuda_data;
}

void MemFree(int ref) {
#ifdef UseHeap
    int totalsize=getTotalSizeFromRefNum(ref) * sizeof(float);
    int similarCnt=0;
    int p;
    for (p=0;p<MAX_HEAP;p++)
    {
        if (memsize_heap[p] == 0)
        {
            mem_heap[p]=(void *) cuda_arrays[ref];
            memsize_heap[p]=totalsize;
            cuda_arrays[ref]=0;
            Dbg_printf3("Array reference %d stored to heap place %d\n",ref,p);
            mem_heap_allocated++;
            return;
        }
        if (totalsize == memsize_heap[p])
            similarCnt++;
    }
    if (similarCnt > (int) sqrt(MAX_HEAP+1))  // enough of these there already
    {
#endif
#ifdef AllocBlas
        cublasFree(cuda_arrays[ref]);
#else
        int custatus=cudaFree(cuda_arrays[ref]);
        if (custatus!=cudaSuccess) { printf("cuda Free: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
#endif
        cuda_arrays[ref]=0;
#ifdef UseHeap

        Dbg_printf2("Array reference %d freed entirely as already in heap\n",ref);
    }
    else  // we need to free another one and keep this one
    {
#ifdef AllocBlas
        cublasFree(mem_heap[mem_heap_pos]);
#else
        custatus=cudaFree(mem_heap[mem_heap_pos]);
        if (custatus!=cudaSuccess) { printf("cuda Free: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
#endif
        mem_heap[mem_heap_pos]=(void *) cuda_arrays[ref];
        memsize_heap[p]=totalsize;
        cuda_arrays[ref]=0;
        Dbg_printf3("Array reference %d kept but a different array %d freed from heap\n",ref,mem_heap_pos);
        mem_heap_pos = (mem_heap_pos+1) % MAX_HEAP;  // next time free a different one
    }
#endif
}



int MatlabTypeFromCuda(int ref) {   /// returns the Matlab class for the specified datatype
    return CUDA_MATLAB_CLASS[cuda_array_type[ref]];
}

int MatlabRealCpxFromCuda(int ref) {  // returns a specifier to indicate whether this is real or complex valued data in Matlab
    if (!isComplexType(ref))
        return mxREAL;
    else
        return mxCOMPLEX;
}

int cudaTypeFromArg(const char * MatlabString) {
    int i;
    /* Copy the string data from prhs[0] into a C string */
    Dbg_printf2("cuda: type to allocate is %s\n",MatlabString);
    // printf("cuda: Number of CUDA_TYPENAMES is %d\n",sizeof(CUDA_TYPE_NAMES));  
    // return 0;
    
    for (i=0;CUDA_TYPE_NAMES[i][0] != 0;i++)
        if (strcmp(MatlabString,CUDA_TYPE_NAMES[i])==0)
        { 
            return i;
        }
    mexErrMsgTxt("cuda: Unknown type!\n");
    return -1;
}

int updataFreeArray() { // looks for the next free position to store and array
    int howmany=0;
    int pos=free_array;
    for(;cuda_array_size[pos] != 0;pos=(pos+1)%MAX_ARRAYS)
        { howmany++;
        if (howmany > MAX_ARRAYS+1) {
            printf("cuda: MAX_ARRAYS is %d\n",MAX_ARRAYS);
            mexErrMsgTxt("cuda: Maximum number of allocatable arrays reached!\n");
            return -1;
            }
        }
    free_array=pos;
    return free_array;
}

 float * getCudaRef(const mxArray * arg) {
    return cuda_arrays[getCudaRefNum(arg)];
 }

 void ReduceToHalfComplex(int pos) {
        cuda_array_origFTsize[pos] = cuda_array_size[pos][0]; // needs to be stored to recover it when doing inverse FTs
        cuda_array_size[pos][0] = ((int) ((cuda_array_size[pos][0]) / 2)) +1;  // reduce size accordingly for half complex values
 }
 void ExpandToFullReal(int pos) {
        cuda_array_size[pos][0] = cuda_array_origFTsize[pos];  // restore size for real space
 }

 void swapMatlabSize(int * mysize, int dims) {
    int tmp=mysize[0];
     if (dims>1)
        {mysize[0]=mysize[1];mysize[1]=tmp;}  // the bad matlab problem with x and y dimensions
 }
 
 void cudaCopySizeVec(int pos, const int * sizevec,int dims) {   // copies a matlab sizevector to the array
    double FTscale = 1.0;
    int d;
    cuda_array_size[pos]=calloc(dims,sizeof(cuda_array_size[0][0]));
    for (d=0;d<dims;d++) {
        cuda_array_size[pos][d]=sizevec[d];  // copy sizes
        FTscale *= sizevec[d];
    }
    //swapMatlabSize(cuda_array_size[pos],dims);
    
    cuda_array_FTscale[pos]=(float) (1.0/sqrt(FTscale));
    cuda_array_dim[pos]=dims;
    Dbg_printf2("CopySizeVec of %d dimensional vector\n",dims);
 }
 void cudaCopySizeVecD(int pos, const double * sizevec,int dims) {    // copies a matlab sizevector to the array
    double FTscale = 1.0;
    int d;
    cuda_array_size[pos]=calloc(dims,sizeof(cuda_array_size[0][0]));
    for (d=0;d<dims;d++) {
        cuda_array_size[pos][d]=(int) sizevec[d];  // copy sizes
        FTscale *= sizevec[d];
    }
    //swapMatlabSize(cuda_array_size[pos],dims);

    cuda_array_FTscale[pos]=(float) (1.0/sqrt(FTscale));
    cuda_array_dim[pos]=dims;
    Dbg_printf2("CopySizeVecD of %d dimensional vector\n",dims);
 }
 
 float * cudaAllocDetailed(int dims, const  int * sizevec, int cuda_type) {   // make a new array
    float * p_cuda_data; // float ** pp_cuda_data=& p_cuda_data;
    int pos=updataFreeArray(),ts;
    cudaCopySizeVec(pos,sizevec,dims);
    
    cuda_array_type[pos]=cuda_type;
    
    ts=getTotalSize(dims,cuda_array_size[pos]);
    if (ts>0) {
        p_cuda_data=MemAlloc(ts*CUDA_TYPE_SIZE[cuda_type]);
    //int custatus = cublasAlloc(ts, CUDA_TYPE_SIZE[cuda_type], (void**) pp_cuda_data);
    //if (custatus != CUBLAS_STATUS_SUCCESS) {
    //    mexErrMsgTxt("cuda: Device memory allocation error (on card)\n");
    //    return 0;
    //}

    }
    else p_cuda_data=0;
    //if (cuda_type==fftHalfSComplex)
    //    cuda_array_size[pos][dims-1] = sizevec[dims-1];   // restore to the original value. Just the allocation needs to be bigger
    
    cuda_arrays[pos]=p_cuda_data; // save it in the array
    cuda_curr_arrays++;
    Dbg_printf3("constructed cuda array nr %d of %d dimensions\n",pos,dims);
    return p_cuda_data;
 }

 float * cudaAlloc(const mxArray * arg) {   // make a new array with same properties as other array
     int ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], cuda_array_type[ref]);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     cuda_array_FTscale[free_array]=cuda_array_FTscale[ref];  // to do the maginitude correction
     cuda_array_type[free_array]=cuda_array_type[ref];  // type tags see CUDA_TYPE definitions above
     return ret;
     }

float * cudaAllocReal(const mxArray * arg) {   // make a new array with same properties as other array, but ignores Complex and makes it Real
     int ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], single);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     cuda_array_FTscale[free_array]=cuda_array_FTscale[ref];  // to do the maginitude correction
     cuda_array_type[free_array]=single;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

float * cudaAllocComplex(const mxArray * arg) {   // make a new array with same properties as other array, but ignores type and makes it Complex
     int ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], scomplex);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     cuda_array_FTscale[free_array]=cuda_array_FTscale[ref];  // to do the maginitude correction
     cuda_array_type[free_array]=scomplex;  // type tags see CUDA_TYPE definitions above
     return ret;
     }
     
 float * getMatlabFloatArray(const mxArray * arg, int * nd) {
  (* nd) = mxGetNumberOfDimensions(arg);
  // int * sz = mxGetDimensions(arg);
  if (*nd > 0) {
  /* Pointer for the real part of the input */
    return (float *) mxGetData(arg);
  }
  mexErrMsgTxt("cuda: getMatlabFloatArray; data is zero dimensional\n");
  return 0;
 }
  
 float * cudaPut(const mxArray * arg) {   // copies Matlab array into cuda
    int dims = mxGetNumberOfDimensions(arg);
    const int * sizevec = mxGetDimensions(arg);
    int ts;
    int cuda_type = -1;
    float * p_cuda_data=0;
    const char * TypeName=mxGetClassName(arg);    

    Dbg_printf2("cudaPut  Classname=%s\n",mxGetClassName(arg));
     
    if (! mxIsSingle(arg))
        mexErrMsgTxt("cuda: Datatype for cuda arrays needs to be single precision, or single precision complex\n");
    ts=getTotalSize(dims,sizevec);
    if (CUDAmaxSize() < ts)
        mexErrMsgTxt("cuda: Array too big for available number of threads\n");
    if (mxIsComplex(arg))
        cuda_type = cudaTypeFromArg("scomplex");
    else
        cuda_type=cudaTypeFromArg(TypeName);  //  "single"  Typename

    if (mxIsComplex(arg))
    {
        p_cuda_data = cudaAllocDetailed(dims,sizevec,cuda_type);
      /* Pointer for the real part of the input */
        if (ts > 0) {
        float * pr= (float *) mxGetPr(arg);
        float * pi= (float *) mxGetPi(arg);
        int custatus=cudaMemcpy2D(p_cuda_data, sizeof(float)*2,pr, sizeof(float),  sizeof(float), ts, cudaMemcpyHostToDevice);
        if (custatus!=cudaSuccess) { printf("cuda Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        //int custatus = cublasSetVector(ts, sizeof(pr[0]), pr, 1, p_cuda_data, 2);
        //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (write C to cuda)\n");return 0;}
        custatus=cudaMemcpy2D(p_cuda_data+1, sizeof(float)*2,pi, sizeof(float),  sizeof(float), ts, cudaMemcpyHostToDevice);
        if (custatus!=cudaSuccess) { printf("cuda Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        //custatus = cublasSetVector(getTotalSize(dims,sizevec), sizeof(pi[0]), pi, 1, p_cuda_data+1, 2);
        //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (write C to cuda)\n");return 0;}
        Dbg_printf("cuda: copied Complex data to device\n");
        }
    }
    else
    {
        int nd;
        /* Initialize the device matrices with the host matrices */
        p_cuda_data = cudaAllocDetailed(dims,sizevec,cuda_type);
        if (ts > 0) {
        float * p_matlab_data = getMatlabFloatArray(arg, & nd);
        int custatus=cudaMemcpy(p_cuda_data, p_matlab_data, sizeof(p_matlab_data[0])*ts, cudaMemcpyHostToDevice);
        if (custatus!=cudaSuccess) { printf("cuda Memcpy: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        //int custatus = cublasSetVector(ts, sizeof(p_matlab_data[0]), p_matlab_data, 1, p_cuda_data, 1);
        //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Memcpy Device access error (write C to cuda)\n");return 0;}
        Dbg_printf("cuda: copied Float data to device\n");
        }
    }
    return p_cuda_data;
}

float * cudaPutVal(float value) {    // writes a single value into a cuda array
    float * p_cuda_data;
    int custatus;

    p_cuda_data=MemAlloc(sizeof(float));

   /* Initialize the device matrices with the host matrices */
    custatus=cudaMemcpy(p_cuda_data, &value, sizeof(value), cudaMemcpyHostToDevice);
    if (custatus!=cudaSuccess) { printf("cuda Memcpy: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 

    // custatus = cublasSetVector(1, sizeof(value), & value, 1, p_cuda_data, 1);
    //if (custatus != CUBLAS_STATUS_SUCCESS) {
    //    mexErrMsgTxt("cuda: Device access error (write C value to cuda)\n");
    //    return 0;
    //}    
    return p_cuda_data;
}

void cudaSetSize(const mxArray * arg, const mxArray * sizes) {  // overwrites current sizes with new sizes
    int pos=getCudaRefNum(arg);
    // int sd = mxGetNumberOfDimensions(sizes);
    const int * sv = mxGetDimensions(sizes);
    int dims = sv[1];
    const double * sizevec = mxGetPr(sizes);
    if (cuda_array_size[pos] != 0)
        { free(cuda_array_size[pos]); cuda_array_size[pos]=0;}
    
    cudaCopySizeVecD(pos,sizevec,dims);

    return;
}

mxArray * cudaGetSize(const mxArray * arg) {  // returns a vector with sizes
    int ref=getCudaRefNum(arg);
    mxArray * ret = mxCreateNumericMatrix(1, cuda_array_dim[ref], mxDOUBLE_CLASS, mxREAL);
    double * ar = mxGetPr(ret); // pointer to real part of array
    int d, dims=cuda_array_dim[ref];
    //swapMatlabSize(cuda_array_size[ref],dims);
    for (d=0;d<dims;d++)
        ar[d] = cuda_array_size[ref][d];
    //swapMatlabSize(cuda_array_size[ref],dims);
    return ret;
}

 mxArray * cudaGet(const mxArray * arg) {  // from device to host
    int ref=getCudaRefNum(arg);
    //int cuda_type=cuda_array_type[ref];
    //int dims=cuda_array_dim[ref];
    //int saveddim=cuda_array_size[ref][dims-1];
    //if (cuda_type==fftHalfSComplex)
    //    cuda_array_size[ref][dims-1] = ((int) saveddim)/2+1;   // restore to the original value. Just the allocation needs to be bigger

    mxArray * ret = 0;
    double * ar =0;
    double * ai=0;
    int custatus;

    /* Copy result back to host */
    // cudaMemcpy( ar, getCudaRef(arg), getTypeSize(arg)*getTotalSizeFromRef(arg), cudaMemcpyDeviceToHost);
    
    int totalsize=getTotalSizeFromRef(arg);  // size in floating point values
    Dbg_printf2("Memcpy totalsize in bytes: %d\n",sizeof(float)*totalsize);
    
    if (totalsize*sizeof(float) < fastMemSize)
        ar=(double *) fastMem;
    else {
        ret=mxCreateNumericArray(cuda_array_dim[ref], cuda_array_size[ref], MatlabTypeFromCuda(ref), MatlabRealCpxFromCuda(ref));
        ar=mxGetPr(ret); // pointer to real part of array
    }
    
    if (MatlabRealCpxFromCuda(ref) == mxCOMPLEX)
        {
            /* Copy result back to host */
            // cudaMemcpy( input_single, rhs_complex_d, sizeof(cufftComplex)*N*M*P, cudaMemcpyDeviceToHost);
            //int custate=cudaMemcpy( ar,  getCudaRef(arg), sizeof(cufftComplex)*N*M*P, cudaMemcpyDeviceToHost);
            //if (custatus != cudaSucecss) {mexErrMsgTxt("cuda: Device access error (read real-part cuda to C)\n");return 0;}
            
            //custatus = cublasGetVector(getTotalSizeFromRef(arg), CUDA_TYPE_SIZE[cuda_array_type[ref]], getCudaRef(arg), 1, ar, 1);
            custatus=cudaMemcpy2D( ar, sizeof(float),getCudaRef(arg), sizeof(float)*2,  sizeof(float), totalsize, cudaMemcpyDeviceToHost);
            if (custatus!=cudaSuccess) { printf("cuda Get Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 

            if (totalsize*sizeof(float) < fastMemSize)
                ai=(double *) fastMemI;
            else
                ai = mxGetPi(ret);
            custatus=cudaMemcpy2D( ai, sizeof(float),getCudaRef(arg)+1, sizeof(float)*2,  sizeof(float), totalsize, cudaMemcpyDeviceToHost);
            if (custatus!=cudaSuccess) { printf("cuda Get Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
            //custatus = cublasGetVector(totalsize, sizeof(float), getCudaRef(arg), 2, ar, 1);
            //custatus = cublasGetVector(totalsize, sizeof(float), getCudaRef(arg)+1, 2, ai, 1);
            //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (real-part cuda to C)\n");return 0;}
        }
    else
        {
            custatus=cudaMemcpy(ar, getCudaRef(arg), CUDA_TYPE_SIZE[cuda_array_type[ref]]*totalsize, cudaMemcpyDeviceToHost);
            if (custatus!=cudaSuccess) { printf("cuda Get Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
            //custatus = cublasGetVector(getTotalSizeFromRef(arg), CUDA_TYPE_SIZE[cuda_array_type[ref]], getCudaRef(arg), 1, ar, 1);
            //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (cuda to C)\n");return 0;}
        }
    //if (cuda_type==fftHalfSComplex)
    //    cuda_array_size[ref][dims-1] = ((int) saveddim)/2+1;   // restore to the original value. Just the allocation needs to be bigger
    if (totalsize*sizeof(float) < fastMemSize) {
        if (totalsize==1 && (MatlabRealCpxFromCuda(ref) != mxCOMPLEX))
            ret = mxCreateDoubleScalar((double)((float *)fastMem)[0]);
        else {
            ret=mxCreateNumericArray(cuda_array_dim[ref], cuda_array_size[ref], MatlabTypeFromCuda(ref), MatlabRealCpxFromCuda(ref));
            memcpy(mxGetPr(ret),fastMem,totalsize*sizeof(float));
            if (MatlabRealCpxFromCuda(ref) == mxCOMPLEX)
                memcpy(mxGetPi(ret),fastMemI,totalsize*sizeof(float));
        }
    }


    Dbg_printf4("cuda: Got type %s sizeX %d sizeY %d\n",CUDA_TYPE_NAMES[cuda_array_type[ref]],cuda_array_size[ref][0],cuda_array_size[ref][1]);
    return ret;
 } 

 void cudaDelete(int cudaref) {
     float * myref=cuda_arrays[cudaref];
     if (cuda_array_size[cudaref] == 0)
        mexErrMsgTxt("cuda: Attempt to delete non-existing reference\n");
     if (myref != 0)
     {
         MemFree(cudaref);
         // cudaFree(myref);
         // cublasFree(myref);     
         cuda_arrays[cudaref]=0;
     }
     free(cuda_array_size[cudaref]);  // free size information
     cuda_array_size[cudaref]=0;
     cuda_array_dim[cudaref]=0;
     if (cuda_FFTplan[cudaref] != 0)
         cufftDestroy(cuda_FFTplan[cudaref]); // delete the current plan (if exists)
     cuda_FFTplan[cudaref]=0;    
     cuda_curr_arrays=cuda_curr_arrays-1;  // reduce number of arrays by one
     //free_array=cudaref; // to keep the array indices low
     Dbg_printf2("cuda: deleted object reference %d\n",cudaref);
 }
 
 float * cloneArray(const mxArray * arg) {
     float * myref= getCudaRef(arg);
     float * newp=cudaAlloc(arg);
     Dbg_printf2("copying array, total size = %d \n",getTotalFloatSizeFromRef(arg));
     cublasScopy(getTotalFloatSizeFromRef(arg), myref, 1, newp, 1);
     if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (cloneArray)\n");return 0;}
     return newp;
 }

 float * copyToCpx(const mxArray * arg) {   // creates an array from real and upcasts it to complex
     int ref=getCudaRefNum(arg);
     float * myref= getCudaRef(arg),* newp;
     if (cuda_array_type[ref] != single)
         mexErrMsgTxt("cuda: Upcasting to complex. Type needs to be single\n");
     cuda_array_type[ref]=scomplex;  // only temporary for allocation below
     newp=cudaAlloc(arg);
     cuda_array_type[ref]=single;  // reset
     Dbg_printf2("copying array to Cpx, total size = %d \n",getTotalFloatSizeFromRef(arg));
     cublasScopy(getTotalFloatSizeFromRef(arg), myref, 1, newp, 2);  // just copy the real parts
     cublasScopy(getTotalFloatSizeFromRef(arg), pZero, 0, newp+1, 2);  // just copy the real parts
     if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (cloneArray)\n");return 0;}
     return newp;
 }

void CreateFFTPlan(int ref) {    
     cufftResult status=0;
     if (cuda_FFTplan[ref] != 0)
     {
         cufftDestroy(cuda_FFTplan[ref]); // for now. Later: keep this plan and have one for forward and one for backward direction
         // return 0;
     }
     //if (cuda_array_type[ref] == single)
     //{
     //    cuda_array_type[ref] = fftSingle;
     //}
      /* Create plan for CUDA FFT */
     
    // printf("FFT Error codes: %d, %d, %d, %d, %d ,%d ,%d, %d, %d, %d\n",CUFFT_SUCCESS,CUFFT_INVALID_PLAN,CUFFT_ALLOC_FAILED,CUFFT_INVALID_TYPE,CUFFT_INVALID_VALUE,CUFFT_INTERNAL_ERROR, CUFFT_EXEC_FAILED, CUFFT_SETUP_FAILED, 0, CUFFT_INVALID_SIZE);
     if (cuda_array_dim[ref] == 1) {
        Dbg_printf3("creating 1D-plan with sizes : %d of type %s\n",cuda_array_size[ref][0],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
         if (cuda_array_type[ref] == single)
             status=cufftPlan1d(&cuda_FFTplan[ref], cuda_array_size[ref][0],CUFFT_R2C,1);
         else if (cuda_array_type[ref] == fftHalfSComplex)
             status=cufftPlan1d(&cuda_FFTplan[ref], cuda_array_origFTsize[ref],CUFFT_C2R,1);
         else if (cuda_array_type[ref] == scomplex)
             status=cufftPlan1d(&cuda_FFTplan[ref], cuda_array_size[ref][0],CUFFT_C2C,1);
         else
             mexErrMsgTxt("cuda: Datatype unsuitable for FFT\n");
     }
     else if (cuda_array_dim[ref] == 2 || (cuda_array_dim[ref] > 2 && cuda_array_size[ref][2] == 1)) {
        Dbg_printf4("creating 2D-plan with sizes : %dx%d of type %s\n",cuda_array_size[ref][0],cuda_array_size[ref][1],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
         if (cuda_array_type[ref] == single)
             status=cufftPlan2d(&cuda_FFTplan[ref], cuda_array_size[ref][1], cuda_array_size[ref][0], CUFFT_R2C);
         else if (cuda_array_type[ref] == fftHalfSComplex)
             status=cufftPlan2d(&cuda_FFTplan[ref], cuda_array_size[ref][1], cuda_array_origFTsize[ref], CUFFT_C2R);
         else if (cuda_array_type[ref] == scomplex)
             status=cufftPlan2d(&cuda_FFTplan[ref], cuda_array_size[ref][1], cuda_array_size[ref][0], CUFFT_C2C);
         else
             mexErrMsgTxt("cuda: Datatype unsuitable for FFT\n");
     }
     else if (cuda_array_dim[ref] > 2) {
        Dbg_printf5("creating 3D-plan with sizes : %dx%dx%d of type %s\n",cuda_array_size[ref][0],cuda_array_size[ref][1],cuda_array_size[ref][2],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
         if (cuda_array_type[ref] == single)
             status=cufftPlan3d(&cuda_FFTplan[ref], cuda_array_size[ref][2], cuda_array_size[ref][1], cuda_array_size[ref][0], CUFFT_R2C);
         else if (cuda_array_type[ref] == fftHalfSComplex)
             status=cufftPlan3d(&cuda_FFTplan[ref], cuda_array_size[ref][2], cuda_array_size[ref][1], cuda_array_size[ref][0], CUFFT_C2R);
         else if (cuda_array_type[ref] == scomplex)
             status=cufftPlan3d(&cuda_FFTplan[ref], cuda_array_size[ref][2], cuda_array_size[ref][1], cuda_array_size[ref][0], CUFFT_C2C);
         else
             mexErrMsgTxt("cuda: Datatype unsuitable for FFT\n");
     }
 
     if (status != CUFFT_SUCCESS) {printf("Error : %s",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error FFT Plan creation failed\n");return;}
}


 
/**************************************************************************/

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  cublasStatus custatus;
  //float * p_matlab_data1=0, * p_cuda_data1=0;
  //float * p_matlab_data2=0, * p_cuda_data2=0;
  //double cudaref1,cudaref2; // will be converted to in on usage
  char *command;
  size_t   buflen;
  //int mstatus;
  
  if (mxIsChar(prhs[0]) != 1)
      mexErrMsgTxt("Input 1 must be a string.");
  if (mxGetM(prhs[0]) != 1)
      mexErrMsgTxt("Input 1 must be a row vector.");
  /* Get the length of the input string. */
  buflen = (mxGetM(prhs[0]) * mxGetN(prhs[0])) + 1;
  /* Allocate memory for input and output strings. */
  command = (char*) mxCalloc(buflen, sizeof(char));
  /* Copy the string data from prhs[0] into a C string */
  mxGetString(prhs[0], command, (int) buflen);  
  
  if (!cuda_initialized) {
    custatus = cublasInit();

    if (custatus != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("cuda: CUBLAS initialization error\n");
        return;
        }
    printf("initializing cuda\n");
    pOne=cudaPutVal(1.0f);
    pZero=cudaPutVal(0.0f);
    custatus = cudaMallocHost(& fastMem, fastMemSize); // for fast copy operations
    if (custatus!=cudaSuccess) { printf("cuda init MallocHost: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
    custatus = cudaMallocHost(& fastMemI, fastMemSize); // for fast copy operations
    if (custatus!=cudaSuccess) { printf("cuda init MallocHost: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
    
#ifndef NOCULA
    {
    culaStatus s;
    s=culaInitialize();
    checkCULAStatus("Cula initialization",s);
    }
#endif
    
    cuda_initialized = 1;
  }
  Dbg_printf2("\ncuda called with command: %s\n",command);
  if (strcmp(command,"put")==0) {  // -------------------------------------------
    float * p_cuda_data1;
    if (nrhs != 2) mexErrMsgTxt("cuda: put needs two arguments\n");

       /* Create plan for CUDA FFT */
   // cufftPlan3d(&rFFT_plan, sizes[0], sizes[1], sizes[2], CUFFT_R2C) ;

   p_cuda_data1=cudaPut(prhs[1]);  // allocate memory and store to graphic card
   if (nlhs > 0)
       plhs[0] =  mxCreateDoubleScalar((double)free_array); // returns the current array, as the free position is not yet updated
   p_cuda_data1=0; // jsut to eliminate a warning
  }
  else  if (strcmp(command,"delete")==0) {      
    int ref=0;
    if (nrhs != 2) mexErrMsgTxt("cuda: delete needs two arguments\n");
    // else if (nlhs > 0)
    ref=getCudaRefNum(prhs[1]);
    if (! (ignoreDelete && ignoreRef==ref))
        cudaDelete(ref);
    else
    {ignoreDelete=0;ignoreRef=-1;}
  }
  else  if (strcmp(command,"shutdown")==0) {      
    if (nrhs != 1) mexErrMsgTxt("cuda: shutdown needs one arguments\n");
    cuda_initialized=0;
    printf("shutting down cuda\n");
    cublasShutdown();
#ifndef NOCULA
    culaShutdown();
#endif

  }
  else  if (strcmp(command,"getSize")==0) {     
    if (nrhs != 2) mexErrMsgTxt("cuda: getSize needs two arguments\n");
    else if (nlhs > 0)
        plhs[0]=cudaGetSize(prhs[1]);
  }
  else  if (strcmp(command,"setSize")==0) {      
    if (nrhs != 3) mexErrMsgTxt("cuda: setSize needs three arguments\n");
    else cudaSetSize(prhs[1],prhs[2]);
  }
  else  if (strcmp(command,"get")==0) {      
    if (nrhs != 2) mexErrMsgTxt("cuda: get needs two arguments\n");
    else if (nlhs > 0)
        plhs[0]=cudaGet(prhs[1]);
  }
  else if (strcmp(command,"swapSize")==0) { // -----------------array + array
    int ref;
    if (nrhs != 2) mexErrMsgTxt("cuda: swapSize needs two arguments\n");
    ref=getCudaRefNum(prhs[1]);
    swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref);
  }
  else  if (strcmp(command,"isCpx")==0) {      // is this data of type complex?  The opposite of the matlab command isreal
    if (nrhs != 2) mexErrMsgTxt("cuda: isCpx needs two arguments\n");
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)isComplexType(getCudaRefNum(prhs[1])));    
  }
  else if (strcmp(command,"complex_alpha")==0) { // ---------------------------------
    CallCUDA_UnaryRealFktConst(complex,cudaAllocComplex)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_complex")==0) { // ---------------------------------
    CallCUDA_UnaryRealFktConstR(complex,cudaAllocComplex)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else  if (strcmp(command,"complex")==0) {      // make complex type from real type
    CallCUDA_BinaryRealFkt(complex,cudaAllocComplex) // creates a complex array from real and imag arrays
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double) free_array);  
  }
  else  if (strcmp(command,"subsref_cuda")==0) {      // makes a copy of this area
    int ref1,mask;
    float * newarr;
    int M=0;const char * ret=0;
    if (nrhs != 3) mexErrMsgTxt("cuda: subsref_cuda needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    mask=getCudaRefNum(prhs[2]);
    if (isComplexType(mask))
        mexErrMsgTxt("cuda_subsref: tried to reference with a complex image\n");
    
    newarr=cudaAlloc(prhs[1]);  // same type, and unfortunately size, as input image
    if (isComplexType(ref1)) {
           Dbg_printf("subsref_cuda complex\n");
           ret=CUDAcarr_subsref_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),newarr,getTotalSizeFromRef(prhs[1]), & M);
    } else {
           Dbg_printf("subsref_cuda real\n");
           ret=CUDAarr_subsref_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),newarr,getTotalSizeFromRef(prhs[1]),& M);
        } 
    if (ret) { printf("cuda: subsref_cuda"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: subsref_cuda\n");
    cuda_array_dim[free_array]=1;
    cuda_array_size[free_array][0]=M;    // shorten this array (if necessary). This is a temporary waste of memory, but probably worth it.
    // printf("Reduced array to %d",M);
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else  if (strcmp(command,"subsasgn_cuda_vec")==0) {      // assings a vector to a masked aread in an image
    int ref1,mask,ref3;
    int M=0;const char * ret=0;
    if (nrhs != 5) mexErrMsgTxt("cuda: subsasgn_cuda_vec needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    mask=getCudaRefNum(prhs[2]);
    ref3=getCudaRefNum(prhs[3]);
    if (isComplexType(mask))
        mexErrMsgTxt("cuda_subsasgn_vec: tried to reference with a complex image\n");
    if ((isComplexType(ref3)  && ! isComplexType(ref1)) || (isComplexType(ref1)  && ! isComplexType(ref3)))
        mexErrMsgTxt("cuda_subsasgn_vec: types must be identical\n");
    
    if (isComplexType(ref1)) {
           Dbg_printf("subsasgn_cuda_vec complex\n");
           ret=CUDAcarr_subsasgn_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),getCudaRef(prhs[3]),getTotalSizeFromRef(prhs[1]), & M);
    } else {
           Dbg_printf("subsasgn_cuda_vec real\n");
           ret=CUDAarr_subsasgn_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),getCudaRef(prhs[3]),getTotalSizeFromRef(prhs[1]),& M);
        } 
    if (ret) { printf("cuda: subsasgn_cuda_vec"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: subsasgn_cuda_vec\n");
    // printf("Reduced array to %d",M);
    
    if(mxGetScalar(prhs[4]) >= 0) {
        Dbg_printf("cuda: subsasgn_cuda_vec next delete will be irgnored\n");
        ignoreDelete=1;ignoreRef=ref1;   // the next delete command will be ignored
    }    else
        Dbg_printf("cuda: subsasgncuda_vec no delete will be irgnored\n");

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref1);
  }
  else if (strcmp(command,"repmat")==0) { // repmat : replicates a matrix
    int ref1,dimsRepsize,d,ndim;
    double *p_Rep;
    float * newarr;
    int dSize[3]={1,1,1},sSize[3];
    const char * ret=0;
    if (nrhs != 3) mexErrMsgTxt("cuda: repmat needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    dimsRepsize=(int) (mxGetM(prhs[2]) * mxGetN(prhs[2]));
    if (dimsRepsize>3)
        mexErrMsgTxt("cuda: repmat is only supported up to 3D. Size vector too long.\n");
    p_Rep=mxGetPr(prhs[2]);
    get3DSize(ref1,sSize);
    for (d=0;d<3;d++) {
        if (d < dimsRepsize){
            dSize[d]=(int) (sSize[d]*p_Rep[d]); 
        }
        else
            dSize[d]=(int) sSize[d];
    }

    ndim=cuda_array_dim[ref1];  // Can change dimensionality of array
    if (dimsRepsize > ndim)
       ndim = dimsRepsize;

    if (isComplexType(ref1))
        newarr=cudaAllocDetailed(ndim,dSize,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,dSize,single);

    if (isComplexType(ref1)) {
           Dbg_printf("repmat complex\n");
           ret=CUDAcarr_repcpy_carr(getCudaRef(prhs[1]),newarr,sSize,dSize);
    } else {
           Dbg_printf("repmat real\n");
           ret=CUDAarr_repcpy_arr(getCudaRef(prhs[1]),newarr,sSize,dSize);
        } 
    if (ret) { printf("cuda: repmat"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: repmat\n");
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);

  }
  else if (strcmp(command,"subsref_block")==0) { // sub referencing a block of data
    int ref1,dimsSoff,dimsSsize,d,ndim;
    double * p_Soffset, * p_Ssize;
    const char * ret=0;
    int nsizes[3]={1,1,1},dOffs[3]={0,0,0},sSize[3]={1,1,1},sOffs[3]={0,0,0};
    float * newarr;
    if (nrhs != 4) mexErrMsgTxt("cuda: csubsref needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    dimsSoff=(int) (mxGetM(prhs[2]) * mxGetN(prhs[2]));
    dimsSsize=(int)(mxGetM(prhs[3]) * mxGetN(prhs[3]));
    if (dimsSoff>3)
        mexErrMsgTxt("cuda: subreferencing is only supported up to 3D. Offset vector too long.\n");
    if (dimsSsize>3)
        mexErrMsgTxt("cuda: subreferencing is only supported up to 3D. Size vector too long.\n");
    p_Soffset=mxGetPr(prhs[2]);
    p_Ssize=mxGetPr(prhs[3]);
    get3DSize(ref1,sSize);
    for (d=0;d<3;d++) {
        if (d<dimsSoff)
            sOffs[d]=(int) p_Soffset[d];
        if (d<dimsSsize)
            nsizes[d]=(int) p_Ssize[d];
        if (sOffs[d] < 0.0 || sOffs[d]+nsizes[d] > sSize[d]) {
            printf("d: %d sOffs %d, sOffs+nsizes %d, sSize %d\n",d,sOffs[d], sOffs[d]+nsizes[d], sSize[d]);
            mexErrMsgTxt("cuda: subreferencing Offset index out of range.\n"); }
        if (nsizes[d] < 0.0)
            mexErrMsgTxt("cuda: subreferencing Negative sizes not allowed.\n");            
    }
    Dbg_printf("subsref_block\n");
    ndim=cuda_array_dim[ref1];  // Will not change dimensionality of array

    Dbg_printf4("s1Size: %d x %d x %d\n",sSize[0],sSize[1],sSize[2]);
    Dbg_printf4("nSize: %d x %d x %d\n",nsizes[0],nsizes[1],nsizes[2]);
    Dbg_printf4("dOffs: %d x %d x %d\n",dOffs[0],dOffs[1],dOffs[2]);
    if (isComplexType(ref1))
        newarr=cudaAllocDetailed(ndim,nsizes,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,nsizes,single);

     if (isComplexType(ref1)) {
           Dbg_printf("subsref_block complex to complex\n");
           ret=CUDAcarr_subcpy_carr(getCudaRef(prhs[1]),newarr,sSize,nsizes,sOffs,nsizes,dOffs);
     } else {
           Dbg_printf("subsref_block real to real\n");
           ret=CUDAarr_subcpy_arr(getCudaRef(prhs[1]),newarr,sSize,nsizes,sOffs,nsizes,dOffs);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: subsref_block "); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: subsref_block\n");
  }
  else if (strcmp(command,"subsasgn_cuda_const")==0) { // conditionally assigns a constant value to an existing array
    const char *ret=0; int ref; double myreal,myimag=0;
    if (nrhs != 5) mexErrMsgTxt("cuda: subsasgn_cuda_const needs four arguments\n");
    ref=getCudaRefNum(prhs[1]);
    myreal = mxGetScalar(prhs[3]);
    if (mxIsComplex(prhs[3])) myimag = * ((double *) (mxGetPi(prhs[3])));

    if (isComplexType(ref)) {
            Dbg_printf("cuda: complex array subsasgn_cuda_const  complex-const\n");
            ret=CUDAcarr_boolassign_const(getCudaRef(prhs[2]),(float) myreal,(float) myimag,getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]));
            if (ret!=(const char *) cudaSuccess) { printf("cuda subsasgn_cuda_const: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error subsasgn_cuda_const: Bailing out");}
        } else {
            if (myimag != 0) mexErrMsgTxt("cuda error subassgn_cuda_const: Tried to assign a complex value to a real array");
            Dbg_printf("cuda: float array subsasgn_cuda_const  complex-const\n");
            ret=CUDAarr_boolassign_const(getCudaRef(prhs[2]),(float) myreal,getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]));
            if (ret!=(const char *) cudaSuccess) { printf("cuda subsasgn_cuda_const: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error subsasgn_cuda_const: Bailing out");}
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref);

    if(mxGetScalar(prhs[4]) >= 0) {
        Dbg_printf("cuda: subsasgn_cuda_const next delete will be irgnored\n");
        ignoreDelete=1;ignoreRef=ref;   // the next delete command will be ignored
    }    else
        Dbg_printf("cuda: subsasgncuda_const no delete will be irgnored\n");
            
    if (ret) { printf("cuda: subsasgn_cuda_const "); mexErrMsgTxt(ret);}
    Dbg_printf("cuda: subsasgn_cuda_const\n");
    } 
  else if (strcmp(command,"subsasgn_block")==0) { // sub referencing a block of data
    int ref2,constmode,dimsDoff,dimsSsize;
    int dOffs[3]={0,0,0},sSize[3]={1,1,1},dSize[3]={1,1,1},noOffs[3]={0,0,0};
    int d;
    double * p_Doffset,* p_sSize;
    const char * ret=0;
    if (nrhs != 6) mexErrMsgTxt("cuda: subsasgn_block needs six arguments\n");  // command, array1 , 3D offset, 3D size
    ref2=getCudaRefNum(prhs[2]); // destination
    constmode=(mxGetScalar(prhs[5]) > 0) ? 1 : 0;  // should prhs[2] be interpreted as a constant to assign? 0 or negative means array
    dimsDoff=(int)(mxGetM(prhs[3]) * mxGetN(prhs[3]));
    if (dimsDoff>3)
        mexErrMsgTxt("cuda: subassigning is only supported up to 3D. Offset vector too long.\n");
    p_Doffset=mxGetPr(prhs[3]);

    dimsSsize=(int)(mxGetM(prhs[4]) * mxGetN(prhs[4]));
    if (dimsSsize>3)
        mexErrMsgTxt("cuda: subassigning is only supported up to 3D. Size vector too long.\n");
    p_sSize=mxGetPr(prhs[4]);

    for (d=0;d<3;d++) {
        if (d<dimsDoff)
            dOffs[d]=(int) p_Doffset[d];
        if (d<dimsSsize)
            sSize[d]=(int) p_sSize[d];
        if (d<cuda_array_dim[ref2])
            dSize[d]=(int) cuda_array_size[ref2][d];
        }
    Dbg_printf("subsasgn_block\n");
    Dbg_printf4("sSize: %d x %d x %d\n",sSize[0],sSize[1],sSize[2]);
    Dbg_printf4("dSize: %d x %d x %d\n",dSize[0],dSize[1],dSize[2]);
    Dbg_printf4("dOffs: %d x %d x %d\n",dOffs[0],dOffs[1],dOffs[2]);

    if (!constmode) {
    int ref1=getCudaRefNum(prhs[1]); // source
    if (isComplexType(ref1)) {
        if (! isComplexType(ref2)) // destination needs to be complex too
            mexErrMsgTxt("cuda: trying to assign complex values to real array.\n");
        Dbg_printf("subsasgn_block complex to complex\n");
        ret=CUDAcarr_subcpy_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),sSize,dSize,noOffs,sSize,dOffs);
     } else {
         if (isComplexType(ref2)) // destination needs to be complex too
            {
             Dbg_printf("subsasgn_block real to complex\n");
             ret=CUDAarr_subcpy_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),sSize,dSize,noOffs,sSize,dOffs);
            }
         else
            {
             Dbg_printf("subsasgn_block real to real\n");
             ret=CUDAarr_subcpy_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),sSize,dSize,noOffs,sSize,dOffs);
            }
        } 
    } else   // const mode (prhs[2] is a matlab constant to assign to array
    {
    if (isComplexType(ref2)) {  // destination
        double  myreal = mxGetScalar(prhs[1]);
        double  myimag = 0.0;
        if (mxIsComplex(prhs[1]))
            myimag=* ((double *) (mxGetPi(prhs[1])));
        Dbg_printf("subsasgn_block complex const to complex\n");
       ret=CUDAcconst_subcpy_carr(getCudaRef(prhs[2]),(float) myreal,(float) myimag,dSize,sSize,dOffs);
     } else {
        double  myreal;
        if (mxIsComplex(prhs[1])) // const is complex but data is real
                mexErrMsgTxt("cuda: trying to assign complex constant to real array.\n");
        myreal = mxGetScalar(prhs[1]);
        Dbg_printf("subsasgn_block real const to complex\n");
        ret=CUDAconst_subcpy_arr(getCudaRef(prhs[2]),(float) myreal,0.0,dSize,sSize,dOffs);
     }}

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref2);
    if(mxGetScalar(prhs[5]) >= 0) {
        Dbg_printf("cuda: subsasgn_block next delete will be irgnored\n");
        ignoreDelete=1;ignoreRef=ref2;   // the next delete command will be ignored
    }    else
        Dbg_printf("cuda: subsasgn_block no delete will be irgnored\n");
            
    if (ret) { printf("cuda: subsasgn_block "); mexErrMsgTxt(ret);}
    Dbg_printf("cuda: subsasgn_block\n");
  }
  else if (strcmp(command,"newarr")==0) { // creates a new array with given sizes and assigns a constant to it. Does not need an input array!
    int dims_sizes;
    int nsizes[10];
    int d,tsize=1;
    double * dsizes;
    float * newarr=0;
    if (nrhs != 3) mexErrMsgTxt("cuda: newarr needs three arguments\n");  
    dims_sizes=(int)(mxGetM(prhs[1]) * mxGetN(prhs[1]));
    dsizes=mxGetPr(prhs[1]);
    if (dims_sizes >= 10)
        mexErrMsgTxt("cuda: newarr to many dimensions (>10)\n");  
    for (d=0;d<dims_sizes;d++) {nsizes[d]=(int) dsizes[d];tsize *= nsizes[d];}
    Dbg_printf5("newarray with dimension %d, sizes %d %d %d\n",dims_sizes,nsizes[0],nsizes[1],nsizes[2]);

    if (mxIsComplex(prhs[2])) {
        float br=(float) mxGetPr(prhs[2])[0];
        float bi=(float) mxGetPi(prhs[2])[0];
        newarr=cudaAllocDetailed(dims_sizes,nsizes,scomplex);
        CUDAset_carr(br, bi, newarr, tsize);
    } else {
        float b=(float) mxGetScalar(prhs[2]);
        newarr=cudaAllocDetailed(dims_sizes,nsizes,single);
        CUDAset_arr(b, newarr, tsize);
    }

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    Dbg_printf("newarr\n");
  }
  else if (strcmp(command,"copy")==0) { // copies an array
    if (nrhs != 2) mexErrMsgTxt("cuda: copy needs two arguments\n");  
    cloneArray(prhs[1]);  // float * newarr=
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    Dbg_printf("copy\n");
  }
  else if (strcmp(command,"transpose")==0) { // transpose an array
    int ref,conjmode,tmp;
    int nsizes[3],ndim,noOffs[3]={0,0,0},sSize[3];
    const char * ret=0;
    float * newarr;

    if (nrhs != 3) mexErrMsgTxt("cuda: transpose needs three arguments\n");  // command, array1 , 3D offset, 3D size
    Dbg_printf("transpose\n");
    ref=getCudaRefNum(prhs[1]);
    conjmode=(mxGetScalar(prhs[2]) > 0) ? 1 : 0;  // conjugate or not, that is the question
    get3DSize(ref,sSize);
    get3DSize(ref,nsizes);
    ndim=cuda_array_dim[ref];
    if (ndim<2) ndim=2;
    tmp=nsizes[1];nsizes[1]=nsizes[0];nsizes[0]=tmp;  // swaps sizes
    
    Dbg_printf4("sSize: %d x %d x %d\n",sSize[0],sSize[1],sSize[2]);
    if (isComplexType(ref))
        newarr=cudaAllocDetailed(ndim,nsizes,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,nsizes,single);

     if (isComplexType(ref)) {
           Dbg_printf("transpose complex to complex\n");
           if (conjmode)
               ret=CUDAcarr_subcpyCT_carr(getCudaRef(prhs[1]),newarr,sSize,nsizes,noOffs,sSize,noOffs);
           else
               ret=CUDAcarr_subcpyT_carr(getCudaRef(prhs[1]),newarr,sSize,nsizes,noOffs,sSize,noOffs);
     } else {
           Dbg_printf("transpose real to real\n");
           ret=CUDAarr_subcpyT_arr(getCudaRef(prhs[1]),newarr,sSize,nsizes,noOffs,sSize,noOffs);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: transpose"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: transpose\n");
  }
  else if (strcmp(command,"diag")==0) { // generates or extracts a diagonal
    int nsizes[3],noOffs[3]={0,0,0},dOffs[3]={0,0,0},sSize[3];
    int diagget=0;
    const char * ret=0;
    int ref=getCudaRefNum(prhs[1]);
    int offset;

    if (nrhs != 3) mexErrMsgTxt("cuda: diag needs two arguments\n");  // array, offset
    offset = (int) mxGetScalar(prhs[2]); // shift off the diagonal
    get3DSize(ref,sSize);
    get3DSize(ref,nsizes);
    if (sSize[0] == 1)
        {sSize[0] = sSize[1];sSize[1]=1;}
    if (sSize[0] > 1 && sSize[1] > 1)  // extract the diagonal
    {
        diagget=1;
        if (offset < 0) {
            dOffs[0]=(int) -offset;
            nsizes[0] -= dOffs[0];
        }
        if (offset > 0) {
            dOffs[1]=(int) offset;
            nsizes[1] -= dOffs[1];
        }
        nsizes[0]= min(nsizes[0],nsizes[1]);
        nsizes[1]= 1; // make it a square matrix
        Dbg_printf("diag set\n");
    }
    else  // generate a diagonal matrix
    {
        nsizes[0]= sSize[0] + (int) abs(offset);
        nsizes[1]= nsizes[0]; // make it a square matrix
    if (offset < 0)
        dOffs[0]=(int) -offset;
    if (offset > 0)
        dOffs[1]=(int) offset;
    Dbg_printf("diag set\n");
    }
    {

    float * newarr;

    if (isComplexType(ref))
    {
        cudaError_t err;
        newarr=cudaAllocDetailed(2,nsizes,scomplex);
        err=cudaMemset(newarr,0,nsizes[0]*nsizes[1]*2*sizeof(float));
        checkCudaError("Memset diag complex",err);
    }
    else
    {
        cudaError_t err;
        newarr=cudaAllocDetailed(2,nsizes,single);
        err=cudaMemset(newarr,0,nsizes[0]*nsizes[1]*sizeof(float)); 
        checkCudaError("Memset diag single",err);
    }
   
    Dbg_printf4("sSize: %d x %d x %d\n",sSize[0],sSize[1],sSize[2]);

    if (isComplexType(ref)) {
           Dbg_printf("diag complex\n");
           if (diagget)
               ret=CUDAcarr_diag_get(newarr,getCudaRef(prhs[1]),nsizes,sSize,noOffs,nsizes,dOffs);
           else
               ret=CUDAcarr_diag_set(getCudaRef(prhs[1]),newarr,sSize,nsizes,noOffs,sSize,dOffs);
     } else {
           Dbg_printf("diag real\n");
           if (diagget)
               ret=CUDAarr_diag_get(newarr,getCudaRef(prhs[1]),nsizes,sSize,noOffs,nsizes,dOffs);
           else
               ret=CUDAarr_diag_set(getCudaRef(prhs[1]),newarr,sSize,nsizes,noOffs,sSize,dOffs);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: diag"); mexErrMsgTxt(ret);}                          
    }
    Dbg_printf("cuda: diag\n");
  }
  else if (strcmp(command,"cat")==0) { // appends arrays along a direciton
    int nsizes[3],dOffs[3],noOffs[3],s1Size[3],s2Size[3],ref1,ref2,ndim;
    int direction;
    const char * ret=0;
    float * newarr;
    if (nrhs != 4) mexErrMsgTxt("cuda: cat needs four arguments\n");  // command, array1 , array2, direction
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    direction=(int) mxGetScalar(prhs[3]);
    get3DSize(ref1,nsizes);get3DSize(ref1,s1Size);get3DSize(ref2,s2Size);
    dOffs[0]=0;dOffs[1]=0;dOffs[2]=0;noOffs[0]=0;noOffs[1]=0;noOffs[2]=0;
    Dbg_printf2("append to direction: %g\n",direction);
    if (direction == 1) {
        nsizes[0]=s1Size[0]+s2Size[0];
        dOffs[0]=s1Size[0];
    } else if (direction == 2) {
        nsizes[1]=s1Size[1]+s2Size[1];
        dOffs[1]=s1Size[1];
    } else if (direction == 3) {
        nsizes[2]=s1Size[2]+s2Size[2];
        dOffs[2]=s1Size[2];
    } else {
        mexErrMsgTxt("cuda: cat. Direction to append along needs to be 1 (x), 2 (y) or 3 (z)\n");    
    }
    ndim=(cuda_array_dim[ref1] > cuda_array_dim[ref2]) ? cuda_array_dim[ref1] : cuda_array_dim[ref2];
    if (direction > ndim) ndim = direction;
        
    Dbg_printf4("s1Size: %d x %d x %d\n",s1Size[0],s1Size[1],s1Size[2]);
    Dbg_printf4("s2Size: %d x %d x %d\n",s2Size[0],s2Size[1],s2Size[2]);
    Dbg_printf4("nSize: %d x %d x %d\n",nsizes[0],nsizes[1],nsizes[2]);
    Dbg_printf4("dOffs: %d x %d x %d\n",dOffs[0],dOffs[1],dOffs[2]);
    if (isComplexType(ref1) || isComplexType(ref2))
        newarr=cudaAllocDetailed(ndim,nsizes,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,nsizes,single);

     if (isComplexType(ref1))
        if (isComplexType(ref2)) {
           Dbg_printf("append complex to complex\n");
           ret=CUDAcarr_subcpy_carr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,noOffs,s1Size,noOffs);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAcarr_subcpy_carr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,noOffs,s2Size,dOffs);
        } else {
           Dbg_printf("append complex to real\n");
           ret=CUDAcarr_subcpy_carr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,noOffs,s1Size,noOffs);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAarr_subcpy_carr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,noOffs,s2Size,dOffs);
        }
    else if (isComplexType(ref2)) {
           Dbg_printf("append real to complex\n");
           ret=CUDAarr_subcpy_carr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,noOffs,s1Size,noOffs);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAcarr_subcpy_carr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,noOffs,s2Size,dOffs);
        } else {
           Dbg_printf("append real to real\n");
           ret=CUDAarr_subcpy_arr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,noOffs,s1Size,noOffs);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAarr_subcpy_arr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,noOffs,s2Size,dOffs);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: cat\n");
  }
  else if (strcmp(command,"subsref_vecs")==0) { 
    int nel,ref;
    float * p_cuda_data1=0;  
    int newref[3]={-1,-1,-1};
    int dSize[3]={1,1,1};
    int d;
    float * newarr;
    const char * ret=0;
    
    mexErrMsgTxt("cuda: subsref_vec is not finnished yet\n");
    if (nrhs != 3) mexErrMsgTxt("cuda: subsref_vec needs three arguments\n");
    nel=(int) mxGetNumberOfElements(prhs[2]);
    if (nel > 3) mexErrMsgTxt("cuda: subsref_vec can reference maximally 3d arrays\n");
    ref=getCudaRefNum(prhs[1]);
    for (d=0;d<nel;d++) {
        p_cuda_data1=cudaPut(mxGetCell(prhs[2],d));  // allocate memory and store to graphic card
        dSize[d]=cuda_array_size[free_array][0];
        newref[d]=free_array;
    }
    newarr=cudaAllocDetailed(cuda_array_dim[ref],dSize,cuda_array_type[ref]);   // new array
    switch (nel) {
        case 1:
            newarr=newarr;
           // ret=CUDAarr_subsref_vec(getCudaRef(prhs[1]),newarr,sSize,dSize,cuda_arrays[newref[0]]);
            break;
        case 2:
           // ret=CUDAarr_subsref_vec(getCudaRef(prhs[1]),newarr,sSize,dSize,cuda_arrays[newref[0]],cuda_arrays[newref[1]]);
            break;
        case 3:
           // ret=CUDAarr_subsref_vec(getCudaRef(prhs[1]),newarr,sSize,dSize,cuda_arrays[newref[0]],cuda_arrays[newref[1]],cuda_arrays[newref[2]]);
            break;
        default:
            mexErrMsgTxt("cuda: subsref_vec can reference maximally 3d arrays\n");
    } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
    for (d=0;d<nel;d++) {
        if (newref[d] >= 0)
                cudaDelete(newref[d]); // delete these arrays again
    }    
  }
  else if (strcmp(command,"equals_alpha")==0) { 
    CallCUDA_UnaryFktConst(equals,cudaAllocReal)  // always returns a real value array
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"equals")==0) { 
    CallCUDA_BinaryFkt(equals,cudaAllocReal)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"unequals_alpha")==0) { 
    CallCUDA_UnaryFktConst(unequals,cudaAllocReal)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"unequals")==0) { 
    CallCUDA_BinaryFkt(unequals,cudaAllocReal)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"or_alpha")==0) {
    CallCUDA_UnaryRealFktConst(or,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_or")==0) {
    CallCUDA_UnaryRealFktConstR(or,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"or")==0) { 
    CallCUDA_BinaryRealFkt(or,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"and_alpha")==0) {
    CallCUDA_UnaryRealFktConst(and,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_and")==0) {
    CallCUDA_UnaryRealFktConstR(and,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"and")==0) { 
    CallCUDA_BinaryRealFkt(and,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"not")==0) { 
      CallCUDA_UnaryRealFkt(not,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"smaller_alpha")==0) {
    CallCUDA_UnaryRealFktConst(smaller,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_smaller")==0) {
    CallCUDA_UnaryRealFktConstR(smaller,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"smaller")==0) { 
    CallCUDA_BinaryRealFkt(smaller,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"larger_alpha")==0) { 
    CallCUDA_UnaryRealFktConst(larger,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_larger")==0) { 
    CallCUDA_UnaryRealFktConstR(larger,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"larger")==0) { 
    CallCUDA_BinaryRealFkt(larger,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"smallerequal_alpha")==0) { 
    CallCUDA_UnaryRealFktConst(smallerequal,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_smallerequal")==0) { 
    CallCUDA_UnaryRealFktConstR(smallerequal,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"smallerequal")==0) { 
    CallCUDA_BinaryRealFkt(smallerequal,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"largerequal_alpha")==0) { 
    CallCUDA_UnaryRealFktConst(largerequal,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_largerequal")==0) {
    CallCUDA_UnaryRealFktConstR(largerequal,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"largerequal")==0) {
    CallCUDA_BinaryRealFkt(largerequal,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"power")==0) { // -----------------array .^ array
    CallCUDA_BinaryRealFkt(power,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"power_alpha")==0) { // ---------------------------------
    CallCUDA_UnaryRealFktConst(power,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_power")==0) { // ---------------------------------
    CallCUDA_UnaryRealFktConstR(power,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"times_alpha")==0) { // ---------------------------------
    CallCUDA_UnaryFktConst(times,cudaAlloc)  // return type is the propagated type
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"times")==0) { 
    CallCUDA_BinaryFkt(times,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"plus_alpha")==0) { // --------------------------------------------
    CallCUDA_UnaryFktConst(plus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"plus")==0) { // -----------------array + array
    CallCUDA_BinaryFkt(plus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"divide_alpha")==0) { // ---------------------------------
    CallCUDA_UnaryFktConst(divide,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_divide")==0) { // ---------------------------------
    CallCUDA_UnaryFktConstR(divide,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"divide")==0) { // ---------------------------------
    CallCUDA_BinaryFkt(divide,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"minus_alpha")==0) { // --------------------------------------------
    CallCUDA_UnaryFktConst(minus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_minus")==0) { // --------------------------------------------
    CallCUDA_UnaryFktConstR(minus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"minus")==0) { // -----------------array - array
    CallCUDA_BinaryFkt(minus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"exp")==0) { // complex exponential
    CallCUDA_UnaryFkt(exp,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"abs")==0) {
      CallCUDA_UnaryFkt(abs,cudaAllocReal)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"log")==0) {
    if (! isComplexType(getCudaRefNum(prhs[1]))) {
        CallCUDA_UnaryFkt(log,cudaAllocReal)
    }
    else
        mexErrMsgTxt("cuda: log not implemented for complex arrays\n");
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"uminus")==0) { 
      CallCUDA_UnaryFkt(uminus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"conj")==0) { 
      CallCUDA_UnaryFkt(conj,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"fftshift")==0) { // -----------------like in matlab
      int ref,mode;
    int ret=0;
    if (nrhs != 3) mexErrMsgTxt("cuda: fftshift needs three arguments\n");
    ref=getCudaRefNum(prhs[1]);
    mode=(mxGetScalar(prhs[2]) > 0) ? 1 : -1;
    if (isComplexType(getCudaRefNum(prhs[1])))
       ret=CUDAarr_times_const_rotate(getCudaRef(prhs[1]),1,cudaAlloc(prhs[1]),cuda_array_size[ref],cuda_array_dim[ref],1,mode);
    else
       ret=CUDAarr_times_const_rotate(getCudaRef(prhs[1]),1,cudaAlloc(prhs[1]),cuda_array_size[ref],cuda_array_dim[ref],0,mode);
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: fftshift\n");
  }
  
  else if (strcmp(command,"fft3d")==0) { // ----------------- carray to carray. Last argument: 1= forward, -1=backward, 2=forward, scramble and scale, -2=backward, scramble & scale
    float * newarr=0;
    int ref,ret;double mode;
    int dev=0;
    struct cudaDeviceProp prop;
    cufftResult status=0;

    if (nrhs != 3) mexErrMsgTxt("cuda: fft needs three arguments\n");
  
    // printf("cuda: Number of CUDA_TYPENAMES is %d\n",sizeof(CUDA_TYPE_NAMES));  
    // return 0;
         
    /* Execute FFT on device */
    ref=getCudaRefNum(prhs[1]);
    
    //int ret=CUDAarr_times_const_rotate(getCudaRef(prhs[1]),cuda_array_FTscale[free_array],getCudaRef(prhs[1]),cuda_array_size[free_array],cuda_array_dim[free_array]); // inplace operation, treats complex as doubles
    if (isComplexType(ref))
        newarr=cloneArray(prhs[1]);
    else
        newarr=copyToCpx(prhs[1]);
    
    CreateFFTPlan(free_array);
    mode=mxGetScalar(prhs[2]);

    ret=0;
    
   // printf("Mode %g Size1 %d Size2 %d Dim %d\n",mode,cuda_array_size[free_array][0],cuda_array_size[free_array][1],cuda_array_dim[free_array]);
   if (mode > 0) {
        if (abs(mode) > 1) 
            ret=CUDAarr_times_const_rotate(newarr,1,newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,-1); // inplace operation, treats complex as doubles
        status=cufftExecC2C(cuda_FFTplan[free_array], (cufftComplex *) newarr, (cufftComplex *) newarr,CUFFT_FORWARD);
        if (abs(mode) > 1)
            ret=CUDAarr_times_const_rotate(newarr,cuda_array_FTscale[free_array],newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,1); // inplace operation, treats complex as doubles
        if (status != CUFFT_SUCCESS) {printf("Error %s",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error complex to complex FFT failed\n");return;}
    }
    else {
        if (abs(mode) > 1) 
            ret=CUDAarr_times_const_rotate(newarr,1,newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,-1);
        status=cufftExecC2C(cuda_FFTplan[free_array], (cufftComplex *) newarr, (cufftComplex *) newarr,CUFFT_INVERSE);
        if (abs(mode) > 1)
            ret=CUDAarr_times_const_rotate(newarr,cuda_array_FTscale[free_array],newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,1); 
        if (status != CUFFT_SUCCESS) {printf("Error %s",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error inverse complex to complex FFT failed\n");return;}
    }
        // CUDAarr_times_const_scramble(newarr,cuda_array_FTscale[free_array],newarr,cuda_array_size[free_array],cuda_array_dim[free_array]); // inplace operation, treats complex as doubles
    cudaGetDevice(&dev);
    status=cudaGetDeviceProperties(&prop,dev);
    if (status!=cudaSuccess) { printf("cuda GetDiviceProperties: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
    //int blockSize=prop.warpSize; int nBlocks=1;	// add extra block if N can't be divided by blockSize
    // printf("BlockSize %d, Threads %d, Mem %d, maj %d, min %d, ret %d\n",prop.warpSize,prop.maxThreadsPerBlock,prop.sharedMemPerBlock,prop.major,prop.minor,ret);
   
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: fft3d\n");
  }
  else if (strcmp(command,"rfft3d")==0) { // -----------------array + array
    float * newarr;int ref;
    cufftResult status=0;
    if (nrhs != 2) mexErrMsgTxt("cuda: fft needs two arguments\n");
    /* Execute FFT on device */
    ref=getCudaRefNum(prhs[1]);
    CreateFFTPlan(ref);

    ReduceToHalfComplex(ref); // restore its size
    newarr=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], fftHalfSComplex);
    ExpandToFullReal(ref);    // hack to get the size correct below
    cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array

     status=cufftExecR2C(cuda_FFTplan[ref], getCudaRef(prhs[1]), (cufftComplex *) newarr);    
    if (status != CUFFT_SUCCESS) {printf("Error: %s",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error FFT failed\n");return;}
    CUDAarr_times_const(newarr,cuda_array_FTscale[free_array],newarr,getTotalSizeFromRefNum(free_array)*2); // inplace operation, treats complex as doubles

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: fft3d\n");
 }
  else if (strcmp(command,"rifft3d")==0) { // -----------------array + array
    float * newarr;
    cufftResult status=0;

    if (nrhs != 2) mexErrMsgTxt("cuda: ifft needs two arguments\n");
    /* Execute FFT on device */
    //int ref=getCudaRefNum(prhs[1]);

    newarr=cloneArray(prhs[1]); // first clone, then do in-place ifft
   
    ExpandToFullReal(free_array);    // hack to get the size correct below
    CreateFFTPlan(free_array);
    //cuda_array_size[free_array][0]+=2;
    
    //float * newarr=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], single);
    // ReduceToHalfComplex(ref); // restore its size

    //int custate=cudaMemcpy(newarr,getCudaRef(prhs[1]), getTotalFloatSizeFromRefNum(free_array)*sizeof(float), cudaMemcpyDeviceToDevice);
    //if (custatus != cudaSuccess) mexErrMsgTxt("cuda: Device access error (read real-part cuda to C)\n");

    status=cufftExecC2R(cuda_FFTplan[free_array], (cufftComplex *) newarr, newarr) ;  // it was necessary to do it in place as the algorithm overwrites the old values getCudaRef(prhs[1])
    //status=cufftExecC2R(cuda_FFTplan[free_array], (cufftComplex *) getCudaRef(prhs[1]), newarr) ;  // it was necessary to do it in place as the algorithm overwrites the old values getCudaRef(prhs[1])
    if (status != CUFFT_SUCCESS) {printf("Error %s",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error FFT failed\n");return;}
    cuda_array_type[free_array]=single;

    //CUDAarr_times_const(newarr,cuda_array_FTscale[free_array],newarr,getTotalSizeFromRefNum(free_array)); // inplace operation
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: ifft3d\n");
  }
  else if (strcmp(command,"real")==0) { // real part
    if (nrhs != 2) mexErrMsgTxt("cuda: real needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        CUDAreal_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    else
        CUDAreal_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
        // cloneArray(prhs[1]);
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: real\n");
  }
  else if (strcmp(command,"imag")==0) { // imaginary part
    if (nrhs != 2) mexErrMsgTxt("cuda: imag needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        CUDAimag_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    else
        CUDAimag_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: imag\n");
  }
  else if (strcmp(command,"phase")==0) { // phase of a complex number
    if (nrhs != 2) mexErrMsgTxt("cuda: phase needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        CUDAphase_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    else
        CUDAphase_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: phase\n");
  }
  else if (strcmp(command,"isnan")==0) { // is not a number
    if (nrhs != 2) mexErrMsgTxt("cuda: isnan needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        CUDAisnan_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    else
        CUDAisnan_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: isnan\n");
  }
  else if (strcmp(command,"isinf")==0) { // is infinite
    if (nrhs != 2) mexErrMsgTxt("cuda: isinf needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        CUDAisinf_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    else
        CUDAisinf_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1])); 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: isinf\n");
  }
  else if (strcmp(command,"max")==0) { // maximum 
    float pres[2];const char * status;
    if (nrhs != 2) mexErrMsgTxt("cuda: max needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        mexErrMsgTxt("cuda: tried to apply max to array of complex datatype\n");
    status=CUDAmax_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), pres);
    if (status) mexErrMsgTxt(status);

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)pres[0]);
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double)pres[1]);
   Dbg_printf("cuda: max\n");
  } 
  else if (strcmp(command,"min")==0) { // minimum 
    float pres[2];const char * status;
    if (nrhs != 2) mexErrMsgTxt("cuda: min needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        mexErrMsgTxt("cuda: tried to apply min to array of complex datatype\n");
    status=CUDAmin_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), pres);
    if (status) mexErrMsgTxt(status);

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)pres[0]);
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double)pres[1]);
   Dbg_printf("cuda: min\n");
  }
  else if (strcmp(command,"part_sum")==0) { // partial sum over array 
    int ref1,ProjDir;
    int sSize[3],dSize[3];
    float * mask = 0;float * new_array;
    if (nrhs != 4) mexErrMsgTxt("cuda: part_sum needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    get3DSize(ref1,sSize);
    ProjDir=(int) mxGetScalar(prhs[3]);
    if (ProjDir < 1 || ProjDir > 3)
         mexErrMsgTxt("cuda: part_sum unsupported projection direction (nees to be between 1 and 3)\n");    

    get3DSize(ref1,dSize);
    dSize[ProjDir-1] = 1;

    if (!mxIsEmpty(prhs[2]))
        mask=getCudaRef(prhs[2]);
    
    Dbg_printf4("dSize is %dx%dx%d\n",dSize[0],dSize[1],dSize[2]);
    new_array=cudaAllocDetailed(cuda_array_dim[ref1], dSize, cuda_array_type[ref1]);

    if (isComplexType(getCudaRefNum(prhs[1])))
    {
          const char * status=CUDApsum_carr(getCudaRef(prhs[1]), mask,new_array, sSize,ProjDir);
          if (status) mexErrMsgTxt(status);
    } else {
            const char * status=CUDApsum_arr(getCudaRef(prhs[1]), mask,new_array, sSize, ProjDir);
            if (status) mexErrMsgTxt(status);
        }
    
    plhs[0] =  mxCreateDoubleScalar((double) free_array);

    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing part_sum\n");return;}
   Dbg_printf("cuda: part_sum\n");
  }
  else if (strcmp(command,"part_max")==0) { // partial maximum over array 
    int ref1,ProjDir,new_array_num;
    int sSize[3],dSize[3];
    float * mask = 0;float * new_array;
    float * new_array_idx=0;const char * status;
    if (nrhs != 4) mexErrMsgTxt("cuda: part_max needs three arguments\n");
    ref1=getCudaRefNum(prhs[1]);
    get3DSize(ref1,sSize);
    ProjDir=(int) mxGetScalar(prhs[3]);
    if (ProjDir < 1 || ProjDir > 3)
         mexErrMsgTxt("cuda: part_max unsupported projection direction (nees to be between 1 and 3)\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
         mexErrMsgTxt("cuda: part_max data to project cannot be complex valued\n");

    get3DSize(ref1,dSize);
    dSize[ProjDir-1] = 1;

    if (!mxIsEmpty(prhs[2]))
        mask=getCudaRef(prhs[2]);
    
    Dbg_printf4("dSize is %dx%dx%d\n",dSize[0],dSize[1],dSize[2]);
    new_array=cudaAllocDetailed(cuda_array_dim[ref1], dSize, cuda_array_type[ref1]);
    new_array_num= free_array;
    if (nlhs > 1)
        new_array_idx=cudaAllocDetailed(cuda_array_dim[ref1], dSize, single);
    
    status=CUDApmax_arr(getCudaRef(prhs[1]), mask,new_array,new_array_idx, sSize, ProjDir);
    if (status) mexErrMsgTxt(status);
    
    plhs[0] =  mxCreateDoubleScalar((double) new_array_num);
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double) free_array);

    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing part_max\n");return;}
   Dbg_printf("cuda: part_max\n");
  }
  else if (strcmp(command,"part_min")==0) { // partial minimum over array 
    int ref1,ProjDir,new_array_num;
    int sSize[3],dSize[3];
    float * mask = 0;float * new_array;
    float * new_array_idx=0;
    const char * status;
    if (nrhs != 4) mexErrMsgTxt("cuda: part_min needs three arguments\n");
    ref1=getCudaRefNum(prhs[1]);
    get3DSize(ref1,sSize);
    ProjDir=(int) mxGetScalar(prhs[3]);
    if (ProjDir < 1 || ProjDir > 3)
         mexErrMsgTxt("cuda: part_min unsupported projection direction (nees to be between 1 and 3)\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
         mexErrMsgTxt("cuda: part_min data to project cannot be complex valued\n");

    get3DSize(ref1,dSize);
    dSize[ProjDir-1] = 1;

    if (!mxIsEmpty(prhs[2]))
        mask=getCudaRef(prhs[2]);
    
    Dbg_printf4("dSize is %dx%dx%d\n",dSize[0],dSize[1],dSize[2]);
    new_array=cudaAllocDetailed(cuda_array_dim[ref1], dSize, cuda_array_type[ref1]);
    new_array_num= free_array;
    if (nlhs > 1)
        new_array_idx=cudaAllocDetailed(cuda_array_dim[ref1], dSize, single);
    
    status=CUDApmin_arr(getCudaRef(prhs[1]), mask,new_array,new_array_idx, sSize, ProjDir);
    if (status) mexErrMsgTxt(status);
    
    plhs[0] =  mxCreateDoubleScalar((double) new_array_num);
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double) free_array);

    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing part_min\n");return;}
   Dbg_printf("cuda: part_min\n");
  }
  else if (strcmp(command,"sum")==0) { // sum over array 
    if (nrhs != 2) mexErrMsgTxt("cuda: sum needs two arguments\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
    {
        if (nlhs > 0)
        { 
          float pres[2]; double * zr,* zi;
          const char * status=CUDAsum_carr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), pres);
          if (status) mexErrMsgTxt(status);
          plhs[0] = mxCreateDoubleMatrix(1, 1, mxCOMPLEX);
          zr = mxGetPr(plhs[0]);
          zi = mxGetPi(plhs[0]);
          zr[0]=(double) pres[0];
          zi[0]=(double) pres[1];
        }
    } else
        if (nlhs > 0)
        {
            float res;
            const char * status=CUDAsum_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), &res);
            if (status) mexErrMsgTxt(status);
            plhs[0] =  mxCreateDoubleScalar((double) res);
        }
        //    plhs[0] =  mxCreateDoubleScalar((double) cublasSasum(getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1]),1));
    
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing sum\n");return;}
   Dbg_printf("cuda: sum\n");
  }
  else if (strcmp(command,"norm")==0) { // norm of array 
    if (nrhs != 2) mexErrMsgTxt("cuda: norm needs two arguments\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
    {
        if (nlhs > 0)
        { 
          double real=(double) cublasSnrm2(getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1]),2);
          double imag=(double) cublasSnrm2(getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1])+1,2);
          plhs[0] =  mxCreateDoubleScalar(sqrt(real*real+imag*imag));
        }
    } else
        if (nlhs > 0)
            plhs[0] =  mxCreateDoubleScalar((double) cublasSnrm2(getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1]),1));
    
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error finding maximum\n");return;}
   Dbg_printf("cuda: norm\n");
  }
  else if (strcmp(command,"mtimes")==0) { // matrix product
    int ref1,ref2;
    int n1=1,m1=1,n2=1,m2=1;
    int dims[2];
    if (nrhs != 3) mexErrMsgTxt("cuda: mtimes needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    if (cuda_array_dim[ref1] > 2)
        mexErrMsgTxt("cuda: matrix multiplication. Object needs to be one or two dimensional\n");
    if (cuda_array_dim[ref2] > 2)
        mexErrMsgTxt("cuda: matrix multiplication. Object needs to be one or two dimensional\n");
    n1=cuda_array_size[ref1][0];n2=cuda_array_size[ref2][0];
    if (cuda_array_dim[ref1] > 1)
        m1=cuda_array_size[ref1][1];
    if (cuda_array_dim[ref2] > 1)
        m2=cuda_array_size[ref2][1];
    if (m1 != n2)
        mexErrMsgTxt("cuda: matrix multiplication. Matrix sizes not matching.\n");
    
    dims[0]=n1;
    dims[1]=m2;
    Dbg_printf("cuda: mtimes\n");
    Dbg_printf3("matrix A: %d x %d\n",n1,m1);
    Dbg_printf3("matrix B: %d x %d\n",n2,m2);
    if (isComplexType(ref1) && isComplexType(ref2))   // both complex
    {
        if (nlhs > 0)
        { 
            float * new_array=cudaAllocDetailed(2, dims, scomplex);
            cuComplex alpha,beta;
            alpha=make_cuComplex(1.0,0.0);
            beta=make_cuComplex(0.0,0.0);
            cublasCgemm('N','N',n1,m2,m1,alpha,(cuComplex *) getCudaRef(prhs[1]),n1,(cuComplex *)getCudaRef(prhs[2]),n2,beta,(cuComplex *) new_array,n1);
            plhs[0] =  mxCreateDoubleScalar((double) free_array);
        }
    } else { 
        if ((!isComplexType(ref1)) && (!isComplexType(ref2)))   // both real
        {
            if (nlhs > 0)
                {
                float * new_array=cudaAllocDetailed(2, dims, single);
                cublasSgemm('N','N',n1,m2,m1,1.0,getCudaRef(prhs[1]),n1,getCudaRef(prhs[2]),n2,0.0,new_array,n1);
                plhs[0] =  mxCreateDoubleScalar((double) free_array);
                }
        }
        else 
            { mexErrMsgTxt("cuda: complex matrix multiplied with real matrix. Not implemented. Please cast to complex type before\n"); return;}
        }
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error in matrix multiplication\n");return;}
   Dbg_printf("cuda: mtimes\n");
  }
  else if (strcmp(command,"mldivide")==0) { // equation systems solving  mldivide(a,b) solves A x = b for x
    int ref1,ref2;
    int n1=1,m1=1,n2=1,m2=1;
    int dims[2];
    if (nrhs != 3) mexErrMsgTxt("cuda: mldivide needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    if (cuda_array_dim[ref1] > 2)
        mexErrMsgTxt("cuda: mldivide; matrix equation solving. Object needs to be one or two dimensional\n");
    if (cuda_array_dim[ref2] > 2)
        mexErrMsgTxt("cuda: mldivide; matrix equation solving. Object needs to be one or two dimensional\n");
    n1=cuda_array_size[ref1][0];n2=cuda_array_size[ref2][0];
    if (cuda_array_dim[ref1] > 1)
        m1=cuda_array_size[ref1][1];
    if (cuda_array_dim[ref2] > 1)
        m2=cuda_array_size[ref2][1];
    if (m1 != n2)
        mexErrMsgTxt("cuda: matrix equation solving. Matrix sizes not matching.\n");
    
    dims[0]=n1;
    dims[1]=m2;
    Dbg_printf("cuda: mldivide\n");
    Dbg_printf3("matrix A: %d x %d\n",n1,m1);
    Dbg_printf3("vector B: %d x %d\n",n2,m2);
    if (isComplexType(ref1) && isComplexType(ref2))   // both complex
    {
        if (nlhs > 0)
        { 
#ifndef NOCULA
            culaStatus s;
            float * tmp_array=cloneArray(prhs[1]);  // copy A to Tmp, because the routine detrois the array
            int tofree=free_array;
            float * new_array=cloneArray(prhs[2]);  // copy B to X
            culaDeviceInt * IPIV;
            cudaError_t err=cudaMalloc((void **) & IPIV,n1*sizeof(int));  // 
            checkCudaError("Allocate mldivide complex IPIV",err);
            err=cudaMemset(IPIV,0,n1*sizeof(int));  // 
            checkCudaError("Memset mldivide complex IPIV",err);

            s=culaDeviceCgesv(n1,m2,(culaDeviceFloatComplex *)  tmp_array,n1,IPIV,(culaDeviceFloatComplex *) new_array,n2);  // replaces new_array with result
            checkCULAStatus("mldivide complex",s);
            err=cudaFree(IPIV);
            checkCudaError("mldivide free IPIV",err);
            cudaDelete(tofree);
#else  // just state error
            float * new_array=cudaAllocDetailed(2, dims, scomplex);
            cuComplex alpha,beta;
            alpha=make_cuComplex(1.0,0.0);
            beta=make_cuComplex(0.0,0.0);
            mexErrMsgTxt("cuda: mldivide not implemented as this version was compiled without Lapack-like CULA support.\n");
            cublasCgemm('N','N',n1,m2,m1,alpha,(cuComplex *) getCudaRef(prhs[1]),n1,(cuComplex *)getCudaRef(prhs[2]),n2,beta,(cuComplex *) new_array,n1);
#endif
            plhs[0] =  mxCreateDoubleScalar((double) free_array);
        }
    } else { 
        if ((!isComplexType(ref1)) && (!isComplexType(ref2)))   // both real
        {
            if (nlhs > 0)
                {
#ifndef NOCULA
            culaStatus s;
            float * tmp_array=cloneArray(prhs[1]);  // copy A to Tmp, because the routine detrois the array
            int tofree=free_array;
            float * new_array=cloneArray(prhs[2]);  // copy B to X
            culaDeviceInt * IPIV;
            cudaError_t err=cudaMalloc((void **) & IPIV,n1*sizeof(int));  // 
            checkCudaError("Allocate mldivide float IPIV",err);
            err=cudaMemset(IPIV,0,n1*sizeof(int));  // 
            checkCudaError("Memset mldivide float IPIV",err);
            
            s=culaDeviceSgesv(n1,m2,(culaDeviceFloat *) tmp_array,n1,IPIV,(culaDeviceFloat *) new_array,n2);  // replaces new_array with result
            checkCULAStatus("mldivide float",s);
            err=cudaFree(IPIV);
            checkCudaError("mldivide free IPIV",err);
            cudaDelete(tofree);
#else  // just state error
                float * new_array=cudaAllocDetailed(2, dims, single);
                mexErrMsgTxt("cuda: mldivide not implemented as this version was compiled without Lapack-like CULA support.\n");
                cublasSgemm('N','N',n1,m2,m1,1.0,getCudaRef(prhs[1]),n1,getCudaRef(prhs[2]),n2,0.0,new_array,n1);
#endif
                //cublasStrsm('L','U','N','N',n1,m2,m1,1.0,getCudaRef(prhs[1]),n1,getCudaRef(prhs[2]),n2,0.0,new_array,n1);  // A * X = 1.0 * B
                plhs[0] =  mxCreateDoubleScalar((double) free_array);
                }
        }
        else 
            { mexErrMsgTxt("cuda: complex matrix solving a with real matrix. Not implemented. Please cast to complex type before\n"); return;}
        }
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error in matrix multiplication\n");return;}
   Dbg_printf("cuda: mldivide\n");
  }
  else if (strcmp(command,"mvtimes")==0) { // matrix times vector
      int ref1,ref2;
    int n1=1,m1=1,n2=1,m2=1;
    int dims[2];
    if (nrhs != 3) mexErrMsgTxt("cuda: mvtimes needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    if (cuda_array_dim[ref1] > 2)
        mexErrMsgTxt("cuda: matrix vector multiplication. Matrix needs to be one or two dimensional\n");
    if (cuda_array_dim[ref2] > 2)
        mexErrMsgTxt("cuda: matrix vector multiplication. Vector needs to be one or two dimensional\n");
    n1=cuda_array_size[ref1][0];n2=cuda_array_size[ref2][0];
    if (cuda_array_dim[ref1] > 1)
        m1=cuda_array_size[ref1][1];
    if (cuda_array_dim[ref2] > 1)
        m2=cuda_array_size[ref2][1];
    if (m1 != n2)
        mexErrMsgTxt("cuda: matrix multiplication. Matrix sizes not matching.\n");
    
    dims[0]=n1;
    dims[1]=m2;
    if (n2 > 1  && m2 > 1)
        mexErrMsgTxt("cuda: matrix vector multiplication. Vector needs to be one dimensional\n");
    Dbg_printf("cuda: mvtimes\n");
    Dbg_printf3("matrix A: %d x %d\n",n1,m1);
    Dbg_printf3("vector B: %d x %d\n",n2,m2);
    if (isComplexType(ref1) && isComplexType(ref2))   // both complex
    {
        if (nlhs > 0)
        { 
            float * new_array=cudaAllocDetailed(2, dims, scomplex);
            cuComplex alpha,beta;
            alpha=make_cuComplex(1.0,0.0);
            beta=make_cuComplex(0.0,0.0);
            cublasCgemv('N',n1,m1,alpha,(cuComplex *) getCudaRef(prhs[1]),n1,(cuComplex *) getCudaRef(prhs[2]),1,beta,(cuComplex *) new_array,1);
            // unfortunately the Cgemv is not implemented yet in cuBLAS
            // cublasCgemm('N','N',n1,m2,m1,alpha,(cuComplex *) getCudaRef(prhs[1]),n1,(cuComplex *)getCudaRef(prhs[2]),n2,beta,(cuComplex *) new_array,n1);
            plhs[0] =  mxCreateDoubleScalar((double) free_array);
        }
    } else { 
        if ((!isComplexType(ref1)) && (!isComplexType(ref2)))   // both real
        {
            if (nlhs > 0)
                {
                float * new_array=cudaAllocDetailed(2, dims, single);
                cublasSgemv('N',n1,m1,1.0,getCudaRef(prhs[1]),n1,getCudaRef(prhs[2]),1,0.0,new_array,1);
                plhs[0] =  mxCreateDoubleScalar((double) free_array);
                }
        }
        else 
            { mexErrMsgTxt("cuda: complex matrix multiplied with real vector. Not implemented. Please cast to complex type before\n"); return;}
        }
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error in matrix with vector multiplication\n");return;}
   Dbg_printf("cuda: mvtimes\n");
  }
  else if (strcmp(command,"sprod")==0) { // matrix product
    int ref1,ref2;
    if (nrhs != 4) mexErrMsgTxt("cuda: sprod needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    if (getTotalSizeFromRefNum(ref1) != getTotalSizeFromRefNum(ref2))
        mexErrMsgTxt("cuda: sprod total size of vectors needs to be identical\n");    
    Dbg_printf("cuda: sprod\n");
    if (isComplexType(ref1) && isComplexType(ref2))   // both complex
    {
        if (nlhs > 0)
        { 
            mwSize dims[]={1,1};
            double mode=mxGetScalar(prhs[3]);   // complex conjugation ?
            cuComplex result;
            float * ar, * ai;
            if (mode == 0)
                result=cublasCdotu(getTotalSizeFromRefNum(ref1),(cuComplex *) getCudaRef(prhs[1]),1,(cuComplex *) getCudaRef(prhs[2]),1);
            else
                result=cublasCdotc(getTotalSizeFromRefNum(ref1),(cuComplex *) getCudaRef(prhs[1]),1,(cuComplex *) getCudaRef(prhs[2]),1);
            plhs[0]=mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxCOMPLEX);
            ar=(float *) mxGetPr(plhs[0]); // pointer to real part of array
            ai=(float *) mxGetPi(plhs[0]); // pointer to real part of array
            ar[0]= cuCrealf(result);
            ai[0]= cuCimagf(result);
            Dbg_printf4("cuda: sprod complex results %d elements: %g + i* %g\n",getTotalSizeFromRefNum(ref1),ar[0],ai[0]);
        }
    } else { 
        if ((!isComplexType(ref1)) && (!isComplexType(ref2)))   // both real
        {
            if (nlhs > 0)
                {
                float result=cublasSdot(getTotalSizeFromRefNum(ref1),getCudaRef(prhs[1]),1,getCudaRef(prhs[2]),1);
                plhs[0] =  mxCreateDoubleScalar(result);
                Dbg_printf3("cuda: sprod results %d elements: %g \n",getTotalSizeFromRefNum(ref1),result);
                }
        }
        else 
            { mexErrMsgTxt("cuda: scalar product between real and complex matrix. Not implemented. Please cast to complex type before\n"); return;}
        }
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error in scalar product\n");return;}
   Dbg_printf("cuda: sprod\n");
  }
  else if (strcmp(command,"minus_alpha_blas")==0) { // --------------------------------------------
    double alpha;float * mynew;
    if (nrhs != 3) mexErrMsgTxt("cuda: minus_alpha needs three arguments\n");
    alpha = mxGetScalar(prhs[2]);
    mynew=cloneArray(prhs[1]);
    cublasSaxpy (getTotalSizeFromRef(prhs[1]), alpha, pOne, 0, mynew, 1);  // y = alpha * x + y
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error adding alpha\n");return;}
   plhs[0] =  mxCreateDoubleScalar((double)free_array);

   Dbg_printf2("cuda: plus_alpha_blas %g\n",alpha);
  }
  else if (strcmp(command,"svd")==0) { // --------------------------------------------
#ifndef NOCULA
{   culaStatus s;
    int ref1=getCudaRefNum(prhs[1]);
    int n1=1,m1=1,dimsS[1],dimsU[2],dimsV[2];
    int tofree;
    float * tmp_array;
    if (cuda_array_dim[ref1] > 2)
        mexErrMsgTxt("cuda: svd; Object needs to be two dimensional\n");
    n1=cuda_array_size[ref1][0];m1=cuda_array_size[ref1][1];
    dimsS[1]=n1;
    dimsU[1]=n1;
    dimsU[2]=n1;
    dimsV[1]=m1;
    dimsV[2]=m1;

    tmp_array=cloneArray(prhs[1]);  // copy A to Tmp, because the routine detrois the array
    tofree=free_array;

    if (isComplexType(ref1))   // matrix is complex
        {
            if (nlhs > 1)  // compute more than just singular values
            {
                float * new_arrayU, * new_arrayS,* new_arrayVp;
                new_arrayU=cudaAllocDetailed(2, dimsU, scomplex);
                plhs[0] =  mxCreateDoubleScalar((double)free_array);

                new_arrayS=cudaAllocDetailed(1, dimsS, single); // Eigenvalues are allways real
                if (nlhs > 1)  // assign S
                    plhs[1] =  mxCreateDoubleScalar((double)free_array);
                new_arrayVp=cudaAllocDetailed(2, dimsV, scomplex);
                if (nlhs > 2)  // assign V'
                    plhs[2] =  mxCreateDoubleScalar((double)free_array);
            

                s=culaDeviceCgesvd('A','A',n1,m1,(culaDeviceFloatComplex *)  tmp_array,n1,(culaDeviceFloatComplex *)  new_arrayS, (culaDeviceFloatComplex *) new_arrayU,m1, (culaDeviceFloatComplex *) new_arrayVp,n1); // A is destroyed.
                checkCULAStatus("svd U S V complex",s);
            }
            else  // just SVDs needed
            {
                s=culaDeviceCgesvd('N','N',n1,m1,(culaDeviceFloatComplex *)  tmp_array,n1,(culaDeviceFloatComplex *)  tmp_array, (culaDeviceFloatComplex *) tmp_array,m1, (culaDeviceFloatComplex *) tmp_array,n1); // A is destroyed.
                checkCULAStatus("svd complex",s);
            }
       }
            else  // real valued
      {
            if (nlhs > 1)  // compute more than just singular values
            {
                float * new_arrayU, * new_arrayS,* new_arrayVp;
                new_arrayU=cudaAllocDetailed(2, dimsU, single);
                plhs[0] =  mxCreateDoubleScalar((double)free_array);

                new_arrayS=cudaAllocDetailed(1, dimsS, single); // Eigenvalues are allways real
                if (nlhs > 1)  // assign S
                    plhs[1] =  mxCreateDoubleScalar((double)free_array);
                new_arrayVp=cudaAllocDetailed(2, dimsV, single);
                if (nlhs > 2)  // assign V'
                    plhs[2] =  mxCreateDoubleScalar((double)free_array);            

                s=culaDeviceSgesvd('A','A',n1,m1,(culaDeviceFloatComplex *)  tmp_array,n1,(culaDeviceFloatComplex *)  new_arrayS, (culaDeviceFloatComplex *) new_arrayU,m1, (culaDeviceFloatComplex *) new_arrayVp,n1); // A is destroyed.
                checkCULAStatus("svd U S V single",s);
            }
            else  // just SVDs needed
            {
                float * new_arrayS=cudaAllocDetailed(1, dimsS, single); // Eigenvalues are allways real
                plhs[0] =  mxCreateDoubleScalar((double)free_array);
                s=culaDeviceSgesvd('N','N',n1,m1,(culaDeviceFloatComplex *)  tmp_array,n1,(culaDeviceFloatComplex *)  new_arrayS, (culaDeviceFloatComplex *) tmp_array,m1, (culaDeviceFloatComplex *) tmp_array,n1); // A is destroyed.
                checkCULAStatus("svd single",s);
            }                
        }

   cudaDelete(tofree);
}
#else  // just state error
            mexErrMsgTxt("cuda: mldivide not implemented as this version was compiled without Lapack-like CULA support.\n");
#endif

   Dbg_printf("cuda: svd\n");
  }
// Now include all the user-defined functions
#include "user/user_c_code.inc"
  else
  {
      printf("Error executing command %s\n",command);
        mexErrMsgTxt("cuda: Unknown command\n");
  }
  mxFree(command);
  Dbg_printf2("cuda: %d arrays in memory\n",cuda_curr_arrays);
 return;
}

