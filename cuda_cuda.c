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
% mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" -lcublas -lcufft -lcudart
mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64" -lcublas -lcufft -lcudart
 * Cula:
mex cuda_cuda.c cudaArith.obj "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" "-LC:\Program Files\CULA\R14\lib64" -lcublas -lcufft -lcudart -lcula_core -lcula_lapack

Linux:
mex -DNOCULA cuda_cuda.c cudaArith.o -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart -v
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

// #define DEBUG
          
#ifdef DEBUG
#define Dbg_printf(arg) printf(arg)
#define Dbg_printf2(arg1,arg2) printf(arg1,arg2)
#define Dbg_printf3(arg1,arg2,arg3) printf(arg1,arg2,arg3)
#define Dbg_printf4(arg1,arg2,arg3,arg4) printf(arg1,arg2,arg3,arg4)
#define Dbg_printf5(arg1,arg2,arg3,arg4,arg5) printf(arg1,arg2,arg3,arg4,arg5)
#define Dbg_printf6(arg1,arg2,arg3,arg4,arg5,arg6) printf(arg1,arg2,arg3,arg4,arg5,arg6)
#define Dbg_printf7(arg1,arg2,arg3,arg4,arg5,arg6,arg7) printf(arg1,arg2,arg3,arg4,arg5,arg6,arg7)
#else
#define Dbg_printf(arg) 
#define Dbg_printf2(arg1,arg2) 
#define Dbg_printf3(arg1,arg2,arg3) 
#define Dbg_printf4(arg1,arg2,arg3,arg4) 
#define Dbg_printf5(arg1,arg2,arg3,arg4,arg5) 
#define Dbg_printf6(arg1,arg2,arg3,arg4,arg5,arg6) 
#define Dbg_printf7(arg1,arg2,arg3,arg4,arg5,arg6,arg7) 
#endif

// https://undocumentedmatlab.com/blog/matlabs-internal-memory-representation
/* Definition of structure mxArray_tag for debugging purposes. Might not be fully correct 
 * for Matlab 2006b or 2007a, but the important things are. Thanks to Peter Boettcher.
 */
typedef struct {
  const char *name;
  mxClassID class_id;
  int vartype;
  mxArray    *crosslink;
  int      number_of_dims;
  int      refcount;
  struct {
    unsigned int    scalar_flag : 1;
    unsigned int    flag1 : 1;
    unsigned int    flag2 : 1;
    unsigned int    flag3 : 1;
    unsigned int    flag4 : 1;
    unsigned int    flag5 : 1;
    unsigned int    flag6 : 1;
    unsigned int    flag7 : 1;
    unsigned int    private_data_flag : 1;
    unsigned int    flag8 : 1;
    unsigned int    flag9 : 1;
    unsigned int    flag10 : 1;
    unsigned int    flag11 : 4;
    unsigned int    flag12 : 8;
    unsigned int    flag13 : 8;
  }   flags;
  int  rowdim;
  int  coldim;
  union {
    struct {
      double  *pdata;       // original: void*
      double  *pimag_data;  // original: void*
      void    *irptr;
      void    *jcptr;
      int     nelements;
      int     nfields;
    }   number_array;
    struct {
      mxArray **pdata;
      char    *field_names;
      void    *dummy1;
      void    *dummy2;
      int     dummy3;
      int     nfields;
    }   struct_array;
    struct {
      void  *pdata;  /*mxGetInfo*/
      char  *field_names;
      char  *name;
      int   checksum;
      int   nelements;
      int   reserved;
    }  object_array;
  }   data;
} mxArray_tag;


void printMem(size_t * start,int num)
{
    int i;
    for (i=0;i<num;i++)
        printf("%lx\n",start[i]);
    printf("\n");
}


#ifndef NOCULA
// #include "culadevice.h"    // Only for old Cula releases
#include "cula.h"   // Later releases such as R14
#include "cula_device.h"   // Later releases such as R14
#endif

#define UseHeap            // if defined the heap allocation and recycling is used. Otherwise normal allocation and free is used
#define AllocBlas          // use the cublasAlloc and free functions instead of cudaMalloc and cudaFree

#define MAX_ARRAYS 65738   // maximum number of arrays simultaneously on card

#ifdef UseHeap
#define MAX_HEAP 25        // size of the heap of arrays to recycle
#endif

// is defined in stdlib.h in Windows which is WRONG!!
#undef min
#define min(a,b) ((a)<(b) ? (a):(b))   // define our own version so that it is compatible with Linux and Windows
// is defined in stdlib.h in Windows which is WRONG!!
#undef max
#define max(a,b) ((a)>(b) ? (a):(b))   // define our own version so that it is compatible with Linux and Windows

#define long_long_abs(a) ((a)>0 ? (a):(-a))   // since the abs function from the library works only with int and not long long
        
#define CHECK_CUDAREF(p)     {if ((((double) (int) p) != p) || (p < 0) || (p >= MAX_ARRAYS)) \
          mexErrMsgTxt("cuda: Reference must be an integer between 0 and max_array\n");}

#define CHECK_CUDASIZES(ref1,ref2)     {if (cuda_array_dim[ref1] != cuda_array_dim[ref2]) mexErrMsgTxt("cuda: Arrays have different dimensionalities\n"); \
         int d; for (d=0;d< cuda_array_dim[ref1]) if (cuda_array_size[ref1][d] != cuda_array_size[ref2][d])\
          mexErrMsgTxt("cuda: Array sizes must be equal in all dimensions\n");}
#define CHECK_CUDATotalSIZES(ref1,ref2) {if (getTotalSizeFromRefNum(ref1) != getTotalSizeFromRefNum(ref2)) mexErrMsgTxt("cuda: Total array sizes are unequal. Bailing out\n");}

#define GET_MAXSIZE(ref,sizeC,numdims,totalResSize,NArgs) {                     \
    int d,a;                                                                    \
    SizeND sizeA;                                                               \
    numdims=1;                                                                  \
    totalResSize=1;                                                             \
    for (a=0;a<NArgs;a++) {                                                     \
        numdims=max(numdims,cuda_array_dim[ref[a]]);                            \
    }                                                                           \
    for (d=0;d<CUDA_MAXDIM;d++) {                                               \
    Dbg_printf2("GET_MAXSIZE d %d \n",d);                                       \
    sizeC.s[d]=1;                                                               \
    for (a=0;a<NArgs;a++) {                                                     \
        sizeA=getSizeFromRefNum(ref[a]);                                        \
        sizeC.s[d]=max((sizeA.s[d]),(sizeC.s[d]));}                             \
        totalResSize=totalResSize*sizeC.s[d];                                   \
        Dbg_printf5("d %d SizeA: %d, SizeC %d, totalRes %d\n",d,sizeA.s[d],sizeC.s[d],totalResSize);    \
    }                                                                           \
}

#define GET_MAXSIZE2(refA,refB,sizeC,numdims,totalResSize) { \
    SizeND sizeA;SizeND sizeB;                                                  \
    int d,dimsA=cuda_array_dim[refA], dimsB=cuda_array_dim[refB];                 \
    totalResSize=1;                                                             \
    numdims=max(dimsA,dimsB);                                                   \
    sizeA=getSizeFromRefNum(refA);                                          \
    sizeB=getSizeFromRefNum(refB);                                          \
    for (d=0;d<CUDA_MAXDIM;d++) {                                               \
        Dbg_printf2("GET_MAXSIZE2 d %d \n",d);                                  \
        sizeC.s[d]=max(sizeA.s[d],sizeB.s[d]);                                  \
        totalResSize*=sizeC.s[d];                                   \
        Dbg_printf7("d %d SizeA: %d, SizeB: %d, SizeC %d, numdims: %d, totalResSize %d\n",d,sizeA.s[d],sizeB.s[d],sizeC.s[d],numdims,totalResSize);\
    }                                                                           \
}


// If sizes not equal: check if one dimension size is 1
// #define CHECK_CUDATotalSIZESSingleton(ref1,ref2,totalResSize, sizeC, singletonA, singletonB, numdims, hasSingleton) { \
//     int d;                                                                      \
//     int dimsA=cuda_array_dim[ref1], dimsB=cuda_array_dim[ref2], SA,SB;          \
//     SizeND sizeA; SizeND sizeB;   \
//     sizeA=getSizeFromRefNum(ref1);sizeB=getSizeFromRefNum(ref2);totalResSize=1;   \
//     numdims=max(dimsA,dimsB);                                                   \
//     Dbg_printf4("CHECK_CUDATotalSIZESSingleton numdims %d, ref1 %d ref2 %d\n",numdims,ref1,ref2); \
//     for (d=0;d<CUDA_MAXDIM;d++) {                                               \
//     Dbg_printf2("CHECK_CUDATotalSIZESSingleton d %d \n",d); \
//         if (d>dimsA) sizeA.s[d]=1;                                                \
//         if (d>dimsB) sizeB.s[d]=1;                                                \
//         singletonA.s[d]=0;singletonB.s[d]=0;                                        \
//         SA=sizeA.s[d]; SB=sizeB.s[d];                                       \
//         sizeC.s[d]=max(SA,SB);                                        \
//     Dbg_printf5("d %d SizeA: %d, SizeB: %d, SizeC %d\n",d,SA,SB,sizeC.s[d]); \
//         totalResSize *= sizeC.s[d];                                               \
//         if (SA != SB)                                               \
//             if (SA != 1 && SB != 1)                                     \
//             {                                                               \
//                 printf("Dimension %d, SizeA: %d, SizeB: %d\n",d,SA,SB);     \
//                 mexErrMsgTxt("cuda: Sizes are incompatible, even considering singleton dimensions. Bailing out\n");} \
//             else if (SA==1) {singletonA.s[d]=1;hasSingleton=1;}             \
//             else {singletonB.s[d]=1;hasSingleton=1;}                              \
//     }                                                                               \
//     Dbg_printf7("CHECK_CUDATotalSIZESSingleton ref1 %d, ref2 %d, totalResSize %d sizeC[0] %d sizeC[1] %d hasSingleton %d\n",ref1,ref2,totalResSize,sizeC.s[0],sizeC.s[1],hasSingleton); \
// }

// If sizes not equal: check if one dimension size is 1
#define CHECK_CUDATotalSIZESSingleton(ref1,sizeC, singletonA, hasSingleton) {   \
    int d;                                                                      \
    int dimsA=cuda_array_dim[ref1]; size_t SA;                                  \
    SizeND sizeA=getSizeFromRefNum(ref1);                                       \
    for (d=0;d<CUDA_MAXDIM;d++) {                                               \
        Dbg_printf2("CHECK_CUDATotalSIZESSingleton d %d \n",d);                 \
        if (d>dimsA) sizeA.s[d]=1;                                              \
        singletonA.s[d]=0;                                                      \
        SA=sizeA.s[d];                                                          \
        Dbg_printf4("d %d SizeA: %d, SizeC %d\n",d,SA,sizeC.s[d]);              \
        if (SA != sizeC.s[d])                                                   \
            if (SA != 1)                                                        \
            {                                                                   \
                printf("Dimension %d, SizeA: %d, SizeC: %d\n",d,SA,sizeC.s[d]); \
                mexErrMsgTxt("cuda: Sizes are incompatible, even considering singleton dimensions. Bailing out\n");} \
            else {singletonA.s[d]=1;hasSingleton=1;}                            \
    }                                                                           \
    Dbg_printf3("CHECK_CUDATotalSIZESSingleton ref1 %d, hasSingleton %d\n",ref1,hasSingleton); \
}

// Generates a new array (float * newarr) from a size vector
#define CUDA_NewArrayFromSize(IsComplex)                                                    \
    int dims_sizes; size_t nsizes[CUDA_MAXDIM],d,tsize=1;                                           \
    double * dsizes;float * newarr=0;                                                       \
    if (nrhs < 2) mexErrMsgTxt("cuda: newarr needs >= 2 arguments\n");                      \
    dims_sizes=(int)(mxGetM(prhs[1]) * mxGetN(prhs[1]));                                    \
    dsizes=mxGetPr(prhs[1]);                                                                \
    if (dims_sizes >= CUDA_MAXDIM)                                                          \
        mexErrMsgTxt("cuda: newarr to many dimensions (>CUDA_MAXDIM)\n");                   \
    for (d=0;d<dims_sizes;d++) {nsizes[d]=(size_t) dsizes[d];tsize *= nsizes[d];}              \
    Dbg_printf5("newarray with dimension %d, sizes %d %d %d\n",dims_sizes,nsizes[0],nsizes[1],nsizes[2]); \
                                                                                            \
    if (IsComplex) {                                                                        \
        newarr=cudaAllocDetailed(dims_sizes,nsizes,scomplex);                               \
    } else {                                                                                \
        newarr=cudaAllocDetailed(dims_sizes,nsizes,single);                                 \
    }

//  ----------------- Macros of code snippets, defining common ways of calling Cuda ---------
// ----------- macro with two arguments. An array and an index. AllocNum is 1 or 2 for first or second argument
#define CallCUDA_IdxFkt(FktName,AllocFkt,AllocNum,FirstSizeNum)                                         \
    const char *ret=0; size_t _ref1,_ref2;                                                     \
    if (nrhs != 3 && nrhs != 4) mexErrMsgTxt("cuda: " #FktName " needs three or four arguments\n");               \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    if (isComplexType(_ref2)) {mexErrMsgTxt("cuda: " #FktName " indexing with complex index arrays is not allowed\n");} \
    if (isComplexType(_ref1)) {                                     \
        Dbg_printf("cuda: complex array " #FktName " index array\n");                     \
        ret=CUDAcarr_##FktName##_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[AllocNum]),getTotalSizeFromRef(prhs[FirstSizeNum]),getTotalSizeFromRef(prhs[2])); } \
    else    {                                                                            \
        Dbg_printf("cuda: array " #FktName " index array\n");                                     \
        ret=CUDAarr_##FktName##_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[AllocNum]),getTotalSizeFromRef(prhs[FirstSizeNum]),getTotalSizeFromRef(prhs[2])); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \

                
#define CallCUDA_IdxFktConst(FktName,AllocFkt)                                            \
    const char *ret=0;                                                                      \
    size_t ref;                                                                                \
    if (nrhs != 4) mexErrMsgTxt("cuda: " #FktName "_1Didx_const needs four arguments\n");        \
    ref=getCudaRefNum(prhs[1]);                                                         \
    if (mxIsComplex(prhs[3])) {                                                             \
        double  myreal = mxGetScalar(prhs[3]);                                              \
        double  myimag = * ((double *) (mxGetPi(prhs[3])));                                 \
        if (isComplexType(ref)) {                                        \
            Dbg_printf3("cuda: complex array " #FktName " complex-const Real: %g Imag: %g\n",myreal,myimag);                 \
            ret=CUDAcarr_##FktName##_const(getCudaRef(prhs[2]),(float) myreal,(float) myimag,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[2])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            Dbg_printf3("cuda: float array " #FktName " complex-const Real: %g Imag: %g\n",myreal,myimag);             \
            ret=CUDAarr_##FktName##_Cconst(getCudaRef(prhs[2]),(float) myreal,(float) myimag,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[2])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[3]);                                                \
        if (isComplexType(ref)) {                                                           \
            Dbg_printf("cuda: complex array " #FktName " real-const\n");                    \
            ret=CUDAcarr_##FktName##_const(getCudaRef(prhs[2]),(float) alpha,0.0,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[2]));    \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName " real-const\n");                      \
            ret=CUDAarr_##FktName##_const(getCudaRef(prhs[2]),(float) alpha,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[2])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } 

    
// Macro for index function with one list per dimension. The indices need to be given as a matrix with the longest index list dominating. The                 
#define CallCUDA_IdxFktND(FktName,AllocFkt)                                         \
    const char *ret=0; size_t _ref1,_ref2; SizeND sizeA,sizeC;float * pC;           \
    if (nrhs != 4 && nrhs != 5) mexErrMsgTxt("cuda: " #FktName " needs four or five (assignment) arguments\n");               \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    if (isComplexType(_ref2)) {mexErrMsgTxt("cuda: " #FktName " indexing with complex index arrays is not allowed\n");} \
    sizeA=getSizeFromRefNum(_ref1);                      \
    sizeC=SizeNDFromRef(prhs[3]);                           \
    if (nrhs ==5)                                           \
        pC=getCudaRef(prhs[4]);                          \
    else                                                    \
        pC=AllocFkt(prhs[1], sizeC, cuda_array_dim[_ref1]);     \
    Dbg_printf7("cuda: sizes of " #FktName " are %dx%dx%d , %dx%dx%d \n",sizeA.s[0],sizeA.s[1],sizeA.s[2],sizeC.s[0],sizeC.s[1],sizeC.s[2]); \
    if (isComplexType(_ref1)) {                                     \
        Dbg_printf("cuda: complex array " #FktName " index array\n");                     \
        ret=CUDAcarr_##FktName##_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),pC,sizeA,sizeC,cuda_array_size[_ref2][0],cuda_array_dim[_ref1]); } \
    else    {                                                                            \
        Dbg_printf("cuda: array " #FktName " index array\n");                                     \
        Dbg_printf4("cuda: ref1 %d, ref2 %d, dim %d \n",_ref1,_ref2,cuda_array_dim[_ref1]); \
        ret=CUDAarr_##FktName##_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),pC,sizeA,sizeC,cuda_array_size[_ref2][0],cuda_array_dim[_ref1]); }\
    Dbg_printf("cuda: came back \n"); \
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
                
// Macro for index function with one list per dimension. The indices need to be given as a matrix with the longest index list dominating. 
// A size vector of the compressed assinged area is needed for the parallelization, even though there is only a single value to be assigned.

#define CallCUDA_IdxFktNDConst(FktName)                                         \
    const char *ret=0; size_t _ref1,_ref2; SizeND sizeA,sizeC;                              \
    if (nrhs != 5) mexErrMsgTxt("cuda: " #FktName " needs five arguments\n");               \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    if (isComplexType(_ref2)) {mexErrMsgTxt("cuda: " #FktName " indexing with complex index arrays is not allowed\n");} \
    sizeA=getSizeFromRefNum(_ref1);                      \
    sizeC=SizeNDFromRef(prhs[4]);                                                             \
    Dbg_printf7("cuda: sizes of " #FktName " are %dx%dx%d , %dx%dx%d \n",sizeA.s[0],sizeA.s[1],sizeA.s[2],sizeC.s[0],sizeC.s[1],sizeC.s[2]); \
    if (mxIsComplex(prhs[3])) {                                                             \
        double  myreal = mxGetScalar(prhs[3]);                                              \
        double  myimag = * ((double *) (mxGetPi(prhs[3])));                                 \
        if (isComplexType(_ref1)) {                                        \
            Dbg_printf3("cuda: complex array " #FktName " complex-const Real: %g Imag: %g\n",myreal,myimag);                 \
            ret=CUDAcarr_##FktName##_const_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),(float) myreal,(float) myimag,sizeA,sizeC,cuda_array_size[_ref2][0],cuda_array_dim[_ref1]);   \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            Dbg_printf3("cuda: float array " #FktName " complex-const Real: %g Imag: %g\n",myreal,myimag);             \
            ret=CUDAarr_##FktName##_Cconst_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),(float) myreal,(float) myimag,sizeA,sizeC,cuda_array_size[_ref2][0],cuda_array_dim[_ref1]);   \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[3]);                                                \
        if (isComplexType(_ref1)) {                                                           \
            Dbg_printf("cuda: complex array " #FktName " real-const\n");                    \
            ret=CUDAcarr_##FktName##_const_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),(float) alpha,0.0,sizeA,sizeC,cuda_array_size[_ref2][0],cuda_array_dim[_ref1]);   \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName " real-const\n");                      \
            ret=CUDAarr_##FktName##_const_ind(getCudaRef(prhs[1]),getCudaRef(prhs[2]),(float) alpha,sizeA,sizeC,cuda_array_size[_ref2][0],cuda_array_dim[_ref1]);  \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } }  \
    Dbg_printf("cuda: came back \n"); \
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
                

            
    /*  This was for debugging purposes. Commented out for now

    //  ----------------- general macro for binary array functions such as plus and minus ---------
#define CallCUDA_BinaryFktOld(FktName,AllocFkt)                                         \
    const char *ret=0; int _ref1,_ref2;                                                     \
    int TotalResSize=0, numdims=0, usedims=0, hasSingleton=0; SizeND sizeC; BoolND singletonA,singletonB;              \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");               \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    CHECK_CUDATotalSIZESSingleton(_ref1,_ref2, TotalResSize, sizeC, singletonA, singletonB, numdims, hasSingleton) \
        Dbg_printf3("TotalSize %d, %d\n",TotalResSize,getTotalSizeFromRef(prhs[1]));                                     \
    if (isComplexType(_ref1) && isComplexType(_ref2)) {                                     \
        Dbg_printf("cuda: complex array " #FktName " complex array\n");                     \
        ret=CUDAcarr_##FktName##_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),getTotalSizeFromRef(prhs[1]),usedims,sizeC,singletonA,singletonB); } \
    else if (isComplexType(_ref1)) {                                       \
        Dbg_printf("cuda: complex array " #FktName " float array\n");                       \
        ret=CUDAcarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),getTotalSizeFromRef(prhs[1]),usedims,sizeC,singletonA,singletonB); } \
    else if (isComplexType(_ref2)) {                                       \
        Dbg_printf("cuda: float array " #FktName " complex array\n");                       \
        ret=CUDAarr_##FktName##_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),getTotalSizeFromRef(prhs[1]),usedims,sizeC,singletonA,singletonB); }\
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        Dbg_printf3("TotalSize %d, %d\n",TotalResSize,getTotalSizeFromRef(prhs[1]));                                     \
        ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),getTotalSizeFromRef(prhs[1]),usedims,sizeC,singletonA,singletonB); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
*/                
                
//  ----------------- general macro for binary array functions such as plus and minus ---------
#define CallCUDA_BinaryFkt(FktName,AllocFkt)                                                \
    const char *ret=0; size_t _ref1,_ref2;                                                     \
    size_t TotalResSize=0; int numdims=0, usedims=0, hasSingleton=0; SizeND sizeC; BoolND singletonA,singletonB;              \
    Dbg_printf("cuda: CallCUDA_BinaryFkt: " #FktName " Checkpoint 1\n");                     \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");              \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    Dbg_printf("cuda: CallCUDA_BinaryFkt: " #FktName " before CHECK_CUDATotalSIZESSingleton\n");                     \
    GET_MAXSIZE2(_ref1,_ref2,sizeC,numdims,TotalResSize)                                      \
    CHECK_CUDATotalSIZESSingleton(_ref1, sizeC, singletonA, hasSingleton)                   \
    CHECK_CUDATotalSIZESSingleton(_ref2, sizeC, singletonB, hasSingleton)                   \
    usedims=numdims;                                                                        \
    if (! hasSingleton) usedims=0;                                                          \
    if (isComplexType(_ref1) && isComplexType(_ref2)) {                                     \
        Dbg_printf("cuda: complex array " #FktName " complex array\n");                     \
        ret=CUDAcarr_##FktName##_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),TotalResSize,usedims,sizeC,singletonA,singletonB); } \
    else if (isComplexType(_ref1)) {                                       \
        Dbg_printf("cuda: complex array " #FktName " float array\n");                       \
        ret=CUDAcarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),TotalResSize,usedims,sizeC,singletonA,singletonB); } \
    else if (isComplexType(_ref2)) {                                       \
        Dbg_printf("cuda: float array " #FktName " complex array\n");                       \
        ret=CUDAarr_##FktName##_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[2],sizeC,numdims),TotalResSize,usedims,sizeC,singletonA,singletonB); }\
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),TotalResSize,usedims,sizeC,singletonA,singletonB); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  

            
//  ----------------- general macro for binary array functions such as plus and minus ---------
#define CallCUDA_NArgsFkt(FktName,AllocFkt,Nargs)                                                \
    const char *ret=0; size_t _refNum[Nargs]; float * _ref[Nargs];                                  \
    size_t n,TotalResSize=0, numdims=0, usedims=0, hasSingleton=0; SizeND sizeC; BoolND singleton[Nargs];\
    Dbg_printf("cuda: CallCUDA_NargsFkt: " #FktName " Checkpoint 1\n");                             \
    if (nrhs != Nargs+1) mexErrMsgTxt("cuda: " #FktName " needs " #Nargs " arguments\n");           \
    for (n=0;n<Nargs;n++) {_refNum[n]=getCudaRefNum(prhs[n+1]);_ref[n]=getCudaRef(prhs[n+1]);}      \
    Dbg_printf("cuda: CallCUDA_NargsFkt: " #FktName " before CHECK_CUDATotalSIZESSingleton\n");     \
    GET_MAXSIZE(_refNum,sizeC,numdims,TotalResSize,Nargs)                                              \
    for (n=1;n<Nargs;n++) {CHECK_CUDATotalSIZESSingleton(_refNum[n], sizeC, singleton[n], hasSingleton) } \
    usedims=numdims;                                                                                \
    if (! hasSingleton) usedims=0;                                                                  \
    Dbg_printf("cuda: array " #FktName " array\n");                                                 \
    ret=CUDAarr_##FktName##_NArgs(_ref,AllocFkt(prhs[1],sizeC,numdims),TotalResSize,usedims,sizeC,singleton); \
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}
        
//  ----------------- for calling with array and constant ---------
#define CallCUDA_UnaryFktConst(FktName,AllocFkt)                                            \
    const char *ret=0;                                                                      \
    size_t ref;                                                                                \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName "_alpha needs three arguments\n");        \
    ref=getCudaRefNum(prhs[1]);                                                         \
    if (mxIsComplex(prhs[2])) {                                                             \
        double  myreal = mxGetScalar(prhs[2]);                                              \
        double  myimag = * ((double *) (mxGetPi(prhs[2])));                                 \
        if (isComplexType(ref)) {                                        \
            Dbg_printf3("cuda: complex array " #FktName " complex-const Real: %g Imag: %g\n",myreal,myimag);                 \
            ret=CUDAcarr_##FktName##_const(getCudaRef(prhs[1]),(float) myreal,(float) myimag,AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        } else {                                                                            \
            float * narr=cudaAllocComplex(prhs[1]);                                         \
            Dbg_printf3("cuda: float array " #FktName " complex-const Real: %g Imag: %g\n",myreal,myimag);             \
            ret=CUDAarr_##FktName##_Cconst(getCudaRef(prhs[1]),(float) myreal,(float) myimag,narr,getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
        }                                                                                   \
    } else {                                                                                \
        double alpha = mxGetScalar(prhs[2]);                                                \
        if (isComplexType(ref)) {                                                           \
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
    size_t ref;                                                                                \
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
            float * narr=cudaAllocComplex(prhs[1]);                                         \
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
    /*
#define CallCUDA_BinaryHRealFkt(FktName,AllocFkt)                                         \
    const char *ret=0; int _ref1,_ref2;                                                     \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");               \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    CHECK_CUDATotalSIZES(_ref1,_ref2);                                                             \
    if (isComplexType(_ref1) && isComplexType(_ref2)) {   \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real argument data.\n");}  \
    else if (isComplexType(_ref1)) {                                       \
        Dbg_printf("cuda: complex array " #FktName " float array\n");                       \
        ret=CUDAcarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); } \
    else if (isComplexType(_ref2)) {                                       \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real argument data.\n");}  \
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1]),getTotalSizeFromRef(prhs[1])); }\
    if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  
*/
    
//  --------- The ones below are FOR REAL-VALUED Functions only --- (e.g. comparison operations) ----------
#define CallCUDA_BinaryRealFkt(FktName,AllocFkt)                                            \
    const char *ret=0; size_t _ref1,_ref2;                                                     \
    size_t TotalResSize=0; int numdims=0, hasSingleton=0; SizeND sizeC; BoolND singletonA,singletonB;              \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");              \
    _ref1=getCudaRefNum(prhs[1]);_ref2=getCudaRefNum(prhs[2]);                              \
    Dbg_printf("cuda: CallCUDA_BinaryFkt: " #FktName " before CHECK_CUDATotalSIZESSingleton\n");                     \
    GET_MAXSIZE2(_ref1,_ref2,sizeC,numdims,TotalResSize)                                      \
    CHECK_CUDATotalSIZESSingleton(_ref1, sizeC, singletonA, hasSingleton)                   \
    CHECK_CUDATotalSIZESSingleton(_ref2, sizeC, singletonB, hasSingleton)                   \
    if (isComplexType(_ref1) || isComplexType(_ref2))                                       \
      mexErrMsgTxt("cuda: Function \"" #FktName "\" is defined only for real valued data.\n");  \
    else {                                                                                  \
        Dbg_printf("cuda: array " #FktName " array\n");                                     \
        if (hasSingleton)                                                                   \
            ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),TotalResSize,numdims,sizeC,singletonA,singletonB);  \
        else                                                                                \
            ret=CUDAarr_##FktName##_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),AllocFkt(prhs[1],sizeC,numdims),TotalResSize,0,sizeC,singletonA,singletonB);  \
        }\
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
    if (nrhs != 2) mexErrMsgTxt("cuda: " #FktName " needs one argument\n");              \
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
    if (nrhs != 2) mexErrMsgTxt("cuda: " #FktName " needs one argument\n");                 \
        if (isComplexType(getCudaRefNum(prhs[1]))) {                                        \
            Dbg_printf("cuda: complex array " #FktName "\n");                               \
            ret=CUDA##FktName##_carr(getCudaRef(prhs[1]),AllocCommand(prhs[1]),getTotalSizeFromRef(prhs[1]));    \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName "\n");                                 \
            ret=CUDA##FktName##_arr(getCudaRef(prhs[1]),AllocCommand(prhs[1]),getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");} \
    } 

//  ----------------- Unary function  ------AllocCommand determines whether the result type is that same, complex or real ----------
#define CallCUDA_UnaryFktSizeConst(FktName,AllocFkt)                                    \
    const char *ret=0;SizeND SC,SA;float * pC;size_t refA;                                                 \
    if (nrhs != 4) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");                 \
        double alpha = mxGetScalar(prhs[2]);                                                \
        refA=getCudaRefNum(prhs[1]);                                                        \
        SC=SizeNDFromRef(prhs[3]);                                                          \
        SA=getSizeFromRefNum(refA);                                          \
        pC=AllocFkt(prhs[1], SC, cuda_array_dim[refA]);                      \
        if (isComplexType(refA)) {                                        \
            Dbg_printf("cuda: complex array " #FktName "\n");                               \
            ret=CUDA##FktName##_carrSizeConst(getCudaRef(prhs[1]),pC,(float) alpha, SA, SC, getTotalSizeFromRef(prhs[1]));    \
        } else {                                                                            \
            Dbg_printf("cuda: float array " #FktName "\n");                                 \
            ret=CUDA##FktName##_arrSizeConst(getCudaRef(prhs[1]),pC,(float) alpha, SA, SC, getTotalSizeFromRef(prhs[1])); \
            if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");} \
    } 
    
// Snippet below expects a vector(CUDA_MAXDIM) and an array as input and generates an array as output. E.g. circshift
#define CallCUDA_ArrVecFkt(FktName,AllocCommand,SetToVal)                                   \
size_t dims_sizes,nshifts[CUDA_MAXDIM], dsize[CUDA_MAXDIM],d,tsize=1,ref;                   \
    double * dshifts;float * newarr=0;                                                      \
    const char * ret;                                                                       \
                                                                                            \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");              \
    dims_sizes=(size_t)(mxGetM(prhs[2]) * mxGetN(prhs[2])); dshifts=mxGetPr(prhs[2]);       \
    if (dims_sizes > CUDA_MAXDIM)                                                           \
        mexErrMsgTxt("cuda: " #FktName " to many dimensions (>CUDA_MAXDIM)\n");             \
                                                                                            \
    ref=getCudaRefNum(prhs[1]);                                                             \
                                                                                            \
    for (d=0;d<CUDA_MAXDIM;d++) {                                                           \
         if (d<dims_sizes)                                                                  \
            nshifts[d]=(size_t) dshifts[d];                                                 \
         else                                                                               \
             nshifts[d]=SetToVal;                                                           \
         if (d<cuda_array_dim[ref])                                                         \
             dsize[d]=cuda_array_size[ref][d];                                              \
         else                                                                               \
             dsize[d]=1;                                                                    \
    }                                                                                       \
    Dbg_printf5("" #FktName " with size %d, shifts %d %d %d\n",dims_sizes,nshifts[0],nshifts[1],nshifts[2]);    \
    if (nrhs != 3) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");                 \
                                                                                            \
    if (isComplexType(ref)) {                                                               \
        ret=CUDAcarr_##FktName##_vec(getCudaRef(prhs[1]),nshifts,AllocCommand(prhs[1]),dsize,getTotalSizeFromRef(prhs[1])); \
        if (ret!=(const char *) cudaSuccess) { printf("cuda complex " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
    } else {                                                                                \
        ret=CUDAarr_##FktName##_vec(getCudaRef(prhs[1]),nshifts,AllocCommand(prhs[1]),dsize,getTotalSizeFromRef(prhs[1])); \
        if (ret!=(const char *) cudaSuccess) { printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("cuda error " #FktName ": Bailing out");}  \
    }                                                                                       \
    Dbg_printf("" #FktName "\n");                                                           \

            
// Snippet below expects a vector(CUDA_MAXDIM) and two vectors as input and generates an array as output. E.g. xx,yy,zz,rr,phiphi
#define CallCUDA_GenArrFkt(FktName)                                                         \
    {VecND vec1,vec2;                                                                       \
    SizeND sSize;                                                                           \
    if (nrhs != 4) mexErrMsgTxt("cuda: " #FktName " needs three arguments\n");              \
    else {CUDA_NewArrayFromSize(0)  /* uses Matlab Ref 1 as a size vector, 0 for no complex number*/   \
    vec1=VecNDFromRef(prhs[2]);                                                             \
    vec2=VecNDFromRef(prhs[3]);                                                             \
    sSize=SizeNDFromRef(prhs[1]);                                                           \
    CUDAarr_##FktName##_2vec(newarr, vec1, vec2, sSize, getTotalSizeFromRefNum(free_array));\
    if (! (cudaGetLastError() == cudaSuccess))                                              \
    	{ printf("cuda " #FktName ": %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt(" " #FktName " Bailing out");} \
    Dbg_printf("" #FktName "\n");                                                           \
    } }
    
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

static int showwarning=1;  // defines whether a warning/error message will be shown when a wrong object is deleted and another one was primed for (protected) deletion

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



#define MAX_FFTPLANS 25
#define MaxCudaDevices 5

static cufftHandle cuda_FFTplans[MAX_FFTPLANS][3][3];  // stores a number of plans for each type of transform and dimension
static size_t cuda_FFTplan_sizes[MAX_FFTPLANS][3][3][CUDA_MAXDIM];    // Stores the associated sizes, to check if plan needs to be re-done
static size_t cuda_FFTplan_dirYes[MAX_FFTPLANS][3][3][CUDA_MAXDIM];    // Stores the information which axes to transform, to check if plan needs to be re-done
static size_t cuda_FFTplan_extraBatch[MAX_FFTPLANS][3][3];
static size_t cuda_FFTplan_extraBatchStride[MAX_FFTPLANS][3][3];
        
static float * cuda_arrays[MAX_ARRAYS];  // static array of cuda arrays
static int cuda_array_dim[MAX_ARRAYS];  // type tags see CUDA_TYPE definitions above
static size_t * cuda_array_size[MAX_ARRAYS];  // dynamically allocated 
//static int cuda_array_origFTsize[MAX_ARRAYS];  // for storing the original data size, when doing FFTs
//static float cuda_array_FTscale[MAX_ARRAYS];  // to do the maginitude correction
static int cuda_array_type[MAX_ARRAYS];  // type tags see CUDA_TYPE definitions above

static size_t free_array=0;   // next number of array to fill
static size_t cuda_curr_arrays=0;
static int cuda_initialized=0;
// static int sizes[100];
// static int dims=0,totalsize=0;
static float * pOne[MaxCudaDevices], * pZero[MaxCudaDevices], * pReturnVal[MaxCudaDevices];
static size_t NumZero[MaxCudaDevices],NumOne[MaxCudaDevices],NumReturnVal[MaxCudaDevices];  // = {-1,-1,-1,-1,-1}
static int currentCudaDevice=0;

static int ignoreDelete=0;  // Needed to avoid delete (or copyiing the whole array) after subassign
static size_t ignoreRef=-2;  // Needed to avoid delete (or copyiing the whole array) after subassign
static void * fastMem=0, * fastMemI=0;
static const int fastMemSize=1024; // number of byte for fast transfer
static size_t SumAllocated=0;   // next number of array to fill

static int forceDeviceSynchronization=0;  // If this is active, synchonizeKernels() will be called after every comand. This is useful for profiling in Matlab.

#ifdef UseHeap
static void * mem_heap[MAX_HEAP];   // memory which  can be reused if size matches
static size_t memsize_heap[MAX_HEAP];  // sizes in bytes
static size_t mem_heap_pos=0;        // position of the next entry to free from the heap
static size_t mem_heap_first_free=0;        // position of the next entry to free from the heap
static int mem_heap_allocated=0;  // filling of the heap. Has to allow negative size!
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
void get3DSize(size_t ref, size_t * mysize) {
    int d;
    for (d=0;d<3;d++)
        if (d< cuda_array_dim[ref])
            mysize[d]=cuda_array_size[ref][d];
        else
            mysize[d]=1;
}

// returns array size in 3D coordinates (expanding with ones)
SizeND getSizeFromRefNum(size_t ref) {
    int d;
    SizeND mysize;
    for (d=0;d<CUDA_MAXDIM;d++) {
    if (d< cuda_array_dim[ref])
            mysize.s[d]=cuda_array_size[ref][d];
        else
            mysize.s[d]=1;
    Dbg_printf4("getSizeFromRefNum d %d ref %d mysize[d] %d\n",d,ref,mysize.s[d]);
    }
    return mysize;
}

// returns array size in 5D coordinates (expanding with ones)
void get5DSize(size_t ref, size_t * mysize) {
    int d;
    for (d=0;d<5;d++)
        if (d< cuda_array_dim[ref])
            mysize[d]=cuda_array_size[ref][d];
        else
            mysize[d]=1;
}

// returns array size in 5D coordinates (expanding with ones)
Size5D getSize5D(size_t ref) {
    Size5D mysize;
    int d;
    for (d=0;d<5;d++)
        if (d< cuda_array_dim[ref])
            mysize.s[d]=cuda_array_size[ref][d];
        else
            mysize.s[d]=1;
    return mysize; 
}

// Constructs a floating point ND vector from a Matlab reference
VecND VecNDFromRef(const mxArray * MatlabRef) {
    VecND ret;
    int d,dim;
    double * pVal=mxGetPr(MatlabRef);
    dim=(int)(mxGetM(MatlabRef) * mxGetN(MatlabRef));
    for (d=0;d<CUDA_MAXDIM;d++)
        if (d< dim)
            ret.s[d]=(float) pVal[d];
        else
            ret.s[d]=0.0f;    
    return ret;
}

// Constructs an integer size vector from a Matlab reference
SizeND SizeNDFromRef(const mxArray * MatlabRef) {
    SizeND ret;
    int d,dim;
    double * pVal=mxGetPr(MatlabRef);
    dim=(int)(mxGetM(MatlabRef) * mxGetN(MatlabRef));
    for (d=0;d<CUDA_MAXDIM;d++)
        if (d< dim)
            ret.s[d]=(int) pVal[d];
        else
            ret.s[d]=1;    
    return ret;
}

 size_t getCudaRefNum(const mxArray * arg) {
    double cudaref;
    if (! mxIsDouble(arg))
      mexErrMsgTxt("cuda: Obtaining reference number. Number must be a double");
    
    cudaref = mxGetScalar(arg);

    /* printf("cuda: get ref %g, next free array %d\n",cudaref,free_array); */

    CHECK_CUDAREF(cudaref);
    if (cuda_array_size[(size_t) cudaref] == 0) {
        printf("While deleting Cuda Reference no. %ld\n",(size_t) cudaref);
        mexErrMsgTxt("cuda: Trying to access non-existing cuda reference.");
    }
        
    return (size_t) cudaref;
 }

bool isComplexType(size_t ref) {
    return (cuda_array_type[ref] >= fftHalfSComplex);
}

size_t getTotalSize(const int dim,const size_t * sizevec) {  // returns the total size in numbers (floats or complex), not accounting for complex size
    size_t totalsize=1,d;
    for (d=0;d<dim;d++) {
        totalsize *= sizevec[d];
        Dbg_printf4("Dimension %d, size %lld, total %lld\n",d,sizevec[d],totalsize);
        if (sizevec[d]==0)
        { 
            // mexErrMsgTxt("cuda: detected a zero in the size of an array. (e.g. in allocation)\n"); 
          return 0;}
    }
    Dbg_printf2("Totalsize = %d\n",totalsize);
    return totalsize;
}

size_t getTotalSizeFromRefNum(size_t pos) {
    return getTotalSize(cuda_array_dim[pos],cuda_array_size[pos]);
}

size_t getTotalSizeFromRef(const mxArray * arg) {
    return getTotalSizeFromRefNum(getCudaRefNum(arg));
}

size_t getTotalFloatSizeFromRef(const mxArray * arg) {   // sizes in floating point numbers
    size_t ref=getCudaRefNum(arg);
    if (isComplexType(ref))  // this is a complex datatyp
        return getTotalSizeFromRefNum(ref)*2;
    else
        return getTotalSizeFromRefNum(ref);
}

void CheckMemoryConsistency();   // Just declare, but no definition yet. See below

void PrintMemoryOverview() {
    size_t p,d,plantypenum,n;
    size_t sumMem=0;
    size_t sumHeap=0;
    size_t size5d[CUDA_MAXDIM];
    printf("Overview of use Memory:\n");
    printf("----------------------------------------------------------------------------------\n");
    printf("Array# Type Dims  Size X\tY   \tZ   \tTotalSize\tAdress\n");
    for (p=0;p<MAX_ARRAYS;p++)
        if (cuda_arrays[p] !=0)
        {
            get5DSize(p,size5d);
            printf(" %4u\t%1d\t%1d\t%4d\t%4d\t%4d\t%9d\t%9d\n",p,cuda_array_type[p],cuda_array_dim[p],size5d[0],size5d[1],size5d[2],getTotalSizeFromRefNum(p),cuda_arrays[p]);
            sumMem+=getTotalSizeFromRefNum(p);
        }

    printf("\n--------------------------------------------------------------------------------\n");
    printf("Overview of Memory Heap (heapsize %u, allocated %u, first_free %u, tofree %u\n",MAX_HEAP,mem_heap_allocated, mem_heap_first_free,mem_heap_pos);
#ifndef UseHeap
    printf("Memory Heap is not used\n");
#else
    printf("HeapPos TotalSize Adress\n");
    for (p=0;p<(size_t) mem_heap_allocated;p++) // find the next free entry
        if (mem_heap[p] !=0)
        {
            printf("   %4d %9d   %9d\n",p,memsize_heap[p],mem_heap[p]);
            sumHeap+=memsize_heap[p];
        }
#endif
    printf("----------------------------------------------------------------------------------\n");
    printf("FFT-Plans (Maximum number is %d):\n",MAX_FFTPLANS);
    for (d=0;d<3;d++)
        for (plantypenum=0;plantypenum<3;plantypenum++)
             for (n=0;n<MAX_FFTPLANS;n++)
                if (cuda_FFTplans[n][d][plantypenum] != 0)
                {
                 if (d==0)
                    printf("1D plan no. %d, Size: %d \n",n,cuda_FFTplan_sizes[n][0][plantypenum][0]);
                 else if (d==1)
                    printf("2D plan no. %d Size: %d x %d\n",n,cuda_FFTplan_sizes[n][1][plantypenum][0],cuda_FFTplan_sizes[n][1][plantypenum][1]);
                 else 
                    printf("3D plan no. %d Size: %d x %d x %d\n",n,cuda_FFTplan_sizes[n][2][plantypenum][0],cuda_FFTplan_sizes[n][2][plantypenum][1],cuda_FFTplan_sizes[n][2][plantypenum][2]);
                 }
    printf("----------------------------------------------------------------------------------\n");
    printf("Summary: Memory used %.7g GB, Heap %.7g GB, Total used %.7g GB, Allocated %.7g GB, Total: %.7g GB, Free %.7g GB\n",sumMem/1.0E9,sumHeap/1.0E9,(sumMem+sumHeap)/1.0E9,SumAllocated/1.0E9,GetDeviceProp().totalGlobalMem/1.0E9,(GetDeviceProp().totalGlobalMem - SumAllocated)/1.0E9);
    printf("Current Reduce Array size %9d byte\n",GetCurrentRedSize());
}

void CheckMemoryConsistency() {
    size_t sumMem=0;
    size_t sumHeap=0;
    size_t p;
    for (p=0;p<MAX_ARRAYS;p++)
        if (cuda_arrays[p] !=0)
            sumMem+=getTotalSizeFromRefNum(p)*CUDA_TYPE_SIZE[cuda_array_type[p]];
        
    for (p=0;p<(size_t) mem_heap_allocated;p++) // find the next free entry
        if (mem_heap[p] !=0)
            sumHeap+=memsize_heap[p];
    if (sumHeap+sumMem != SumAllocated)
       { printf("Memory Consitency Check failed for total allocation: %d, (Heap+Arrays): %d\n",SumAllocated,sumHeap+sumMem); 
         PrintMemoryOverview();
         SumAllocated=sumHeap+sumMem;
         mexErrMsgTxt("Adjusting Total Size and ... Bailing out");
         return;
        }
}

int ClearHeap()  // release all resources
{
    int custatus;
#ifndef UseHeap
    return;  // do nothing
#endif
Dbg_printf2("Out of Memory. Clearing heap of size %d\n",mem_heap_allocated);    
 
for (mem_heap_allocated--;mem_heap_allocated>=0;mem_heap_allocated--) // find the next free entry
     if (mem_heap[mem_heap_allocated] != 0)
         {
          Dbg_printf3("Freeing array %d of size %d\n",mem_heap_allocated,mem_heap[mem_heap_allocated]);
#ifdef AllocBlas
        custatus=cublasFree(mem_heap[mem_heap_allocated]);
#else
        custatus=cudaFree(mem_heap[mem_heap_allocated]);
#endif
        SumAllocated -= memsize_heap[mem_heap_allocated];
        if (custatus!=cudaSuccess) 
            { printf("ERROR cuda Free: %s\n",cudaGetErrorString(cudaGetLastError())); 
              mexErrMsgTxt("Bailing out");
              return custatus;
            } 
        mem_heap[mem_heap_allocated]=0;
        memsize_heap[mem_heap_allocated]=0;
        }
  mem_heap_first_free=0;
  mem_heap_allocated=0;
  return cudaSuccess;
}


float * MemAlloc(size_t mysize) {   // returns an array from the heap or a fresh array, mysize given in bytes. DOES NOT REGISTER THE ARRAY!
    float * p_cuda_data; float ** pp_cuda_data= & p_cuda_data;
    int custatus;
#ifdef UseHeap
    long long p;  // needs to be able to become negative
    float * tmp=0;
    for (p=(long long) mem_heap_allocated-1;p>=0;p--)   // Look in the heap, whether there is something of the right size
    {
        Dbg_printf4("Scanning heap for memory %d, has size %d, looking for %d\n",p,memsize_heap[p],mysize);    
        if (mysize == memsize_heap[p])  // found the right size (in bytes) on the heap
        {
            memsize_heap[p]=0;
            tmp=(float *) mem_heap[p];
            mem_heap[p]=0;
            if (p==mem_heap_allocated-1)  // Decrease heapsize only, if the last entry was removed
                mem_heap_allocated--;   // heap is less full now
            if (p< (long long) mem_heap_first_free) mem_heap_first_free=p; // lowest position free space needs to be updated 
            Dbg_printf3("Array allocated from heap %d, heap filling is %d\n",p,mem_heap_allocated);    
            return tmp;  // found something
        }
    }
    Dbg_printf2("No matching size in heap. To allocate: %d \n",mysize);    
#endif

#ifdef AllocBlas
    custatus = cublasAlloc((int) (mysize+3)/sizeof(float), sizeof(float), (void**) pp_cuda_data);
    if (custatus != CUBLAS_STATUS_SUCCESS) {
        const char * dummy=cudaGetErrorString(cudaGetLastError());  // just to clear the error
        custatus=cublasGetError();  // also to clear the error
        custatus=ClearHeap();  // Clear heap and try again
        custatus = cublasAlloc((int) (mysize+3)/sizeof(float), sizeof(float), (void**) pp_cuda_data);
        if (custatus != CUBLAS_STATUS_SUCCESS) {
            printf("cuda Malloc: %s\n",cudaGetErrorString(cudaGetLastError())); 
            mexErrMsgTxt("cuda: cublasAlloc Device memory allocation error (on card)\n");
           return 0;
        }
    }
#else
    custatus=cudaMalloc((void **) pp_cuda_data, mysize);
    if (custatus!=cudaSuccess) { 
        const char * dummy=cudaGetErrorString(cudaGetLastError());  // just to clear the error
        custatus=ClearHeap();   // Clear heap and try again
        custatus=cudaMalloc((void **) pp_cuda_data, mysize);
        if (custatus!=cudaSuccess) { 
            printf("cuda Malloc: %s\n",cudaGetErrorString(cudaGetLastError())); 
            mexErrMsgTxt("cuda: cudaMallox Device memory allocation error (on card)\n");
            return 0;
        }
    } 
#endif
    SumAllocated += mysize;

    Dbg_printf2("Allocated %d bytes\n",mysize);    
    return p_cuda_data;
}

void MemFree(size_t ref) {
#ifdef UseHeap
    size_t totalsize=getTotalSizeFromRefNum(ref) * CUDA_TYPE_SIZE[cuda_array_type[ref]];  // size in bytes
    size_t similarCnt=0,p=0;
    if (mem_heap_first_free<MAX_HEAP)  // this is the free space. Deposite the array here
    {
        mem_heap[mem_heap_first_free]=(void *) cuda_arrays[ref];
        memsize_heap[mem_heap_first_free]=totalsize;
        cuda_arrays[ref]=0;
        if (mem_heap_first_free==mem_heap_allocated)  // Increase heapsize only, if the last entry was first_free
            mem_heap_allocated++;  // keeps track of the laset used entry on the heap
        for (;mem_heap_first_free<MAX_HEAP;mem_heap_first_free++) // find the next free entry
        	if (memsize_heap[mem_heap_first_free]==0)
                break;  // found the next empty space
        Dbg_printf4("Array reference %d stored to heap place %d, filling is %d\n",ref,mem_heap_first_free,mem_heap_allocated);
        return;
        //if (totalsize == memsize_heap[p])
        //    similarCnt++;
    }  // No empty space there. Will have to free this array or another one from the heap
    // if (similarCnt > (int) sqrt(MAX_HEAP+1))  // enough of these there already

    // printf("Warning: Cuda Heap is full, %d arrays on stack\n",mem_heap_allocated);

    for (p=0;p<MAX_HEAP;p++)
        if (memsize_heap[p]==totalsize) similarCnt++;  // count the number of arrays of this type

    if (similarCnt > (size_t)((MAX_HEAP+1)/2))  // enough of these there already, free this space
    {
#endif
#ifdef AllocBlas
        int custatus=cublasFree(cuda_arrays[ref]);
#else
        int custatus=cudaFree(cuda_arrays[ref]);
#endif
        SumAllocated -= totalsize;

        if (custatus!=cudaSuccess) { printf("Error cuda Free: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        cuda_arrays[ref]=0;
#ifdef UseHeap

        Dbg_printf2("Array reference %d freed entirely as already in heap\n",ref);
        // ClearHeap();
    }
    else  // we need to free another one on the heap and keep this one as a replacement
    {
        int custatus;
        //mem_heap_first_free=MAX_HEAP-1;  // always take the last on the heap to free, as this is most recent one
#ifdef AllocBlas
        custatus=cublasFree(mem_heap[mem_heap_pos]);
#else
        custatus=cudaFree(mem_heap[mem_heap_pos]);
#endif
        if (custatus!=cudaSuccess) { printf("cuda Free: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        SumAllocated -= memsize_heap[mem_heap_pos];

        mem_heap[mem_heap_pos]=(void *) cuda_arrays[ref];
        memsize_heap[mem_heap_pos]=totalsize;
        cuda_arrays[ref]=0;
        Dbg_printf3("Array reference %d kept but a different array %d freed from heap\n",ref,mem_heap_first_free);
        // mem_heap_first_free++;  // which means no free one available again
        mem_heap_pos = (mem_heap_pos+1) % MAX_HEAP;  // next time free a different one
    }
#endif
}


int MatlabTypeFromCuda(size_t ref) {   /// returns the Matlab class for the specified datatype
    return CUDA_MATLAB_CLASS[cuda_array_type[ref]];
}

int MatlabRealCpxFromCuda(size_t ref) {  // returns a specifier to indicate whether this is real or complex valued data in Matlab
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

size_t updateFreeArray() { // looks for the next free position to store and array
    size_t howmany=0;
    size_t pos=free_array;
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

 // The idea below cannot be used, as otherwise all the functions need to copy the origFT sizes along
/* void ReduceToHalfComplex(int pos) {
        int mydim=cuda_array_dim[pos];
        if (mydim>3) mydim=3;  // because maximally 3d ffts are allowed
        cuda_array_origFTsize[pos] = cuda_array_size[pos][mydim-1]; // last dimension needs to be stored to recover it when doing inverse FTs
        cuda_array_size[pos][mydim-1] = ((int) ((cuda_array_size[pos][mydim-1]) / 2)) +1;  // reduce size accordingly for half complex values
 }
 void ExpandToFullReal(int pos) {
        int mydim=cuda_array_dim[pos];
        if (mydim>3) mydim=3;  // because maximally 3d ffts are allowed
        cuda_array_size[pos][mydim-1] = cuda_array_origFTsize[pos];  // restore size for real space
 } */

 void swapMatlabSize(size_t * mysize, int dims) {
    size_t tmp=mysize[0];
     if (dims>1)
        {mysize[0]=mysize[1];mysize[1]=tmp;}  // the bad matlab problem with x and y dimensions
 }

 void swapMatlabSizeBin(unsigned char * mysize, int dims) {
    unsigned char tmp=mysize[0];
     if (dims>1)
        {mysize[0]=mysize[1];mysize[1]=tmp;}  // the bad matlab problem with x and y dimensions
 }
 
 void cudaCopySizeVec(size_t pos, const size_t * sizevec,int dims) {   // copies a matlab sizevector to the array
 //   double FTscale = 1.0;
    int d;
    cuda_array_size[pos]=calloc(dims,sizeof(cuda_array_size[0][0]));
    for (d=0;d<dims;d++) {
        cuda_array_size[pos][d]=sizevec[d];  // copy sizes
//        FTscale *= sizevec[d];
    }
    //swapMatlabSize(cuda_array_size[pos],dims);
    
//    cuda_array_FTscale[pos]=(float) (1.0/sqrt(FTscale));
    cuda_array_dim[pos]=dims;
    Dbg_printf2("CopySizeVec of %d dimensional vector\n",dims);
 }
 void cudaCopySizeVecD(size_t pos, const double * sizevec,int dims) {    // copies a matlab sizevector to the array (by creating a new one using calloc)
//    double FTscale = 1.0;
    int d;
    cuda_array_size[pos]=calloc(dims,sizeof(cuda_array_size[0][0]));
    for (d=0;d<dims;d++) {
        cuda_array_size[pos][d]=(size_t) sizevec[d];  // copy sizes
//        FTscale *= sizevec[d];
    }
    //swapMatlabSize(cuda_array_size[pos],dims);

//    cuda_array_FTscale[pos]=(float) (1.0/sqrt(FTscale));
    cuda_array_dim[pos]=dims;
    Dbg_printf2("CopySizeVecD of %d dimensional vector\n",dims);
 }
 
 float * cudaAllocDetailed(int dims, const  size_t * sizevec, int cuda_type) {   // make a new array
    float * p_cuda_data; // float ** pp_cuda_data=& p_cuda_data;
    size_t pos=updateFreeArray(),ts;
    cudaCopySizeVec(pos,sizevec,dims);   // copy the size information to the array
    
    cuda_array_type[pos]=cuda_type;
    
    ts=getTotalSize(dims,cuda_array_size[pos]);
    if (ts>0) {
        p_cuda_data=MemAlloc(ts*CUDA_TYPE_SIZE[cuda_type]);
    }
    else p_cuda_data=0;
    
    cuda_arrays[pos]=p_cuda_data; // save it in the array
    cuda_curr_arrays++;
    Dbg_printf3("constructed cuda array nr %d of %d dimensions\n",pos,dims);
    return p_cuda_data;
 }

 float * cudaReturnVal(const mxArray * arg) {   // sets the result value to zero and returns the pointer
    cudaError_t err;    
    err=cudaMemset(pReturnVal[currentCudaDevice],0,2*sizeof(float));
    checkCudaError("cudaReturnVal: ",err);
    return pReturnVal[currentCudaDevice];
 }
 
 
 float * cudaAllocNum(const mxArray * arg) {   // make a new result float number 
     cudaError_t err;
     size_t mysizes[3]={1,0,0};
     float * ret=cudaAllocDetailed(1, mysizes, single);
     err=cudaMemset(cuda_arrays[free_array],0,sizeof(float));checkCudaError("cudaAllocNum single: ",err);

//     cuda_array_FTscale[free_array]=1;  // to do the maginitude correction
     cuda_array_type[free_array]=single;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

 float * cudaAllocCNum(const mxArray * arg) {   // make a new result complex number 
     cudaError_t err;
     size_t mysizes[3]={1,0,0};
     float * ret=cudaAllocDetailed(1, mysizes, scomplex);
     err=cudaMemset(cuda_arrays[free_array],0,sizeof(cufftComplex));checkCudaError("cudaAllocNum complex: ",err);

//     cuda_array_FTscale[free_array]=1;  // to do the maginitude correction
     cuda_array_type[free_array]=1;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

 float * cudaNoAlloc(const mxArray * arg) {   // just use the pointer
     free_array=getCudaRefNum(arg);  // updates this as this is often used to define the return value
     return getCudaRef(arg);
     }

 float * cudaAlloc(const mxArray * arg) {   // make a new array with same properties as other array
     size_t ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], cuda_array_type[ref]);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     //cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
//     cuda_array_FTscale[free_array]=cuda_array_FTscale[ref];  // to do the maginitude correction
     cuda_array_type[free_array]=cuda_array_type[ref];  // type tags see CUDA_TYPE definitions above
     return ret;
     }

float * cudaAllocReal(const mxArray * arg) {   // make a new array with same properties as other array, but ignores Complex and makes it Real
     size_t ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], single);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     //cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
 //    cuda_array_FTscale[free_array]=cuda_array_FTscale[ref];  // to do the maginitude correction
     cuda_array_type[free_array]=single;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

float * cudaAllocComplex(const mxArray * arg) {   // make a new array with same properties as other array, but ignores type and makes it Complex
     size_t ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], scomplex);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     //cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     //cuda_array_FTscale[free_array]=cuda_array_FTscale[ref];  // to do the maginitude correction
     cuda_array_type[free_array]=scomplex;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

 float * cudaAllocSized(const mxArray * arg, SizeND mysize, int mydims) {   // make a new array with same properties as other array
     size_t ref=getCudaRefNum(arg);
     Dbg_printf2("cudaAllocSized ref is %d\n",ref);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(mydims, mysize.s, cuda_array_type[ref]);
     Dbg_printf("Step 2\n");
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     //cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     //cuda_array_FTscale[free_array]=(float) (1.0/sqrt(getTotalSize(CUDA_MAXDIM,mysize.s)));  // to do the maginitude correction
     cuda_array_type[free_array]=cuda_array_type[ref];  // type tags see CUDA_TYPE definitions above
     Dbg_printf2("cudaAllocSized adress is %p\n",(void *) ret);
     return ret;
     }
 float * cudaNoAllocSized(const mxArray * arg, SizeND mysize, int mydims) {   // just use the pointer
     free_array=getCudaRefNum(arg);  // updates this as this is often used to define the return value
     return getCudaRef(arg);
     }
 
 float * cudaNoAllocFree(const mxArray * arg) {   // just use the pointer of the last allocated array (free_array)
     return cuda_arrays[free_array];
     }
 
float * cudaAllocRealSized(const mxArray * arg, SizeND mysize, int mydims) {   // make a new array with same properties as other array, but ignores Complex and makes it Real
     size_t ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(mydims, mysize.s, single);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     //cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     //cuda_array_FTscale[free_array]=(float) (1.0/sqrt(getTotalSize(CUDA_MAXDIM,mysize.s)));  // to do the maginitude correction
     cuda_array_type[free_array]=single;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

float * cudaAllocComplexSized(const mxArray * arg, SizeND mysize, int mydims) {   // make a new array with same properties as other array, but ignores type and makes it Complex
     size_t ref=getCudaRefNum(arg);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // pretend this is a matlab array until allocation is done
     float * ret=cudaAllocDetailed(mydims, mysize.s, scomplex);
     //swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);   // swap back
     //cuda_array_origFTsize[free_array]=cuda_array_origFTsize[ref]; // needs to be copied when creating another HalfFourier array
     //cuda_array_FTscale[free_array]=(float) (1.0/sqrt(getTotalSize(CUDA_MAXDIM,mysize.s)));  // to do the maginitude correction
     cuda_array_type[free_array]=scomplex;  // type tags see CUDA_TYPE definitions above
     return ret;
     }

 float * getMatlabFloatArray(const mxArray * arg, mwSize * nd) {
  (* nd) = mxGetNumberOfDimensions(arg);
  // int * sz = mxGetDimensions(arg);
  if (*nd > 0) {
  /* Pointer for the real part of the input */
    return (float *) mxGetData(arg);
  }
  mexErrMsgTxt("cuda: getMatlabFloatArray; data is zero dimensional\n");
  return 0;
 }

 void ConvertToSize_t(const mwSize * mwSizeVec, size_t * sizevec,mwSize dims) {
     mwSize d;
     for (d=0; d<dims;d++) {
         if (d >= CUDA_MAXDIM)
             mexErrMsgTxt("CudaPut: The array has more dimension than supported by CudaMat\n");
         else
             sizevec[d] = (size_t) mwSizeVec[d];
     }
 }

 void ConvertToMwSize(const size_t * sizevec, mwSize * mwSizeVec, mwSize dims) {
     mwSize d;
     for (d=0; d<dims;d++) {
         if (d >= CUDA_MAXDIM)
             mexErrMsgTxt("CudaPut: The array has more dimension than supported by CudaMat\n");
         else
             mwSizeVec[d] = (mwSize) sizevec[d];
     }
 }
  
 float * cudaPut(const mxArray * arg) {   // copies Matlab array into cuda
    mwSize dims = mxGetNumberOfDimensions(arg);
    const mwSize * mwSizeVec = mxGetDimensions(arg);
    size_t sizevec[CUDA_MAXDIM];
    size_t ts,maxarr;
    int cuda_type = -1;
    float * p_cuda_data=0;
    const char * TypeName=mxGetClassName(arg);    

    ConvertToSize_t(mwSizeVec,sizevec,dims);

    Dbg_printf2("cudaPut  Classname=%s\n",mxGetClassName(arg));
     
    if (! mxIsSingle(arg))
        mexErrMsgTxt("cuda: Datatype for cuda arrays needs to be single precision, or single precision complex\n");
    ts=getTotalSize((const int) dims,sizevec);  // Size of the array to allocate
    maxarr=CUDAmaxSize();
    Dbg_printf5("Total size ts = %lld, dims = %d, Max Size = %lld, MAXINT = %d\n",ts,dims, maxarr, INT_MAX);
    if (ts > maxarr)
        mexErrMsgTxt("cuda: Array too big for available number of threads\n");
    if (mxIsComplex(arg))
        cuda_type = cudaTypeFromArg("scomplex");
    else
        cuda_type=cudaTypeFromArg(TypeName);  //  "single"  Typename

    if (mxIsComplex(arg))
    {
        p_cuda_data = cudaAllocDetailed((int) dims,sizevec,cuda_type);
      /* Pointer for the real part of the input */
        if (ts > 0) {
        float * pr= (float *) mxGetPr(arg);
        float * pi= (float *) mxGetPi(arg);
        int custatus=cudaMemcpy2D(p_cuda_data, sizeof(cufftComplex),pr, sizeof(float),  sizeof(float), ts, cudaMemcpyHostToDevice);
        if (custatus!=cudaSuccess) { printf("cuda Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        //int custatus = cublasSetVector(ts, sizeof(pr[0]), pr, 1, p_cuda_data, 2);
        //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (write C to cuda)\n");return 0;}
        custatus=cudaMemcpy2D(p_cuda_data+1, sizeof(cufftComplex),pi, sizeof(float),  sizeof(float), ts, cudaMemcpyHostToDevice);
        if (custatus!=cudaSuccess) { printf("cuda Memcpy2D: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 
        //custatus = cublasSetVector(getTotalSize(dims,sizevec), sizeof(pi[0]), pi, 1, p_cuda_data+1, 2);
        //if (custatus != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (write C to cuda)\n");return 0;}
        Dbg_printf("cuda: copied Complex data to device\n");
        }
    }
    else
    {
        mwSize nd;
        /* Initialize the device matrices with the host matrices */
        p_cuda_data = cudaAllocDetailed((int) dims,sizevec,cuda_type);
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
    size_t mysize[3]={1,1,1};
    
    p_cuda_data=cudaAllocDetailed(1, mysize, single);

   /* Initialize the device matrices with the host matrices */
    custatus=cudaMemcpy(p_cuda_data, &value, sizeof(value), cudaMemcpyHostToDevice);
    if (custatus!=cudaSuccess) { printf("cuda Memcpy: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 

    return p_cuda_data;
}

float * cudaPutCVal(float Rvalue,float Ivalue) {    // writes a single value into a cuda array
    float * p_cuda_data;
    float value[2]={Rvalue,Ivalue};
    int custatus;
    size_t mysize[3]={1,1,1};
    
    p_cuda_data=cudaAllocDetailed(1, mysize, scomplex);

   /* Initialize the device matrices with the host matrices */
    custatus=cudaMemcpy(p_cuda_data, value, sizeof(value), cudaMemcpyHostToDevice);
    if (custatus!=cudaSuccess) { printf("cuda Memcpy: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 

    return p_cuda_data;
}

void cudaSetSize(const mxArray * arg, const mxArray * sizes) {  // overwrites current sizes with new sizes
    size_t pos=getCudaRefNum(arg);
    // int sd = mxGetNumberOfDimensions(sizes);
    const mwSize * sv = mxGetDimensions(sizes);
    mwSize dims = sv[1];
    const double * sizevec = mxGetPr(sizes);
    if (cuda_array_size[pos] != 0)
        { free(cuda_array_size[pos]); cuda_array_size[pos]=0;}
    
    cudaCopySizeVecD(pos,sizevec,(int) dims);

    return;
}

mxArray * cudaGetSize(const mxArray * arg) {  // returns a vector with sizes
    size_t ref=getCudaRefNum(arg);
    mxArray * ret = mxCreateNumericMatrix(1, cuda_array_dim[ref], mxDOUBLE_CLASS, mxREAL);
    double * ar = mxGetPr(ret); // pointer to real part of array
    int d, dims=cuda_array_dim[ref];
    //swapMatlabSize(cuda_array_size[ref],dims);
    for (d=0;d<dims;d++)
        ar[d] = (double) cuda_array_size[ref][d];  // Caution! This could lead to a loss of size accuracy! It would be better to create a size_t Matlab matrix
    //swapMatlabSize(cuda_array_size[ref],dims);
    return ret;
}

 mxArray * cudaGet(const mxArray * arg) {  // from device to host
    size_t ref=getCudaRefNum(arg);
    //int cuda_type=cuda_array_type[ref];
    //int dims=cuda_array_dim[ref];
    //int saveddim=cuda_array_size[ref][dims-1];
    //if (cuda_type==fftHalfSComplex)
    //    cuda_array_size[ref][dims-1] = ((int) saveddim)/2+1;   // restore to the original value. Just the allocation needs to be bigger

    mxArray * ret = 0;
    double * ar =0;
    double * ai=0;
    int custatus;
    mwSize MyMwSize[CUDA_MAXDIM];

    /* Copy result back to host */
    // cudaMemcpy( ar, getCudaRef(arg), getTypeSize(arg)*getTotalSizeFromRef(arg), cudaMemcpyDeviceToHost);
    
    size_t totalsize=getTotalSizeFromRef(arg);  // size in floating point values
    Dbg_printf2("Memcpy totalsize in bytes: %d\n",sizeof(float)*totalsize);
    
    if (totalsize*sizeof(float) < fastMemSize)
        ar=(double *) fastMem;
    else {
        ConvertToMwSize(cuda_array_size[ref],MyMwSize,cuda_array_dim[ref]);
        ret=mxCreateNumericArray(cuda_array_dim[ref], MyMwSize, MatlabTypeFromCuda(ref), MatlabRealCpxFromCuda(ref));
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
            ConvertToMwSize(cuda_array_size[ref],MyMwSize,cuda_array_dim[ref]);
            ret=mxCreateNumericArray(cuda_array_dim[ref], MyMwSize, MatlabTypeFromCuda(ref), MatlabRealCpxFromCuda(ref));
            memcpy(mxGetPr(ret),fastMem,totalsize*sizeof(float));
            if (MatlabRealCpxFromCuda(ref) == mxCOMPLEX)
                memcpy(mxGetPi(ret),fastMemI,totalsize*sizeof(float));
        }
    }


    Dbg_printf4("cuda: Got type %s sizeX %d sizeY %d\n",CUDA_TYPE_NAMES[cuda_array_type[ref]],cuda_array_size[ref][0],cuda_array_size[ref][1]);
    return ret;
 } 

float cudaGetVal(float * p_cuda_data) {    // gets a single value from a cuda array
    int custatus; float value;
    
   /* Initialize the device matrices with the host matrices */
    custatus=cudaMemcpy(&value, p_cuda_data , sizeof(value), cudaMemcpyDeviceToHost);
    if (custatus!=cudaSuccess) { printf("cuda Memcpy: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 

    return value;
}

cufftComplex cudaGetCVal(float * p_cuda_data) {    // gets a single value from a cuda array
    int custatus; cufftComplex value;
    
   /* Initialize the device matrices with the host matrices */
    custatus=cudaMemcpy(&value, p_cuda_data , sizeof(value), cudaMemcpyDeviceToHost);
    if (custatus!=cudaSuccess) { printf("cuda Memcpy: %s\n",cudaGetErrorString(cudaGetLastError())); mexErrMsgTxt("Bailing out");} 

    return value;
}
 
 void cudaDelete(size_t cudaref) {
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
     cuda_curr_arrays=cuda_curr_arrays-1;  // reduce number of arrays by one
     //free_array=cudaref; // to keep the array indices low
     Dbg_printf2("cuda: deleted object reference %d\n",cudaref);
 }
 
 float * cloneArray(const mxArray * arg) {
     float * myref= getCudaRef(arg);
     float * newp=cudaAlloc(arg);
     Dbg_printf2("copying array, total size = %d \n",getTotalFloatSizeFromRef(arg));
     cublasScopy((int) getTotalFloatSizeFromRef(arg), myref, 1, newp, 1);
     if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (cloneArray)\n");return 0;}
     return newp;
 }

 float * shrink2OneDArray(size_t ref, size_t newsize) {
     float * newp=0;
     size_t asize[2]; asize[0]=newsize; asize[1]=1;
     newp=cudaAllocDetailed(1, asize, cuda_array_type[ref]);
     Dbg_printf2("copy shrinking array, total size = %d \n",asize);
     if (isComplexType(free_array))
         cublasScopy((int) (2* getTotalSizeFromRefNum(free_array)), cuda_arrays[ref], 1, newp, 1);
     else
         cublasScopy((int) getTotalSizeFromRefNum(free_array), cuda_arrays[ref], 1, newp, 1);         
     if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (cloneArray)\n");return 0;}
     return newp;
 }

 float * copyToCpx(const mxArray * arg) {   // creates an array from real and upcasts it to complex
     size_t ref=getCudaRefNum(arg);
     float * myref= getCudaRef(arg),* newp;
     if (cuda_array_type[ref] != single)
         mexErrMsgTxt("cuda: Upcasting to complex. Type needs to be single\n");
     cuda_array_type[ref]=scomplex;  // only temporary for allocation below
     newp=cudaAlloc(arg);
     cuda_array_type[ref]=single;  // reset
     Dbg_printf2("copying array to Cpx, total size = %d \n",getTotalFloatSizeFromRef(arg));
     cublasScopy((int) getTotalFloatSizeFromRef(arg), myref, 1, newp, 2);  // just copy the real parts
     cublasScopy((int) getTotalFloatSizeFromRef(arg), pZero[currentCudaDevice], 0, newp+1, 2);  // just copy the real parts
     if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Device access error (cloneArray)\n");return 0;}
     return newp;
 }

 void ReleaseAllFFTPlans()
 {
     int n,d,plantypenum;
     Dbg_printf("Releasing all FFT-Plans\n");
     for (d=0;d<3;d++)
         for (plantypenum=0;plantypenum<3;plantypenum++)
             for (n=0;n<MAX_FFTPLANS;n++)
                if (cuda_FFTplans[n][d][plantypenum] != 0)
                {
                    cufftDestroy(cuda_FFTplans[n][d][plantypenum]); // for now. Later: keep this plan and have one for forward and one for backward direction
                    cuda_FFTplans[n][d][plantypenum]=0;
                    Dbg_printf4("Released Plan. no: %d dim: %d, type: %d\n",n,d,plantypenum);
                }
 }
 
cufftHandle CreateFFTPlan(size_t ref, int PlanType, int * dirYes, float * FTScale, size_t * extraBatches, size_t * extraBatchStride) {    // returns the correct plan to use, or creates a new plan and in FTScale calculates the correct scaling factor
     cufftResult status=0;
     static int triedOnce=0;
     int numdims=cuda_array_dim[ref];
     int plantypenum=0;
     int n=0,d=0;
     int p=MAX_FFTPLANS-1;
     int matched=1;
     
     size_t istride=1; size_t idist=1;
     int yesdim=0;
     // int ostride=1, int odist; 
     int batch=1;  // how many times to repeat the transform
     int batchused=0;  // was the batch processing feature already exploited?
     int batchstarted=0;  // did a cuda-internal batch initiate?
     int tdims=0;
     int TransformSizes[CUDA_MAXDIM];  // maximal number of dimensions. Unfortunately Cuda FFT does not seem to support larger sizes
     int TransformStorageSizes[CUDA_MAXDIM];  // maximal number of dimensions
     size_t AllSizes=1;

     if (PlanType == CUFFT_R2C) plantypenum=0;
     else if (PlanType == CUFFT_C2R) plantypenum=1;
     else if (PlanType == CUFFT_C2C) plantypenum=2;
     else {mexErrMsgTxt("cuda: Error Unknown Plan type found\n");return 0;}
     
     if (dirYes == NULL && numdims>3) numdims=3;

     if (FTScale != NULL) {
         (* FTScale)=1.0f;
         for (n=0;n<numdims;n++) // CUDA_MAXDIM
             if (dirYes == NULL || dirYes[n] == 1) 
                  (* FTScale) *= cuda_array_size[ref][n];
         (* FTScale) = (float) (1.0/sqrt(FTScale[0]));
     }
     if (extraBatches != NULL)
         (* extraBatches) = 1;
     if (extraBatchStride != NULL)
         (* extraBatchStride) = 0;
     
     Dbg_printf4("Creating/Looking for Plan. ref: %d dim: %d, type: %d\n",ref,numdims,plantypenum);

     for (n=0;n<MAX_FFTPLANS;n++) {
         if (dirYes == NULL) {
         if (numdims==1)
         {if (cuda_FFTplan_sizes[n][0][plantypenum][0] == cuda_array_size[ref][0])  // found a suitable plan
          {Dbg_printf2("Found matching 1D plan no. %d \n",n);
          p=n;return cuda_FFTplans[n][0][plantypenum];}}
         else if (numdims==2)
         {if (cuda_FFTplan_sizes[n][1][plantypenum][0] == cuda_array_size[ref][0] && cuda_FFTplan_sizes[n][1][plantypenum][1] == cuda_array_size[ref][1])  // found a suitable plan
          {Dbg_printf2("Found matching 2D plan no. %d \n",n);
          p=n;return cuda_FFTplans[n][1][plantypenum];}}
         else 
         if (cuda_FFTplan_sizes[n][2][plantypenum][0] == cuda_array_size[ref][0] && cuda_FFTplan_sizes[n][2][plantypenum][1] == cuda_array_size[ref][1] && cuda_FFTplan_sizes[n][2][plantypenum][2] == cuda_array_size[ref][2])  // found a suitable plan
          {Dbg_printf2("Found matching 3D plan no. %d \n",n);
          p=n;return cuda_FFTplans[n][2][plantypenum];}
         }
         else {
             matched=1;
             for (d=0;d<CUDA_MAXDIM;d++) {  // here all sizes and dirYes information has to be checked.
                 if (d < numdims) {
                     matched = matched && (cuda_FFTplan_sizes[n][2][plantypenum][d] == cuda_array_size[ref][d]);
                     matched = matched && (cuda_FFTplan_dirYes[n][2][plantypenum][d] == dirYes[d]); }
                 else {
                     matched = matched && (cuda_FFTplan_sizes[n][2][plantypenum][d] == 1);
                     matched = matched && (cuda_FFTplan_dirYes[n][2][plantypenum][d] == 0);  
                 }
                 Dbg_printf4("Comparing plan no. %d, dimension %d, matched: %d \n",n,d,matched);
             }
             if (matched)  // found a suitable plan
                  {Dbg_printf2("Found matching PartialTransform plan no. %d \n",n);
                  p=n;
                  if (extraBatches != NULL)
                    (* extraBatches) = cuda_FFTplan_extraBatch[p][2][plantypenum];
                  if (extraBatchStride != NULL)
                    (* extraBatchStride) = cuda_FFTplan_extraBatchStride[p][2][plantypenum];
                  return cuda_FFTplans[n][2][plantypenum];}
         }
         
         if (cuda_FFTplans[n][numdims-1][plantypenum] == 0)  //  found a free plan, which has to be created
         {Dbg_printf2("Found a free plan no. %d to use!\n",n);
          p=n;n=MAX_FFTPLANS;break;}
     }
                
     if (cuda_FFTplans[p][numdims-1][plantypenum] != 0) {
         cufftDestroy(cuda_FFTplans[p][numdims-1][plantypenum]); // for now. Later: keep this plan and have one for forward and one for backward direction
         cuda_FFTplans[p][numdims-1][plantypenum]=0;
         printf("WARNING: No more free plans in cuda. Had to destroy an existing fft-plan\n");
        // return 0;
     }
    // printf("FFT Error codes: %d, %d, %d, %d, %d ,%d ,%d, %d, %d, %d\n",CUFFT_SUCCESS,CUFFT_INVALID_PLAN,CUFFT_ALLOC_FAILED,CUFFT_INVALID_TYPE,CUFFT_INVALID_VALUE,CUFFT_INTERNAL_ERROR, CUFFT_EXEC_FAILED, CUFFT_SETUP_FAILED, 0, CUFFT_INVALID_SIZE);
     cuda_FFTplan_extraBatch[p][2][plantypenum] = 1;   // always set these, even if they are not used.
     cuda_FFTplan_extraBatchStride[p][2][plantypenum] = 1;

     if (dirYes == NULL) {  // p contains the plan number to use     
     if (numdims == 1) {
        Dbg_printf3("creating 1D-plan with sizes : %d of type %s\n",cuda_array_size[ref][0],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
        status=cufftPlan1d(&cuda_FFTplans[p][numdims-1][plantypenum], (int) cuda_array_size[ref][0],PlanType,1);   
        cuda_FFTplan_sizes[p][0][plantypenum][0] = cuda_array_size[ref][0];
     }
     else if (numdims  == 2) { //  DO NOT USE, AS THIS CAUSES a MIXUP: || (cuda_array_dim[ref] > 2 && cuda_array_size[ref][2] == 1))
        Dbg_printf4("creating 2D-plan with sizes : %dx%d of type %s\n",cuda_array_size[ref][0],cuda_array_size[ref][1],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
        status=cufftPlan2d(&cuda_FFTplans[p][numdims-1][plantypenum], (int) cuda_array_size[ref][1], (int) cuda_array_size[ref][0], PlanType);
        cuda_FFTplan_sizes[p][1][plantypenum][0] = cuda_array_size[ref][0];
        cuda_FFTplan_sizes[p][1][plantypenum][1] = cuda_array_size[ref][1];
        if (cuda_array_dim[ref] > 2)
            cuda_FFTplan_sizes[p][2][plantypenum][2] = cuda_array_size[ref][2];
     }
     else if (numdims > 2) {
        Dbg_printf5("creating 3D-plan with sizes : %dx%dx%d of type %s\n",cuda_array_size[ref][0],cuda_array_size[ref][1],cuda_array_size[ref][2],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
        status=cufftPlan3d(&cuda_FFTplans[p][numdims-1][plantypenum], (int) cuda_array_size[ref][2], (int) cuda_array_size[ref][1], (int) cuda_array_size[ref][0], PlanType);
        cuda_FFTplan_sizes[p][2][plantypenum][0] = cuda_array_size[ref][0];
        cuda_FFTplan_sizes[p][2][plantypenum][1] = cuda_array_size[ref][1];
        cuda_FFTplan_sizes[p][2][plantypenum][2] = cuda_array_size[ref][2];
     } }
     else  // user supplied information about which directions to transform
    {
        tdims=0;
        for (n=0;n<CUDA_MAXDIM;n++) { // CUDA_MAXDIM
            tdims += (dirYes[n]>0);  // count how many dimensions need to be transformed in total
            if (n >= numdims) { // numdims corresponds to dimension number of this array
                cuda_array_size[ref][n]=1;  // to fix unassigned sizes in the cuda_array
            }
            TransformSizes[n]=1;
            TransformStorageSizes[n]=1;
        }
        yesdim=tdims-1;  // couting backwards
        istride=1;
        idist=1;
        batchused=0;
        batchstarted = 0;
        if (tdims > 3)
            { mexErrMsgTxt("cuda: FFT Error. The maximally allowed number of dimensions to transform is 3.\n"); return 0;}

        AllSizes = 1;
        for (n=0;n<CUDA_MAXDIM;n++) { // CUDA_MAXDIM            // The code below converts the DirYes information into strides to use for the plan creation
            if (dirYes[n]) {
                TransformStorageSizes[yesdim]=(int) cuda_array_size[ref][n];  // unfortunately Cuda FFT does not support 64 bit sizes 
                TransformSizes[yesdim]=(int) cuda_array_size[ref][n];
                yesdim -= 1;
                if (batchused==0 && batchstarted==0) {
                    idist *= cuda_array_size[ref][n];  // increase the distance to the next start of the FFT in case that zeros follow
                }
                if ((numdims-1 > 2 && cuda_FFTplan_extraBatch[p][2][plantypenum] != 1) 
                || (numdims-1 < 2 && cuda_FFTplan_extraBatch[p][numdims-1][plantypenum] != 1)) {
                    Dbg_printf2("Problem with dirYes[%d]\n",n);
                    mexErrMsgTxt("cuda: FFT Error this particular combination of transform directions is not supported in cuda.\n");
                    return 0;
                }
                AllSizes *= cuda_array_size[ref][n];
            }
            else {
                if (yesdim == tdims-1) { // there was no yes before
                    idist = 1;  // This is the innermost dimension to apply the batch processing to
                    istride *= cuda_array_size[ref][n]; // increase the stepsize to the next datapoint belonging to the same FFT
                    batch *= (int) cuda_array_size[ref][n]; 
                    AllSizes *= cuda_array_size[ref][n];
                    batchused=1; }
                else {
                    if (batchused != 0) { // to reach this point, at least as single transform direction had to be present before
                        cuda_FFTplan_extraBatch[p][2][plantypenum] *= cuda_array_size[ref][n];   // these are all trailing zeros, which should be batched by a for loop
                        cuda_FFTplan_extraBatchStride[p][2][plantypenum] = AllSizes; // just needed for the adressing
                        // mexErrMsgTxt("cuda: FFT Error this particular combination of transform directions [0 1 0] (matlab notation) is not supported in cuda.\n");
                    }
                    else {
                        batch *= (int) cuda_array_size[ref][n];   // these are all trailing zeros, which should be batched alltogether
                        TransformStorageSizes[yesdim+1] *= (int) cuda_array_size[ref][n]; // just needed for the adressing. Make the previous Yes-Dim Include this size when jumping
                        // Note that TransformSizes are not changed here
                        AllSizes *= cuda_array_size[ref][n];
                        batchstarted = 1;
                        }
                }
            }
         //if (n < numdims) {
            cuda_FFTplan_sizes[p][2][plantypenum][n] = cuda_array_size[ref][n];  // save this information to compare plans later. Independend of whether these are used like this
            cuda_FFTplan_dirYes[p][2][plantypenum][n] = dirYes[n]; }
         //else {
         //   cuda_FFTplan_dirYes[p][2][plantypenum][n] = 0;
         //   cuda_FFTplan_sizes[p][2][plantypenum][n] = 1; }
         //}
        Dbg_printf5("creating partial transform plan with sizes: %dx%dx%d of type %s\n",cuda_array_size[ref][0],cuda_array_size[ref][1],cuda_array_size[ref][2],CUDA_TYPE_NAMES[cuda_array_type[ref]]);
        Dbg_printf4("Transform direction (dirYes) Vector: %dx%dx%d \n", dirYes[0],dirYes[1],dirYes[2]);
// cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch);
        Dbg_printf7("tdims: %d, TransformSizes: %dx%dx%d,istride %d, idist: %d \n", tdims,TransformSizes[0],TransformSizes[1],TransformSizes[2],istride,idist);
        Dbg_printf7("tdims: %d, TransformStorageSizes: %dx%dx%d,istride %d, idist: %d \n", tdims,TransformStorageSizes[0],TransformStorageSizes[1],TransformStorageSizes[2],istride,idist);
        Dbg_printf2("batch: %d\n", batch);
        Dbg_printf2("Extra batches: %d\n", cuda_FFTplan_extraBatch[p][numdims-1][plantypenum]);
        if (extraBatches != NULL)
            (* extraBatches) = cuda_FFTplan_extraBatch[p][2][plantypenum];
        if (extraBatchStride != NULL)
            (* extraBatchStride) = cuda_FFTplan_extraBatchStride[p][2][plantypenum];
            
        // mexErrMsgTxt("cuda: Stop here.");
        status=cufftPlanMany(&cuda_FFTplans[p][numdims-1][plantypenum], (int) tdims, TransformSizes, TransformStorageSizes, (int) istride, (int) idist, TransformStorageSizes, (int) istride, (int) idist, PlanType, batch);
    }
 
     if (status != CUFFT_SUCCESS) {
        cufftHandle myhandle;
        if (triedOnce) {
             printf("Error : %s\n",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error FFT Plan creation failed\n");
             return 0;
         }
         else {
         triedOnce=1;
         ClearHeap();   // maybe some memory can be freed. Try again
         myhandle=CreateFFTPlan(ref, PlanType, NULL, FTScale, extraBatches, extraBatchStride);
         triedOnce=0;
         return myhandle;
         }
     }

     Dbg_printf4("Return Plan no. %d,dims %d, type %d \n",p,numdims,plantypenum);
     return cuda_FFTplans[p][numdims-1][plantypenum];
}

void activateCudaDevice(int devnum)
{
      int devCount;
      Dbg_printf2("activateCudaDevice: %d\n",devnum);
      cudaGetDeviceCount(&devCount);
      if ((devnum < 0) || (devnum >= devCount) || (devnum >= MaxCudaDevices))
      {printf("# Devices : %d, setDevice to %d\n" ,devCount,devnum);
         mexErrMsgTxt("cuda: activateCudaDevice, DeviceNumber too high or below zero\n");}
      else
       cudaSetDevice(devnum);

      currentCudaDevice=devnum;
      
      if (pOne[currentCudaDevice] == 0)  // Device was never activated before
      {
          pOne[currentCudaDevice]=cudaPutVal(1.0f);NumOne[currentCudaDevice]=free_array;  // keep pointers and reference
          pZero[currentCudaDevice]=cudaPutVal(0.0f);NumZero[currentCudaDevice]=free_array; // keep pointers and reference
          pReturnVal[currentCudaDevice]=cudaPutCVal(0.0f,0.0f);NumReturnVal[currentCudaDevice]=free_array; // keep pointers and reference
          // for (i=0;i<devCount;i++)
          //    cudaDeviceEnablePeerAccess(i,0)
      }
      SetDeviceProperties();   // to ensure that the correct device properties are used
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
  mxGetString(prhs[0], command, (mwSize) buflen);  
  
  Dbg_printf4("Pos 1: ignoreDelete state is : %d, command %s, ignoreRef: %d\n",ignoreDelete, command, ignoreRef);

  if (!cuda_initialized) {
      const char * anerror=0;
      int devCount;
      cudaError_t err;
      Dbg_printf2("int is %d\n",sizeof(int));
      Dbg_printf2("Long is %d\n",sizeof(long));
      Dbg_printf2("mwSize is %d\n",sizeof(mwSize));
      Dbg_printf2("size_T is %d\n",sizeof(size_t));

      err=cudaGetDeviceCount(&devCount);
      checkCudaError("GetDeviceCount",err);
      
      printf("initializing cuda ... ");
      if (devCount > 1)
      {
          printf("Found more than one cuda Device. Activating Device %d, Use 'setDevice' to change\n",devCount-1);
          activateCudaDevice(devCount-1);
          // cudaDeviceEnablePeerAccess(0,0)
      }
      else
          activateCudaDevice(0);

      custatus = cublasInit();
      if (custatus != CUBLAS_STATUS_SUCCESS) {
          mexErrMsgTxt("cuda: CUBLAS initialization error\n");
          return;
      }
      anerror=SetDeviceProperties();
      if (ReduceThreadsDef() != GetMaxThreads()) {
          printf("\nWARNING: cuda CUIMAGE_REDUCE_THREADS is set to %d but the card supports %d threads\n",ReduceThreadsDef(),GetMaxThreads());
          printf("Fix this by copying to the startup.m file:\nglobal NVCCFLAGS;NVCCFLAGS='-D CUIMAGE_REDUCE_THREADS=%d';\n",GetMaxThreads());
          if (ReduceThreadsDef() > GetMaxThreads())
              mexErrMsgTxt("Number of defined CUIMAGE_REDUCE_THREADS is too large");
      }
      if (anerror!=0) { printf("cuda SetDeviceProperties failed: %s\n",anerror); mexErrMsgTxt("SetDeviceProperties Bailing out");}
      printf("DeviceProperties: Maximal threads per block: %d, maximal blocks: %d\n",GetMaxThreads(),(size_t) GetMaxBlocksX());
      
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
Dbg_printf4("Pos 2: ignoreDelete state is : %d, command %s, ignoreRef: %d\n",ignoreDelete, command, ignoreRef);


// mexErrMsgTxt("Blubb ");

#ifdef DEBUG
CheckMemoryConsistency();
#endif

Dbg_printf4("ignoreDelete state is : %d, command %s, ignoreRef: %d\n",ignoreDelete, command, ignoreRef);

if ((ignoreDelete!=0) && strcmp(command,"set_ignoreDelete")!=0 && strcmp(command,"clear_ignoreDelete")!=0 && strcmp(command,"setSize")!=0 && strcmp(command,"forceDelete")!=0 && ((strcmp(command,"delete")!=0) || ((nrhs>1) && ignoreRef != getCudaRefNum(prhs[1]))))
{
    if (showwarning) {
    printf("WARNING! Unwanted call command: %s: nrhs: %d, argument is %d\n",command,nrhs, getCudaRefNum(prhs[1]));
    printf("After the subassign function the first command to expect is a delete command for the same reference. However the above command was initiated.\n");
    printf("The subsasg function was previously called for an object which has another reference. Cuda did directly assign to this object (for speed reasons) without making an extra copy.\n Please make sure that no other object exists when subassigning to an array.\n");
    printf("To avoid this warning, you may want to replace a(10:20,:)=a(60:70); with tmp=a(10:20,:);a(60:70,:)=tmp;\n... continuing execution hoping no further references exist.\n");
    printf("Also changing a range of an argument inside a function call causes this warning: function hello(a) ... a(10:20,:)=33;.\n");
    printf("... continuing execution hoping no further references exist. This warning is now disabled\n");
    showwarning=0;
    if (strcmp(command,"delete")==0 && (nrhs>1) && ignoreRef != getCudaRefNum(prhs[1]))
        printf("\n........................\n DISAGREEMENT of delete references! Stored: %d, toDelete: %d\n",ignoreRef, getCudaRefNum(prhs[1]));
    mexErrMsgTxt("cuda error: The subsasg function was previously called for an object which has another reference. Cuda did directly assign to this object (for speed reasons) without making an extra copy. Please make sure that no other object exists when subassigning to an array. Next time this error is disabled. So call this funtion again if you feel brave.");        
    } 
    //ignoreDelete=0; ignoreRef=-1;  // still continue and hope it is fine.
}

  if (strcmp(command,"put")==0) {  // -------------------------------------------
    float * p_cuda_data1;
    if (nrhs != 2) mexErrMsgTxt("cuda: put needs two arguments\n");

   p_cuda_data1=cudaPut(prhs[1]);  // allocate memory and store to graphic card
   if (nlhs > 0)
       plhs[0] =  mxCreateDoubleScalar((double)free_array); // returns the current array, as the free position is not yet updated
   p_cuda_data1=0; // jsut to eliminate a warning
  }
  else  if (strcmp(command,"getSize")==0) {     
    if (nrhs != 2) mexErrMsgTxt("cuda: getSize needs two arguments\n");
    else if (nlhs > 0)
        plhs[0]=cudaGetSize(prhs[1]);
  }
  else  if (strcmp(command,"delete")==0) {      
    size_t ref=0;
    if (nrhs != 2) mexErrMsgTxt("cuda: delete needs two arguments\n");
    // else if (nlhs > 0)
    ref=getCudaRefNum(prhs[1]);
    if (! (ignoreDelete && ignoreRef==ref))
    {
        cudaDelete(ref);
        Dbg_printf2("delete: deleted ref %d.\n",ref);
    }
    else
        {ignoreDelete=0;ignoreRef=-1;
         Dbg_printf2("delete: This delete (of ref %d) is ignored as a previous call to subsasg asked to ignore it.\n",ref);
        }
  }
  else  if (strcmp(command,"forceDelete")==0) {      // does not care about ignoreDelete
    size_t ref=0;
    if (nrhs != 2) mexErrMsgTxt("cuda: forceDelete needs two arguments\n");
    // else if (nlhs > 0)
    ref=getCudaRefNum(prhs[1]);
    cudaDelete(ref);
    Dbg_printf2("forceDelete: forceDelete ref %d.\n",ref);
  }
  else  if (strcmp(command,"cuda_memory")==0) {      
      PrintMemoryOverview();
  }
  else  if (strcmp(command,"cuda_clearheap")==0) {      
    ClearHeap();
  }
  else  if (strcmp(command,"shutdown")==0) {      
    int i;
    if (nrhs != 1) mexErrMsgTxt("cuda: shutdown needs one arguments\n");
    cuda_initialized=0;
    printf("shutting down cuda\n");
    ReleaseAllFFTPlans();
    for (i=0;i<MaxCudaDevices;i++) 
    if (pOne[i]) {
        cudaDelete(NumOne[i]);
        cudaDelete(NumReturnVal[i]);  // release the fixed numbers
        cudaDelete(NumZero[i]);  // release the fixed numbers
    }
    ClearHeap();
    cublasShutdown();
#ifndef NOCULA
    culaShutdown();
#endif
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
  else  if (strcmp(command,"getVal")==0) {      // accepts only linear indexing
    size_t myindex=0,maxsize;
    if (nrhs != 3) mexErrMsgTxt("cuda: get needs two arguments\n");
    myindex = (size_t) mxGetScalar(prhs[2]);
    maxsize=getTotalSizeFromRef(prhs[1]); // last index of array
    if (myindex < 0)
        if (isComplexType(getCudaRefNum(prhs[1])))
            myindex=(maxsize-1)/2; // last index of array
        else
            myindex=maxsize+myindex; // last index of array backwards
    if (myindex >= maxsize || myindex < 0)
        mexErrMsgTxt("cuda: getVal index out of range\n");
    if (nlhs > 0)
        if (isComplexType(getCudaRefNum(prhs[1]))) {
        double *zr,*zi;
          cufftComplex val=cudaGetCVal(getCudaRef(prhs[1])+myindex*2);
          plhs[0] = mxCreateDoubleMatrix(1, 1, mxCOMPLEX);
          zr = mxGetPr(plhs[0]);
          zi = mxGetPi(plhs[0]);
          zr[0]=(double) val.x;
          zi[0]=(double) val.y; }
        else
            plhs[0]=mxCreateDoubleScalar(cudaGetVal(getCudaRef(prhs[1])+myindex));
  }
  else if (strcmp(command,"swapSize")==0) {
    size_t ref;
    if (nrhs != 2) mexErrMsgTxt("cuda: swapSize needs two arguments\n");
    ref=getCudaRefNum(prhs[1]);
    swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref);
  }
  else if (strcmp(command,"swapSizeForceDim2")==0) {
    size_t ref;
    if (nrhs != 2) mexErrMsgTxt("cuda: swapSizeForceDim2 needs two arguments\n");
    ref=getCudaRefNum(prhs[1]);
    if (cuda_array_dim[ref] <2)
        {cuda_array_dim[ref]=2;cuda_array_size[ref][1]=1;}
    swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref);
  }
  else if (strcmp(command,"swapSizeForceDim1")==0) {
    size_t ref;
    if (nrhs != 2) mexErrMsgTxt("cuda: swapSizeForceDim1 needs two arguments\n");
    ref=getCudaRefNum(prhs[1]);
    if (cuda_array_dim[ref] != 2 || cuda_array_size[ref][0]!=1)
        mexErrMsgTxt("cuda: swapSizeForceDim1 array needs to be 2D of size 1 along first dimension\n");
    swapMatlabSize(cuda_array_size[ref],cuda_array_dim[ref]);
    cuda_array_dim[ref]=1;
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref);
  }
  else if (strcmp(command,"enableWarning")==0) { 
    showwarning=1;
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
    CallCUDA_BinaryRealFkt(complex,cudaAllocComplexSized) // creates a complex array from real and imag arrays
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double) free_array);  
  }
  else  if (strcmp(command,"subsref_cuda")==0) {      // makes a copy of this area
    size_t ref1,mask;
    float * newarr;
    size_t firstres=0;const char * ret=0;
    size_t totalSize=0;
    size_t M=0;   // this is the cuda result size datatype
    if (nrhs != 3) mexErrMsgTxt("cuda: subsref_cuda needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    mask=getCudaRefNum(prhs[2]);
    if (isComplexType(mask))
        mexErrMsgTxt("subsref_cuda: tried to reference with a complex image\n");
    
    totalSize=getTotalSizeFromRef(prhs[1]);
    newarr=cudaAlloc(prhs[1]);  // same type, and unfortunately also size, as input image
    if (isComplexType(ref1)) {
           Dbg_printf("subsref_cuda complex\n");
           ret=CUDAcarr_subsref_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),newarr,totalSize, & M);
    } else {
           Dbg_printf("subsref_cuda real\n");
           ret=CUDAarr_subsref_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),newarr,totalSize,& M);
        } 
    Dbg_printf2("cuda: subsref_cuda: Written one-D array of size %d\n",M);
    if (M == 0)  // This is an empty array!
    {
        cudaDelete(free_array);
        if (nlhs > 0)
            plhs[0] =  mxCreateDoubleScalar((double)-1.0);   // Indicate this to the outside world
        return;
    }
    if (ret) { printf("cuda: subsref_cuda"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: subsref_cuda\n");

    firstres=free_array;
    shrink2OneDArray(firstres, M);
    cudaDelete(firstres);
    // cuda_array_dim[free_array]=1;
    // cuda_array_size[free_array][0]=M;    // shorten this array (if necessary). This is a temporary waste of memory, but probably worth it.
    // printf("Reduced array to %d",M);
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else  if (strcmp(command,"subsasgn_cuda_vec")==0) {      // assings a vector to a masked aread in an image
    size_t ref1,mask,ref3;
    const char * ret=0;
    size_t M=0;
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
        Dbg_printf("cuda: subsasgn_cuda_vec next delete will be ignored\n");
        ignoreDelete=1;ignoreRef=ref1;   // the next delete command will be ignored
    }    else
        Dbg_printf("cuda: subsasgncuda_vec no delete will be ignored\n");

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref1);
  }
  else if (strcmp(command,"repmat")==0) { // repmat : replicates a matrix
    size_t ref1,dimsRepsize;
    int d,ndim;
    double *p_Rep;
    float * newarr;
    size_t dSize[CUDA_MAXDIM]={1,1,1,1,1},sSize[CUDA_MAXDIM];
    const char * ret=0;
    if (nrhs != 3) mexErrMsgTxt("cuda: repmat needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    dimsRepsize=(size_t) (mxGetM(prhs[2]) * mxGetN(prhs[2]));
    if (dimsRepsize>5)
        mexErrMsgTxt("cuda: repmat is only supported up to 5D. Size vector too long.\n");
    p_Rep=mxGetPr(prhs[2]);
    get5DSize(ref1,sSize);
    for (d=0;d<5;d++) {
        if (d < dimsRepsize){
            dSize[d]=(size_t) (sSize[d]*p_Rep[d]); 
        }
        else
            dSize[d]=(size_t) sSize[d];
    }

    ndim=cuda_array_dim[ref1];  // Can change dimensionality of array
    if (dimsRepsize > ndim)
       ndim = (int) dimsRepsize;

    if (isComplexType(ref1))
        newarr=cudaAllocDetailed(ndim,dSize,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,dSize,single);

    if (isComplexType(ref1)) {
           Dbg_printf("repmat complex\n");
           ret=CUDAcarr_5drepcpy_carr(getCudaRef(prhs[1]),newarr,sSize,dSize);
    } else {
           Dbg_printf("repmat real\n");
           ret=CUDAarr_5drepcpy_arr(getCudaRef(prhs[1]),newarr,sSize,dSize);
        } 
    if (ret) { printf("cuda: repmat"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: repmat\n");
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);

  }
  else if (strcmp(command,"subsref_block")==0) { // sub referencing a block of data
    size_t ref1,dimsSoff,dimsSsize,dimsSstep,EndPos;
    int ndim,d;
    double * p_Soffset, * p_SROI, * p_Sstep;
    const char * ret=0;
    Size5D nsizes,sROI,dOffs,sSize,sOffs,sStep;
    // int nsizes[CUDA_MAXDIM]={1,1,1,1,1},sROI[CUDA_MAXDIM]={1,1,1,1,1},dOffs[CUDA_MAXDIM]={0,0,0,0,0},sSize[CUDA_MAXDIM]={1,1,1,1,1},sOffs[CUDA_MAXDIM]={0,0,0,0,0};
    float * newarr;
    if (nrhs != 5) mexErrMsgTxt("cuda: csubsref needs four arguments\n");  // command, array1 , 3D offset, 3D size
    ref1=getCudaRefNum(prhs[1]);
    dimsSoff=(size_t) (mxGetM(prhs[2]) * mxGetN(prhs[2]));
    dimsSsize=(size_t)(mxGetM(prhs[3]) * mxGetN(prhs[3]));
    dimsSstep=(size_t)(mxGetM(prhs[4]) * mxGetN(prhs[4]));
    if (dimsSoff>5)
        mexErrMsgTxt("cuda: subreferencing is only supported up to 5D. Offset vector too long.\n");
    if (dimsSsize>5)
        mexErrMsgTxt("cuda: subreferencing is only supported up to 5D. Size vector too long.\n");
    if (dimsSstep>5)
        mexErrMsgTxt("cuda: subreferencing is only supported up to 5D. Step vector too long.\n");
    p_Soffset=mxGetPr(prhs[2]);
    p_SROI=mxGetPr(prhs[3]);
    p_Sstep=mxGetPr(prhs[4]);
    sSize=getSize5D(ref1);  // from the reference number in 
    for (d=0;d<5;d++) {
        dOffs.s[d]=0;
        if (d<dimsSoff)
            sOffs.s[d]=(size_t) p_Soffset[d];
        else
            sOffs.s[d]=0;
        if (d<dimsSsize)
            sROI.s[d]=(size_t) p_SROI[d];
        else
            sROI.s[d]=1;
        nsizes.s[d]=sROI.s[d];
        if (d<dimsSstep)
            sStep.s[d]=(size_t) p_Sstep[d];
        else
            sStep.s[d]=1;
        EndPos=sOffs.s[d]+(nsizes.s[d]-1)*sStep.s[d];
        if (sOffs.s[d] < 0.0 || sOffs.s[d] >= sSize.s[d] || EndPos < 0 || EndPos >= sSize.s[d]) {
            printf("d: %d sOffs %d, EndPos %d, sSize %d\n",d,sOffs.s[d], EndPos, sSize.s[d]);
            mexErrMsgTxt("cuda: subreferencing index out of range.\n"); }
        if (sROI.s[d] < 0.0)
            mexErrMsgTxt("cuda: subreferencing Negative sizes not allowed.\n");            // Negative sizes mean negative indexing
    }
    Dbg_printf("subsref_block\n");
    ndim=cuda_array_dim[ref1];  // Will not change dimensionality of array

    Dbg_printf6("s1Size: %d x %d x %d x %d x %d\n",sSize.s[0],sSize.s[1],sSize.s[2],sSize.s[3],sSize.s[4]);
    Dbg_printf6("nSize: %d x %d x %d x %d x %d\n",nsizes.s[0],nsizes.s[1],nsizes.s[2],nsizes.s[3],nsizes.s[4]);
    Dbg_printf6("dOffs: %d x %d x %d x %d x %d\n",dOffs.s[0],dOffs.s[1],dOffs.s[2],dOffs.s[3],dOffs.s[4]);
    if (isComplexType(ref1))
        newarr=cudaAllocDetailed(ndim,nsizes.s,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,nsizes.s,single);

     if (isComplexType(ref1)) {
           Dbg_printf("subsref_block complex to complex\n");
           ret=CUDAcarr_5dsubcpy_carr(getCudaRef(prhs[1]),newarr,sSize,nsizes,sOffs,sROI,dOffs,sStep,Size5DOnes); 
     } else {
           Dbg_printf("subsref_block real to real\n");
           ret=CUDAarr_5dsubcpy_arr(getCudaRef(prhs[1]),newarr,sSize,nsizes,sOffs,sROI,dOffs,sStep,Size5DOnes);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: subsref_block "); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: subsref_block\n");
  }
  else if (strcmp(command,"subsasgn_cuda_const")==0) { // conditionally assigns a constant value to an existing array
    const char *ret=0; size_t ref; double myreal,myimag=0;
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
        Dbg_printf("cuda: subsasgn_cuda_const next delete will be ignored\n");
        ignoreDelete=1;ignoreRef=ref;   // the next delete command will be ignored
    }    else
        Dbg_printf("cuda: subsasgncuda_const no delete will be ignored\n");
            
    if (ret) { printf("cuda: subsasgn_cuda_const "); mexErrMsgTxt(ret);}
    Dbg_printf("cuda: subsasgn_cuda_const\n");
    } 
  else if (strcmp(command,"subsasgn_block")==0) { // sub referencing a block of data
    size_t ref2,constmode,dimsDoff,dimsSsize,dimsDstep;
    Size5D sSize,dSize,dOffs,dStep; // noOffs,
    // int dOffs[CUDA_MAXDIM]={0,0,0,0,0},sSize[CUDA_MAXDIM]={1,1,1,1,1},dSize[CUDA_MAXDIM]={1,1,1,1,1},noOffs[CUDA_MAXDIM]={0,0,0,0,0};
    int d;
    double * p_Doffset,* p_sSize, * p_dStep;
    const char * ret=0;
    if (nrhs != 7) mexErrMsgTxt("cuda: subsasgn_block needs seven arguments\n");  // command, array1 , 5D offset, 5D size, 5D step
    ref2=getCudaRefNum(prhs[2]); // destination
    constmode=(mxGetScalar(prhs[5]) > 0) ? 1 : 0;  // should prhs[2] be interpreted as a constant to assign? 0 or negative means array
    dimsDoff=(size_t)(mxGetM(prhs[3]) * mxGetN(prhs[3]));
    if (dimsDoff>5)
        mexErrMsgTxt("cuda: subassigning is only supported up to 5D. Offset vector too long.\n");
    p_Doffset=mxGetPr(prhs[3]);

    dimsSsize=(size_t)(mxGetM(prhs[4]) * mxGetN(prhs[4]));
    if (dimsSsize>5)
        mexErrMsgTxt("cuda: subassigning is only supported up to 5D. Size vector too long.\n");
    p_sSize=mxGetPr(prhs[4]);

    dimsDstep=(size_t)(mxGetM(prhs[6]) * mxGetN(prhs[6]));
    if (dimsSsize>5)
        mexErrMsgTxt("cuda: subassigning is only supported up to 5D. Size vector too long.\n");
    p_dStep=mxGetPr(prhs[6]);

    for (d=0;d<CUDA_MAXDIM;d++) {
        // noOffs.s[d]=0;
        if (d<dimsDoff)
            dOffs.s[d]=(size_t) p_Doffset[d];
        else
            dOffs.s[d]=0;
        if (d<dimsSsize)
            sSize.s[d]=(size_t) p_sSize[d];
        else
            sSize.s[d]=1;
        if (d<dimsDstep)
            dStep.s[d]=(size_t) p_dStep[d];
        else
            dStep.s[d]=1;
        if (d<cuda_array_dim[ref2])
            dSize.s[d]=(size_t) cuda_array_size[ref2][d];
        else
            dSize.s[d]=1;
        }
    Dbg_printf("subsasgn_block\n");
    Dbg_printf6("sSize: %d x %d x %d x %d x %d\n",sSize.s[0],sSize.s[1],sSize.s[2],sSize.s[3],sSize.s[4]);
    Dbg_printf6("dSize: %d x %d x %d x %d x %d\n",dSize.s[0],dSize.s[1],dSize.s[2],dSize.s[3],dSize.s[4]);
    Dbg_printf6("dOffs: %d x %d x %d x %d x %d\n",dOffs.s[0],dOffs.s[1],dOffs.s[2],dOffs.s[3],dOffs.s[4]);
    Dbg_printf6("dStep: %d x %d x %d x %d x %d\n",dStep.s[0],dStep.s[1],dStep.s[2],dStep.s[3],dStep.s[4]);

    if (!constmode) {
    size_t ref1=getCudaRefNum(prhs[1]); // source
    if (isComplexType(ref1)) {
        if (! isComplexType(ref2)) // destination needs to be complex too
            mexErrMsgTxt("cuda: trying to assign complex values to real array.\n");
        Dbg_printf("subsasgn_block complex to complex\n");
        ret=CUDAcarr_5dsubcpy_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),sSize,dSize,Size5DZeros,sSize,dOffs,Size5DOnes,dStep);
     } else {
         if (isComplexType(ref2)) // destination needs to be complex too
            {
             Dbg_printf("subsasgn_block real to complex\n");
             ret=CUDAarr_5dsubcpy_carr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),sSize,dSize,Size5DZeros,sSize,dOffs,Size5DOnes,dStep);
            }
         else
            {
             Dbg_printf("subsasgn_block real to real\n");
             ret=CUDAarr_5dsubcpy_arr(getCudaRef(prhs[1]),getCudaRef(prhs[2]),sSize,dSize,Size5DZeros,sSize,dOffs,Size5DOnes,dStep);
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
       ret=CUDAcconst_5dsubcpy_carr(getCudaRef(prhs[2]),(float) myreal,(float) myimag,dSize,sSize,dOffs,dStep);
     } else {
        double  myreal;
        if (mxIsComplex(prhs[1])) // const is complex but data is real
                mexErrMsgTxt("cuda: trying to assign complex constant to real array.\n");
        myreal = mxGetScalar(prhs[1]);
        Dbg_printf("subsasgn_block real const to complex\n");
        ret=CUDAconst_5dsubcpy_arr(getCudaRef(prhs[2]),(float) myreal,0.0,dSize,sSize,dOffs,dStep);
     }}

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)ref2);
    if(mxGetScalar(prhs[5]) >= 0) {
        ignoreDelete=1;
        ignoreRef=ref2;   // the next delete command will be ignored
        Dbg_printf("cuda: subsasgn_block next delete will be ignored (why not?)\n");
    }    else {
        Dbg_printf("cuda: subsasgn_block no delete will be ignored\n");
    }
            
    if (ret) { printf("cuda: subsasgn_block "); mexErrMsgTxt(ret);}
    Dbg_printf("cuda: subsasgn_block\n");
  }
  else if (strcmp(command,"newarr")==0) { // creates a new array with given sizes and assigns a constant to it. Does not need an input array!
    if (nrhs != 3) mexErrMsgTxt("cuda: newarr needs three arguments\n");                    \
    else {CUDA_NewArrayFromSize(((int) mxIsComplex(prhs[2])))

    if (mxIsComplex(prhs[2])) {
        float br=(float) mxGetPr(prhs[2])[0];
        float bi=(float) mxGetPi(prhs[2])[0];
        CUDAset_carr(br, bi, newarr, tsize);
    } else {
        float b=(float) mxGetScalar(prhs[2]);
        CUDAset_arr(b, newarr, tsize);}
            
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    Dbg_printf("newarr\n");
    }
  }
  else if (strcmp(command,"xyz")==0) { // creates a new array with given sizes and assigns a constant to it. Does not need an input array!
    CallCUDA_GenArrFkt(xyz)    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"rr")==0) { // creates a new array with given sizes and assigns a constant to it. Does not need an input array!
    CallCUDA_GenArrFkt(rr)    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"phiphi")==0) { // creates a new array with given sizes and assigns a constant to it. Does not need an input array!
    CallCUDA_GenArrFkt(phiphi)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"copy")==0) { // copies (dublicates) an array
    if (nrhs != 2) mexErrMsgTxt("cuda: copy needs two arguments\n");  
    cloneArray(prhs[1]);  // float * newarr=
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    Dbg_printf("copy\n");
  }
  else if (strcmp(command,"transpose")==0) { // transpose an array
    size_t ref,conjmode,tmp;
    int ndim,d;
    Size5D nsizes,sSize;
    const char * ret=0;
    float * newarr;

    if (nrhs != 3) mexErrMsgTxt("cuda: transpose needs three arguments\n");  // command, array1 , 3D offset, 3D size
    Dbg_printf("transpose\n");
    ref=getCudaRefNum(prhs[1]);
    conjmode=(mxGetScalar(prhs[2]) > 0) ? 1 : 0;  // conjugate or not, that is the question
    sSize=getSize5D(ref);
    nsizes=getSize5D(ref);
    ndim=cuda_array_dim[ref];
    if (ndim<2) ndim=2;
    tmp=nsizes.s[1];nsizes.s[1]=nsizes.s[0];nsizes.s[0]=tmp;  // swaps sizes

    // for (d=0;d<CUDA_MAXDIM;d++) {noOffs.s[d]=0;}  // noStep.s[d]=1; 
    
    Dbg_printf6("sSize: %d x %d x %d x %d x %d\n",sSize.s[0],sSize.s[1],sSize.s[2],sSize.s[3],sSize.s[4]);
    if (isComplexType(ref))
        newarr=cudaAllocDetailed(ndim,nsizes.s,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,nsizes.s,single);

     if (isComplexType(ref)) {
           Dbg_printf("transpose complex to complex\n");
           if (conjmode)
               ret=CUDAcarr_5dsubcpyCT_carr(getCudaRef(prhs[1]),newarr,sSize,nsizes,Size5DZeros,sSize,Size5DZeros,Size5DOnes,Size5DOnes);
           else
               ret=CUDAcarr_5dsubcpyT_carr(getCudaRef(prhs[1]),newarr,sSize,nsizes,Size5DZeros,sSize,Size5DZeros,Size5DOnes,Size5DOnes);
     } else {
           Dbg_printf("transpose real to real\n");
           ret=CUDAarr_5dsubcpyT_arr(getCudaRef(prhs[1]),newarr,sSize,nsizes,Size5DZeros,sSize,Size5DZeros,Size5DOnes,Size5DOnes);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: transpose"); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: transpose\n");
  }
  else if (strcmp(command,"diag")==0) { // generates or extracts a diagonal
    size_t nsizes[3],noOffs[3]={0,0,0},sSize[3];
    long long dOffs[3]={0,0,0};  // a singed 64 bit variable
    int diagget=0;
    const char * ret=0;
    size_t ref=getCudaRefNum(prhs[1]);
    long long offset;

    if (nrhs != 3) mexErrMsgTxt("cuda: diag needs two arguments\n");  // array, offset
    offset = (long long) mxGetScalar(prhs[2]); // shift off the diagonal
    get3DSize(ref,sSize);
    get3DSize(ref,nsizes);
    if (sSize[0] == 1)
        {sSize[0] = sSize[1];sSize[1]=1;}
    if (sSize[0] > 1 && sSize[1] > 1)  // extract the diagonal
    {
        diagget=1;
        if (offset < 0) {
            dOffs[0]=(size_t) -offset;
            nsizes[0] -= dOffs[0];
        }
        if (offset > 0) {
            dOffs[1]=(size_t) offset;
            nsizes[1] -= dOffs[1];
        }
        nsizes[0]= min(nsizes[0],nsizes[1]);
        nsizes[1]= 1; // make it a square matrix
        Dbg_printf("diag set\n");
    }
    else  // generate a diagonal matrix
    {
        nsizes[0]= sSize[0] + (size_t) long_long_abs(offset);
        nsizes[1]= nsizes[0]; // make it a square matrix
    if (offset < 0)
        dOffs[0]=(size_t) -offset;
    if (offset > 0)
        dOffs[1]=(size_t) offset;
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
    Size5D nsizes,dOffs=Size5DZeros,s1Size,s2Size;
    size_t ref1,ref2;
    int ndim,direction,d;
    const char * ret=0;
    float * newarr;
    if (nrhs != 4) mexErrMsgTxt("cuda: cat needs four arguments\n");  // command, array1 , array2, direction
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    direction=(int) mxGetScalar(prhs[3]);
    nsizes=getSize5D(ref1);s1Size=getSize5D(ref1);s2Size=getSize5D(ref2);
    // for (d=0;d<CUDA_MAXDIM;d++) {dOffs.s[d]=0;noOffs.s[d]=0;noStep.s[d]=1;}
    Dbg_printf2("append to direction: %g\n",direction);
    if (direction == 1) {
        nsizes.s[0]=s1Size.s[0]+s2Size.s[0];
        dOffs.s[0]=s1Size.s[0];
    } else if (direction == 2) {
        nsizes.s[1]=s1Size.s[1]+s2Size.s[1];
        dOffs.s[1]=s1Size.s[1];
    } else if (direction == 3) {
        nsizes.s[2]=s1Size.s[2]+s2Size.s[2];
        dOffs.s[2]=s1Size.s[2];
    } else if (direction == 4) {
        nsizes.s[3]=s1Size.s[3]+s2Size.s[3];
        dOffs.s[3]=s1Size.s[3];
    } else if (direction == 5) {
        nsizes.s[4]=s1Size.s[4]+s2Size.s[4];
        dOffs.s[4]=s1Size.s[4];
    } else {
        mexErrMsgTxt("cuda: cat. Direction to append along needs to be 1 (x), 2 (y), 3 (z), 4 (t) or 5 (e) \n");    
    }
    ndim=(cuda_array_dim[ref1] > cuda_array_dim[ref2]) ? cuda_array_dim[ref1] : cuda_array_dim[ref2];
    if (direction > ndim) ndim = direction;
        
    Dbg_printf6("s1Size: %d x %d x %d x %d x %d\n",s1Size.s[0],s1Size.s[1],s1Size.s[2],s1Size.s[3],s1Size.s[4]);
    Dbg_printf6("s2Size: %d x %d x %d x %d x %d\n",s2Size.s[0],s2Size.s[1],s2Size.s[2],s2Size.s[3],s2Size.s[4]);
    Dbg_printf6("nSize: %d x %d x %d x %d x %d\n",nsizes.s[0],nsizes.s[1],nsizes.s[2],nsizes.s[3],nsizes.s[4]);
    Dbg_printf6("dOffs: %d x %d x %d x %d x %d\n",dOffs.s[0],dOffs.s[1],dOffs.s[2],dOffs.s[3],dOffs.s[4]);
    if (isComplexType(ref1) || isComplexType(ref2))
        newarr=cudaAllocDetailed(ndim,nsizes.s,scomplex);
    else
        newarr=cudaAllocDetailed(ndim,nsizes.s,single);

     if (isComplexType(ref1))
        if (isComplexType(ref2)) {
           Dbg_printf("append complex to complex\n");
           ret=CUDAcarr_5dsubcpy_carr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,Size5DZeros,s1Size,Size5DZeros,Size5DOnes,Size5DOnes);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAcarr_5dsubcpy_carr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,Size5DZeros,s2Size,dOffs,Size5DOnes,Size5DOnes);
        } else {
           Dbg_printf("append complex to real\n");
           ret=CUDAcarr_5dsubcpy_carr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,Size5DZeros,s1Size,Size5DZeros,Size5DOnes,Size5DOnes);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAarr_5dsubcpy_carr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,Size5DZeros,s2Size,dOffs,Size5DOnes,Size5DOnes);
        }
    else if (isComplexType(ref2)) {
           Dbg_printf("append real to complex\n");
           ret=CUDAarr_5dsubcpy_carr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,Size5DZeros,s1Size,Size5DZeros,Size5DOnes,Size5DOnes);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAcarr_5dsubcpy_carr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,Size5DZeros,s2Size,dOffs,Size5DOnes,Size5DOnes);
        } else {
           Dbg_printf("append real to real\n");
           ret=CUDAarr_5dsubcpy_arr(getCudaRef(prhs[1]),newarr,s1Size,nsizes,Size5DZeros,s1Size,Size5DZeros,Size5DOnes,Size5DOnes);
           if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
           ret=CUDAarr_5dsubcpy_arr(getCudaRef(prhs[2]),newarr,s2Size,nsizes,Size5DZeros,s2Size,dOffs,Size5DOnes,Size5DOnes);
        } 
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
    if (ret) { printf("cuda: cat "); mexErrMsgTxt(ret);}                          
    Dbg_printf("cuda: cat\n");
  }
  else if (strcmp(command,"subsref_vecs")==0) { 
    size_t nel,ref;
    float * p_cuda_data1=0;  
    size_t newref[3]={-1,-1,-1};
    size_t dSize[3]={1,1,1};
    int d;
    float * newarr;
    const char * ret=0;
    
    mexErrMsgTxt("cuda: subsref_vec is not finnished yet\n");
    if (nrhs != 3) mexErrMsgTxt("cuda: subsref_vec needs three arguments\n");
    nel=(size_t) mxGetNumberOfElements(prhs[2]);
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
  else if (strcmp(command,"subsref_NDidx")==0) {   // reference with a matrix where lowest dimension corresponds to vectors of indices
    CallCUDA_IdxFktND(subsrefND,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"subsasgn_NDidx")==0) {   // assing with a matrix where lowest dimension corresponds to vectors of indices
    CallCUDA_IdxFktND(subsasgnND,cudaNoAllocSized)
    ignoreDelete=1;ignoreRef=_ref1;   // the next delete command will be ignored
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)_ref1);
  }
  else if (strcmp(command,"subsasgn_NDidx_const")==0) {   // assing with a matrix where lowest dimension corresponds to vectors of indices
    CallCUDA_IdxFktNDConst(subsasgnND)   // will not allocate 
    ignoreDelete=1;ignoreRef=_ref1;   // the next delete command will be ignored
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)_ref1);
  }
  else if (strcmp(command,"subsref_1Didx")==0) {   // just reference with a one-dimensional index already as a cuda array.
    if (isComplexType(getCudaRefNum(prhs[1]))) {
        CallCUDA_IdxFkt(subsref,cudaAllocComplex,2,1) }
    else {
        CallCUDA_IdxFkt(subsref,cudaAlloc,2,1)        }
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"subsasgn_1Didx")==0) {   // just reference with a one-dimensional index already as a cuda array.
    CallCUDA_IdxFkt(subsasgn,cudaNoAlloc,3,3)
    ignoreDelete=1;ignoreRef=free_array;   // the next delete command will be ignored
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"subsasgn_1Didx_const")==0) {   // just reference with a one-dimensional index already as a cuda array.
    CallCUDA_IdxFktConst(subsasgn,cudaNoAlloc)   // misuses this function. free_array is also updated.
    ignoreDelete=1;ignoreRef=ref;   // the next delete command will be ignored
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"clear_ignoreDelete")==0) {   // just reference with a one-dimensional index already as a cuda array.
    ignoreDelete=0;ignoreRef=-1;   // the next delete will be executed
  }
  else if (strcmp(command,"set_ignoreDelete")==0) {   // just reference with a one-dimensional index already as a cuda array.
    ignoreDelete=1;ignoreRef=getCudaRefNum(prhs[1]);   // the next delete will be executed
  }
  else if (strcmp(command,"diff")==0) { 
    CallCUDA_UnaryFktSizeConst(diff,cudaAllocSized)   // uses the pointer to the previously allocated array
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"equals_alpha")==0) { 
    CallCUDA_UnaryFktConst(equals,cudaAllocReal)  // always returns a real value array
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"equals")==0) { 
    CallCUDA_BinaryFkt(equals,cudaAllocRealSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"unequals_alpha")==0) { 
    CallCUDA_UnaryFktConst(unequals,cudaAllocReal)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"unequals")==0) { 
    CallCUDA_BinaryFkt(unequals,cudaAllocRealSized)
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
    CallCUDA_BinaryRealFkt(or,cudaAllocSized)
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
    CallCUDA_BinaryRealFkt(and,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"not")==0) { 
      CallCUDA_UnaryRealFkt(not,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"sign")==0) { 
      CallCUDA_UnaryFkt(sign,cudaAlloc)
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
    CallCUDA_BinaryRealFkt(smaller,cudaAllocSized)
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
    CallCUDA_BinaryRealFkt(larger,cudaAllocSized)
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
    CallCUDA_BinaryRealFkt(smallerequal,cudaAllocSized)
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
    CallCUDA_BinaryRealFkt(largerequal,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"isLegal")==0) { // Checks whether there is any zero or illegal value in the array
    CallCUDA_UnaryFkt(isIllegal,cudaReturnVal)

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double) cudaGetVal(pReturnVal[currentCudaDevice]) == 0);
        // plhs[0] =  mxCreateDoubleScalar((double) cudaGetVal(cuda_arrays[free_array]) == 0);
    // cudaDelete(free_array);
  }
  else if (strcmp(command,"any")==0) { // Checks whether there is any is non-zero
    CallCUDA_UnaryFkt(any,cudaReturnVal)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double) cudaGetVal(pReturnVal[currentCudaDevice]));
        // plhs[0] =  mxCreateDoubleScalar((double) cudaGetVal(cuda_arrays[free_array]));
    // cudaDelete(free_array);
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
  else if (strcmp(command,"power")==0) { 
    CallCUDA_BinaryRealFkt(power,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"times_alpha")==0) { // ---------------------------------
    CallCUDA_UnaryFktConst(times,cudaAlloc)  // return type is the propagated type
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"times")==0) { 
    CallCUDA_BinaryFkt(times,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"plus_alpha")==0) { // --------------------------------------------
    CallCUDA_UnaryFktConst(plus,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"plus")==0) { // -----------------array + array
    CallCUDA_BinaryFkt(plus,cudaAllocSized)
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
    CallCUDA_BinaryFkt(divide,cudaAllocSized)
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
    CallCUDA_BinaryFkt(minus,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"exp")==0) { // complex exponential
    CallCUDA_UnaryFkt(exp,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"sin")==0) {
    CallCUDA_UnaryFkt(sin,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"cos")==0) {
    CallCUDA_UnaryFkt(cos,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"tan")==0) {
    CallCUDA_UnaryRealFkt(tan,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"sinc")==0) {
    CallCUDA_UnaryFkt(sinc,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"sinh")==0) {
    CallCUDA_UnaryFkt(sinh,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"cosh")==0) {
    CallCUDA_UnaryFkt(cosh,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"besselj_alpha")==0) {
    CallCUDA_UnaryRealFktConst(besselj,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_besselj")==0) {
    CallCUDA_UnaryRealFktConstR(besselj,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"besselj")==0) { 
    CallCUDA_BinaryRealFkt(besselj,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"atan2_alpha")==0) {
    CallCUDA_UnaryRealFktConst(atan2,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"alpha_atan2")==0) {
    CallCUDA_UnaryRealFktConstR(atan2,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"atan2")==0) { 
    CallCUDA_BinaryRealFkt(atan2,cudaAllocSized)
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
  else if (strcmp(command,"round")==0) { 
      CallCUDA_UnaryFkt(round,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"floor")==0) { 
      CallCUDA_UnaryFkt(floor,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"ceil")==0) { 
      CallCUDA_UnaryFkt(ceil,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"conj")==0) { 
      CallCUDA_UnaryFkt(conj,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"sqrt")==0) { 
      CallCUDA_UnaryFkt(sqrt,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"circshift")==0) { // - vector of CUDA_MAXDIM as input and an array .. outputs the shifted array
    CallCUDA_ArrVecFkt(circshift,cudaAlloc,0)  // set shift to zero for dimensions larger than requested
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"permute")==0) { // from outside it needs to be assured that the indices are unique. However the number of dimensions can expand
    size_t nsizes[CUDA_MAXDIM]; int maxdim=1;
    CallCUDA_ArrVecFkt(permute,cudaAlloc,d)   // executes the permutation to a new array,  set assignment dimension to same dimension for dimensions larger than requested

    for (d=0;d<CUDA_MAXDIM;d++)  // iterates over new dimensions
        nsizes[d]=1;
    for (d=0;d<CUDA_MAXDIM;d++)  // iterates over old dimensions
    {
        if (d < dims_sizes)
            if (nshifts[d] >= 0 && nshifts[d] < CUDA_MAXDIM)   // dimension numbering has to start with zero
            { 
                nsizes[d]=dsize[nshifts[d]];
                if (nshifts[d] > maxdim)
                    maxdim = (int) nshifts[d];
                Dbg_printf6("cuda: permute shifing dimension %d to %d, oldsize %d, newsize is %d maxdim= %d\n",d,nshifts[d],dsize[d],nsizes[d],maxdim);
            }
            else
            {
                mexErrMsgTxt("cuda: permutation contains index beyond acceptable array dimension CUDA_MAXDIM or below zero\n");
            }
    }
            
    if (cuda_array_size[free_array] != 0)
        { free(cuda_array_size[free_array]); cuda_array_size[free_array]=0;}

    cudaCopySizeVec(free_array,nsizes,maxdim+1);
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"fftshift")==0) { // -----------------like in matlab
      size_t ref,size3d=0;
      int mode,mydim;
      float * srcstart,*dststart, * newarr;
    int ret=0;
    if (nrhs != 3) mexErrMsgTxt("cuda: fftshift needs three arguments\n");
    ref=getCudaRefNum(prhs[1]);
    mode=(mxGetScalar(prhs[2]) > 0) ? 1 : -1;

    mydim=cuda_array_dim[ref];
    if (mydim>3) mydim=3;
    if (mydim == 1)
        size3d=cuda_array_size[ref][0];
    if (mydim == 2)
        {size3d=cuda_array_size[ref][0]*cuda_array_size[ref][1]; }
    if (mydim == 3)
        {size3d=cuda_array_size[ref][0]*cuda_array_size[ref][1]*cuda_array_size[ref][2]; }

    newarr=cudaAlloc(prhs[1]);
    if (isComplexType(getCudaRefNum(prhs[1])))
        for (dststart=newarr,srcstart=getCudaRef(prhs[1]);dststart<newarr+getTotalSizeFromRefNum(ref);srcstart += size3d*2,dststart+=size3d*2)
           ret=CUDAarr_times_const_rotate(srcstart,1,dststart,cuda_array_size[ref],cuda_array_dim[ref],1,mode);
    else
        for (dststart=newarr,srcstart=getCudaRef(prhs[1]);dststart<newarr+getTotalSizeFromRefNum(ref);srcstart += size3d,dststart+=size3d)
           ret=CUDAarr_times_const_rotate(srcstart,1,dststart,cuda_array_size[ref],cuda_array_dim[ref],0,mode);
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: fftshift\n");
  }
  
  else if (strcmp(command,"fft3d")==0) { // ----------------- carray to carray. Last argument: 1= forward, -1=backward, 2=forward, scramble and scale, -2=backward, scramble & scale
    float * newarr=0;
    float FTScale;
    size_t ref;
    int ret,n;
    double mode;
    int dev=0;
    int eb=0; 
    size_t ExtraBatch=1;
    size_t ExtraBatchStride=1;
    struct cudaDeviceProp prop;
    int * dirYes=NULL;
    int myDirYes[CUDA_MAXDIM];
    const mwSize * sv;
    const double * transvec;
    cufftHandle myPlan;
    cufftResult status=0;

    if (nrhs < 3 || nrhs > 4) mexErrMsgTxt("cuda: fft needs three or four arguments\n");
  
    // printf("cuda: Number of CUDA_TYPENAMES is %d\n",sizeof(CUDA_TYPE_NAMES));  
    // return 0;
         
    /* Execute FFT on device */
    ref=getCudaRefNum(prhs[1]);


    //int ret=CUDAarr_times_const_rotate(getCudaRef(prhs[1]),cuda_array_FTscale[free_array],getCudaRef(prhs[1]),cuda_array_size[free_array],cuda_array_dim[free_array]); // inplace operation, treats complex as doubles
    if (isComplexType(ref))
        newarr=cloneArray(prhs[1]);
    else
        newarr=copyToCpx(prhs[1]);
    
    mode=mxGetScalar(prhs[2]);
    if (nrhs > 3) {  // four arguments including a dirYes vector
            sv = mxGetDimensions(prhs[3]);       
            // if (sv[1] != cuda_array_dim[ref] || sv[0] != 1) mexErrMsgTxt("cuda: fft needs dirYes vector to be of the same size as the number of dimensions in the array\n");
            if (sv[0] != 1) mexErrMsgTxt("cuda: fft needs dirYes vector to be oriented correctly\n");
            transvec = mxGetPr(prhs[3]); // get pointer to vector data
            dirYes=& myDirYes[0];
            // Dbg_printf4("Transform direction (transvec) Vector: %gx%gx%g \n", transvec[0],transvec[1],transvec[2]);
            for (n=0;n<CUDA_MAXDIM;n++)
                if (n < sv[1])
                     dirYes[n]=(int) (transvec[n] > 0.5);  // copy to integer size vector
                else
                     dirYes[n]=0;  // all other dimenstions are not transformed
            Dbg_printf5("Transform direction (dirYes) Vector: %dx%dx%dx%d \n", dirYes[0],dirYes[1],dirYes[2],dirYes[3]);      
    }
        
    myPlan=CreateFFTPlan(free_array,CUFFT_C2C, dirYes, & FTScale, & ExtraBatch, & ExtraBatchStride);
    if (myPlan == 0) {
        mexErrMsgTxt("cuda: fft plan creation failed\n");
        return;
    }        
    ret=0;
    
   // printf("Mode %g Size1 %d Size2 %d Dim %d\n",mode,cuda_array_size[free_array][0],cuda_array_size[free_array][1],cuda_array_dim[free_array]);
   if (mode > 0) {
        if (fabs(mode) > 1.0) 
            ret=CUDAarr_times_const_rotate(newarr,1,newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,-1); // inplace operation, treats complex as doubles
        for (eb=0;eb<ExtraBatch;eb++) {  // only for the case where the FFT Plan cannot cover all batches (e.g. transforming along [0 1 0])
            status=cufftExecC2C(myPlan, ((cufftComplex *) newarr)+eb*ExtraBatchStride, ((cufftComplex *) newarr)+eb*ExtraBatchStride,CUFFT_FORWARD);
            if (status != CUFFT_SUCCESS) {printf("Error %s\n",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error complex to complex FFT failed\n");return;}
        }
        if (fabs(mode) > 1.0)
            ret=CUDAarr_times_const_rotate(newarr,FTScale,newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,1); // inplace operation, treats complex as doubles
    }
    else {
        if (fabs(mode) > 1.0) 
            ret=CUDAarr_times_const_rotate(newarr,1,newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,-1);
        for (eb=0;eb<ExtraBatch;eb++) {  // only for the case where the FFT Plan cannot cover all batches (e.g. transforming along [0 1 0])
            status=cufftExecC2C(myPlan, ((cufftComplex *) newarr)+eb*ExtraBatchStride, ((cufftComplex *) newarr)+ eb*ExtraBatchStride,CUFFT_INVERSE);
            if (status != CUFFT_SUCCESS) {printf("Error %s\n",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error inverse complex to complex FFT failed\n");return;}
        }
        if (fabs(mode) > 1.0)
            ret=CUDAarr_times_const_rotate(newarr,FTScale,newarr,cuda_array_size[free_array],cuda_array_dim[free_array],1,1); 
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
  else if (strcmp(command,"rfft3d")==0) { // half complex fft up to 3d. Other dimensions are treated as cosecutive transforms of sub-volumes
    float * srcstart;
    cufftComplex * newarr, * dststart; 
    size_t ref,size3d=0,dstsize3d=0;
    size_t oldsize=0;
    int mydim=0;
    double mode=1.0;
    cufftResult status=0;
    cufftHandle myPlan;

    if (nrhs != 3) mexErrMsgTxt("cuda: rfft3d needs two arguments\n");
    /* Execute FFT on device */
    ref=getCudaRefNum(prhs[1]);
    mode=mxGetScalar(prhs[2]);  // 1: sqrt scaling, 2: no scaling
    myPlan=CreateFFTPlan(ref,CUFFT_R2C, NULL, NULL, NULL, NULL);

    //ReduceToHalfComplex(ref); // restore its size
    mydim=cuda_array_dim[ref];
    if (mydim>3) mydim=3;
    
    oldsize=cuda_array_size[ref][0];
    //SumAllocated -= getTotalSizeFromRefNum(ref)*4;  // nasty trick to make the debug mode not bail out
    cuda_array_size[ref][0] /= 2;cuda_array_size[ref][0] += 1;
    //SumAllocated += getTotalSizeFromRefNum(ref)*4;
    newarr=(cufftComplex *) cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], fftHalfSComplex);  // allocates the result
    //SumAllocated -= getTotalSizeFromRefNum(ref)*4;
    cuda_array_size[ref][0]=oldsize;
    //SumAllocated += getTotalSizeFromRefNum(ref)*4;

    if (mydim == 1)
    {   size3d=cuda_array_size[ref][0]; dstsize3d=cuda_array_size[free_array][0];}
    if (mydim == 2)
    {   size3d=cuda_array_size[ref][0]*cuda_array_size[ref][1]; dstsize3d=cuda_array_size[free_array][0]*cuda_array_size[free_array][1];}
    if (mydim == 3)
    {   size3d=cuda_array_size[ref][0]*cuda_array_size[ref][1]*cuda_array_size[ref][2]; dstsize3d=cuda_array_size[free_array][0]*cuda_array_size[free_array][1]*cuda_array_size[free_array][2];}
    
    Dbg_printf3("rfft newarr has size %d %d\n",cuda_array_size[free_array][0],cuda_array_size[free_array][1]);

    for (dststart=newarr,srcstart=getCudaRef(prhs[1]);srcstart<getCudaRef(prhs[1])+getTotalSizeFromRefNum(ref);srcstart += size3d,dststart+=dstsize3d)
    // (myPlan, dststart, srcstart);  // Out-of-place transform
    {   status=cufftExecR2C(myPlan, srcstart, dststart);  // Out-of-place transform
        if (status != CUFFT_SUCCESS) {printf("Error: %s\n",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error FFT failed\n");return;}
    }

    // Multiply complex array with a constant
    if (fabs(mode) == 1.0)    // 1: sqrt scaling, 2: no scaling
        CUDAarr_times_const((float *) newarr,(float) (1/sqrt((double) size3d)),(float *) newarr,getTotalSizeFromRefNum(free_array)*2); // inplace operation, treats complex a 2 reals
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: rfft3d\n");
 }
  else if (strcmp(command,"rifft3d")==0) { // -----------------array + array
    float * newarr,* dststart;
    cufftComplex * srcstart; 
    size_t ref,size3d=0,dstsize3d=0;
    size_t oldsize=0;
    int mydim=0;
    double mode=1.0;
    cufftResult status=0;
    cufftHandle myPlan;

    if (nrhs != 3) mexErrMsgTxt("cuda: rifft3d needs three arguments\n");
    /* Execute FFT on device */
    ref=getCudaRefNum(prhs[1]);
    mode=mxGetScalar(prhs[2]);  // 1: sqrt scaling, 2: no scaling

    mydim=cuda_array_dim[ref];
    if (mydim>3) mydim=3;

    //SumAllocated -= getTotalSizeFromRefNum(ref)*4;  // nasty trick to make the debug mode not bail out
    oldsize=cuda_array_size[ref][0];
    cuda_array_size[ref][0] -= 1;cuda_array_size[ref][0] *= 2;
    //SumAllocated += getTotalSizeFromRefNum(ref)*4;  // nasty trick to make the debug mode not bail out

    newarr=cudaAllocDetailed(cuda_array_dim[ref], cuda_array_size[ref], single);  // allocates the result array
    //SumAllocated -= getTotalSizeFromRefNum(ref)*4;  // nasty trick to make the debug mode not bail out
    cuda_array_size[ref][0]=oldsize;
    //SumAllocated += getTotalSizeFromRefNum(ref)*4;  // nasty trick to make the debug mode not bail out

    if (mydim == 1)
    {   size3d=cuda_array_size[ref][0]; dstsize3d=cuda_array_size[free_array][0];}
    if (mydim == 2)
    {   size3d=cuda_array_size[ref][0]*cuda_array_size[ref][1]; dstsize3d=cuda_array_size[free_array][0]*cuda_array_size[free_array][1];}
    if (mydim == 3)
    {   size3d=cuda_array_size[ref][0]*cuda_array_size[ref][1]*cuda_array_size[ref][2]; dstsize3d=cuda_array_size[free_array][0]*cuda_array_size[free_array][1]*cuda_array_size[free_array][2];}

    myPlan=CreateFFTPlan(free_array,CUFFT_C2R, NULL, NULL, NULL, NULL);  // use the sizes of the new array for the plan
    Dbg_printf3("rfft newarr has size %d %d\n",cuda_array_size[free_array][0],cuda_array_size[free_array][1]);

    for (dststart=newarr,srcstart=(cufftComplex *) getCudaRef(prhs[1]);srcstart<((cufftComplex *) getCudaRef(prhs[1]))+getTotalSizeFromRefNum(ref);srcstart += size3d,dststart+=dstsize3d)
    {   status=cufftExecC2R(myPlan, srcstart, dststart);  // Out-of-place transform  (cufftComplex *) 
        if (status != CUFFT_SUCCESS) {printf("Error: %s\n",ERROR_NAMES[status]);mexErrMsgTxt("cuda: Error rifft3d failed\n");return;}
    }

    // Multiply complex array with a constant
    if (fabs(mode) == 1.0)    // 1: sqrt scaling, 
        CUDAarr_times_const(newarr,(float) (1.0/sqrt((double)dstsize3d)),newarr,getTotalSizeFromRefNum(free_array)); // inplace operation
    else if (fabs(mode) == 0.0)    // 0: scaling on the way backwards only
        CUDAarr_times_const(newarr,(float) (1.0/((double)dstsize3d)),newarr,getTotalSizeFromRefNum(free_array)); // inplace operation

    // cuda_array_FTscale[free_array]
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
   Dbg_printf("cuda: rifft3d\n");
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
  else if (strcmp(command,"max")==0) { // maximum as an operation over the whole array
    ACCUTYPE pres[2];const char * status;
    if (nrhs != 2) mexErrMsgTxt("cuda: max needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        { mexErrMsgTxt("cuda: tried to apply max to array of complex datatype\n"); return;}
    status=CUDAmax_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), pres);
    if (status) mexErrMsgTxt(status);
    
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)pres[0]);  // max value
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double)pres[1]); // for the maximum position
   Dbg_printf("cuda: max\n");
  } 
  else if (strcmp(command,"max_alpha")==0) { // --------------array > const
    CallCUDA_UnaryFktConst(max,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"max_arr")==0) { // -----------------array > array
    CallCUDA_BinaryFkt(max,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"min")==0) { // minimum 
    ACCUTYPE pres[2];const char * status;
    if (nrhs != 2) mexErrMsgTxt("cuda: min needs two arguments\n");
    if (isComplexType(getCudaRefNum(prhs[1])))
        { mexErrMsgTxt("cuda: tried to apply min to array of complex datatype\n"); return;}
    status=CUDAmin_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), pres);
    if (status) mexErrMsgTxt(status);

    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)pres[0]);
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double)pres[1]);
   Dbg_printf("cuda: min\n");
  }
  else if (strcmp(command,"min_alpha")==0) { // --------------array > const
    CallCUDA_UnaryFktConst(min,cudaAlloc)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"min_arr")==0) { // -----------------array > array
    CallCUDA_BinaryFkt(min,cudaAllocSized)
    if (nlhs > 0)
        plhs[0] =  mxCreateDoubleScalar((double)free_array);
  }
  else if (strcmp(command,"part_sum")==0) { // partial sum over array 
    size_t ref1;
    int ProjDir;
    size_t sSize[CUDA_MAXDIM],dSize[CUDA_MAXDIM];
    float * mask = 0;float * new_array;
    if (nrhs != 4) mexErrMsgTxt("cuda: part_sum needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    get5DSize(ref1,sSize);
    ProjDir=(int) mxGetScalar(prhs[3]);
    if (ProjDir < 1 || ProjDir > 5)
         mexErrMsgTxt("cuda: part_sum unsupported projection direction (nees to be between 1 and 5)\n");    

    get5DSize(ref1,dSize);
    dSize[ProjDir-1] = 1;

    if (!mxIsEmpty(prhs[2]))    // otherwise leave the mask to be zero
        mask=getCudaRef(prhs[2]);
    
    Dbg_printf6("dSize is %dx%dx%d\n",dSize[0],dSize[1],dSize[2],dSize[3],dSize[4]);
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
    size_t ref1,new_array_num;
    int ProjDir;
    size_t sSize[CUDA_MAXDIM],dSize[CUDA_MAXDIM];
    float * mask = 0;float * new_array;
    float * new_array_idx=0;const char * status;
    if (nrhs != 4) mexErrMsgTxt("cuda: part_max needs three arguments\n");
    ref1=getCudaRefNum(prhs[1]);
    get5DSize(ref1,sSize);
    ProjDir=(int) mxGetScalar(prhs[3]);
    if (ProjDir < 1 || ProjDir > 5)
         mexErrMsgTxt("cuda: part_max unsupported projection direction (nees to be between 1 and 5)\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
         mexErrMsgTxt("cuda: part_max data to project cannot be complex valued\n");

    get5DSize(ref1,dSize);
    dSize[ProjDir-1] = 1;

    if (!mxIsEmpty(prhs[2]))
        mask=getCudaRef(prhs[2]);
    
    Dbg_printf6("dSize is %dx%dx%dx%dx%d\n",dSize[0],dSize[1],dSize[2],dSize[3],dSize[4]);
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
    size_t ref1,new_array_num;
    int ProjDir;
    size_t sSize[CUDA_MAXDIM],dSize[CUDA_MAXDIM];
    float * mask = 0;float * new_array;
    float * new_array_idx=0;
    const char * status;
    if (nrhs != 4) mexErrMsgTxt("cuda: part_min needs three arguments\n");
    ref1=getCudaRefNum(prhs[1]);
    get5DSize(ref1,sSize);
    ProjDir=(int) mxGetScalar(prhs[3]);
    if (ProjDir < 1 || ProjDir > 5)
         mexErrMsgTxt("cuda: part_min unsupported projection direction (nees to be between 1 and 5)\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
         mexErrMsgTxt("cuda: part_min data to project cannot be complex valued\n");

    get5DSize(ref1,dSize);
    dSize[ProjDir-1] = 1;

    if (!mxIsEmpty(prhs[2]))
        mask=getCudaRef(prhs[2]);
    
    Dbg_printf6("dSize is %dx%dx%dx%dx%d\n",dSize[0],dSize[1],dSize[2],dSize[3],dSize[4]);
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
          ACCUTYPE pres[2]; double * zr,* zi;
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
            ACCUTYPE res;
            const char * status=CUDAsum_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), &res);
            if (status) mexErrMsgTxt(status);
            plhs[0] =  mxCreateDoubleScalar((double) res);
        }
        //    plhs[0] =  mxCreateDoubleScalar((double) cublasSasum(getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1]),1));
    
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing sum\n");return;}
   Dbg_printf("cuda: sum\n");
  }
  else if (strcmp(command,"sumpos")==0) { // sum over array 
    if (nrhs != 2) mexErrMsgTxt("cuda: sumpos needs two arguments\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
    	mexErrMsgTxt("cuda: sumpos needs a real datatyp as mask to sum over\n");    
    else
        if (nlhs > 0)
        {
            ACCUTYPE res;
            const char * status=CUDAsumpos_arr(getCudaRef(prhs[1]),getTotalSizeFromRef(prhs[1]), &res);
            if (status) mexErrMsgTxt(status);
            plhs[0] =  mxCreateDoubleScalar((double) res);
        }    
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing sumpos\n");return;}
   Dbg_printf("cuda: sumpos\n");
  }
  else if (strcmp(command,"norm")==0) { // norm of array 
    if (nrhs != 2) mexErrMsgTxt("cuda: norm needs two arguments\n");    
    if (isComplexType(getCudaRefNum(prhs[1])))
    {
        if (nlhs > 0)
        { 
          double real=(double) cublasSnrm2((int) getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1]),2);
          double imag=(double) cublasSnrm2((int) getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1])+1,2);
          plhs[0] =  mxCreateDoubleScalar(sqrt(real*real+imag*imag));
        }
    } else
        if (nlhs > 0)
            plhs[0] =  mxCreateDoubleScalar((double) cublasSnrm2((int) getTotalSizeFromRef(prhs[1]),getCudaRef(prhs[1]),1));
    
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error finding maximum\n");return;}
   Dbg_printf("cuda: norm\n");
  }
  else if (strcmp(command,"mtimes")==0) { // matrix product
    size_t ref1,ref2;
    size_t n1=1,m1=1,n2=1,m2=1;
    size_t dims[2];
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
            cublasCgemm('N','N',(int) n1,(int) m2,(int) m1,alpha,(cuComplex *) getCudaRef(prhs[1]),(int) n1,(cuComplex *)getCudaRef(prhs[2]),(int) n2,beta,(cuComplex *) new_array,(int) n1);
            plhs[0] =  mxCreateDoubleScalar((double) free_array);
        }
    } else { 
        if ((!isComplexType(ref1)) && (!isComplexType(ref2)))   // both real
        {
            if (nlhs > 0)
                {
                float * new_array=cudaAllocDetailed(2, dims, single);
                cublasSgemm('N','N',(int) n1,(int) m2,(int) m1,1.0,getCudaRef(prhs[1]),(int) n1,getCudaRef(prhs[2]),(int) n2,0.0,new_array,(int) n1);
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
    size_t ref1,ref2;
    size_t n1=1,m1=1,n2=1,m2=1;
    size_t dims[2];
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
            size_t tofree=free_array;
            float * new_array=cloneArray(prhs[2]);  // copy B to X
            // culaDeviceInt * IPIV;
            // cudaError_t err=cudaMalloc((void **) & IPIV,n1*sizeof(int)); 
            float * IPIV = MemAlloc(n1*sizeof(float)); // sizeof(int) ?? // use the heap if required.
            checkCudaError("Allocate mldivide complex IPIV",err);
            err=cudaMemset(IPIV,0,n1*sizeof(float));  // 
            checkCudaError("Memset mldivide complex IPIV",err);

            s=culaDeviceCgesv(n1,m2,(culaDeviceFloatComplex *)  tmp_array,n1,IPIV,(culaDeviceFloatComplex *) new_array,n2);  // replaces new_array with result
            checkCULAStatus("mldivide complex",s);
            // err=cudaFree(IPIV);
            err=MemFree(free_array);  // since this was the last to allocate
            checkCudaError("mldivide free IPIV",err);
            cudaDelete(tofree);
#else  // just state error
            float * new_array=cudaAllocDetailed(2, dims, scomplex);
            cuComplex alpha,beta;
            alpha=make_cuComplex(1.0,0.0);
            beta=make_cuComplex(0.0,0.0);
            mexErrMsgTxt("cuda: mldivide not implemented as this version was compiled without Lapack-like CULA support.\n");
            cublasCgemm('N','N',(int) n1,(int) m2,(int) m1,alpha,(cuComplex *) getCudaRef(prhs[1]),(int) n1,(cuComplex *)getCudaRef(prhs[2]),(int) n2,beta,(cuComplex *) new_array,(int) n1);
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
            size_t tofree=free_array;
            float * new_array=cloneArray(prhs[2]);  // copy B to X
            //culaDeviceInt * IPIV;
            //cudaError_t err=cudaMalloc((void **) & IPIV,n1*sizeof(int));  // 
            float * IPIV = MemAlloc(n1*sizeof(float));  // sizeof(int) ? sizeof(size_t) ?// use the heap if required.
            checkCudaError("Allocate mldivide float IPIV",err);
            err=cudaMemset(IPIV,0,n1*sizeof(float));  // 
            checkCudaError("Memset mldivide float IPIV",err);
            
            s=culaDeviceSgesv(n1,m2,(culaDeviceFloat *) tmp_array,n1,IPIV,(culaDeviceFloat *) new_array,n2);  // replaces new_array with result
            checkCULAStatus("mldivide float",s);
            // err=cudaFree(IPIV);
            err=MemFree(free_array);  // since this was the last to allocate
            checkCudaError("mldivide free IPIV",err);
            cudaDelete(tofree);
#else  // just state error
                float * new_array=cudaAllocDetailed(2, dims, single);
                mexErrMsgTxt("cuda: mldivide not implemented as this version was compiled without Lapack-like CULA support.\n");
                cublasSgemm('N','N',(int) n1,(int) m2,(int) m1,1.0,getCudaRef(prhs[1]),(int) n1,getCudaRef(prhs[2]),(int) n2,0.0,new_array,(int) n1);
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
      size_t ref1,ref2;
    size_t n1=1,m1=1,n2=1,m2=1;
    size_t dims[2];
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
            cublasCgemv('N',(int) n1,(int) m1,alpha,(cuComplex *) getCudaRef(prhs[1]),(int) n1,(cuComplex *) getCudaRef(prhs[2]),1,beta,(cuComplex *) new_array,1);
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
                cublasSgemv('N',(int) n1,(int) m1,1.0,getCudaRef(prhs[1]),(int) n1,getCudaRef(prhs[2]),1,0.0,new_array,1);
                plhs[0] =  mxCreateDoubleScalar((double) free_array);
                }
        }
        else 
            { mexErrMsgTxt("cuda: complex matrix multiplied with real vector. Not implemented. Please cast to complex type before\n"); return;}
        }
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error in matrix with vector multiplication\n");return;}
   Dbg_printf("cuda: mvtimes\n");
  }
  else  if (strcmp(command,"getLinkedVarAddr")==0) {  // This allows to detect whether there is a linked MatlabVariable     
    if (nrhs != 2) mexErrMsgTxt("cuda: getLinkedVarAddr needs two arguments\n");
    else if (nlhs > 0) {
       unsigned long int * ar;
       plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);
       ar = (unsigned long int *) mxGetPr(plhs[0]); 
       ar[0] = ((unsigned long int *) prhs[1])[0];
    }
  }
  else  if (strcmp(command,"getLinkedVarNum")==0) {  // This allows to detect whether there is a linked MatlabVariable     
    if (nrhs != 2) mexErrMsgTxt("cuda: getLinkedVarAddr needs two arguments\n");
    else {
    size_t n=0;
    size_t RefCnt=1;
    // mxArray_tag * start= ((mxArray_tag *) prhs[1]);
    size_t * start= ((size_t *) prhs[1]);
    // printf("Traced Var %d, is %s, %d, refcnt: %d\n",n,start->name,start,start->refcount);
    // mxArray_tag * cur=((mxArray_tag *) start->crosslink);  // get next one
    size_t * cur=((size_t *) start[0]);  // get next one
    RefCnt=start[4];
    // printf("Traced Var %d, addr. %lx, Ref. %ld\n",n,start,RefCnt);
    // printMem(start,5);
    while (cur != start && cur !=0) {
        // printf("Traced Var %d, is %s, %d, refcnt: %d\n",n,cur->name,cur,cur->refcount);
        n+=1;
        RefCnt=cur[4];
        // printf("Traced Var %d, Addr. %lx, Ref. %ld\n",n,cur, RefCnt);
        // printMem(cur,5);
        // cur=cur->crosslink;  // iterates over the linked list until it arrives at the end
        cur=((size_t *) cur[0]);  // iterates over the linked list until it arrives at the end
    }
    // plhs[0]=mxCreateDoubleScalar((double) n);
    plhs[0]=mxCreateDoubleScalar((double) RefCnt);
    }
  }
  else if (strcmp(command,"sprod")==0) { // matrix product
    size_t ref1,ref2;
    if (nrhs != 4) mexErrMsgTxt("cuda: sprod needs three arguments\n");    
    ref1=getCudaRefNum(prhs[1]);
    ref2=getCudaRefNum(prhs[2]);
    CHECK_CUDATotalSIZES(ref1,ref2);

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
                result=cublasCdotu((int) getTotalSizeFromRefNum(ref1),(cuComplex *) getCudaRef(prhs[1]),1,(cuComplex *) getCudaRef(prhs[2]),1);
            else
                result=cublasCdotc((int) getTotalSizeFromRefNum(ref1),(cuComplex *) getCudaRef(prhs[1]),1,(cuComplex *) getCudaRef(prhs[2]),1);
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
                float result=cublasSdot((int) getTotalSizeFromRefNum(ref1),getCudaRef(prhs[1]),1,getCudaRef(prhs[2]),1);
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
    cublasSaxpy ((int) getTotalSizeFromRef(prhs[1]), (float) alpha, pOne[currentCudaDevice], 0, mynew, 1);  // y = alpha * x + y
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error adding alpha\n");return;}
   plhs[0] =  mxCreateDoubleScalar((double)free_array);

   Dbg_printf2("cuda:minus_alpha_blas %g\n",alpha);
  }
  else if (strcmp(command,"svd3D_last")==0) { // singular value decomposition along the last dimension of an array. Code by E. Soubies emmanuel.soubies@epfl.ch and stamatis.lefkimmiatis@epfl.ch
    size_t ref1,new_array_num, dSize[CUDA_MAXDIM],LastDim,k,N;
    float * new_array_Ye;  // Eigenvalues
    float * new_array_Yv;  // Eigenvectors
    const char * status;
    if (nrhs != 2) mexErrMsgTxt("cuda: svd_last needs one arguments\n");
    ref1=getCudaRefNum(prhs[1]);  // Input array
    if (isComplexType(getCudaRefNum(prhs[1])))
         mexErrMsgTxt("cuda svd_last: Input cannot be complex valued\n");

    get5DSize(ref1,dSize);
    Dbg_printf6("dSize is %dx%dx%dx%dx%d\n",dSize[0],dSize[1],dSize[2],dSize[3],dSize[4]);

    LastDim=0;
    for (k=0;k<CUDA_MAXDIM;k++)
        if (dSize[k]>1)  LastDim=k;
    if (dSize[LastDim] != 6)
         mexErrMsgTxt("cuda svd_last: The last input dimension has to be of size 6\n");

    dSize[LastDim]=3;    // for 3 Eignevalues
    new_array_Ye=cudaAllocDetailed(cuda_array_dim[ref1], dSize, cuda_array_type[ref1]);
    new_array_num= free_array;
    // if (nlhs > 1)
    dSize[LastDim]=9;   // for the 9 components of the Eigenvectors
    new_array_Yv=cudaAllocDetailed(cuda_array_dim[ref1], dSize, cuda_array_type[ref1]);
    
    N = getTotalSizeFromRef(prhs[1])/6;
    Dbg_printf2("N is %d\n",N);
    status=CUDAsvd_last(getCudaRef(prhs[1]), new_array_Ye,new_array_Yv, N);
    if (status) mexErrMsgTxt(status);
    
    plhs[0] =  mxCreateDoubleScalar((double) new_array_num);
    if (nlhs > 1)
        plhs[1] =  mxCreateDoubleScalar((double) free_array);

    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing svd_last\n");return;}
   Dbg_printf("cuda: svd_last\n");
  }
  else if (strcmp(command,"svd3D_recomp")==0) { // singular value decomposition along the last dimension of an array. Code by E. Soubies emmanuel.soubies@epfl.ch and stamatis.lefkimmiatis@epfl.ch
    size_t ref1,ref2,dSize[CUDA_MAXDIM],LastDim,k,N;
    float * new_array_Y;  // Eigenvalues
    const char * status;
    if (nrhs != 3) mexErrMsgTxt("cuda: svd_last needs two arguments\n");
    ref1=getCudaRefNum(prhs[1]);  // Input array
    ref2=getCudaRefNum(prhs[2]);  // Input array
    if (isComplexType(getCudaRefNum(prhs[1])))
         mexErrMsgTxt("cuda svd_last: Input 1 cannot be complex valued\n");
    if (isComplexType(getCudaRefNum(prhs[2])))
         mexErrMsgTxt("cuda svd_last: Input 2 cannot be complex valued\n");


    get5DSize(ref1,dSize);
    Dbg_printf6("dSize is %dx%dx%dx%dx%d\n",dSize[0],dSize[1],dSize[2],dSize[3],dSize[4]);

    LastDim=0;
    for (k=0;k<CUDA_MAXDIM;k++)
        if (dSize[k]>1)  LastDim=k;
    if (dSize[LastDim] != 3)
         mexErrMsgTxt("cuda svd_last: The Eigenvalue (input1) dimension has to be of size 3\n");

    get5DSize(ref2,dSize);
    Dbg_printf6("dSize is %dx%dx%dx%dx%d\n",dSize[0],dSize[1],dSize[2],dSize[3],dSize[4]);
    if (dSize[LastDim] != 9)
         mexErrMsgTxt("cuda svd_last: The last Eigenvector (input2) dimension has to be of size 9\n");
    
    dSize[LastDim]=6;    // for 6 matrix entries
    new_array_Y=cudaAllocDetailed(cuda_array_dim[ref1], dSize, cuda_array_type[ref1]);
    
    N = getTotalSizeFromRef(prhs[1])/3;
    Dbg_printf2("N is %d\n",N);
    status=CUDAsvd3D_recomp(new_array_Y, getCudaRef(prhs[1]), getCudaRef(prhs[2]), N);
    if (status) mexErrMsgTxt(status);
    
    plhs[0] =  mxCreateDoubleScalar((double) free_array);

    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {mexErrMsgTxt("cuda: Error computing svd_recomp\n");return;}
    Dbg_printf("cuda: svd_recomp\n");
  }
  else if (strcmp(command,"svd")==0) { // --------------------------------------------
#ifndef NOCULA
{   culaStatus s;
    size_t ref1=getCudaRefNum(prhs[1]);
    size_t n1=1,m1=1,dimsS[1],dimsU[2],dimsV[2];
    size_t tofree;
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
//#include "user/user_c_code.inc"
#include "user_c_code.inc"
  else  if (strcmp(command,"setDevice")==0) {     
    int devNum=0;
    if (nrhs != 2) mexErrMsgTxt("cuda: setDevice needs two argument\n");
    devNum=(int) mxGetScalar(prhs[1]);
    activateCudaDevice(devNum);
  }
  else  if (strcmp(command,"setForceSynchronization")==0) {     
    int devNum=0;
    if (nrhs != 2) mexErrMsgTxt("cuda: setForceSynchronization needs two argument\n");
    forceDeviceSynchronization=(int) mxGetScalar(prhs[1]);
    if (forceDeviceSynchronization) {
         cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    } else {
        cudaSetDeviceFlags(cudaDeviceScheduleAuto);
    }
  }
  else
  {
      printf("Error executing command %s\n",command);
        mexErrMsgTxt("cuda: Unknown command\n");
  }

if (forceDeviceSynchronization) { // If this is active, synchonizeKernels() will be called after every comand. This is useful for profiling in Matlab.
    cudaDeviceSynchronize();   
}

#ifdef DEBUG
    printf("Executed command %s, rechecking memory consistency ...\n",command);
    CheckMemoryConsistency();
#endif
Dbg_printf4("Pos 3: ignoreDelete state is : %d, command %s, ignoreRef: %d\n",ignoreDelete, command, ignoreRef);


  mxFree(command);
  Dbg_printf2("cuda: %d arrays in memory\n",cuda_curr_arrays);
 return;
}
