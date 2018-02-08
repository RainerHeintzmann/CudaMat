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
 * Compile with:
 * Windows:
system('"c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"')
system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin')

Window 64 bit:
system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" -I"c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include" ')

Linux:
 * File sudo vi /usr/local/cuda/bin/nvcc.profile
 * needs the flag -fPIC  in the include line
system('nvcc -c cudaArith.cu -v -I/usr/local/cuda/include/')
 */

// To suppress the unused variable argument for ARM targets
#pragma diag_suppress 177

#include <cuda.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>

#ifndef NAN   // should be part of math.h
#define NAN (0.0/0.0)
#endif

#include "cufft.h"
#include "cudaArith.h"
#define ACCU_ARRTYPE double  // Type of the tempory arrays for reduce operations
#define IMUL(a, b) __mul24(a, b)

//#define BLOCKSIZE 512
//#define BLOCKSIZE 512
// below is blocksize for temporary array for reduce operations. Has to be a power of 2 in size
#ifndef CUIMAGE_REDUCE_THREADS  // this can be defined at compile time via the flag NVCCFLAG='-D CUIMAGE_REDUCE_THREADS=512'
#define CUIMAGE_REDUCE_THREADS 512
#endif
// (prop.maxThreadsPerBlock)
// #define CUIMAGE_REDUCE_THREADS 512
// #define CUIMAGE_REDUCE_THREADS 128
//#define CUIMAGE_REDUCE_BLOCKS  64

#define NBLOCKS(N,blockSize) (N/blockSize+(N%blockSize==0?0:1))

#define NBLOCKSL(N,blockSize) 1
// min((N/blockSize+(N%blockSize==0?0:1)),prop.maxGridSize[0])


#define MemoryLayout(N,blockSize,nBlocks)	blockSize=prop.maxThreadsPerBlock; \
{ size_t numb=NBLOCKS(N,blockSize);                    \
    if (numb<prop.maxGridSize[0])                   \
    nBlocks.x=numb;                                 \
else                                                \
    {nBlocks.x=(size_t)(sqrt((float)numb)+1);          \
    nBlocks.y=(size_t)(sqrt((float)numb)+1);}}

// the real part is named ".x" and the imaginary ".y" in the cufftComplex datatype
__device__ cufftComplex cuda_resultVal;   // here real and complex valued results can be stored to be then transported to the host
__device__ size_t cuda_resultInt;   // here size-valued results can be stored to be then transported to the host
static ACCU_ARRTYPE * TmpRedArray=0;   // This temporary array will be constructed on the device, whenever the first reduce operation is performed
static ACCU_ARRTYPE * accum = 0;       // This is the corresponding array on the host side
static size_t CurrentRedSize=0;    // Keeps track of how much reduce memory is allocated on the device
static const int MinRedBlockSize=65536;    // defines the chunks of memory (in floats) which will be used in reduce operations
static struct cudaDeviceProp prop;  // Defined in cudaArith.h: contains the cuda Device properties. is set during initialisation
    // prop.maxThreadsPerBlock;  // 512
    // prop.multiProcessorCount;   // 30
    // prop.warpSize;   // 32
    // prop.maxThreadsDim[0];   // 512  = max blocksize
    // prop.maxGridSize[0];   // 65535  = max GridSize = max nBlocks?


#define mysumpos(a,b) ((a)+((b)>0))
#define mysum(a,b) ((a)+(b))
#define maxCond(a,b) (((b)>(a)))
#define minCond(a,b) (((b)<(a)))

#define Sqr(a) ((a)*(a))

#define sign(x) (((x) > 0) - ((x) < 0))

// below are code snippets used in other macros 
#define Coords3DFromIdx(idx,sSize)                                      \
  size_t x=(idx)%sSize.s[0];                                               \
  size_t y=(idx/sSize.s[0])%sSize.s[1];                                    \
  size_t z=(idx/(sSize.s[0]*sSize.s[1]))%sSize.s[2];                       

#define IdxFromCoords3D(x,y,z,dSize,dOffs) \
  unsigned size_t idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]);  \


#define CoordsNDFromIdx(idx,sSize,pos)                                  \
   IntND pos;                                               \
   { size_t resid=idx;                                               \
  for(int _d=0;_d<CUDA_MAXDIM;_d++)                                     \
      if (resid > 0)                                                    \
        { pos.s[_d]=resid%sSize.s[_d];                                    \
          resid/=sSize.s[_d]; }                                         \
      else                                                              \
          pos.s[_d]=0;                                                    \
  }

// since the c- modula function does not wrap to positive number we define our own modula function
#define MyModulo(x,N) (((x) % (N) + (N)) % (N))

#define IdxNDFromCoords(pos,dSize,idd)                                   \
  (idd)=0;                                                              \
  {                                                                     \
  size_t _Stride=1;                                                \
  for(int _d=0;_d<CUDA_MAXDIM;_d++)                                      \
  if (dSize.s[_d]>0) {                                                   \
              long long N=dSize.s[_d];                                  \
              {(idd) += MyModulo(pos.s[_d],N) *_Stride;}  \
        _Stride *= dSize.s[_d]; }                                        \
}
// This was removed when changed from int to size_t to accomodate 64 bits properly:
// if (pos.s[_d] < 0)                                              
//              {(idd) += (dSize.s[_d]-((-pos.s[_d]) % dSize.s[_d])) *_Stride;}          
//          else                                                          
   
// The macro below converts an ND memory position into a memory position that may have singleton dimensions
// numdims: number of dimensions
// posOrig: original index in ND array (without singleton)
// isSingleton: boolean array denoting whether a dimension needs to be reduced to singleton (size 1)
// stridesOrig: strides of the original array
// posSingleton: resulting Singleton index which can be used
// stridesSingleton: the strides in the result array
// 
// The algorithms goes through all dimensions and allways assumes that the rest of (yet untreated) dimensions is of the same type as the state variable _state: 1 = singleton dimension
#define Original2Singleton(numdims, isSingleton, posOrig, sizesOrig, posSingleton) { \
    posSingleton=posOrig;                                       \
    int _d,_state=0, stridesOrig=1, stridesSingleton=1;         \
    for (_d = 0;_d<numdims;_d++){                               \
        if (_state == 0)    {                                   \
            if (isSingleton.s[_d] != _state)    {               \
                posSingleton = (posSingleton % stridesSingleton);\
                _state = 1;}                                    \
            else stridesSingleton *= sizesOrig.s[_d];                \
        } else {                                                \
            if (isSingleton.s[_d] != _state)     {              \
                posSingleton +=  (posOrig / stridesOrig) * stridesSingleton;      \
            	stridesSingleton *= sizesOrig.s[_d];                \
                _state=0; }                                     \
        }                                                       \
        stridesOrig *= sizesOrig.s[_d];                         \
    }                                                           \
}                                                               \


// The partial reduction function below projects the data along one dimension
// the processors are assigned to the result image pixels
// CAVE: These versions can be slow, if the resulting data has is smaller than the number of processors
#define CUDA_PartRedMask(FktName, OP)               \
__global__ void FktName (float *in, float *out, float * mask, size_t N, size_t ProjStride, size_t ProjSize){      \
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  if(idd>=N) return;                                                    \
  size_t p;                                                                \
  size_t ids=((idd%ProjStride) + (idd/ProjStride)*(ProjStride*ProjSize));  \
  ACCUTYPE accu=0.0;                                                       \
  int laterPix=0;                                                       \
  for (p=0;p<ProjSize;p++)                                              \
    {                                                                   \
      if (mask == 0 || mask[ids] != 0.0)                                \
        if (! laterPix)  {                                              \
            accu=in[ids];                                               \
            laterPix=1;                                                 \
        } else {                                                        \
            accu=OP(accu,(ACCUTYPE) in[ids]);                             \
        }                                                               \
      ids += ProjStride;                                                \
    }                                                                   \
 out[idd] = (float) accu;                                               \
}                                                                       \
\
extern "C" const char * CUDA ## FktName(float *a, float * mask, float * c, size_t sSize[CUDA_MAXPROJ], int ProjDir)\
{                                                                       \
    cudaError_t myerr;                                                  \
    size_t d,N=1;                                                       \
	size_t blockSize;dim3 nBlocks;                                      \
    size_t ProjStride=1,ProjSize=1;                                     \
    if (ProjDir>CUDA_MAXPROJ)                                           \
        return "Error: Unsupported projection direction";               \
    for (d=0;d<CUDA_MAXPROJ;d++)  {                                     \
        if (d < ProjDir-1)  ProjStride *= sSize[d];                     \
        if (d != ProjDir-1) N*=sSize[d];                                \
    }                                                                   \
    ProjSize=sSize[ProjDir-1];                                          \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName<<<nBlocks,blockSize>>>(a,c,mask,N,ProjStride,ProjSize);     \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}

//   This is the same as the above but suited for complex numbers
#define CUDA_PartRedMaskCpx(FktName, OP)               \
__global__ void FktName (float *in, float *out, float * mask, size_t N, size_t ProjStride, size_t ProjSize){      \
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  if(idd>=N) return;                                                    \
  size_t p;                                                                \
  size_t ids=((idd%ProjStride) + (idd/ProjStride)*(ProjStride*ProjSize));  \
  ACCUTYPE accu=0.0;                                                       \
  ACCUTYPE accuI=0.0;                                                      \
  int laterPix=0;                                                       \
  for (p=0;p<ProjSize;p++)                                              \
    {                                                                   \
      if (mask == 0 || mask[ids] != 0.0)                                \
        if (! laterPix)  {                                              \
            accu=in[2*ids];                                             \
            accuI=in[2*ids+1];                                          \
            laterPix=1;                                                 \
        } else {                                                        \
            accu=OP(accu,(ACCUTYPE)in[2*ids]);                           \
            accuI=OP(accuI,(ACCUTYPE)in[2*ids+1]);                       \
        }                                                               \
      ids += ProjStride;                                                \
    }                                                                   \
 out[2*idd] = (float) accu;                                             \
 out[2*idd+1] = (float) accuI;                                          \
}                                                                       \
\
extern "C" const char * CUDA ## FktName(float *a, float * mask, float * c, size_t sSize[3], int ProjDir)\
{                                                                       \
     cudaError_t myerr;                                                \
    size_t d,N=1;                                                       \
	size_t blockSize;dim3 nBlocks;                                      \
    size_t ProjStride=1,ProjSize=1;                                     \
    if (ProjDir>CUDA_MAXPROJ)                                           \
        return "Error: Unsupported projection direction";               \
    for (d=0;d<CUDA_MAXPROJ;d++)  {                                     \
        if (d < ProjDir-1)  ProjStride *= sSize[d];                     \
        if (d != ProjDir-1) N*=sSize[d];                                \
    }                                                                   \
    ProjSize=sSize[ProjDir-1];                                          \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName<<<nBlocks,blockSize>>>(a,c,mask,N,ProjStride,ProjSize);     \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}

// This partial reduction code keeps track of the index
#define CUDA_PartRedMaskIdx(FktName, OP)               \
__global__ void FktName (float *in, float *out, float * outIdx, float * mask, size_t N, size_t ProjStride, size_t ProjSize){      \
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  if(idd>=N) return;                                                    \
  size_t p;                                                                \
  size_t ids=((idd%ProjStride) + (idd/ProjStride)*(ProjStride*ProjSize));  \
  float accu=0.0;                                                       \
  float accuIdx=-1;                                                     \
  int laterPix=0;                                                       \
  for (p=0;p<ProjSize;p++)                                              \
    {                                                                   \
      if (mask == 0 || mask[ids] != 0.0)                                \
        if (! laterPix)  {                                              \
            accu=in[ids];                                               \
            accuIdx=p;                                                  \
            laterPix=1;                                                 \
        } else {                                                        \
            if (OP(accu,in[ids])) {accu=in[ids];accuIdx=p;}             \
        }                                                               \
      ids += ProjStride;                                                \
    }                                                                   \
 out[idd] = accu;                                                       \
 if (outIdx != 0)                                                       \
    outIdx[idd] = accuIdx;                                                 \
}                                                                       \
\
extern "C" const char * CUDA ## FktName(float *a, float * mask, float * c, float * cIdx, size_t sSize[5], int ProjDir)\
{                                                                       \
    cudaError_t myerr;                                                  \
    size_t d,N=1;                                                       \
	size_t blockSize;dim3 nBlocks;                                      \
    size_t ProjStride=1,ProjSize=1;                                     \
    if (ProjDir>CUDA_MAXPROJ)                                           \
        return "Error: Unsupported projection direction";               \
    for (d=0;d<CUDA_MAXPROJ;d++)  {                                     \
        if (d < ProjDir-1)  ProjStride *= sSize[d];                     \
        if (d != ProjDir-1) N*=sSize[d];                                \
    }                                                                   \
    ProjSize=sSize[ProjDir-1];                                          \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName<<<nBlocks,blockSize>>>(a,c,cIdx,mask,N,ProjStride,ProjSize);\
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}

// Below is some reduction code adapted from the tips and tricks tutorial 
// https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
// FUNCTION BELOW IS SLOW AND DOES NOT WORK PROPERLY YET
#define CUDA_FullRedBin(FktName, OP)                                    \
__global__ void FktName (float *in, size_t N){                             \
  const size_t stride = CUIMAGE_REDUCE_THREADS;                    \
  const size_t start  = threadIdx.x;\
  __shared__ float accum[CUIMAGE_REDUCE_THREADS];               \
  ACCUTYPE tmp=0;                                                 \
  size_t nTotalThreads=CUIMAGE_REDUCE_THREADS;                     \
  size_t thread2;                                                  \
                                                                \
  if (start >= CUIMAGE_REDUCE_THREADS) return;                   \
  if (start >= N) {accum[start]=0;return;}                      \
                                                                \
  tmp = in[start];                               \
  for (size_t ii=start+stride; ii < N; ii += CUIMAGE_REDUCE_THREADS)  { \
    tmp = OP(tmp, (ACCUTYPE) in[ii]);        \
  }                                                             \
  accum[threadIdx.x]=tmp;                                       \
  __syncthreads();                                              \
                                                                \
/* Now entering the logaritmic reduction phase of the algorithm*/       \
while(nTotalThreads > 1)                                                \
{                                                                       \
  size_t halfPoint = (nTotalThreads >> 1);	/* divide by two */             \
  /* only the first half of the threads will be active. */              \
                                                                        \
  if (threadIdx.x < halfPoint)                                          \
  {  thread2 = threadIdx.x + halfPoint;                                   \
   /* Skipping the fictious threads blockDim.x ... blockDim_2-1 */      \
   if (thread2 < stride)                                            \
      accum[threadIdx.x]=OP(accum[threadIdx.x],accum[thread2]);         \
  }                                                                     \
  __syncthreads();                                                      \
  /* Reducing the binary tree size by two:  */                          \
  nTotalThreads = halfPoint;                                            \
}                                                                       \
  __syncthreads();                                              \
  if (threadIdx.x == 0)                                         \
          cuda_resultVal.x=accum[0];                            \
}                                                               \
extern "C" const char * CUDA ## FktName(float * a, size_t N, float * resp) \
{                                                               \
  int CUIMAGE_REDUCE_BLOCKS;                                    \
  dim3 threadBlock;                                             \
  dim3 blockGrid;                                               \
  CUIMAGE_REDUCE_BLOCKS=NBLOCKSL(N,CUIMAGE_REDUCE_THREADS);     \
  threadBlock.x=CUIMAGE_REDUCE_THREADS;                         \
  blockGrid.x=CUIMAGE_REDUCE_BLOCKS;                            \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, N);                    \
  if (cudaGetLastError() != cudaSuccess)                        \
      return cudaGetErrorString(cudaGetLastError());            \
                                                                \
  cudaMemcpyFromSymbol(resp, cuda_resultVal, sizeof(* resp));   \
  if (cudaGetLastError() != cudaSuccess)                        \
      return cudaGetErrorString(cudaGetLastError());            \
  return 0;                                                     \
}


// Below is the reduction code of Wouter Caarls, modified
// This could potentially also be run sequentially over the remaining dimension

#define CUDA_FullRed(FktName, OP1,OP2)                               \
__global__ void FktName (float *in, ACCU_ARRTYPE *out, size_t N){         \
  const size_t stride = blockDim.x * gridDim.x;                    \
  const size_t start  = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;\
  __shared__ ACCU_ARRTYPE accum[CUIMAGE_REDUCE_THREADS];               \
  ACCUTYPE tmp=0;                                                 \
  if (start >= N) return;                                       \
                                                                \
  tmp = in[start];                                              \
  for (size_t ii=start+stride; ii < N; ii += stride)  {            \
    tmp = OP1(tmp, (ACCUTYPE) in[ii]);                             \
  }                                                             \
  accum[threadIdx.x]=tmp;                                       \
  __syncthreads();                                              \
  if (threadIdx.x == 0)                                         \
  {                                                             \
    ACCUTYPE res = accum[0];                                      \
    size_t limit;                                                  \
    if (start+blockDim.x > N) limit=(N-start);                  \
    else limit=blockDim.x;                                      \
    for (size_t ii = 1; ii < limit; ii++) {                        \
      res=OP2(res,(ACCUTYPE) accum[ii]);                           \
     }                                                          \
    out[blockIdx.x] = res;                                      \
  }                                                             \
}                                                               \
                                                                \
extern "C" const char * CUDA ## FktName(float * a, size_t N, ACCUTYPE * resp) \
{                                                               \
  cudaError_t myerr;                                            \
  const char * myerrStr;                                        \
  ACCUTYPE res;                                                    \
  int CUIMAGE_REDUCE_BLOCKS;                                    \
  dim3 threadBlock;                                             \
  dim3 blockGrid;                                               \
  CUIMAGE_REDUCE_BLOCKS=NBLOCKSL(N,CUIMAGE_REDUCE_THREADS);     \
  threadBlock.x=CUIMAGE_REDUCE_THREADS;                         \
  blockGrid.x=CUIMAGE_REDUCE_BLOCKS;                            \
                                                                \
  myerrStr=CheckReduceAllocation(2*CUIMAGE_REDUCE_BLOCKS);      \
  if (myerrStr) return myerrStr;                                \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, TmpRedArray, N);       \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpy(accum, TmpRedArray, CUIMAGE_REDUCE_BLOCKS*sizeof(ACCU_ARRTYPE), cudaMemcpyDeviceToHost);\
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  res = (ACCUTYPE) accum[0];                                      \
  for (size_t ii=1; ii < CUIMAGE_REDUCE_BLOCKS; ii++)  {           \
    res=(ACCUTYPE) OP2(res,(ACCUTYPE) accum[ii]);                    \
   }                                                            \
  /* cudaFree(TmpRedArray); */                                  \
  /* free(accum); */                                            \
                                                                \
  (* resp)=res;                                                 \
  return 0;                                                     \
}

// The version below is for complex valued arrays

#define CUDA_FullRedCpx(FktName, OP)               \
__global__ void FktName (float *in, ACCU_ARRTYPE *out, size_t N){      \
  const size_t stride = blockDim.x * gridDim.x;                    \
  const size_t start  = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;\
  ACCUTYPE tmpR=0,tmpI=0;                                         \
  __shared__ ACCU_ARRTYPE accum[CUIMAGE_REDUCE_THREADS];               \
  __shared__ ACCU_ARRTYPE accumI[CUIMAGE_REDUCE_THREADS];              \
  if (start >= N) return;                                    \
                                                                \
  tmpR = in[2*start];                             \
  tmpI = in[2*start+1];                          \
  for (size_t ii=start+stride; ii < N; ii += stride)  {         \
    tmpR = OP(tmpR, (ACCUTYPE) in[2*ii]);      \
    tmpI = OP(tmpI, (ACCUTYPE) in[2*ii +1]); \
  }                                                             \
  accum[threadIdx.x]=tmpR;                                       \
  accumI[threadIdx.x]=tmpI;                                     \
  __syncthreads();                                              \
  if (!threadIdx.x)                                             \
  {                                                             \
    ACCUTYPE res = accum[0];                                       \
    ACCUTYPE resI = accumI[0];                                     \
    size_t limit;                                                  \
    if (start+blockDim.x > N) limit=(N-start);  \
    else limit=blockDim.x;                                      \
    for (size_t ii = 1; ii < limit; ii++) {                        \
      res=OP(res,(ACCUTYPE) accum[ii]);                           \
      resI=OP(resI,(ACCUTYPE) accumI[ii]);                        \
     }                                                          \
    out[2*blockIdx.x] = res;                                    \
    out[2*blockIdx.x + 1] = resI;                               \
  }                                                             \
}  \
\
extern "C" const char * CUDA ## FktName(float * a, size_t N, ACCUTYPE * resp) \
{                                                               \
    cudaError_t myerr;                                          \
  const char * myerrStr;                                              \
  ACCUTYPE res, resI;                                              \
  int CUIMAGE_REDUCE_BLOCKS;                                    \
  dim3 threadBlock;                                             \
  dim3 blockGrid;                                               \
  CUIMAGE_REDUCE_BLOCKS=NBLOCKSL(N,CUIMAGE_REDUCE_THREADS);     \
  threadBlock.x=CUIMAGE_REDUCE_THREADS;                         \
  blockGrid.x=CUIMAGE_REDUCE_BLOCKS;                            \
                                                                \
  myerrStr=CheckReduceAllocation(2*CUIMAGE_REDUCE_BLOCKS);      \
  if (myerrStr) return myerrStr;                                \
                                                                \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, TmpRedArray, N);       \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpy(accum, TmpRedArray, 2*CUIMAGE_REDUCE_BLOCKS*sizeof(ACCU_ARRTYPE), cudaMemcpyDeviceToHost);\
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
                                                                \
  res = (ACCUTYPE) accum[0];                                               \
  resI = (ACCUTYPE) accum[1];                                              \
  for (size_t ii=1; ii < CUIMAGE_REDUCE_BLOCKS; ii++)  {           \
    res=(ACCUTYPE) OP(res,(ACCUTYPE) accum[2*ii]);                                    \
    resI=(ACCUTYPE) OP(resI,(ACCUTYPE) accum[2*ii + 1]);                              \
   }                                                            \
  /* cudaFree(interm);  */                                      \
  /* free(accum); */                                            \
                                                                \
  (* resp)=res;                                                 \
  (* (resp+1))=resI;                                            \
  return 0;                                                     \
}

// The version below is for remembering the index (e.g. max and min)

#define CUDA_FullRedIdx(FktName, OP)               \
__global__ void FktName (float *in, ACCU_ARRTYPE *out, size_t size){      \
  const size_t stride = blockDim.x * gridDim.x;                    \
  const size_t start  = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;\
  __shared__ ACCU_ARRTYPE accum[CUIMAGE_REDUCE_THREADS];               \
  __shared__ ACCU_ARRTYPE accumI[CUIMAGE_REDUCE_THREADS];              \
  if (start >= size) return;                                    \
                                                                \
  accum[threadIdx.x] = in[start];                               \
  accumI[threadIdx.x] = start;                                  \
  for (size_t ii=start+stride; ii < size; ii += stride)  {         \
    if OP(accum[threadIdx.x], in[ii]) { accum[threadIdx.x]= in[ii]; accumI[threadIdx.x]= ii; }      \
  }                                                             \
  __syncthreads();                                              \
  if (!threadIdx.x)                                             \
  {                                                             \
    ACCUTYPE res = (ACCUTYPE) accum[0];                         \
    ACCUTYPE resI = (ACCUTYPE) accumI[0];                       \
    size_t limit;                                                  \
    if (start+blockDim.x > size) limit=1+(size-start-1)/gridDim.x;  \
    else limit=blockDim.x;                                      \
    for (size_t ii = 1; ii < limit; ii++) {                        \
    if OP(res, (ACCUTYPE) accum[ii]){ res= (ACCUTYPE) accum[ii]; resI= (ACCUTYPE) accumI[ii]; }  \
     }                                                          \
    out[2*blockIdx.x] = res;                                    \
    out[2*blockIdx.x + 1] = resI;                               \
  }                                                             \
}  \
\
extern "C" const char * CUDA ## FktName(float * a, size_t N, ACCUTYPE * resp) \
{                                                               \
  ACCUTYPE res, resI;                                              \
  cudaError_t myerr;                                            \
  const char * myerrStr;                                        \
  int CUIMAGE_REDUCE_BLOCKS;                                    \
  dim3 threadBlock;                                             \
  dim3 blockGrid;                                               \
  CUIMAGE_REDUCE_BLOCKS=NBLOCKSL(N,CUIMAGE_REDUCE_THREADS);     \
  threadBlock.x=CUIMAGE_REDUCE_THREADS;                         \
  blockGrid.x=CUIMAGE_REDUCE_BLOCKS;                            \
                                                                \
  myerrStr=CheckReduceAllocation(2*CUIMAGE_REDUCE_BLOCKS);      \
  if (myerrStr) return myerrStr;                                \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, TmpRedArray, N);       \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpy(accum, TmpRedArray, 2*CUIMAGE_REDUCE_BLOCKS*sizeof(ACCU_ARRTYPE), cudaMemcpyDeviceToHost);\
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  res = (ACCUTYPE) accum[0];                                               \
  resI = (ACCUTYPE)accum[1];                                              \
  for (size_t ii=1; ii < CUIMAGE_REDUCE_BLOCKS; ii++)  {           \
    if OP(res, (ACCUTYPE) accum[2*ii]) {res=(ACCUTYPE)accum[2*ii]; resI= (ACCUTYPE) accum[2*ii+1];  }  \
   }                                                            \
  /* cudaFree(TmpRedArray); */                                  \
  /* free(accum); */                                            \
                                                                \
  (* resp)=res;                                                 \
  (* (resp+1))=resI;                                            \
  return 0;                                                     \
}


// Allows to work with the linear index image from a binary mask image.
// useful for: a(mask) = 2*a(mask)
// Algorithm: pass1 : count ones in your area
// pass 2: integrate accum over thread number to get block ones offset
// pass 3: Apply index

#define CUDA_MaskIdx(FktName, EXPRESSIONS)                      \
__global__ void FktName (float *a, float * mask,float *c, size_t N){ \
  size_t Blocksize = N/CUIMAGE_REDUCE_THREADS + 1;                 \
  size_t start = Blocksize * threadIdx.x;                          \
  __shared__ size_t accum[CUIMAGE_REDUCE_THREADS+1];               \
  if (start >= N) return;                                       \
                                                                \
  { size_t SumMask=0;                                              \
  for (size_t ii=start; ii < start+Blocksize; ii ++)  {            \
    if (ii < N)                                                 \
        SumMask += (mask[ii] != 0);                             \
  }                                                             \
  accum[threadIdx.x+1] = SumMask;                               \
  }                                                             \
  __syncthreads();                                              \
  if (threadIdx.x == 0)                                         \
  {                                                             \
    accum[0] = 0;                                               \
    size_t res = 0;                                                \
    for (size_t ii = 0; ii*Blocksize < N; ii++) {                  \
      res += accum[ii+1];                                       \
      accum[ii+1] = res;                                        \
     }                                                          \
    cuda_resultInt = res;                                       \
  }                                                             \
  __syncthreads();                                              \
  size_t mask_idx= accum[threadIdx.x];                             \
  for (size_t idx=start; idx < start+Blocksize; idx ++)  {         \
    if ((idx < N) && (mask[idx] != 0))                          \
      {                                                         \
        EXPRESSIONS                                             \
        mask_idx ++;                                            \
      }                                                         \
  }                                                             \
}                                                               \
                                                                \
extern "C" const char * CUDA ## FktName(float * in, float * mask, float *  out, size_t N, size_t * pM) \
{                                                               \
  int CUIMAGE_REDUCE_BLOCKS=1;                                  \
  cudaError_t myerr;                                            \
  dim3 threadBlock(CUIMAGE_REDUCE_THREADS);                     \
  dim3 blockGrid(CUIMAGE_REDUCE_BLOCKS);                        \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(in, mask, out, N);        \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpyFromSymbol(pM, cuda_resultInt, sizeof(* pM));       \
                                                                \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
  return 0;                                                     \
}

/*  This was for debugging purposes. Commented out for now
#define CUDA_BinaryFktOld(FktName,expression)                          \
__global__ void                                                     \
FktName(float*a,float *b, float * c, size_t N)                         \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=N) return;                                              \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB)  \
{                                                                       \
    cudaError_t myerr;                                          \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,b,c,N);                            \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       
*/

// In the expression one can use the variables idx (for real valued arrays) and idc (for complex valued arrays)
// -------------- caller function is also generated -------------
// 
// The 2 macros below treat binary functions such as plus (as the one before)
// but singleton dimensions will be wrapped just like in Python or DIPImage
#define CUDA_BinaryFkt(FktName,expression)                          \
__global__ void                                                     \
FktName(float*a,float *b, float * c, size_t N)                         \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
    size_t idxA=idx, idxB=idx;                                         \
	if(idx>=N) return;                                              \
	expression                                                      \
}                                                                   \
__global__ void                                                     \
FktName ##_S(float*a,float *b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB) \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
    size_t idxA,idxB;                                                  \
	if(idx>=N) return;                                              \
    Original2Singleton(numdims, isSingletonA, idx,sizesC,idxA)     \
    Original2Singleton(numdims, isSingletonB, idx,sizesC,idxB)     \
    expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB)  \
{                                                                       \
    cudaError_t myerr;                                                  \
	size_t blockSize;dim3 nBlocks;                                         \
    myerr=cudaGetLastError();                                           \
    if (numdims==0) {                                                            \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName<<<nBlocks,blockSize>>>(a,b,c,N);                            \
    } else                                                              \
    {                                                                   \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName ## _S<<<nBlocks,blockSize>>>(a,b,c,N, numdims, sizesC, isSingletonA, isSingletonB);  \
    }                                                                   \
    myerr=cudaGetLastError();                                           \
    if (myerr != cudaSuccess)                                           \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

// In the expression one can use the variables idx (for real valued arrays) and idc (for complex valued arrays)
// -------------- caller function is also generated -------------
// 
// The 2 macros below treat functions with an arbitrary number of reference arrays
// but singleton dimensions will be wrapped just like in Python or DIPImage

#define CUDA_NArgsFkt(FktName,expression,NArgs)                     \
typedef struct {                                                    \
    float * s[NArgs];                                                  \
} FktName ##_ARGTYPE ;                                             \
__global__ void                                                     \
FktName(FktName ##_ARGTYPE f,float * c, size_t N)                         \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
    size_t idxArg[NArgs],myarg;                                        \
	if(idx>=N) return;                                              \
     for (myarg=0;myarg<NArgs;myarg++)                              \
        idxArg[myarg]=idx;                                          \
	expression                                                      \
}                                                                   \
__global__ void                                                     \
FktName ##_S(FktName ##_ARGTYPE f,float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[NArgs]) \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
    size_t idxArg[NArgs],myarg;                                              \
	if(idx>=N) return;                                              \
    for (myarg=0;myarg<NArgs;myarg++)                               \
        {Original2Singleton(numdims, isSingleton[NArgs], idx,sizesC,idxArg[myarg]) }\
    expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * f[NArgs], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[NArgs])  \
{                                                                       \
    cudaError_t myerr;                                                  \
	size_t blockSize,n;dim3 nBlocks;                                         \
    FktName ##_ARGTYPE F;                                               \
    for (n=0;n<NArgs;n++) F.s[n]=f[n];                                  \
    myerr=cudaGetLastError();                                           \
    if (numdims==0) {                                                   \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName<<<nBlocks,blockSize>>>(F,c,N);                            \
    } else                                                              \
    {                                                                   \
    MemoryLayout(N,blockSize,nBlocks)                                   \
	FktName ## _S<<<nBlocks,blockSize>>>(F,c,N, numdims, sizesC, isSingleton);  \
    }                                                                   \
    myerr=cudaGetLastError();                                           \
    if (myerr != cudaSuccess)                                           \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

// In the expression one can use the variables idx (for real valued arrays) 
// -------------- caller function is also generated -------------
#define CUDA_IndexFkt(FktName,expression)                          \
__global__ void                                                     \
FktName(float*a,float *b, float * c, size_t N, size_t M)                         \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=M) return;                                              \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float * b, float * c, size_t N, size_t M)  \
{                                                                       \
    cudaError_t myerr;                                          \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(M,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,b,c,N,M);                            \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

// --------------Macro generating operation of array with real constant -------------

#define CUDA_UnaryFktConst(FktName,expression)                      \
__global__ void                                                     \
FktName(float*a,float b, float * c, size_t N)                          \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=N) return;                                              \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float b, float * c, size_t N)  \
{                                                                       \
    cudaError_t myerr;                                          \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,b,c,N);                            \
    myerr=cudaGetLastError();                                             \
    if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

// --------------Macro generating operation with complex array and constant -------------
#define CUDA_UnaryFktConstC(FktName,expression)                      \
__global__ void                                                     \
FktName(float*a,float br, float bi, float * c, size_t N)               \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=N) return;                                              \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float br, float bi, float * c, size_t N)  \
{                                                                       \
    cudaError_t myerr;                                                  \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,br,bi,c,N);                        \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

        

// ----------- Makro for function with an integer Vector ---- e.g.- for cyclic shifts etc. -----
#define CUDA_UnaryFktIntVec(FktName,expression)                      \
__global__ void                                                     \
FktName(float*a, SizeND b, float * c, SizeND sSize, size_t N)          \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=N) return;                                              \
    expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, size_t b[CUDA_MAXDIM], float * c, size_t mySize[CUDA_MAXDIM], size_t N)  \
{                                                                       \
  cudaError_t myerr;                                          \
	size_t blockSize;dim3 nBlocks;                                         \
    SizeND sb,sSize;                                                    \
    for (size_t d=0;d<CUDA_MAXDIM;d++)                                     \
    { sb.s[d]=b[d];sSize.s[d]=mySize[d]; }                              \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,sb,c,sSize,N);                     \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

// ----------- Makro for function with an integer Vector ---- e.g.- for cyclic shifts etc. -----
#define CUDA_Fkt2Vec(FktName,expression)                            \
__global__ void                                                     \
FktName(float * c, VecND vec1, VecND vec2, SizeND sSize, size_t N)     \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=N) return;                                              \
    expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * c, VecND vec1, VecND vec2, SizeND sSize, size_t N)  \
{                                                                       \
    cudaError_t myerr;                                          \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(c,vec1,vec2,sSize,N);                \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}



// --------------Macro generating unary operation with complex array  -------------
#define CUDA_UnaryFkt(FktName,expression)                     \
__global__ void                                                     \
FktName(float*a, float * c, size_t N)                                  \
{                                                                   \
    size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
	if(idx>=N) return;                                              \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float * c, size_t N)         \
{                                                                       \
    cudaError_t myerr;                                          \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,c,N);                              \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

// ---------------------- Some functions which know about x, and z position --------
// gets two sources and one destination, the two sources are assumed to have the same size
// sx,sy,sz : Source array sizes (total)
// sox,soy,soy : offsets
// ssx, ssy,ssz : source (or destination) subarray sizes
// dx,dy,dz: destination total array sizes
// dox,doy,doz : destination offsets


// Line below is used as an add-on to the 3d function below in case 3d assignment is needed
#define GET3DIDD size_t idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]);

#define CUDA_3DFkt(FktName,expressions)                                  \
__global__ void                                                         \
FktName(float *a, float *c, Size3D sSize,Size3D dSize,Size3D sOffs, Size3D sROI, Size3D dOffs) \
{                                                                       \
  size_t N=sROI.s[0]*sROI.s[1]*sROI.s[2];                                        \
  size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  size_t x=(idx)%sROI.s[0];                                                  \
  size_t y=(idx/sROI.s[0])%sROI.s[1];                                          \
  size_t z=(idx/(sROI.s[0]*sROI.s[1]))%sROI.s[2];                                \
  size_t ids=x+sOffs.s[0]+sSize.s[0]*(y+sOffs.s[1])+sSize.s[0]*sSize.s[1]*(z+sOffs.s[2]);                               \
  if(idx>=N) return;                                              \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * a, float *c, size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3])  \
{                                                                       \
    cudaError_t myerr;                                                \
    size_t N=sROI[0]*sROI[1]*sROI[2];                                      \
	size_t blockSize;dim3 nBlocks;                                         \
     Size3D sS,dS,sO,sR,dO;                                              \
    int d;                                                              \
    for (d=0;d<3;d++)                                                   \
        {sS.s[d]=sSize[d];dS.s[d]=dSize[d];sO.s[d]=sOffs[d];sR.s[d]=sROI[d];dO.s[d]=dOffs[d];} \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,c,sS,dS,sO,sR,dO); \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                     

// --- macros for sub-assigning a block with vectors in each dimension -----
#define CUDA_3DAsgFkt(FktName,expressions)                                  \
__global__ void                                                         \
FktName(float *c, float br, float bi, Size3D dSize, Size3D dROI, Size3D dOffs) \
{                                                                       \
  size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  size_t N=dROI.s[0]*dROI.s[1]*dROI.s[2];                                        \
  if(idx>=N) return;                                                    \
  size_t x=(idx)%dROI.s[0];                                               \
  size_t y=(idx/dROI.s[0])%dROI.s[1];                                    \
  size_t z=(idx/(dROI.s[0]*dROI.s[1]))%dROI.s[2];                       \
  size_t idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]);                               \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * c, float br, float bi, size_t dSize[3], size_t dROI[3], size_t dOffs[3])  \
{                                                                       \
    cudaError_t myerr;                                                \
    size_t N=dROI[0]*dROI[1]*dROI[2];                                      \
	size_t blockSize;dim3 nBlocks;                                         \
    Size3D dR,dS,dO;                                              \
    int d;                                                              \
    for (d=0;d<3;d++)                                                   \
        {dS.s[d]=dSize[d];dR.s[d]=dROI[d];dO.s[d]=dOffs[d];} \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(c,br,bi,dS,dR,dO); \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                     

// --- macros for sub-assigning a block with vectors in each dimension - Extended version to be suitable for repmat
#define CUDA_3DWrapAsgFkt(FktName,expressions)                          \
__global__ void                                                         \
FktName(float *a, float *c, Size3D dSize, Size3D sSize)       \
{                                                                       \
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  size_t N=dSize.s[0]*dSize.s[1]*dSize.s[2];                               \
  size_t x=(idd)%dSize.s[0];                                                \
  size_t y=(idd/dSize.s[0])%dSize.s[1];                                    \
  size_t z=(idd/(dSize.s[0]*dSize.s[1]))%dSize.s[2];                       \
  size_t ids=x%sSize.s[0]+sSize.s[0]*(y%sSize.s[1])+sSize.s[0]*sSize.s[1]*(z%sSize.s[2]); \
  if(idd>=N) return;                                                    \
  expressions                                                           \
}                                                                       \
extern "C" const char * CUDA ## FktName(float *a, float * c, size_t sSize[3], size_t dSize[3])  \
{                                                                       \
    cudaError_t myerr;                                                \
    size_t N=dSize[0]*dSize[1]*dSize[2];                                      \
	size_t blockSize;dim3 nBlocks;                                         \
    int d;                                                              \
    Size3D sS,dS;                                              \
    for (d=0;d<3;d++)                                                   \
        {dS.s[d]=dSize[d];sS.s[d]=sSize[d];} \
   MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,c,dS,sS); \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

//  Now the 5D Versions of the same code


#define GETXYZET(aSize,idx)                                             \
  size_t x=(idx)%aSize.s[0];                                          \
  size_t y=(idx/aSize.s[0])%aSize.s[1];                                    \
  size_t z=(idx/(aSize.s[0]*aSize.s[1]))%aSize.s[2];                       \
  size_t t=(idx/(aSize.s[0]*aSize.s[1]*aSize.s[2]))%aSize.s[3];            \
  size_t e=(idx/(aSize.s[0]*aSize.s[1]*aSize.s[2]*aSize.s[3]))%aSize.s[4]; \

#define GET5DIDS size_t ids=x*sStep.s[0]+sOffs.s[0]+sSize.s[0]*(y*sStep.s[1]+sOffs.s[1])+sSize.s[0]*sSize.s[1]*(z*sStep.s[2]+sOffs.s[2])+sSize.s[0]*sSize.s[1]*sSize.s[2]*(t*sStep.s[3]+sOffs.s[3])+sSize.s[0]*sSize.s[1]*sSize.s[2]*sSize.s[3]*(e*sStep.s[4]+sOffs.s[4]);   \

// Line below is used as an add-on to the 5d function below in case 5d assignment is needed
#define GET5DIDD_STEP size_t idd=x*dStep.s[0]+dOffs.s[0]+dSize.s[0]*(y*dStep.s[1]+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z*dStep.s[2]+dOffs.s[2])+dSize.s[0]*dSize.s[1]*dSize.s[2]*(t*dStep.s[3]+dOffs.s[3])+dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]*(e*dStep.s[4]+dOffs.s[4]);

#define GET5DIDD size_t idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])+dSize.s[0]*dSize.s[1]*dSize.s[2]*(t+dOffs.s[3])+dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]*(e+dOffs.s[4]);

#define CUDA_5DFkt(FktName,expressions)                                 \
__global__ void                                                         \
FktName(float *a, float *c, Size5D sSize,Size5D dSize,Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep) \
{                                                                     \
  size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  size_t N=sROI.s[0]*sROI.s[1]*sROI.s[2]*sROI.s[3]*sROI.s[4];            \
  GETXYZET(sROI,idx)                                                      \
  GET5DIDS;                                                               \
  if(idx>=N) return;                                                  \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * a, float *c, Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep)  \
{                                                                       \
    cudaError_t myerr;                                                \
    size_t N=sROI.s[0]*sROI.s[1]*sROI.s[2]*sROI.s[3]*sROI.s[4];                      \
	size_t blockSize;dim3 nBlocks;                                         \
    MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,c,sSize,dSize,sOffs,sROI,dOffs,sStep,dStep); \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      
                
// --- macros for sub-assigning a block with vectors in each dimension -----

#define CUDA_5DAsgFkt(FktName,expressions)                                  \
__global__ void                                                         \
FktName(float *c, float br, float bi, Size5D dSize, Size5D dROI, Size5D dOffs, Size5D dStep) \
{                                                                       \
  size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  size_t N=dROI.s[0]*dROI.s[1]*dROI.s[2]*dROI.s[3]*dROI.s[4];              \
  GETXYZET(dROI,idx)                                                      \
  GET5DIDD_STEP                                                                \
  if(idx>=N) return;                                                    \
   expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * c, float br, float bi, Size5D dSize, Size5D dROI, Size5D dOffs, Size5D dStep)  \
{                                                                       \
    cudaError_t myerr;                                                \
    size_t N=dROI.s[0]*dROI.s[1]*dROI.s[2]*dROI.s[3]*dROI.s[4];            \
	size_t blockSize;dim3 nBlocks;                                         \
   MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(c,br,bi,dSize,dROI,dOffs,dStep);       \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

// --- macros for sub-assigning a block with vectors in each dimension - Extended version to be suitable for repmat
#define CUDA_5DWrapAsgFkt(FktName,expressions)                          \
__global__ void                                                         \
FktName(float *a, float *c, Size5D dSize, Size5D sSize)       \
{                                                                       \
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);  \
  size_t N=dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]*dSize.s[4];         \
  GETXYZET(dSize,idd)                                                   \
  size_t ids=x%sSize.s[0]+sSize.s[0]*(y%sSize.s[1])+sSize.s[0]*sSize.s[1]*(z%sSize.s[2]) + sSize.s[0]*sSize.s[1]*sSize.s[2]*(t%sSize.s[3])+sSize.s[0]*sSize.s[1]*sSize.s[2]*sSize.s[3]*(e%sSize.s[4]); \
  if(idd>=N) return;                                                    \
  expressions                                                           \
}                                                                       \
extern "C" const char * CUDA ## FktName(float *a, float * c, size_t sSize[5], size_t dSize[5])  \
{                                                                       \
    cudaError_t myerr;                                                  \
    size_t N=dSize[0]*dSize[1]*dSize[2]*dSize[3]*dSize[4];                 \
	size_t blockSize;dim3 nBlocks;                                         \
    Size5D sS,dS;                                                       \
    int d;                                                              \
    for (d=0;d<5;d++)                                                   \
        {dS.s[d]=dSize[d];sS.s[d]=sSize[d];}                            \
   MemoryLayout(N,blockSize,nBlocks)                                     \
	FktName<<<nBlocks,blockSize>>>(a,c,dS,sS);                          \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      
        
// The function below checks whether the size of allocated reduce arrays is sufficient and reallocates if needed be
// The arrays are "accum" and "TmpRedArray"
const char * CheckReduceAllocation(size_t asize) {
    cudaError_t myerr;
    asize=((asize/MinRedBlockSize) + 1)*MinRedBlockSize;  // round it up to the nearest multiple of MinRedSize
    if (! accum){
       accum = (ACCU_ARRTYPE *) malloc(asize*sizeof(ACCU_ARRTYPE));
       if (! accum)
       return "CheckReduceAllocation: Malloc failed";
    }    
    if (! TmpRedArray) {
        cudaMalloc((void **) &TmpRedArray, asize*sizeof(ACCU_ARRTYPE));
        CurrentRedSize=asize;
        myerr=cudaGetLastError();
        if (myerr != cudaSuccess)
          return cudaGetErrorString(myerr);
    }
    
    if (asize > CurrentRedSize)
    {
        free(accum);
        accum = (ACCU_ARRTYPE *) malloc(asize*sizeof(ACCU_ARRTYPE));
        if (! accum)
            return "CheckReduceAllocation: ReMalloc failed";
        cudaFree(TmpRedArray);
        myerr=cudaGetLastError();
        if (myerr != cudaSuccess)
            return cudaGetErrorString(myerr);

        cudaMalloc((void **) &TmpRedArray, asize*sizeof(ACCU_ARRTYPE));
        myerr=cudaGetLastError();
        if (myerr != cudaSuccess)
            return cudaGetErrorString(myerr);
        CurrentRedSize=asize;
    }
    return 0;
}

extern "C" size_t GetCurrentRedSize(void) {
    return CurrentRedSize;
}

int GetMaxThreads(void) {
    return prop.maxThreadsPerBlock;
}

long GetMaxBlocksX(void) {
    // return min(prop.maxGridSize[0],65535);  // Why does it not work, if this is bigger than 65535 ??
    return prop.maxGridSize[0];  // Why does it not work, if this is bigger than 65535 ??
}

cudaDeviceProp GetDeviceProp(void) {
    return prop;
}

/*__global__ void                                                         \
bla_ ## FktName(float*a, float * c, int N,  Size3D sSize,Size3D dSize,Size3D sOffs, Size3D sROI, Size3D dOffs) {                                    \
  int idx=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int idcd=0,idcs=0,ids=0;                                                    \
  if(idx>=N) return;                                                    \
    expression                                                          \
}   \  */

//	FktName<<<nBlocks,blockSize>>>(a,c,sSize,dSize,sOffs, sROI, dOffs); \


CUDA_FullRed(sumpos_arr,mysumpos,mysum)  // only sums over the number of positive values
CUDA_FullRed(sum_arr,mysum,mysum)
//CUDA_FullRedBin(sum_arr,mysum)
CUDA_FullRedCpx(sum_carr,mysum)
// CUDA_FullRed(sum_carr,res+=accum[ii];)
CUDA_FullRedIdx(max_arr,maxCond)
CUDA_FullRedIdx(min_arr,minCond)

CUDA_PartRedMask(psum_arr,mysum)
CUDA_PartRedMaskCpx(psum_carr,mysum)
CUDA_PartRedMaskIdx(pmax_arr,maxCond)
CUDA_PartRedMaskIdx(pmin_arr,minCond)

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
// CUDA_3DFkt(arr_subcpy_arr,c[idd]=a[ids];)
CUDA_3DAsgFkt(const_3dsubcpy_arr,c[idd]=br;)
CUDA_3DAsgFkt(cconst_3dsubcpy_carr,c[2*idd]=br;c[2*idd+1]=bi;)

// repcopy for repmat command
CUDA_3DWrapAsgFkt(arr_3drepcpy_arr,c[idd]=a[ids];)
CUDA_3DWrapAsgFkt(crepcpy_carr,c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign

CUDA_5DAsgFkt(const_5dsubcpy_arr,c[idd]=br;)
CUDA_5DAsgFkt(cconst_5dsubcpy_carr,c[2*idd]=br;c[2*idd+1]=bi;)

// repcopy for repmat command
CUDA_5DWrapAsgFkt(arr_5drepcpy_arr,c[idd]=a[ids];)
CUDA_5DWrapAsgFkt(carr_5drepcpy_carr,c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)

// Assigning constant values to arrays accessed with a boolean array
CUDA_UnaryFktConst(arr_boolassign_const,if (a[idx]!=0) c[idx]=b;)

CUDA_UnaryFktConstC(carr_boolassign_const,if (a[idx]!=0) {c[2*idx]=br;c[2*idx+1]=bi;})

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
// CUDA_3DFkt(arr_subcpy_arr,c[idd]=a[ids];)
CUDA_3DFkt(subcpy_arr, GET3DIDD; c[idd]=a[ids];)
CUDA_3DFkt(carr_3dsubcpy_carr, GET3DIDD; c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)
CUDA_3DFkt(arr_3dsubcpy_carr, GET3DIDD; c[2*idd]=a[ids];c[2*idd+1]=0;)

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
// These versions intoduce a transpose operation
CUDA_3DFkt(arr_3dsubcpyT_arr,  size_t iddt=y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]); c[iddt]=a[ids];)
CUDA_3DFkt(carr_3dsubcpyT_carr,size_t idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=a[2*ids+1];)
// with conjugation
CUDA_3DFkt(carr_3dsubcpyCT_carr,size_t idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=-a[2*ids+1];)

//CUDA_3DFkt(arr_subref_arr3d,c[idd]=)
//getCudaRef(prhs[1]),newarr,sSize,dSize,cuda_array[newref[0]],cuda_array[newref[1]],cuda_array[newref[2]]);

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
CUDA_5DFkt(arr_5dsubcpy_arr, GET5DIDD_STEP; c[idd]=a[ids];)
CUDA_5DFkt(carr_5dsubcpy_carr, GET5DIDD_STEP; c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)
CUDA_5DFkt(arr_5dsubcpy_carr, GET5DIDD_STEP; c[2*idd]=a[ids];c[2*idd+1]=0;)

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
// These versions intoduce a transpose operation
CUDA_5DFkt(arr_5dsubcpyT_arr,  size_t iddt=y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]); c[iddt]=a[ids];)
CUDA_5DFkt(carr_5dsubcpyT_carr,size_t idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=a[2*ids+1];)
 // with conjugation
CUDA_5DFkt(carr_5dsubcpyCT_carr,size_t idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=-a[2*ids+1];)  


// Power
CUDA_BinaryFkt(arr_power_arr,c[idx]=pow(a[idxA],b[idxB]);)
CUDA_UnaryFktConst(arr_power_const,c[idx]=pow(a[idx],b);)
CUDA_UnaryFktConst(const_power_arr,c[idx]=pow(b,a[idx]);)

// Multiplications
CUDA_BinaryFkt(arr_times_arr,c[idx]=a[idxA]*b[idxB];)
CUDA_BinaryFkt(carr_times_carr,
    size_t idc=2*idx;size_t idcA=2*idxA;size_t idcB=2*idxB;
    float myr=a[idcA]*b[idcB]-a[idcA+1]*b[idcB+1];float myi=a[idcA]*b[idcB+1]+a[idcA+1]*b[idcB];
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_BinaryFkt(arr_times_carr,size_t idc=2*idx;size_t idcB=2*idxB;c[idc]=a[idxA]*b[idcB];c[idc+1]=a[idxA]*b[idcB+1];)
CUDA_BinaryFkt(carr_times_arr,size_t idc=2*idx;size_t idcA=2*idxA;c[idc]=a[idcA]*b[idxB];c[idc+1]=a[idcA+1]*b[idxB];)
//CUDA_BinaryFkt(arr_times_carr,c[2*idx]=a[idx]*b[2*idx];c[2*idx+1]=a[idx+1]*b[2*idx];)
CUDA_UnaryFktConst(arr_times_const,c[idx]=a[idx]*b;)
CUDA_UnaryFktConst(const_times_arr,c[idx]=a[idx]*b;)
CUDA_UnaryFktConstC(carr_times_const,
    size_t idc=2*idx;
    float myr=a[idc]*br-a[idc+1]*bi;float myi=a[idc]*bi+a[idc+1]*br;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(const_times_carr,
    size_t idc=2*idx;
    float myr=a[idc]*br-a[idc+1]*bi;float myi=a[idc]*bi+a[idc+1]*br;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(arr_times_Cconst,c[2*idx]=a[idx]*br;c[2*idx+1]=a[idx]*bi;)
CUDA_UnaryFktConstC(Cconst_times_arr,c[2*idx]=br*a[idx];c[2*idx+1]=bi*a[idx];)

// Divisions
CUDA_BinaryFkt(arr_divide_arr,c[idx]=a[idxA]/b[idxB];)
CUDA_BinaryFkt(carr_divide_carr,
    size_t idc=2*idx;size_t idcA=2*idxA;size_t idcB=2*idxB;
    float tmp=b[idcB]*b[idcB]+b[idcB+1]*b[idcB+1];
    float myr=(a[idcA]*b[idcB]+a[idcA+1]*b[idcB+1])/tmp;float myi=(a[idcA+1]*b[idcB]-a[idcA]*b[idcB+1])/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_BinaryFkt(carr_divide_arr,size_t idc=2*idx;size_t idcA=2*idxA; c[idc]=a[idcA]/b[idxB];c[idc+1]=a[idcA+1]/b[idxB];)
CUDA_BinaryFkt(arr_divide_carr,
    size_t idc=2*idx;size_t idcB=2*idxB;
    float tmp=b[idcB]*b[idcB]+b[idcB+1]*b[idcB+1];
    float myr=(a[idxA]*b[idcB]+a[idxA+1]*b[idcB+1])/tmp;float myi=(a[idxA+1]*b[idcB]-a[idxA]*b[idcB+1])/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConst(arr_divide_const,c[idx]=a[idx]/b;)
CUDA_UnaryFktConst(const_divide_arr,c[idx]=b/a[idx];)
CUDA_UnaryFktConstC(carr_divide_const,
    size_t idc=2*idx;
    float tmp=br*br+bi*bi;
    float myr=(a[idc]*br+a[idc+1]*bi)/tmp;float myi=(a[idc+1]*br-a[idc]*bi)/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(const_divide_carr,
    size_t idc=2*idx;
    float tmp=a[idc]*a[idc]+a[idc+1]*a[idc+1];
    float myr=(br*a[idc]+bi*a[idc+1])/tmp;float myi=(bi*a[idc]-br*a[idc+1])/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(arr_divide_Cconst,
    float tmp=br*br+bi*bi;
    float myr=a[idx]*br/tmp;float myi= -a[idx]*bi/tmp;
    c[2*idx]=myr;c[2*idx+1]=myi;
)
CUDA_UnaryFktConstC(Cconst_divide_arr,c[2*idx]=br/a[idx];c[2*idx+1]=bi/a[idx];)

// Element-wise maximum operations
CUDA_BinaryFkt(arr_max_arr,c[idx]=a[idxA]>b[idxB]?a[idxA]:b[idxB];)
CUDA_BinaryFkt(carr_max_carr, size_t idc=2*idx;size_t idcA=2*idxA;size_t idcB=2*idxB; if (a[idcA]*a[idcA]+a[idcA+1]*a[idcA+1] > b[idcB]*b[idcB]+b[idcB+1]*b[idcB+1]) {c[idc]=a[idcA];c[idc+1]=a[idcA+1];}else{ c[idc]=b[idcB];c[idc+1]=b[idcB+1];})
CUDA_BinaryFkt(carr_max_arr,size_t idc=2*idx;size_t idcA=2*idxA; if (a[idcA]*a[idcA]+a[idcA+1]*a[idcA+1] > b[idxB]*b[idxB]) {c[idc]=a[idcA];c[idc+1]=a[idcA+1];}else{ c[idc]=b[idxB];c[idc+1]=0;})
CUDA_BinaryFkt(arr_max_carr,size_t idc=2*idx;size_t idcB=2*idxB; if (a[idxA]*a[idxA] > b[idcB]*b[idcB]+b[idcB+1]*b[idcB+1]) {c[idc]=a[idxA];c[idc+1]=0;}else{ c[idc]=b[idcB];c[idc+1]=b[idcB+1];})
CUDA_UnaryFktConst(arr_max_const,c[idx]=a[idx]>b?a[idx]:b;)
CUDA_UnaryFktConst(const_max_arr,c[idx]=a[idx]>b?a[idx]:b;)
CUDA_UnaryFktConstC(carr_max_const,size_t idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] > br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(const_max_carr,size_t idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] > br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(arr_max_Cconst,size_t idc=2*idx;if (a[idx]*a[idx] > br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(Cconst_max_arr,size_t idc=2*idx;if (a[idx]*a[idx] > br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})

// Element-wise minimum operations
CUDA_BinaryFkt(arr_min_arr,c[idx]=a[idxA]<b[idxB]?a[idxA]:b[idxB];)
CUDA_BinaryFkt(carr_min_carr, size_t idc=2*idx;size_t idcA=2*idxA;size_t idcB=2*idxB; if (a[idcA]*a[idcA]+a[idcA+1]*a[idcA+1] < b[idcB]*b[idcB]+b[idcB+1]*b[idcB+1]) {c[idc]=a[idcA];c[idc+1]=a[idcA+1];}else{ c[idc]=b[idcB];c[idc+1]=b[idcB+1];})
CUDA_BinaryFkt(carr_min_arr,size_t idc=2*idx;size_t idcA=2*idxA; if (a[idcA]*a[idcA]+a[idcA+1]*a[idcA+1] < b[idxB]*b[idxB]) {c[idc]=a[idcA];c[idc+1]=a[idcA+1];}else{ c[idc]=b[idxB];c[idc+1]=0;})
CUDA_BinaryFkt(arr_min_carr,size_t idc=2*idx;size_t idcB=2*idxB; if (a[idxA]*a[idxA] < b[idcB]*b[idcB]+b[idcB+1]*b[idcB+1]) {c[idc]=a[idxA];c[idc+1]=0;}else{ c[idc]=b[idcB];c[idc+1]=b[idcB+1];})
CUDA_UnaryFktConst(arr_min_const,c[idx]=a[idx]<b?a[idx]:b;)
CUDA_UnaryFktConst(const_min_arr,c[idx]=a[idx]<b?a[idx]:b;)
CUDA_UnaryFktConstC(carr_min_const,size_t idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] < br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(const_min_carr,size_t idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] < br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(arr_min_Cconst,size_t idc=2*idx;if (a[idx]*a[idx] < br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(Cconst_min_arr,size_t idc=2*idx;if (a[idx]*a[idx] < br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})

// Additions
CUDA_BinaryFkt(arr_plus_arr,c[idx]=a[idxA]+b[idxB];)
CUDA_BinaryFkt(carr_plus_carr, size_t idc=2*idx;size_t idcA=2*idxA;size_t idcB=2*idxB; c[idc]=a[idcA]+b[idcB];c[idc+1]=a[idcA+1]+b[idcB+1];)
CUDA_BinaryFkt(carr_plus_arr,size_t idc=2*idx;size_t idcA=2*idxA;c[idc]=a[idcA]+b[idxB];c[idc+1]=a[idcA+1];)
CUDA_BinaryFkt(arr_plus_carr,size_t idc=2*idx;size_t idcB=2*idxB;c[idc]=a[idxA]+b[idcB];c[idc+1]=b[idcB+1];)
CUDA_UnaryFktConst(arr_plus_const,c[idx]=a[idx]+b;)
CUDA_UnaryFktConst(const_plus_arr,c[idx]=a[idx]+b;)
CUDA_UnaryFktConstC(carr_plus_const,size_t idc=2*idx;c[idc]=a[idc]+br;c[idc+1]=a[idc+1]+bi;)
CUDA_UnaryFktConstC(const_plus_carr,size_t idc=2*idx;c[idc]=a[idc]+br;c[idc+1]=a[idc+1]+bi;)
CUDA_UnaryFktConstC(arr_plus_Cconst,size_t idc=2*idx;c[idc]=a[idx]+br;c[idc+1]=bi;)
CUDA_UnaryFktConstC(Cconst_plus_arr,size_t idc=2*idx;c[idc]=br+a[idx];c[idc+1]=bi;)

// Subtractions
CUDA_BinaryFkt(arr_minus_arr,c[idx]=a[idxA]-b[idxB];)
CUDA_BinaryFkt(carr_minus_carr,size_t idc=2*idx;size_t idcA=2*idxA;size_t idcB=2*idxB; c[idc]=a[idcA]-b[idcB];c[idc+1]=a[idcA+1]-b[idcB+1];)
CUDA_BinaryFkt(carr_minus_arr,size_t idc=2*idx;size_t idcA=2*idxA;c[idc]=a[idcA]-b[idxB];c[idc+1]=a[idcA+1];)
CUDA_BinaryFkt(arr_minus_carr,size_t idc=2*idx;size_t idcB=2*idxB;c[idc]=a[idxA]-b[idcB];c[idc+1]=-b[idcB+1];)
CUDA_UnaryFktConst(arr_minus_const,c[idx]=a[idx]-b;)
CUDA_UnaryFktConst(const_minus_arr,c[idx]=b-a[idx];)
CUDA_UnaryFktConstC(carr_minus_const,size_t idc=2*idx;c[idc]=a[idc]-br;c[idc+1]=a[idc+1]-bi;)
CUDA_UnaryFktConstC(const_minus_carr,size_t idc=2*idx;c[idc]=br-a[idc];c[idc+1]=bi-a[idc+1];)
CUDA_UnaryFktConstC(arr_minus_Cconst,size_t idc=2*idx;c[idc]=a[idx]-br;c[idc+1]=-bi;)
CUDA_UnaryFktConstC(Cconst_minus_arr,size_t idc=2*idx;c[idc]=br-a[idx];c[idc+1]=bi;)

// Referencing and assignment  // STILL NEEDS SOME WORK
// CUDA_BinaryFkt(arr_subsref_arr,c[idx]=(b[idx] == 0) ? 0 : a[idx];)
// CUDA_BinaryFkt(carr_subsref_arr,c[idc]=(b[idx] == 0) ? 0 : a[idc]; c[idc+1]=(b[idx] == 0) ? 0 : a[idc+1];)
// CUDA_BinaryFkt(arr_subsasgn_arr,if (b[idx] == 0) c[idx] = a[idx];)
// CUDA_BinaryFkt(carr_subsasgn_arr,if (b[idx] == 0) {c[idc] = a[idc];c[idc+1] = a[idc+1];})
CUDA_MaskIdx(arr_subsref_arr,c[mask_idx]=a[idx];)
CUDA_MaskIdx(carr_subsref_arr,c[2*mask_idx]=a[2*idx]; c[2*mask_idx+1]=a[2*idx+1];)
CUDA_MaskIdx(arr_subsasgn_arr,a[idx]=c[mask_idx];)
CUDA_MaskIdx(carr_subsasgn_arr,a[2*idx]=c[2*mask_idx]; a[2*idx+1]=c[2*mask_idx+1];)

// diagonal matrix generation
CUDA_3DFkt(arr_diag_set,  size_t iddt=ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1]); c[iddt]=a[ids];)
CUDA_3DFkt(carr_diag_set,  size_t idcdt=2*(ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1])); c[idcdt]=a[2*ids];c[idcdt+1]=a[2*ids*1];)
CUDA_3DFkt(arr_diag_get,  size_t iddt=ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1]); a[ids]=c[iddt];)
CUDA_3DFkt(carr_diag_get,  size_t idcdt=2*(ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1])); a[2*ids]=c[idcdt];a[2*ids*1]=c[idcdt+1];)

// referencing and assignment with index vectors.No Index checking performed
// The code below "misuses" the CUDA_BinaryFkt macro to subreference or sub-assign from index lists
CUDA_BinaryFkt(arr_subsref_vec,{c[idx]=a[(size_t) b[idx]];})
CUDA_BinaryFkt(carr_subsref_vec,{c[2*idx]=a[2*((size_t) b[idx])];c[2*idx+1]=a[2*((size_t) b[idx])+1];})

CUDA_BinaryFkt(arr_subsasg_vec,{c[(size_t) b[idx]]=a[idx];})
CUDA_BinaryFkt(carr_subsasg_vec,{c[2*((size_t) b[idx])]=a[2*idx];c[2*((size_t) b[idx])+1]=a[2*idx+1];})

// one-D index operations. Note that the order is changed to remain compatible in the allocation of array c
// The NAN are needed to generate NaNs for wrong accesses.
//CUDA_IndexFkt(arr_subsref_ind,{if ((idx<M)&&(idx>=0)) {size_t myind=(size_t) b[idx];((myind<N)&&(myind>=0))?(c[idx]=a[myind]):c[idx]=NAN;} else c[idx]=NAN;})
//CUDA_IndexFkt(carr_subsref_ind,{if ((idx<M)&&(idx>=0)) {size_t myindC=2*(size_t) b[idx];((myindC<2*N)&&(myindC>=0))?(c[2*idx]=a[myindC],c[2*idx+1]=a[myindC+1]):(c[2*idx]=NAN,c[2*idx+1]=NAN);} else {c[2*idx]=NAN;c[2*idx+1]=NAN;}})

// The function below accepts a 2D index matrix (b) where each row is a list of indices corresponding to this dimension. The size of this matrix should have been adapted to the longest index list.

//CUDA_IndexFktND(arr_subsrefND_ind,CoordsNDFromIdx(idx,sSize,pos);for(int _d=0;_d<CUDA_MAXDIM;_d++){if ((idx<M)) {size_t myind=(size_t) b[idx];((myind<N))?(c[idx]=a[myind]):c[idx]=NAN;} else c[idx]=NAN;})

// CUDA_UnaryFktIntVec(arr_circshift_vec,CoordsNDFromIdx(idx,sSize,pos);for(int _d=0;_d<CUDA_MAXDIM;_d++){pos.s[_d]-=b.s[_d];}long long ids=0;IdxNDFromCoords(pos,sSize,ids);c[idx]=a[ids];)  // a[idx]


CUDA_IndexFkt(arr_subsref_ind,{if ((idx<M)) {size_t myind=(size_t) b[idx];((myind<N))?(c[idx]=a[myind]):c[idx]=NAN;} else c[idx]=NAN;})
//CUDA_IndexFkt(carr_subsref_ind,{if ((idx<M)&&(idx>=0)) {size_t myindC=2*(size_t) b[idx];((myindC<2*N)&&(myindC>=0))?(c[2*idx]=a[myindC],c[2*idx+1]=a[myindC+1]):(c[2*idx]=NAN,c[2*idx+1]=NAN);} else {c[2*idx]=NAN;c[2*idx+1]=NAN;}})
CUDA_IndexFkt(carr_subsref_ind,{if ((idx<M)) {size_t myindC=2*(size_t) b[idx];((myindC<2*N))?(c[2*idx]=a[myindC],c[2*idx+1]=a[myindC+1]):(c[2*idx]=NAN,c[2*idx+1]=NAN);} else {c[2*idx]=NAN;c[2*idx+1]=NAN;}})

CUDA_IndexFkt(arr_subsasgn_ind,{size_t myind=(size_t) b[idx]; if ((idx<M)) {((myind<N))?(c[myind]=a[idx]):0;} else c[myind]=NAN;})
CUDA_IndexFkt(carr_subsasgn_ind,{size_t myindC=2*(size_t) b[idx]; if ((idx<M)) {((myindC<2*N))?(c[myindC]=a[2*idx],c[myindC+1]=a[2*idx+1]):0;} else {c[idx]=NAN;c[2*idx+1]=NAN;}})

CUDA_UnaryFktConst(arr_subsasgn_const,{((idx<N))?(c[(size_t) a[idx]]=b):0;})
CUDA_UnaryFktConstC(arr_subsasgn_Cconst,{((idx<N))?(c[(size_t) a[idx]]=br):0;})  // This should not happen. If it does only the real part is kept.
CUDA_UnaryFktConstC(carr_subsasgn_const,{((idx<N))?(c[2*((size_t) a[idx])]=br,c[2*((size_t) a[idx])+1])=bi:0;})

// binary logical operations

CUDA_BinaryFkt(arr_or_arr,{c[idx]=(float) (a[idxA]!=0) || (b[idxB]!=0);})
CUDA_UnaryFktConst(arr_or_const,{c[idx]=(float) (a[idx]!=0) || (b!=0);})
CUDA_UnaryFktConst(const_or_arr,{c[idx]=(float) (b!=0) || (a[idx]!=0);})

CUDA_BinaryFkt(arr_and_arr,{c[idx]=(float) (a[idxA]!=0) && (b[idxB]!=0);})
CUDA_UnaryFktConst(arr_and_const,{c[idx]=(float) (a[idx]!=0) && (b!=0);})
CUDA_UnaryFktConst(const_and_arr,{c[idx]=(float) (b!=0) && (a[idx]!=0);})

// Unary logical operations
CUDA_UnaryFkt(not_arr,c[idx]=(a[idx] == 0);)

// Unary sign operation
CUDA_UnaryFkt(sign_arr,c[idx]=sign(a[idx]);)  // (a[idx] > 0)?1 :((a[idx]<0)?-1:0);
CUDA_UnaryFkt(sign_carr,size_t idc=2*idx; float absc=sqrt(a[idc]*a[idc]+a[idc+1]*a[idc+1]); if (absc==0) {c[idc]=0;c[idc+1]=0;} else {c[idc]=a[idc]/absc;c[idc+1]=a[idc+1]/absc;})

// Comparison
CUDA_BinaryFkt(arr_smaller_arr,c[idx]=a[idxA]<b[idxB];)
CUDA_UnaryFktConst(arr_smaller_const,c[idx]=a[idx]<b;)
CUDA_UnaryFktConst(const_smaller_arr,c[idx]=b<a[idx];)

CUDA_BinaryFkt(arr_larger_arr,c[idx]=a[idxA]>b[idxB];)
CUDA_UnaryFktConst(arr_larger_const,c[idx]=a[idx]>b;)
CUDA_UnaryFktConst(const_larger_arr,c[idx]=b>a[idx];)

CUDA_BinaryFkt(arr_smallerequal_arr,c[idx]=a[idxA]<=b[idxB];)
CUDA_UnaryFktConst(arr_smallerequal_const,c[idx]=a[idx]<=b;)
CUDA_UnaryFktConst(const_smallerequal_arr,c[idx]=b<=a[idx];)

CUDA_BinaryFkt(arr_largerequal_arr,c[idx]=a[idxA]>=b[idxB];)
CUDA_UnaryFktConst(arr_largerequal_const,c[idx]=a[idx]>=b;)
CUDA_UnaryFktConst(const_largerequal_arr,c[idx]=b>=a[idx];)

// equals will always output a real valued array
CUDA_BinaryFkt(arr_equals_arr,c[idx]=(a[idxA]==b[idxB]);)
CUDA_BinaryFkt(carr_equals_carr, size_t idcA=2*idxA;size_t idcB=2*idxB; c[idx]=(a[idcA]==b[idcB]) && (a[idcA+1]==b[idcB+1]);)
CUDA_BinaryFkt(carr_equals_arr,size_t idcA=2*idxA; c[idx]=(a[idcA]==b[idxB]) && (a[idcA+1] == 0);)
CUDA_BinaryFkt(arr_equals_carr,size_t idcB=2*idxB; c[idx]=(a[idxA]==b[idcB]) && (b[idcB+1] == 0);)
CUDA_UnaryFktConst(arr_equals_const,c[idx]=(a[idx]==b);)
CUDA_UnaryFktConst(const_equals_arr,c[idx]=(b==a[idx]);)
CUDA_UnaryFktConstC(carr_equals_const,size_t idc=2*idx; c[idx]=(a[idc]==br) && (a[idc+1]==bi);)
CUDA_UnaryFktConstC(const_equals_carr,size_t idc=2*idx; c[idx]=(br==a[idc]) && (bi==a[idc+1]);)
CUDA_UnaryFktConstC(arr_equals_Cconst,c[idx]=(a[idx]==br) && (bi==0);)
CUDA_UnaryFktConstC(Cconst_equals_arr,c[idx]=(br==a[idx]) && (bi==0);)

// not equals will always output a real valued array
CUDA_BinaryFkt(arr_unequals_arr,c[idx]=(a[idxA]!=b[idxB]);)
CUDA_BinaryFkt(carr_unequals_carr, size_t idcA=2*idxA;size_t idcB=2*idxB; c[idx]=(a[idcA]!=b[idcB]) || (a[idcA+1]!=b[idcB+1]);)
CUDA_BinaryFkt(carr_unequals_arr, size_t idcA=2*idxA;c[idx]=(a[idcA]!=b[idxB]) || (a[idcA+1] != 0);)
CUDA_BinaryFkt(arr_unequals_carr,size_t idcB=2*idxB;c[idx]=(a[idxA]!=b[idcB]) || (b[idcB+1] != 0);)
CUDA_UnaryFktConst(arr_unequals_const,c[idx]=(a[idx]!=b);)
CUDA_UnaryFktConst(const_unequals_arr,c[idx]=(b!=a[idx]);)
CUDA_UnaryFktConstC(carr_unequals_const,c[idx]=(a[2*idx]!=br) || (a[2*idx+1]!=bi);)
CUDA_UnaryFktConstC(const_unequals_carr,c[idx]=(br!=a[2*idx]) || (bi!=a[2*idx+1]);)
CUDA_UnaryFktConstC(arr_unequals_Cconst,c[idx]=(a[idx]!=br) || (bi!=0);)
CUDA_UnaryFktConstC(Cconst_unequals_arr,c[idx]=(br!=a[idx]) || (bi!=0);)

// other Unary oparations
CUDA_UnaryFkt(uminus_arr,c[idx]=-a[idx];)
CUDA_UnaryFkt(uminus_carr,size_t idc=2*idx; c[idc]=-a[idc];c[idc+1]=-a[idc+1];)   // negates real and imaginary part

CUDA_UnaryFkt(round_arr,c[idx]=round(a[idx]);)
CUDA_UnaryFkt(round_carr,size_t idc=2*idx; c[idc]=round(a[idc]);c[idc+1]=round(a[idc+1]);)   // negates real and imaginary part

CUDA_UnaryFkt(floor_arr,c[idx]=floor(a[idx]);)
CUDA_UnaryFkt(floor_carr,size_t idc=2*idx; c[idc]=floor(a[idc]);c[idc+1]=floor(a[idc+1]);)   // negates real and imaginary part

CUDA_UnaryFkt(ceil_arr,c[idx]=ceil(a[idx]);)
CUDA_UnaryFkt(ceil_carr,size_t idc=2*idx; c[idc]=ceil(a[idc]);c[idc+1]=ceil(a[idc+1]);)   // negates real and imaginary part

CUDA_UnaryFkt(exp_arr,c[idx]= exp(a[idx]);)
CUDA_UnaryFkt(exp_carr,size_t idc=2*idx; float len=exp(a[idc]);c[idc]=len*cos(a[idc+1]);c[idc+1]=len*sin(a[idc+1]);)

CUDA_UnaryFkt(sin_arr,c[idx]= sin(a[idx]);)
CUDA_UnaryFkt(sin_carr,size_t idc=2*idx; c[idc]=sin(a[idc])*cosh(a[idc+1]);c[idc+1]=cos(a[idc])*sinh(a[idc+1]);)

CUDA_UnaryFkt(cos_arr,c[idx]= cos(a[idx]);)
CUDA_UnaryFkt(cos_carr,size_t idc=2*idx; c[idc]=cos(a[idc])*cosh(a[idc+1]);c[idc+1]=sin(a[idc])*sinh(a[idc+1]);)

CUDA_UnaryFkt(tan_arr,c[idx]= tan(a[idx]);)

CUDA_UnaryFkt(sinh_arr,c[idx]= sinh(a[idx]);)
CUDA_UnaryFkt(sinh_carr,size_t idc=2*idx; c[idc]=sinh(a[idc])*cos(a[idc+1]);c[idc+1]=cosh(a[idc])*sin(a[idc+1]);)

CUDA_UnaryFkt(cosh_arr,c[idx]= cosh(a[idx]);)
CUDA_UnaryFkt(cosh_carr,size_t idc=2*idx; c[idc]=cosh(a[idc])*cos(a[idc+1]);c[idc+1]=sinh(a[idc])*sin(a[idc+1]);)

CUDA_UnaryFkt(sinc_arr, c[idx]= (a[idx] != 0) ? sin(a[idx])/a[idx] : 1.0;)
CUDA_UnaryFkt(sinc_carr,size_t idc=2*idx; c[idc]=0;c[idc+1]=0;) 
// c[idc]= (a[idc] == 0) ? sin(a[idc])*cosh(a[idc+1])/a[idc] : cosh(a[idc+1]);c[idc+1]= (a[idc] == 0) ? cos(a[idc])*sinh(a[idc+1])/a[idc] : sinh(a[idc+1]);)

// besselj, but order will be integer only:
CUDA_BinaryFkt(arr_besselj_arr,{c[idx]=jnf(size_t(a[idxA]),b[idxB]);})
CUDA_UnaryFktConst(arr_besselj_const,{c[idx]=jnf(size_t(a[idx]),b);})
CUDA_UnaryFktConst(const_besselj_arr,{c[idx]=jnf(size_t(b),a[idx]);})

// atan2 only for real inputs
CUDA_BinaryFkt(arr_atan2_arr,{c[idx]=atan2(b[idx],a[idx]);})
CUDA_UnaryFktConst(arr_atan2_const,{c[idx]=atan2(a[idx],b);})
CUDA_UnaryFktConst(const_atan2_arr,{c[idx]=atan2(b,a[idx]);})

CUDA_UnaryFkt(log_arr,c[idx]=log(a[idx]);)
CUDA_UnaryFkt(log_carr,c[2*idx]=log(a[2*idx]);c[2*idx+1]=0;)   //  not implemented

CUDA_UnaryFkt(abs_arr,c[idx]= (a[idx] > 0) ? a[idx] : -a[idx];)
CUDA_UnaryFkt(abs_carr,size_t idc=2*idx; c[idx]=sqrt(a[idc]*a[idc]+a[idc+1]*a[idc+1]);)

CUDA_UnaryFkt(conj_arr,c[idx]=a[idx];)
CUDA_UnaryFkt(conj_carr,size_t idc=2*idx; c[idc]=a[idc];c[idc+1]=-a[idc+1];)  // only affects the imaginary part

CUDA_UnaryFkt(sqrt_arr,c[idx]= sqrt(a[idx]);)
// funny expression below is the sign function ((x>0)-(x<0))
CUDA_UnaryFkt(sqrt_carr,size_t idc=2*idx; float L=sqrt(a[idc]*a[idc]+a[idc+1]*a[idc+1]); c[idc]=sqrt((L+a[idc])/2);c[idc+1]=((a[idc+1]>0)-(a[idc+1])<0)*sqrt((L-a[idc])/2);)

// Unary functions resulting in just a single value
CUDA_UnaryFkt(isIllegal_arr,if (isnan(a[idx]) || isinf(a[idx]) ) c[0]=1;)
CUDA_UnaryFkt(isIllegal_carr,if (a[2*idx+1]!=0 || isnan(a[2*idx]) || isnan(a[2*idx+1]) || isinf(a[2*idx]) || isinf(a[2*idx+1]) ) c[0]=1;)

CUDA_UnaryFkt(any_arr,if (a[idx]!=0) c[0]=1;)
CUDA_UnaryFkt(any_carr,if (a[2*idx]!=0 || a[2*idx+1]!=0) c[0]=1;)

// Binary functions with real valued input returning always complex arrays
CUDA_BinaryFkt(arr_complex_arr,c[2*idx]=a[idxA];c[2*idx+1]=b[idxB];)
CUDA_UnaryFktConst(arr_complex_const,c[2*idx]=a[idx];c[2*idx+1]=b;)
CUDA_UnaryFktConst(const_complex_arr,c[2*idx]=b;c[2*idx+1]=a[idx];)

// unary functions returning always real valued arrays

CUDA_UnaryFkt(real_arr,c[idx]=a[idx];)
CUDA_UnaryFkt(real_carr,c[idx]=a[2*idx];)

CUDA_UnaryFkt(imag_arr,c[idx]=0;)
CUDA_UnaryFkt(imag_carr,c[idx]=a[2*idx+1];)

CUDA_UnaryFkt(phase_arr,c[idx]=0;)
CUDA_UnaryFkt(phase_carr,c[idx]=atan2(a[2*idx+1],a[2*idx]);)

CUDA_UnaryFkt(isnan_arr,c[idx]=(float) isnan(a[idx]);)
CUDA_UnaryFkt(isnan_carr,c[idx]=(float) (isnan(a[2*idx])||isnan(a[2*idx+1]));)   // is not a number

CUDA_UnaryFkt(isinf_arr,c[idx]=(float) isinf(a[idx]);)
CUDA_UnaryFkt(isinf_carr,c[idx]=(float) (isinf(a[2*idx])||isinf(a[2*idx+1]));)   // is infinite

CUDA_UnaryFktIntVec(arr_circshift_vec,CoordsNDFromIdx(idx,sSize,pos);for(int _d=0;_d<CUDA_MAXDIM;_d++){pos.s[_d]-=b.s[_d];}long long ids=0;IdxNDFromCoords(pos,sSize,ids);c[idx]=a[ids];)  // a[idx]
CUDA_UnaryFktIntVec(carr_circshift_vec,CoordsNDFromIdx(idx,sSize,pos);for(int _d=0;_d<CUDA_MAXDIM;_d++){pos.s[_d]-=b.s[_d];}long long ids=0;IdxNDFromCoords(pos,sSize,ids);c[2*idx]=a[2*ids];c[2*idx+1]=a[2*ids+1];)

// In code below, the loop runs over the source dimensions. The array sizes are still set to the source sizes and will be (again) adjusted later
CUDA_UnaryFktIntVec(arr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos);
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=0;}
        for(_d=0;_d<CUDA_MAXDIM;_d++){
                if (b.s[_d]<CUDA_MAXDIM) {
                        dSize.s[_d]=sSize.s[b.s[_d]]; posnew.s[_d] = pos.s[b.s[_d]];}
                } 
        size_t idd=0;IdxNDFromCoords(posnew,dSize,idd);c[idd]=a[idx];}) // a[idx]

CUDA_UnaryFktIntVec(carr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos);
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=0;}
        for(_d=0;_d<CUDA_MAXDIM;_d++){
                if (b.s[_d]<CUDA_MAXDIM)  {
                        dSize.s[_d]=sSize.s[b.s[_d]]; posnew.s[_d] = pos.s[b.s[_d]];}
                }
        size_t idd=0;IdxNDFromCoords(posnew,dSize,idd);c[2*idd]=a[2*idx];c[2*idd+1]=a[2*idx+1];}) 
/*
CUDA_UnaryFktIntVec(arr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos); \
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=pos.s[_d];} \
        for(_d=0;_d<CUDA_MAXDIM;_d++){ \
                if (b.s[_d]<CUDA_MAXDIM && b.s[_d]>=0) { \
                        dSize.s[b.s[_d]]=sSize.s[_d]; posnew.s[b.s[_d]] = pos.s[_d];} \
                } \
        size_t idd=0;IdxNDFromCoords(posnew,dSize,idd);c[idd]=a[idx];}) 

CUDA_UnaryFktIntVec(carr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos); \
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=pos.s[_d];} \
        for(_d=0;_d<CUDA_MAXDIM;_d++){ \
                if (b.s[_d]<CUDA_MAXDIM && b.s[_d]>=0) { \
                        dSize.s[b.s[_d]]=sSize.s[_d]; posnew.s[b.s[_d]] = pos.s[_d];} \
                } \
        size_t idd=0;IdxNDFromCoords(posnew,dSize,idd);c[2*idd]=a[2*idx];c[2*idd+1]=a[2*idx+1];}) 
*/
        
CUDA_Fkt2Vec(arr_xyz_2vec,CoordsNDFromIdx(idx,sSize,pos);float val=0;for(int _d=0;_d<CUDA_MAXDIM;_d++){val += vec1.s[_d]+pos.s[_d]*(vec2.s[_d]-vec1.s[_d])/sSize.s[_d];} c[idx]=val;)  // a[idx]
CUDA_Fkt2Vec(arr_rr_2vec,CoordsNDFromIdx(idx,sSize,pos);float val=0;for(int _d=0;_d<CUDA_MAXDIM;_d++){val += Sqr(vec1.s[_d]+pos.s[_d]*(vec2.s[_d]-vec1.s[_d])/sSize.s[_d]);} c[idx]=sqrt(val);)  // a[idx]
CUDA_Fkt2Vec(arr_phiphi_2vec,CoordsNDFromIdx(idx,sSize,pos); c[idx]=atan2(vec1.s[0]+pos.s[0]*(vec2.s[0]-vec1.s[0])/sSize.s[0],vec1.s[1]+pos.s[1]*(vec2.s[1]-vec1.s[1])/sSize.s[1]);)  // phiphi

// Now include all the user-defined functions
// #include "user/user_cu_code.inc"
#include "user_cu_code.inc"


__global__ void set_arr(float b, float * c, size_t N)                          
{                                                          
   size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); 
   if(idx>=N) return;   
   c[idx]=b;
}                                                                   
extern "C" const char * CUDAset_arr(float b, float * c, size_t N)  
{                                                                       
    cudaError_t myerr;                                                
	size_t blockSize;dim3 nBlocks;                                         
    MemoryLayout(N,blockSize,nBlocks)                                    
	set_arr<<<nBlocks,blockSize>>>(b,c,N);                            
  myerr=cudaGetLastError();                                             
  if (myerr != cudaSuccess)                                             
      return cudaGetErrorString(myerr);                                 
  return 0;                                                                   
}                                                                       

__global__ void set_carr(float br, float bi, float * c, size_t N)               
{                                                                   
   size_t idx=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); 
    if(idx>=N) return;   
    size_t idc=idx*2;                                                  
	c[idc]=br;c[idc+1]=bi;
}

extern "C" const char * CUDAset_carr(float br, float bi, float * c, size_t N)
{                                                                       
    cudaError_t myerr;                                             
	size_t blockSize;dim3 nBlocks;                                         
    MemoryLayout(N,blockSize,nBlocks)                                    
	set_carr<<<nBlocks,blockSize>>>(br,bi,c,N);                        
  myerr=cudaGetLastError();                                             
  if (myerr != cudaSuccess)                                             
      return cudaGetErrorString(myerr);                                 
  return 0;                                                             
}                                                                       

// function below is used to check whether CUIMAGE_REDUCE_THREADS is set correctly 
extern "C" int ReduceThreadsDef(void) {
    return CUIMAGE_REDUCE_THREADS;
}

extern "C" const char * SetDeviceProperties(void) {
    cudaError_t myerr;                                             
    int dev=0;
    cudaGetDevice(&dev);
    myerr=cudaGetDeviceProperties(&prop,dev);
    if (myerr != cudaSuccess)  
        return cudaGetErrorString(myerr);
    return 0;
}


extern "C" size_t CUDAmaxSize(void) {
    int dev=0;
    cudaGetDevice(&dev);
    struct cudaDeviceProp prop;
    int status=cudaGetDeviceProperties(&prop,dev);

    // return prop.maxThreadsPerBlock;  // 512
    // return prop.multiProcessorCount;   // 30
    // return prop.warpSize;   // 32
    // return prop.maxThreadsDim[0];   // 512  = max blocksize
    // return prop.maxGridSize[0];   // 65535  = max GridSize = max nBlocks?
    return ((size_t)prop.maxGridSize[0])*((size_t)prop.maxGridSize[1])*((size_t)prop.maxThreadsDim[0]);   // maximally 2D grids are currently allowed.
}


__global__ void
arr_times_const_checkerboard(float*a,float b, float * c, size_t N, size_t sx,size_t sy,size_t sz)
{
    size_t ids=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); // which source array element do I have to deal with?
	if(ids>=N) return;  // not in range ... quit

	size_t px=(ids/2)%sx;   // my x pos
	size_t py=(ids/2)/sx;   // my y pos
    float minus1=(1-2*((px+py)%2));
	c[ids]=a[ids]*b*minus1;
}

extern "C" int CUDAarr_times_const_checkerboard(float * a, float b, float * c, size_t * sizes, int dims)  // multiplies with a constand and scrambles the array
{
    size_t sx=sizes[0],sy=1,sz=1;
    if (dims>1)
        sy=sizes[1];
    if (dims>2)
        sz=sizes[2];
    size_t N=sx*sy*sz*2;  // every pair will be processed exactly once
	size_t blockSize;dim3 nBlocks;                                         
    MemoryLayout(N,blockSize,nBlocks)                                    
	arr_times_const_checkerboard<<<nBlocks,blockSize>>>(a,b,c,N,sx,sy,sz);
	return 0;
}


/// cyclicly rotates datastack cyclic into positive direction in all coordinates by (dx,dy,dz) voxels
/// simple version with all processors dealing with exactly one element
__global__ void
rotate2(float*a,float b, float * c, size_t sx,size_t sy,size_t sz, long long dx, long long dy, long long dz)
{
  size_t ids=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); // id of this processor
  long long x=(ids + dx)%sx;  // advance by the offset steps along the chain
  long long y=(ids/sx + dy)%sy;
  long long z=(ids/(sx*sy) + dz)%sz;
  size_t idd=x+sx*y+sx*sy*z;
  if(ids>=sx*sy*sz) return;
  // float tmp = a[ids];
  // __syncthreads();             // nice try but does not work !
  c[idd] = b*a[ids];
}

/// cyclicly rotates datastack cyclic into positive direction in all coordinates by (dx,dy,dz) voxels
__global__ void
rotate(float*a,float b, float * c, size_t sx,size_t sy,size_t sz, size_t dx, size_t dy, size_t dz, size_t ux, size_t uy, size_t uz)
{
  // id of this processor
  size_t id=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); 

  size_t Processes=blockDim.x * gridDim.x;
  size_t chains=ux*uy*uz; // total number of independent chains
  size_t N=sx*sy*sz;  // total size of array, has to be chains*length_of_chain
  size_t length=N/chains;  // chain length
  size_t steps=N/Processes;  // this is how many steps each processor has to do

  size_t step,nl,nx,ny,nz,x,y,z,i,idd;
  float swp, nswp;

//if (id != 0)   return;
//for (id=0;id<Processes;id++)
{
  step=steps*id;   // my starting step as the id times the number of steps
  nl=step%length;  // current position in chain length
  nx=(step/length)%ux;  // current position in unit cell x
  ny=(step/(length*ux))%uy;  // current position in unit cell y
  nz=(step/(length*ux*uy))%uz;  // current position in unit cell z
  i=0;

  //if (step/steps != 4 && step/steps != 5) return;

  while(nz<uz)
   {
      while(ny<uy)
        {
          while (nx<ux)
            {
            x=(nx+nl*dx)%sx;  // advance by the offset steps along the chain
            y=(ny+nl*dy)%sy;
            z=(nz+nl*dz)%sz;
            idd=x+sx*y+sx*sy*z;
            if (i < steps) {
                swp=a[idd]; 
                // a[idd]=a[idd]+0.1;
                __syncthreads();
            }
            while (nl<length-1)
                {
                  if (i > steps-1)
                    goto nextProcessor; // return;
                  if (step >= N)  // this thread has reached the end of the total data to process
                    goto nextProcessor; // return;
                  step++;
                  x = (x+dx)%sx; // new position
                  y = (y+dy)%sy;
                  z = (z+dz)%sz;
                  idd=x+sx*y+sx*sy*z;
                  if (i < steps-1) {
                    nswp=a[idd];
                    __syncthreads();
                    //a[idd]=a[idd]+0.1;
                    }

                  c[idd]=swp+0.1; // c[idd]+ny+0.1; // c[idd]+i; // swp+0.1; // c[idd]+(step/steps);
                  i++; // counts number of writes
                  if (i > steps-1)
                    goto nextProcessor; // return;
                  nl++;
                  if (i < steps) {
                  swp=nswp;
                  }
                }
            nx++; nl=0;
            //if (nx < ux) {
            x = (x+dx)%sx; // new position
            y = (y+dy)%sy;
            z = (z+dz)%sz;
            idd=x+sx*y+sx*sy*z;
            c[idd]=swp+0.1; // no need to save this value as this is the end of the line
            //}
            i++; 
            if (i > steps-1)
                goto nextProcessor; // return;
            // if (nx <ux) x=(x+1)%sx;
            }
        ny++;
        // if (ny <uy) y=(y+1)%sy;
        nx=0;x=0;
        }
    nz++;
    // if (nz <uz) z=(z+1)%sz;
    ny=0;y=0;
    }
nextProcessor:
nz=0;
}
return;
}

size_t gcd(size_t a, size_t b) // greatest commod divisor by recursion
{ 
   return ( b == 0 ? a : gcd(b, a % b) ); 
}

extern "C" int CUDAarr_times_const_rotate(float * a, float b, float * c, size_t * sizes, int dims, int complex,int direction)  // multiplies with a constand and cyclilcally rotates the array using the chain algorithm
{
    // printf("TestING\n");   % Does NOT work!
    long long sx=1,sy=1,sz=1;
    if (dims>0)
        sx=sizes[0];
    if (dims>1)
        {sx=sizes[0];sy=sizes[1];}
    if (dims>2)
        sz=sizes[2]; 

    long long dx=(sx+direction*sx/2)%sx,dy=(sy+direction*sy/2)%sy,dz=(sz+direction*sz/2)%sz;  // how much to cyclically rotate
    if (complex) {sx=sx*2;dx=dx*2;}
    //printf("sx %d sy %d dx %d dy %d\n",sx,sy,dx,dy);

    // calculate the length of each swapping chain
    long long ux=gcd(sx,dx);  // unit cell in x. Any repeat along y directions will be also a repeat in x. Chain length is sx/ux
    // size_t lx=sx/ux; // how many accesses to get one round in x
    long long uy=gcd(((sx/ux)*dy%sy),sy); // how many times must the first chain be repeated to form a longer chain. This defines unit cell y
    long long uz=gcd(((sy/uy)*dz%sz),sz); // similar for z
    long long length=sx*sy*sz/(ux*uy*uz);  // chain length

    // in one dimension the gcd=greatest common divisor, would mean that one has to start task at position 0 ... gcd-1
    // in several dimensions even completing one round leaving a spacing at gcd does not mean that this is a complete loop
    // however it could be a complete loop. The number of steps that where performed in the lower dimension are s/gcd before reaching the beginning again
    // with the size of the dimension s. If we are at the same startingpoint in the next dimension the chain is complete.
    // So the number of times a super chain (in 2D) must be executed is sy/gcd(sy,s/gcd(sx,dx))
    int dev=0;
    cudaGetDevice(&dev);
    struct cudaDeviceProp prop;
    int status=cudaGetDeviceProperties(&prop,dev);

    long long m=1;
    if (ux>uy)
        m=ux;
    else
        m=uy;
    if (uz>m)
        m=uz;
    if (length>m)
        m=length;

    //size_t blockSize=1; // prop.warpSize; // ux*uy*uz;
    //size_t nBlocks=m;	// add extra block if N can't be divided by blockSize
    
    //    rotate<<<nBlocks,blockSize>>>(a,b,c,sx,sy,sz,dx,dy,dz,ux,uy,uz);  // get unit cell sizes

    size_t N=sx*sy*sz; // includes the space for complex numbers
	size_t blockSize;dim3 nBlocks;                                         
                                                                //    printf("BlockSize %d, ux %d, uy %d, uz %d\n",blockSize,ux,uy,uz);
    // unfortunately we have to do it out of place.
    MemoryLayout(N,blockSize,nBlocks)                                    
    // printf("rotate 2 call: (%zd %zd %zd %lld %lld %lld)\n",sx,sy,sz,dx,dy,dz);
    if (a == c)
    {
        float * d =0;
        int status=cudaMalloc((void **) &d, N*sizeof(float));
        cudaMemcpy(d,a, N*sizeof(float),cudaMemcpyDeviceToDevice);
        rotate2 <<<nBlocks,blockSize>>>(d,b,c,sx,sy,sz,dx,dy,dz);
        cudaFree(d);
    }
    else
        rotate2 <<<nBlocks,blockSize>>>(a,b,c,sx,sy,sz,dx,dy,dz);  // get unit cell sizes

	return prop.maxThreadsPerBlock;
}



__global__ void
arr_times_const_scramble(float*a,float b, float * c, size_t sx,size_t sy,size_t sz, size_t ox, size_t oy, size_t oz)
{
	// which source array element do I have to deal with?
    size_t pnum=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); 

	size_t px=pnum%(sx/2);   // my x pos of a complex number in the subarray
	size_t py=pnum/(sx/2);   // my y pos of a complex number
	if(px>=(sx/2) || py >= (sy/2)) return;  // not in range ... quit
    size_t ids=2*(px+py*sx);  /// offset to array start in floats
    size_t idd=2*((ox+px)+(oy+py)*sx);

    // echange two values using a tmp
    float tmpR = c[idd];
    float tmpI = c[idd+1];
    c[idd]=a[ids]; // (float)(ox+px); // 
    c[idd+1]=a[ids+1]; // (float)(oy+py); // 
    a[ids]=tmpR;
    a[ids+1]=tmpI;
}

__global__ void
array_copy(float*a, float * c, size_t mx, size_t my, size_t mz, size_t sx,size_t sy,size_t sz, size_t ox, size_t oy, size_t oz)  // copies between two memories with different strides
{
    size_t pnum=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x); 

	size_t px=pnum%(sx/2);   // my x pos of a complex number in the subarray
	size_t py=pnum/(sx/2);   // my y pos of a complex number
	if(px>=sx || py >= (sy/2)) return;  // not in range ... quit
    size_t ids=2*(px+py*sx);  /// offset to array start in floats
    size_t idd=2*((ox+px)+(oy+py)*sx);

    // echange two values using a tmp
    float tmpR = c[idd];
    float tmpI = c[idd+1];
    c[idd]=a[ids]; // (float)(ox+px); // 
    c[idd+1]=a[ids+1]; // (float)(oy+py); // 
    a[ids]=tmpR;
    a[ids+1]=tmpI;
}


extern "C" int CUDAarr_times_const_scramble(float * a, float b, float * c, size_t * sizes, int dims)  // multiplies with a constand and scrambles the array
{
    size_t sx=sizes[0],sy=1,sz=1, iseven=1;
	size_t blockSize;dim3 nBlocks;                                         
    if (sx%2 == 1) iseven=0;
    if (dims>1) {
        sy=sizes[1];
        if (sy%2 == 1) iseven=0;
        }

    if (dims>2) {
        sz=sizes[2];
        if (sz%2 == 1) iseven=0;
        }
    size_t N=sx*sy*sz*2;  // every pair will be processed exactly once
    MemoryLayout(N,blockSize,nBlocks)                                    

    if (! iseven)
        {
            float * tmp=0;
            cudaMalloc((void **) &tmp,sizeof(tmp[0])*(1+sx/2)*(1+sy/2));
        }
	arr_times_const_scramble<<<nBlocks,blockSize>>>(a,b,c,sx,sy,sz,sx/2,sy/2,0);
	arr_times_const_scramble<<<nBlocks,blockSize>>>(a+2*(sx/2),b,c+2*(sx/2),sx,sy,sz,-sx/2,sy/2,0);
	return 0;
}


// Here is some code for calculating the singular value decomposition of the trailing dimension in an array
// This code is adopted from matLib3D.h by stamatis.lefkimmiatis@epfl.ch
// and svd3D_decomp.cpp by emmanuel.soubies@epfl.ch

/***************************************************************************
  Let X be a NxMxKx6 matrix such that:
  
  P_mn = [X(n,m,k,1) X(n,m,k,2) X(n,m,k,3)
          X(n,m,k,2) X(n,m,k,4) X(n,m,k,5)
          X(n,m,k,3) X(n,m,k,5) X(n,m,k,6)] 
          
  is a symmetric matrix. Then the present function computes the eigenvalues
  E(n,m,k,1) E(n,m,k,2) E(n,m,k,3) and the eigenvector 
          V1 = [V(n,m,k,1) V(n,m,k,2)  V(n,m,k,3)] 
          V2 = [V(n,m,k,4) V(n,m,k,5)  V(n,m,k,6)] 
          V2 = [V(n,m,k,5) V(n,m,k,8)  V(n,m,k,9)]  
  Hence the function outputs two matrices E of size NxMxKx3 and V of size NxMxKx9.
  
  Copyright (C) 2017 E. Soubies emmanuel.soubies@epfl.ch

****************************************************************************/

#include "matlib3D.h"

__global__ void core_svd3D(float *X, float *Ye, float * Yv, size_t N){   // N is NOT the total size, but only the size excluding the last dimension (of size 3)
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);
  if(idd>=N) return;
  int k;
    double E[3];
	double D[3];   
    double V[9];
  
        for (k=0;k<3;k++)   // get the matrix value [X(1,1) X(2,1)=X(1,2), X(2,2)]
            V[k]=X[idd+N*k];
        for (k = 4; k < 6; k++)
            V[k] = X[idd+N*(k-1)];
        for (k = 7; k < 9; k++)
            V[k] = X[idd+N*(k-3)];
        V[3]=X[idd+N];
        V[6]=X[idd+N*2];
        
        tred2(V, D, E);
        tql2(V, D, E);
        
  		for (k=0;k<3;k++)  // set result
        	Ye[idd+N*k]=D[k];
        
        for (k=0;k<9;k++){
            Yv[idd+N*k]=V[k];
        }
}

extern "C" const char * CUDAsvd_last(float *X, float *Ye, float * Yv, size_t N)  // N is NOT the total size, but only the size excluding the last dimension (of size 3)
{
    cudaError_t myerr;
	dim3 nBlocks;
    size_t blockSize=prop.maxThreadsPerBlock / 2; // To account for the many registers needed
    size_t numb=NBLOCKS(N,blockSize);
    if (numb<prop.maxGridSize[0])
        nBlocks.x=numb;
    else
        {nBlocks.x=(size_t)(sqrt((float)numb)+1);
    nBlocks.y=(size_t)(sqrt((float)numb)+1);}

	core_svd3D<<<nBlocks,blockSize>>>(X,Ye,Yv,N);
    myerr=cudaGetLastError();
    if (myerr != cudaSuccess)
        return cudaGetErrorString(myerr);
    return 0;
}


__global__ void core_svd3D_recomp(float *Y, float *E, float * V, size_t N){   // N is NOT the total size, but only the size excluding the last dimension (of size 3)
  size_t idd=((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x);
  if(idd>=N) return;
  int k;
  double ee[3];
	double vv[9];
	double tmp[6];  
  
    	for (k=0;k<3;k++)   // get the eigenvalues 
        	ee[k]=E[idd+N*k];
        for (k=0;k<9;k++)   // get the eigenvectors
        	vv[k]=V[idd+N*k];

        eigen3x3SymRec(tmp,vv,ee);
  		for (k=0;k<6;k++){  // set result
        	Y[idd+N*k]=tmp[k];
  		}
}

extern "C" const char * CUDAsvd3D_recomp(float *Y, float *E, float * V, size_t N)  // N is NOT the total size, but only the size excluding the last dimension (of size 3)
{
    cudaError_t myerr;
	dim3 nBlocks;
    size_t blockSize=prop.maxThreadsPerBlock / 2; // To account for the many registers needed
    size_t numb=NBLOCKS(N,blockSize);
    if (numb<prop.maxGridSize[0])
        nBlocks.x=numb;
    else
        {nBlocks.x=(size_t)(sqrt((float)numb)+1);
    nBlocks.y=(size_t)(sqrt((float)numb)+1);}

	core_svd3D_recomp<<<nBlocks,blockSize>>>(Y,E,V,N);
    myerr=cudaGetLastError();
    if (myerr != cudaSuccess)
        return cudaGetErrorString(myerr);
    return 0;
}

