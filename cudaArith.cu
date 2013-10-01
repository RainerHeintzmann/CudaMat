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

#include <cuda.h>
#include <stdio.h>

#include "cufft.h"
#include "cudaArith.h"
#define IMUL(a, b) __mul24(a, b)

#define BLOCKSIZE 1024
//#define BLOCKSIZE 512
#define NBLOCKS(N,blockSize) (N/blockSize+(N%blockSize==0?0:1))

// the real part is named ".x" and the imaginary ".y" in the cufftComplex datatype
__device__ cufftComplex cuda_resultVal;   // here real and complex valued results can be stored to be then transported to the host
__device__ int cuda_resultInt;   // here real and complex valued results can be stored to be then transported to the host
static float * TmpRedArray=0;   // This temporary array will be constructed on the device, whenever the first reduce operation is performed
static float * accum = 0;       // This is the corresponding array on the host side
static int CurrentRedSize=0;    // Keeps track of how much reduce memory is allocated on the device
static const int MinRedBlockSize=65536;    // defines the chunks of memory (in floats) which will be used

// below is blocksize for temporary array for reduce operations. Has to be a power of 2 in size
#define CUIMAGE_REDUCE_THREADS 512
// #define CUIMAGE_REDUCE_THREADS 512
// #define CUIMAGE_REDUCE_THREADS 128
//#define CUIMAGE_REDUCE_BLOCKS  64

#define mysum(a,b) ((a)+(b))
#define maxCond(a,b) (((b)>(a)))
#define minCond(a,b) (((b)<(a)))

#define Sqr(a) ((a)*(a))

// below are code snippets used in other macros 
#define Coords3DFromIdx(idx,sSize)                                      \
  int x=(idx)%sSize.s[0];                                               \
  int y=(idx/sSize.s[0])%sSize.s[1];                                    \
  int z=(idx/(sSize.s[0]*sSize.s[1]))%sSize.s[2];                       

#define IdxFromCoords3D(x,y,z,dSize,dOffs) \
  unsigned int idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]);  \
  
#define CoordsNDFromIdx(idx,sSize,pos)                                  \
   SizeND pos;                                               \
   { unsigned int resid=idx;                                               \
  for(int _d=0;_d<CUDA_MAXDIM;_d++)                                     \
      if (resid > 0)                                                    \
        { pos.s[_d]=resid%sSize.s[_d];                                    \
          resid/=sSize.s[_d]; }                                         \
      else                                                              \
          pos.s[_d]=0;                                                    \
  }

#define IdxNDFromCoords(pos,dSize,idd)                                   \
  (idd)=0;                                                              \
  {                                                                     \
  unsigned int _Stride=1;                                                \
  for(int _d=0;_d<CUDA_MAXDIM;_d++)                                      \
  if (dSize.s[_d]>0) {                                                   \
          if (pos.s[_d] < 0)                                              \
              {(idd) += (dSize.s[_d]-((-pos.s[_d]) % dSize.s[_d])) *_Stride;}          \
          else                                                          \
              {(idd) += (pos.s[_d] % dSize.s[_d]) *_Stride;}  \
        _Stride *= dSize.s[_d]; }                                        \
}
   

// The partial reduction funciton below projects the data along one dimension
// the processors are assigned to the result image pixels
// CAVE: These versions can be slow, if the resulting data has is smaller than the number of processors
#define CUDA_PartRedMask(FktName, OP)               \
__global__ void FktName (float *in, float *out, float * mask, int N, int dSizeX, int sStrideX, int sStrideY, int ProjStride, int ProjSize){      \
  int idd=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  if(idd>=N) return;                                                    \
  int p;                                                                \
  int ids=(idd%dSizeX)*sStrideX+(idd/dSizeX)*sStrideY;                  \
  float accu=0.0;                                                       \
  int laterPix=0;                                                       \
  for (p=0;p<ProjSize;p++)                                              \
    {                                                                   \
      if (mask == 0 || mask[ids] != 0.0)                                \
        if (! laterPix)  {                                              \
            accu=in[ids];                                               \
            laterPix=1;                                                 \
        } else {                                                        \
            accu=OP(accu,in[ids]);                                      \
        }                                                               \
      ids += ProjStride;                                                \
    }                                                                   \
 out[idd] = accu;                                                       \
}                                                                       \
\
extern "C" const char * CUDA ## FktName(float *a, float * mask, float * c, int sSize[CUDA_MAXPROJ], int ProjDir)\
{                                                                       \
    cudaError_t myerr;                                                  \
    int dSize[CUDA_MAXPROJ],d,N=1;                                      \
    for (d=0;d<CUDA_MAXPROJ;d++)  {dSize[d]=sSize[d]; }                 \
    dSize[ProjDir-1]=1;                                                 \
    for (d=0;d<CUDA_MAXPROJ;d++)  {N*=dSize[d];}                         \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    int ProjStride=0,ProjSize=0,sStrideX=0,sStrideY=0,dSizeX=0;         \
    if (ProjDir==1)                                                     \
        {ProjStride=1;ProjSize=sSize[0];dSizeX=sSize[1];sStrideX=sSize[0];sStrideY=sSize[0]*sSize[1];}\
    else if (ProjDir == 2)                                              \
        {ProjStride=sSize[0];ProjSize=sSize[1];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0]*sSize[1];}\
    else if (ProjDir == 3)                                              \
        {ProjStride=sSize[0]*sSize[1];ProjSize=sSize[2];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else if (ProjDir == 4)                                              \
        {ProjStride=sSize[0]*sSize[1]*sSize[2];ProjSize=sSize[3];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else if (ProjDir == 5)                                              \
        {ProjStride=sSize[0]*sSize[1]*sSize[2]*sSize[3];ProjSize=sSize[4];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else                                                                \
        return "Error: Unsupported projection direction";               \
	FktName<<<nBlocks,blockSize>>>(a,c,mask,N,dSizeX,sStrideX,sStrideY,ProjStride,ProjSize);\
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}

//   This is the same as the above but suited for complex numbers
#define CUDA_PartRedMaskCpx(FktName, OP)               \
__global__ void FktName (float *in, float *out, float * mask, int N, int dSizeX, int sStrideX, int sStrideY, int ProjStride, int ProjSize){      \
  int idd=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  if(idd>=N) return;                                                    \
  int p;                                                                \
  int ids=(idd%dSizeX)*sStrideX+(idd/dSizeX)*sStrideY;                  \
  float accu=0.0;                                                       \
  float accuI=0.0;                                                      \
  int laterPix=0;                                                       \
  for (p=0;p<ProjSize;p++)                                              \
    {                                                                   \
      if (mask == 0 || mask[ids] != 0.0)                                \
        if (! laterPix)  {                                              \
            accu=in[2*ids];                                             \
            accuI=in[2*ids+1];                                          \
            laterPix=1;                                                 \
        } else {                                                        \
            accu=OP(accu,in[2*ids]);                                      \
            accuI=OP(accuI,in[2*ids+1]);                                    \
        }                                                               \
      ids += ProjStride;                                                \
    }                                                                   \
 out[2*idd] = accu;                                                       \
 out[2*idd+1] = accuI;                                                       \
}                                                                       \
\
extern "C" const char * CUDA ## FktName(float *a, float * mask, float * c, int sSize[3], int ProjDir)\
{                                                                       \
     cudaError_t myerr;                                                \
    int dSize[CUDA_MAXPROJ],d,N=1;                                      \
    for (d=0;d<CUDA_MAXPROJ;d++)  {dSize[d]=sSize[d]; }                 \
    dSize[ProjDir-1]=1;                                                 \
    for (d=0;d<CUDA_MAXPROJ;d++)  {N*=dSize[d];}                         \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    int ProjStride=0,ProjSize=0,sStrideX=0,sStrideY=0,dSizeX=0;         \
    if (ProjDir==1)                                                     \
        {ProjStride=1;ProjSize=sSize[0];dSizeX=sSize[1];sStrideX=sSize[0];sStrideY=sSize[0]*sSize[1];}\
    else if (ProjDir == 2)                                              \
        {ProjStride=sSize[0];ProjSize=sSize[1];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0]*sSize[1];}\
    else if (ProjDir == 3)                                              \
        {ProjStride=sSize[0]*sSize[1];ProjSize=sSize[2];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else if (ProjDir == 4)                                              \
        {ProjStride=sSize[0]*sSize[1]*sSize[2];ProjSize=sSize[3];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else if (ProjDir == 5)                                              \
        {ProjStride=sSize[0]*sSize[1]*sSize[2]*sSize[3];ProjSize=sSize[4];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else                                                                \
        return "Error: Unsupported projection direction";                     \
	FktName<<<nBlocks,blockSize>>>(a,c,mask,N,dSizeX,sStrideX,sStrideY,ProjStride,ProjSize);\
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}

// This partial reduction code keeps track of the index

#define CUDA_PartRedMaskIdx(FktName, OP)               \
__global__ void FktName (float *in, float *out, float * outIdx, float * mask, int N, int dSizeX, int sStrideX, int sStrideY, int ProjStride, int ProjSize){      \
  int idd=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  if(idd>=N) return;                                                    \
  int p;                                                                \
  int ids=(idd%dSizeX)*sStrideX+(idd/dSizeX)*sStrideY;                  \
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
extern "C" const char * CUDA ## FktName(float *a, float * mask, float * c, float * cIdx, int sSize[5], int ProjDir)\
{                                                                       \
    cudaError_t myerr;                                                  \
    int dSize[CUDA_MAXPROJ],d,N=1;                                      \
    for (d=0;d<CUDA_MAXPROJ;d++)  {dSize[d]=sSize[d]; }                 \
    dSize[ProjDir-1]=1;                                                 \
    for (d=0;d<CUDA_MAXPROJ;d++)  {N*=dSize[d];}                         \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    int ProjStride=0,ProjSize=0,sStrideX=0,sStrideY=0,dSizeX=0;         \
    if (ProjDir==1)                                                     \
        {ProjStride=1;ProjSize=sSize[0];dSizeX=sSize[1];sStrideX=sSize[0];sStrideY=sSize[0]*sSize[1];}\
    else if (ProjDir == 2)                                              \
        {ProjStride=sSize[0];ProjSize=sSize[1];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0]*sSize[1];}\
    else if (ProjDir == 3)                                              \
        {ProjStride=sSize[0]*sSize[1];ProjSize=sSize[2];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else if (ProjDir == 4)                                              \
        {ProjStride=sSize[0]*sSize[1]*sSize[2];ProjSize=sSize[3];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else if (ProjDir == 5)                                              \
        {ProjStride=sSize[0]*sSize[1]*sSize[2]*sSize[3];ProjSize=sSize[4];dSizeX=sSize[0];sStrideX=1;sStrideY=sSize[0];}\
    else                                                                \
        return "Error: Unsupported projection direction";               \
	FktName<<<nBlocks,blockSize>>>(a,c,cIdx,mask,N,dSizeX,sStrideX,sStrideY,ProjStride,ProjSize);\
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}

// Below is some reduction code adapted from the tips and tricks tutorial 
// https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
// FUNCTION BELOW IS SLOW AND DOES NOT WORK PROPERLY YET
#define CUDA_FullRedBin(FktName, OP)                                    \
__global__ void FktName (float *in, int N){                             \
  const int stride = CUIMAGE_REDUCE_THREADS;                    \
  const int start  = threadIdx.x;\
  __shared__ float accum[CUIMAGE_REDUCE_THREADS];               \
  int nTotalThreads=CUIMAGE_REDUCE_THREADS;                     \
  int thread2;                                                  \
                                                                \
  if (start >= CUIMAGE_REDUCE_THREADS) return;                   \
  if (start >= N) {accum[start]=0;return;}                      \
                                                                \
  accum[threadIdx.x] = in[start];                               \
  for (int ii=start+stride; ii < N; ii += CUIMAGE_REDUCE_THREADS)  { \
    accum[threadIdx.x] = OP(accum[threadIdx.x], in[ii]);        \
  }                                                             \
  __syncthreads();                                              \
                                                                \
/* Now entering the logaritmic reduction phase of the algorithm*/       \
while(nTotalThreads > 1)                                                \
{                                                                       \
  int halfPoint = (nTotalThreads >> 1);	/* divide by two */             \
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
extern "C" const char * CUDA ## FktName(float * a, int N, float * resp) \
{                                                               \
  int CUIMAGE_REDUCE_BLOCKS=NBLOCKS(N,CUIMAGE_REDUCE_THREADS);  \
  dim3 threadBlock(CUIMAGE_REDUCE_THREADS);                     \
  dim3 blockGrid(CUIMAGE_REDUCE_BLOCKS);                        \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, N);                    \
  if (cudaGetLastError() != cudaSuccess)                        \
      return cudaGetErrorString(cudaGetLastError());            \
                                                                \
  cudaMemcpyFromSymbol(resp, cuda_resultVal, sizeof(* resp));\
  if (cudaGetLastError() != cudaSuccess)                        \
      return cudaGetErrorString(cudaGetLastError());            \
  return 0;                                                     \
}


// Below is the reduction code of Wouter Caarls, modified
// This could potentially also be run sequentially over the remaining dimension

#define CUDA_FullRed(FktName, OP)                               \
__global__ void FktName (float *in, float *out, int N){      \
  const int stride = blockDim.x * gridDim.x;                    \
  const int start  = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;\
  __shared__ float accum[CUIMAGE_REDUCE_THREADS];               \
  if (start >= N) return;                                    \
                                                                \
  accum[threadIdx.x] = in[start];                               \
  for (int ii=start+stride; ii < N; ii += stride)  {         \
    accum[threadIdx.x] = OP(accum[threadIdx.x], in[ii]);        \
  }                                                             \
  __syncthreads();                                              \
  if (threadIdx.x == 0)                                         \
  {                                                             \
    float res = accum[0];                                       \
    int limit;                                                  \
    if (start+blockDim.x > N) limit=(N-start);  \
    else limit=blockDim.x;                                      \
    for (int ii = 1; ii < limit; ii++) {                  \
      res=OP(res,accum[ii]);                                    \
     }                                                          \
    out[blockIdx.x] = res;                                      \
  }                                                             \
}                                                               \
                                                                \
extern "C" const char * CUDA ## FktName(float * a, int N, float * resp) \
{                                                               \
  cudaError_t myerr;                                            \
  const char * myerrStr;                                              \
  float res;                                                    \
  int CUIMAGE_REDUCE_BLOCKS=NBLOCKS(N,CUIMAGE_REDUCE_THREADS);  \
  dim3 threadBlock(CUIMAGE_REDUCE_THREADS);                     \
  dim3 blockGrid(CUIMAGE_REDUCE_BLOCKS);                        \
  myerrStr=CheckReduceAllocation(2*CUIMAGE_REDUCE_BLOCKS);      \
  if (myerrStr) return myerrStr;                                \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, TmpRedArray, N);       \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpy(accum, TmpRedArray, CUIMAGE_REDUCE_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);\
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  res = accum[0];                                               \
  for (int ii=1; ii < CUIMAGE_REDUCE_BLOCKS; ii++)  {           \
    res=OP(res,accum[ii]);                                      \
   }                                                            \
  /* cudaFree(TmpRedArray); */                                  \
  /* free(accum); */                                            \
                                                                \
  (* resp)=res;                                                 \
  return 0;                                                     \
}

// The version below is for complex valued arrays

#define CUDA_FullRedCpx(FktName, OP)               \
__global__ void FktName (float *in, float *out, int N){      \
  const int stride = blockDim.x * gridDim.x;                    \
  const int start  = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;\
  __shared__ float accum[CUIMAGE_REDUCE_THREADS];               \
  __shared__ float accumI[CUIMAGE_REDUCE_THREADS];              \
  if (start >= N) return;                                    \
                                                                \
  accum[threadIdx.x] = in[2*start];                             \
  accumI[threadIdx.x] = in[2*start+1];                          \
  for (int ii=start+stride; ii < N; ii += stride)  {         \
    accum[threadIdx.x] = OP(accum[threadIdx.x], in[2*ii]);      \
    accumI[threadIdx.x] = OP(accumI[threadIdx.x], in[2*ii +1]); \
  }                                                             \
  __syncthreads();                                              \
  if (!threadIdx.x)                                             \
  {                                                             \
    float res = accum[0];                                       \
    float resI = accumI[0];                                     \
    int limit;                                                  \
    if (start+blockDim.x > N) limit=(N-start);  \
    else limit=blockDim.x;                                      \
    for (int ii = 1; ii < limit; ii++) {                        \
      res=OP(res,accum[ii]);                                    \
      resI=OP(resI,accumI[ii]);                                 \
     }                                                          \
    out[2*blockIdx.x] = res;                                    \
    out[2*blockIdx.x + 1] = resI;                               \
  }                                                             \
}  \
\
extern "C" const char * CUDA ## FktName(float * a, int N, float * resp) \
{                                                               \
    cudaError_t myerr;                                          \
  const char * myerrStr;                                              \
  float res, resI;                                              \
  int CUIMAGE_REDUCE_BLOCKS=NBLOCKS(N,CUIMAGE_REDUCE_THREADS);  \
  dim3 threadBlock(CUIMAGE_REDUCE_THREADS);                     \
  dim3 blockGrid(CUIMAGE_REDUCE_BLOCKS);                        \
  myerrStr=CheckReduceAllocation(2*CUIMAGE_REDUCE_BLOCKS);      \
  if (myerrStr) return myerrStr;                                \
                                                                \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, TmpRedArray, N);       \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpy(accum, TmpRedArray, 2*CUIMAGE_REDUCE_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);\
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
                                                                \
  res = accum[0];                                               \
  resI = accum[1];                                              \
  for (int ii=1; ii < CUIMAGE_REDUCE_BLOCKS; ii++)  {           \
    res=OP(res,accum[2*ii]);                                    \
    resI=OP(resI,accum[2*ii + 1]);                              \
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
__global__ void FktName (float *in, float *out, int size){      \
  const int stride = blockDim.x * gridDim.x;                    \
  const int start  = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;\
  __shared__ float accum[CUIMAGE_REDUCE_THREADS];               \
  __shared__ float accumI[CUIMAGE_REDUCE_THREADS];              \
  if (start >= size) return;                                    \
                                                                \
  accum[threadIdx.x] = in[start];                               \
  accumI[threadIdx.x] = start;                                  \
  for (int ii=start+stride; ii < size; ii += stride)  {         \
    if OP(accum[threadIdx.x], in[ii]) { accum[threadIdx.x]= in[ii]; accumI[threadIdx.x]= ii; }      \
  }                                                             \
  __syncthreads();                                              \
  if (!threadIdx.x)                                             \
  {                                                             \
    float res = accum[0];                                       \
    float resI = accumI[0];                                     \
    int limit;                                                  \
    if (start+blockDim.x > size) limit=1+(size-start-1)/gridDim.x;  \
    else limit=blockDim.x;                                      \
    for (int ii = 1; ii < limit; ii++) {                        \
    if OP(res, accum[ii]){ res= accum[ii]; resI= accumI[ii]; }  \
     }                                                          \
    out[2*blockIdx.x] = res;                                    \
    out[2*blockIdx.x + 1] = resI;                               \
  }                                                             \
}  \
\
extern "C" const char * CUDA ## FktName(float * a, int N, float * resp) \
{                                                               \
  float res, resI;                                              \
  cudaError_t myerr;                                            \
  const char * myerrStr;                                              \
  int CUIMAGE_REDUCE_BLOCKS=NBLOCKS(N,CUIMAGE_REDUCE_THREADS);  \
  dim3 threadBlock(CUIMAGE_REDUCE_THREADS);                     \
  dim3 blockGrid(CUIMAGE_REDUCE_BLOCKS);                        \
  myerrStr=CheckReduceAllocation(2*CUIMAGE_REDUCE_BLOCKS);      \
  if (myerrStr) return myerrStr;                                \
                                                                \
  FktName<<<blockGrid, threadBlock>>>(a, TmpRedArray, N);       \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  cudaMemcpy(accum, TmpRedArray, 2*CUIMAGE_REDUCE_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);\
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
                                                                \
  res = accum[0];                                               \
  resI = accum[1];                                              \
  for (int ii=1; ii < CUIMAGE_REDUCE_BLOCKS; ii++)  {           \
    if OP(res, accum[2*ii]) {res= accum[2*ii]; resI= accum[2*ii+1];  }  \
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
__global__ void FktName (float *a, float * mask,float *c, int N){ \
  int Blocksize = N/CUIMAGE_REDUCE_THREADS + 1;                 \
  int start = Blocksize * threadIdx.x;                          \
  __shared__ int accum[CUIMAGE_REDUCE_THREADS+1];               \
  if (start >= N) return;                                       \
                                                                \
  { int SumMask=0;                                              \
  for (int ii=start; ii < start+Blocksize; ii ++)  {            \
    if (ii < N)                                                 \
        SumMask += (mask[ii] != 0);                             \
  }                                                             \
  accum[threadIdx.x+1] = SumMask;                               \
  }                                                             \
  __syncthreads();                                              \
  if (threadIdx.x == 0)                                         \
  {                                                             \
    accum[0] = 0;                                               \
    int res = 0;                                                \
    for (int ii = 0; ii*Blocksize < N; ii++) {                  \
      res += accum[ii+1];                                       \
      accum[ii+1] = res;                                        \
     }                                                          \
    cuda_resultInt = res;                                       \
  }                                                             \
  __syncthreads();                                              \
  int mask_idx= accum[threadIdx.x];                             \
  for (int idx=start; idx < start+Blocksize; idx ++)  {         \
    if ((idx < N) && (mask[idx] != 0))                          \
      {                                                         \
        EXPRESSIONS                                             \
        mask_idx ++;                                            \
      }                                                         \
  }                                                             \
}                                                               \
                                                                \
extern "C" const char * CUDA ## FktName(float * in, float * mask, float *  out, int N, int * pM) \
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
  cudaMemcpyFromSymbol(pM, cuda_resultInt, sizeof(* pM));\
                                                                \
  myerr=cudaGetLastError();                                     \
  if (myerr != cudaSuccess)                                     \
      return cudaGetErrorString(myerr);                         \
  return 0;                                                     \
}



// In the expression one can use the variables idx (for real valued arrays) and idc (for complex valued arrays)
// -------------- caller function is also generated -------------
#define CUDA_BinaryFkt(FktName,expression)                          \
__global__ void                                                     \
FktName(float*a,float *b, float * c, int N)                         \
{                                                                   \
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float * b, float * c, int N)  \
{                                                                       \
    cudaError_t myerr;                                          \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
	FktName<<<nBlocks,blockSize>>>(a,b,c,N);                            \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

// --------------Macro generating operation of array with real constant -------------

#define CUDA_UnaryFktConst(FktName,expression)                      \
__global__ void                                                     \
FktName(float*a,float b, float * c, int N)                          \
{                                                                   \
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float b, float * c, int N)  \
{                                                                       \
    cudaError_t myerr;                                          \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
	FktName<<<nBlocks,blockSize>>>(a,b,c,N);                            \
    myerr=cudaGetLastError();                                             \
    if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

// --------------Macro generating operation with complex array and constant -------------
#define CUDA_UnaryFktConstC(FktName,expression)                      \
__global__ void                                                     \
FktName(float*a,float br, float bi, float * c, int N)               \
{                                                                   \
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float br, float bi, float * c, int N)  \
{                                                                       \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
   cudaError_t myerr;                                          \
	FktName<<<nBlocks,blockSize>>>(a,br,bi,c,N);                        \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                       

        

// ----------- Makro for function with an integer Vector ---- e.g.- for cyclic shifts etc. -----
#define CUDA_UnaryFktIntVec(FktName,expression)                      \
__global__ void                                                     \
FktName(float*a, SizeND b, float * c, SizeND sSize, int N)          \
{                                                                   \
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   \
    expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, int b[CUDA_MAXDIM], float * c, int mySize[CUDA_MAXDIM], int N)  \
{                                                                       \
  cudaError_t myerr;                                          \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    SizeND sb,sSize;                                                    \
    for (int d=0;d<CUDA_MAXDIM;d++)                                     \
    { sb.s[d]=b[d];sSize.s[d]=mySize[d]; }                              \
	FktName<<<nBlocks,blockSize>>>(a,sb,c,sSize,N);                     \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

// ----------- Makro for function with an integer Vector ---- e.g.- for cyclic shifts etc. -----
#define CUDA_Fkt2Vec(FktName,expression)                            \
__global__ void                                                     \
FktName(float * c, VecND vec1, VecND vec2, SizeND sSize, int N)     \
{                                                                   \
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   \
    expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * c, VecND vec1, VecND vec2, SizeND sSize, int N)  \
{                                                                       \
    cudaError_t myerr;                                          \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
	FktName<<<nBlocks,blockSize>>>(c,vec1,vec2,sSize,N);                \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}



// --------------Macro generating unary operation with complex array  -------------
#define CUDA_UnaryFkt(FktName,expression)                     \
__global__ void                                                     \
FktName(float*a, float * c, int N)                                  \
{                                                                   \
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   \
	expression                                                      \
}                                                                   \
extern "C" const char * CUDA ## FktName(float * a, float * c, int N)         \
{                                                                       \
    cudaError_t myerr;                                          \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
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

// THESE STRUCT DEVINITION ARE NEEDED, AS CUDA CANNOT DEAL CORRECTLY WITH FIXED LENGTH ARRAYS IN THE ARGUMENT
// ACCESING THEM WILL CAUSE A CRASH!
// HOWEVER, STRUCTS WITH THE ARRAY INSIDE ARE OK
typedef struct {
    int s[3];
} Size3D ;

// Line below is used as an add-on to the 3d function below in case 3d assignment is needed
#define GET3DIDD int idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]);

#define CUDA_3DFkt(FktName,expressions)                                  \
__global__ void                                                         \
FktName(float *a, float *c, Size3D sSize,Size3D dSize,Size3D sOffs, Size3D sROI, Size3D dOffs) \
{                                                                       \
  int idx=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int N=sROI.s[0]*sROI.s[1]*sROI.s[2];                                        \
  if(idx>=N) return;                                                    \
  int x=(idx)%sROI.s[0];                                                  \
  int y=(idx/sROI.s[0])%sROI.s[1];                                          \
  int z=(idx/(sROI.s[0]*sROI.s[1]))%sROI.s[2];                                \
  int ids=x+sOffs.s[0]+sSize.s[0]*(y+sOffs.s[1])+sSize.s[0]*sSize.s[1]*(z+sOffs.s[2]);                               \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * a, float *c, int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3])  \
{                                                                       \
    cudaError_t myerr;                                                \
    int N=sROI[0]*sROI[1]*sROI[2];                                      \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    Size3D sS,dS,sO,sR,dO;                                              \
    int d;                                                              \
    for (d=0;d<3;d++)                                                   \
        {sS.s[d]=sSize[d];dS.s[d]=dSize[d];sO.s[d]=sOffs[d];sR.s[d]=sROI[d];dO.s[d]=dOffs[d];} \
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
  int idx=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int N=dROI.s[0]*dROI.s[1]*dROI.s[2];                                        \
  if(idx>=N) return;                                                    \
  int x=(idx)%dROI.s[0];                                               \
  int y=(idx/dROI.s[0])%dROI.s[1];                                    \
  int z=(idx/(dROI.s[0]*dROI.s[1]))%dROI.s[2];                       \
  int idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]);                               \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * c, float br, float bi, int dSize[3], int dROI[3], int dOffs[3])  \
{                                                                       \
    cudaError_t myerr;                                                \
    int N=dROI[0]*dROI[1]*dROI[2];                                      \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    Size3D dR,dS,dO;                                              \
    int d;                                                              \
    for (d=0;d<3;d++)                                                   \
        {dS.s[d]=dSize[d];dR.s[d]=dROI[d];dO.s[d]=dOffs[d];} \
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
  int idd=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int N=dSize.s[0]*dSize.s[1]*dSize.s[2];                               \
  if(idd>=N) return;                                                    \
  int x=(idd)%dSize.s[0];                                                \
  int y=(idd/dSize.s[0])%dSize.s[1];                                    \
  int z=(idd/(dSize.s[0]*dSize.s[1]))%dSize.s[2];                       \
  int ids=x%sSize.s[0]+sSize.s[0]*(y%sSize.s[1])+sSize.s[0]*sSize.s[1]*(z%sSize.s[2]); \
  expressions                                                           \
}                                                                       \
extern "C" const char * CUDA ## FktName(float *a, float * c, int sSize[3], int dSize[3])  \
{                                                                       \
    cudaError_t myerr;                                                \
    int N=dSize[0]*dSize[1]*dSize[2];                                      \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    Size3D sS,dS;                                              \
    int d;                                                              \
    for (d=0;d<3;d++)                                                   \
        {dS.s[d]=dSize[d];sS.s[d]=sSize[d];} \
	FktName<<<nBlocks,blockSize>>>(a,c,dS,sS); \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

//  Now the 5D Versions of the same code

// THESE STRUCT DEVINITION ARE NEEDED, AS CUDA CANNOT DEAL CORRECTLY WITH FIXED LENGTH ARRAYS IN THE ARGUMENT
// ACCESING THEM WILL CAUSE A CRASH!
// HOWEVER, STRUCTS WITH THE ARRAY INSIDE ARE OK
typedef struct {
    int s[5];
} Size5D ;

// Line below is used as an add-on to the 3d function below in case 3d assignment is needed
#define GET5DIDD   int idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])+dSize.s[0]*dSize.s[1]*dSize.s[2]*(t+dOffs.s[3])+dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]*(e+dOffs.s[4]);

#define CUDA_5DFkt(FktName,expressions)                                 \
__global__ void                                                         \
FktName(float *a, float *c, Size5D sSize,Size5D dSize,Size5D sOffs, Size5D sROI, Size5D dOffs) \
{                                                                     \
  int idx=(blockIdx.x*blockDim.x+threadIdx.x);                        \
  int N=sROI.s[0]*sROI.s[1]*sROI.s[2]*sROI.s[3]*sROI.s[4];            \
  if(idx>=N) return;                                                  \
  int x=(idx)%sROI.s[0];                                              \
  int y=(idx/sROI.s[0])%sROI.s[1];                                    \
  int z=(idx/(sROI.s[0]*sROI.s[1]))%sROI.s[2];                        \
  int t=(idx/(sROI.s[0]*sROI.s[1]*sROI.s[2]))%sROI.s[3];              \
  int e=(idx/(sROI.s[0]*sROI.s[1]*sROI.s[2]*sROI.s[3]))%sROI.s[4];    \
  int ids=x+sOffs.s[0]+sSize.s[0]*(y+sOffs.s[1])+sSize.s[0]*sSize.s[1]*(z+sOffs.s[2])+sSize.s[0]*sSize.s[1]*sSize.s[2]*(t+sOffs.s[3])+sSize.s[0]*sSize.s[1]*sSize.s[2]*sSize.s[3]*(e+sOffs.s[4]);   \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * a, float *c, int sSize[5], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5])  \
{                                                                       \
    cudaError_t myerr;                                                \
    int N=sROI[0]*sROI[1]*sROI[2]*sROI[3]*sROI[4];                      \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    Size5D sS,dS,sO,sR,dO;                                              \
    int d;                                                              \
    for (d=0;d<5;d++)                                                   \
        {sS.s[d]=sSize[d];dS.s[d]=dSize[d];sO.s[d]=sOffs[d];sR.s[d]=sROI[d];dO.s[d]=dOffs[d];} \
	FktName<<<nBlocks,blockSize>>>(a,c,sS,dS,sO,sR,dO); \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      
                
// --- macros for sub-assigning a block with vectors in each dimension -----

#define CUDA_5DAsgFkt(FktName,expressions)                                  \
__global__ void                                                         \
FktName(float *c, float br, float bi, Size5D dSize, Size5D dROI, Size5D dOffs) \
{                                                                       \
  int idx=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int N=dROI.s[0]*dROI.s[1]*dROI.s[2]*dROI.s[3]*dROI.s[4];              \
  if(idx>=N) return;                                                    \
  int x=(idx)%dROI.s[0];                                               \
  int y=(idx/dROI.s[0])%dROI.s[1];                                    \
  int z=(idx/(dROI.s[0]*dROI.s[1]))%dROI.s[2];                       \
  int t=(idx/(dROI.s[0]*dROI.s[1]*dROI.s[2]))%dROI.s[3];              \
  int e=(idx/(dROI.s[0]*dROI.s[1]*dROI.s[2]*dROI.s[3]))%dROI.s[4];    \
  int idd=x+dOffs.s[0]+dSize.s[0]*(y+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])+dSize.s[0]*dSize.s[1]*dSize.s[2]*(t+dOffs.s[3])+dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]*(e+dOffs.s[4]);   \
  expressions                                                            \
}                                                                       \
extern "C" const char * CUDA ## FktName(float * c, float br, float bi, int dSize[5], int dROI[5], int dOffs[5])  \
{                                                                       \
    cudaError_t myerr;                                                \
    int N=dROI[0]*dROI[1]*dROI[2]*dROI[3]*dROI[4];                      \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    Size5D dR,dS,dO;                                                    \
    int d;                                                              \
    for (d=0;d<5;d++)                                                   \
        {dS.s[d]=dSize[d];dR.s[d]=dROI[d];dO.s[d]=dOffs[d];}            \
	FktName<<<nBlocks,blockSize>>>(c,br,bi,dS,dR,dO);                   \
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
  int idd=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int N=dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]*dSize.s[4];                               \
  if(idd>=N) return;                                                    \
  int x=(idd)%dSize.s[0];                                                \
  int y=(idd/dSize.s[0])%dSize.s[1];                                    \
  int z=(idd/(dSize.s[0]*dSize.s[1]))%dSize.s[2];                       \
  int t=(idd/(dSize.s[0]*dSize.s[1]*dSize.s[2]))%dSize.s[3];              \
  int e=(idd/(dSize.s[0]*dSize.s[1]*dSize.s[2]*dSize.s[3]))%dSize.s[4];    \
  int ids=x%sSize.s[0]+sSize.s[0]*(y%sSize.s[1])+sSize.s[0]*sSize.s[1]*(z%sSize.s[2]) + sSize.s[0]*sSize.s[1]*sSize.s[2]*(t%sSize.s[3])+sSize.s[0]*sSize.s[1]*sSize.s[2]*sSize.s[3]*(e%sSize.s[4]); \
  expressions                                                           \
}                                                                       \
extern "C" const char * CUDA ## FktName(float *a, float * c, int sSize[5], int dSize[5])  \
{                                                                       \
    cudaError_t myerr;                                                  \
    int N=dSize[0]*dSize[1]*dSize[2]*dSize[3]*dSize[4];                 \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          \
    Size5D sS,dS;                                                       \
    int d;                                                              \
    for (d=0;d<5;d++)                                                   \
        {dS.s[d]=dSize[d];sS.s[d]=sSize[d];}                            \
	FktName<<<nBlocks,blockSize>>>(a,c,dS,sS);                          \
  myerr=cudaGetLastError();                                             \
  if (myerr != cudaSuccess)                                             \
      return cudaGetErrorString(myerr);                                 \
  return 0;                                                             \
}                                                                      

// The funciton below checks whether the size of allocated reduce arrays is sufficient and reallocates if needed be
// The arrays are "accum" and "TmpRedArray"
const char * CheckReduceAllocation(int asize) {
    cudaError_t myerr;
    asize=((asize/MinRedBlockSize) + 1)*MinRedBlockSize;  // round it up to the nearest multiple of MinRedSize
    if (! accum){
       accum = (float *) malloc(asize*sizeof(float));
       if (! accum)
       return "CheckReduceAllocation: Malloc failed";
    }    
    if (! TmpRedArray) {
        cudaMalloc((void **) &TmpRedArray, asize*sizeof(float));
        CurrentRedSize=asize;
        myerr=cudaGetLastError();
        if (myerr != cudaSuccess)
          return cudaGetErrorString(myerr);
    }
    
    if (asize > CurrentRedSize)
    {
        free(accum);
        accum = (float *) malloc(asize*sizeof(float));
        if (! accum)
            return "CheckReduceAllocation: ReMalloc failed";
        cudaFree(TmpRedArray);
        myerr=cudaGetLastError();
        if (myerr != cudaSuccess)
            return cudaGetErrorString(myerr);

        cudaMalloc((void **) &TmpRedArray, asize*sizeof(float));
        myerr=cudaGetLastError();
        if (myerr != cudaSuccess)
            return cudaGetErrorString(myerr);
        CurrentRedSize=asize;
    }
    return 0;
}

extern "C" int GetCurrentRedSize(void) {
    return CurrentRedSize;
}

/*__global__ void                                                         \
bla_ ## FktName(float*a, float * c, int N,  Size3D sSize,Size3D dSize,Size3D sOffs, Size3D sROI, Size3D dOffs) {                                    \
  int idx=(blockIdx.x*blockDim.x+threadIdx.x);                          \
  int idcd=0,idcs=0,ids=0;                                                    \
  if(idx>=N) return;                                                    \
    expression                                                          \
}   \  */

//	FktName<<<nBlocks,blockSize>>>(a,c,sSize,dSize,sOffs, sROI, dOffs); \


CUDA_FullRed(sum_arr,mysum)
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
CUDA_3DWrapAsgFkt(carr_3drepcpy_carr,c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)

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
CUDA_3DFkt(arr_3dsubcpy_arr, GET3DIDD; c[idd]=a[ids];)
CUDA_3DFkt(carr_3dsubcpy_carr, GET3DIDD; c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)
CUDA_3DFkt(arr_3dsubcpy_carr, GET3DIDD; c[2*idd]=a[ids];c[2*idd+1]=0;)

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
// These versions intoduce a transpose operation
CUDA_3DFkt(arr_3dsubcpyT_arr,  int iddt=y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]); c[iddt]=a[ids];)
CUDA_3DFkt(carr_3dsubcpyT_carr,int idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=a[2*ids+1];)
// with conjugation
CUDA_3DFkt(carr_3dsubcpyCT_carr,int idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=-a[2*ids+1];)

//CUDA_3DFkt(arr_subref_arr3d,c[idd]=)
//getCudaRef(prhs[1]),newarr,sSize,dSize,cuda_array[newref[0]],cuda_array[newref[1]],cuda_array[newref[2]]);

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
CUDA_5DFkt(arr_5dsubcpy_arr, GET5DIDD; c[idd]=a[ids];)
CUDA_5DFkt(carr_5dsubcpy_carr, GET5DIDD; c[2*idd]=a[2*ids];c[2*idd+1]=a[2*ids+1];)
CUDA_5DFkt(arr_5dsubcpy_carr, GET5DIDD; c[2*idd]=a[ids];c[2*idd+1]=0;)

// Sub copying, copies a source area into a destination area. Can be used for cat and subassign
// These versions intoduce a transpose operation
CUDA_5DFkt(arr_5dsubcpyT_arr,  int iddt=y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2]); c[iddt]=a[ids];)
CUDA_5DFkt(carr_5dsubcpyT_carr,int idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=a[2*ids+1];)
 // with conjugation
CUDA_5DFkt(carr_5dsubcpyCT_carr,int idcdt=2*(y+dOffs.s[0]+dSize.s[0]*(x+dOffs.s[1])+dSize.s[0]*dSize.s[1]*(z+dOffs.s[2])); c[idcdt]=a[2*ids];c[idcdt+1]=-a[2*ids+1];)  


// Power
CUDA_BinaryFkt(arr_power_arr,c[idx]=pow(a[idx],b[idx]);)
CUDA_UnaryFktConst(arr_power_const,c[idx]=pow(a[idx],b);)
CUDA_UnaryFktConst(const_power_arr,c[idx]=pow(b,a[idx]);)

// Multiplications
CUDA_BinaryFkt(arr_times_arr,c[idx]=a[idx]*b[idx];)
CUDA_BinaryFkt(carr_times_carr,
    int idc=2*idx;
    float myr=a[idc]*b[idc]-a[idc+1]*b[idc+1];float myi=a[idc]*b[idc+1]+a[idc+1]*b[idc];
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_BinaryFkt(arr_times_carr,c[2*idx]=a[idx]*b[2*idx];c[2*idx+1]=a[idx]*b[2*idx+1];)
CUDA_BinaryFkt(carr_times_arr,c[2*idx]=a[2*idx]*b[idx];c[2*idx+1]=a[2*idx+1]*b[idx];)
//CUDA_BinaryFkt(arr_times_carr,c[2*idx]=a[idx]*b[2*idx];c[2*idx+1]=a[idx+1]*b[2*idx];)
CUDA_UnaryFktConst(arr_times_const,c[idx]=a[idx]*b;)
CUDA_UnaryFktConst(const_times_arr,c[idx]=a[idx]*b;)
CUDA_UnaryFktConstC(carr_times_const,
    int idc=2*idx;
    float myr=a[idc]*br-a[idc+1]*bi;float myi=a[idc]*bi+a[idc+1]*br;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(const_times_carr,
    int idc=2*idx;
    float myr=a[idc]*br-a[idc+1]*bi;float myi=a[idc]*bi+a[idc+1]*br;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(arr_times_Cconst,c[2*idx]=a[idx]*br;c[2*idx+1]=a[idx]*bi;)
CUDA_UnaryFktConstC(Cconst_times_arr,c[2*idx]=br*a[idx];c[2*idx+1]=bi*a[idx];)

// Divisions
CUDA_BinaryFkt(arr_divide_arr,c[idx]=a[idx]/b[idx];)
CUDA_BinaryFkt(carr_divide_carr,
    int idc=2*idx;
    float tmp=b[idc]*b[idc]+b[idc+1]*b[idc+1];
    float myr=(a[idc]*b[idc]+a[idc+1]*b[idc+1])/tmp;float myi=(a[idc+1]*b[idc]-a[idc]*b[idc+1])/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_BinaryFkt(carr_divide_arr,c[2*idx]=a[2*idx]/b[idx];c[2*idx+1]=a[2*idx+1]/b[idx];)
CUDA_BinaryFkt(arr_divide_carr,
    int idc=2*idx;
    float tmp=b[idc]*b[idc]+b[idc+1]*b[idc+1];
    float myr=(a[idx]*b[idc]+a[idx+1]*b[idc+1])/tmp;float myi=(a[idx+1]*b[idc]-a[idx]*b[idc+1])/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConst(arr_divide_const,c[idx]=a[idx]/b;)
CUDA_UnaryFktConst(const_divide_arr,c[idx]=b/a[idx];)
CUDA_UnaryFktConstC(carr_divide_const,
    int idc=2*idx;
    float tmp=br*br+bi*bi;
    float myr=(a[idc]*br+a[idc+1]*bi)/tmp;float myi=(a[idc+1]*br-a[idc]*bi)/tmp;
    c[idc]=myr;c[idc+1]=myi;
)
CUDA_UnaryFktConstC(const_divide_carr,
    int idc=2*idx;
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
CUDA_BinaryFkt(arr_max_arr,c[idx]=a[idx]>b[idx]?a[idx]:b[idx];)
CUDA_BinaryFkt(carr_max_carr, int idc=2*idx; if (a[idc]*a[idc]+a[idc+1]*a[idc+1] > b[idc]*b[idc]+b[idc+1]*b[idc+1]) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=b[idc];c[idc+1]=b[idc+1];})
CUDA_BinaryFkt(carr_max_arr,int idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] > b[idx]*b[idx]) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=b[idx];c[idc+1]=0;})
CUDA_BinaryFkt(arr_max_carr,int idc=2*idx;if (a[idx]*a[idx] > b[idc]*b[idc]+b[idc+1]*b[idc+1]) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=b[idc];c[idc+1]=b[idc+1];})
CUDA_UnaryFktConst(arr_max_const,c[idx]=a[idx]>b?a[idx]:b;)
CUDA_UnaryFktConst(const_max_arr,c[idx]=a[idx]>b?a[idx]:b;)
CUDA_UnaryFktConstC(carr_max_const,int idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] > br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(const_max_carr,int idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] > br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(arr_max_Cconst,int idc=2*idx;if (a[idx]*a[idx] > br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(Cconst_max_arr,int idc=2*idx;if (a[idx]*a[idx] > br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})

// Element-wise minimum operations
CUDA_BinaryFkt(arr_min_arr,c[idx]=a[idx]<b[idx]?a[idx]:b[idx];)
CUDA_BinaryFkt(carr_min_carr, int idc=2*idx; if (a[idc]*a[idc]+a[idc+1]*a[idc+1] < b[idc]*b[idc]+b[idc+1]*b[idc+1]) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=b[idc];c[idc+1]=b[idc+1];})
CUDA_BinaryFkt(carr_min_arr,int idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] < b[idx]*b[idx]) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=b[idx];c[idc+1]=0;})
CUDA_BinaryFkt(arr_min_carr,int idc=2*idx;if (a[idx]*a[idx] < b[idc]*b[idc]+b[idc+1]*b[idc+1]) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=b[idc];c[idc+1]=b[idc+1];})
CUDA_UnaryFktConst(arr_min_const,c[idx]=a[idx]<b?a[idx]:b;)
CUDA_UnaryFktConst(const_min_arr,c[idx]=a[idx]<b?a[idx]:b;)
CUDA_UnaryFktConstC(carr_min_const,int idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] < br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(const_min_carr,int idc=2*idx;if (a[idc]*a[idc]+a[idc+1]*a[idc+1] < br*br+bi*bi) {c[idc]=a[idc];c[idc+1]=a[idc+1];}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(arr_min_Cconst,int idc=2*idx;if (a[idx]*a[idx] < br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})
CUDA_UnaryFktConstC(Cconst_min_arr,int idc=2*idx;if (a[idx]*a[idx] < br*br+bi*bi) {c[idc]=a[idx];c[idc+1]=0;}else{ c[idc]=br;c[idc+1]=bi;})

// Additions
CUDA_BinaryFkt(arr_plus_arr,c[idx]=a[idx]+b[idx];)
CUDA_BinaryFkt(carr_plus_carr, int idc=2*idx; c[idc]=a[idc]+b[idc];c[idc+1]=a[idc+1]+b[idc+1];)
CUDA_BinaryFkt(carr_plus_arr,int idc=2*idx;c[idc]=a[idc]+b[idx];c[idc+1]=a[idc+1];)
CUDA_BinaryFkt(arr_plus_carr,int idc=2*idx;c[idc]=a[idx]+b[idc];c[idc+1]=b[idc+1];)
CUDA_UnaryFktConst(arr_plus_const,c[idx]=a[idx]+b;)
CUDA_UnaryFktConst(const_plus_arr,c[idx]=a[idx]+b;)
CUDA_UnaryFktConstC(carr_plus_const,int idc=2*idx;c[idc]=a[idc]+br;c[idc+1]=a[idc+1]+bi;)
CUDA_UnaryFktConstC(const_plus_carr,int idc=2*idx;c[idc]=a[idc]+br;c[idc+1]=a[idc+1]+bi;)
CUDA_UnaryFktConstC(arr_plus_Cconst,int idc=2*idx;c[idc]=a[idx]+br;c[idc+1]=bi;)
CUDA_UnaryFktConstC(Cconst_plus_arr,int idc=2*idx;c[idc]=br+a[idx];c[idc+1]=bi;)

// Subtractions
CUDA_BinaryFkt(arr_minus_arr,c[idx]=a[idx]-b[idx];)
CUDA_BinaryFkt(carr_minus_carr,int idc=2*idx; c[idc]=a[idc]-b[idc];c[idc+1]=a[idc+1]-b[idc+1];)
CUDA_BinaryFkt(carr_minus_arr,int idc=2*idx;c[idc]=a[idc]-b[idx];c[idc+1]=a[idc+1];)
CUDA_BinaryFkt(arr_minus_carr,int idc=2*idx;c[idc]=a[idx]-b[idc];c[idc+1]=-b[idc+1];)
CUDA_UnaryFktConst(arr_minus_const,c[idx]=a[idx]-b;)
CUDA_UnaryFktConst(const_minus_arr,c[idx]=b-a[idx];)
CUDA_UnaryFktConstC(carr_minus_const,int idc=2*idx;c[idc]=a[idc]-br;c[idc+1]=a[idc+1]-bi;)
CUDA_UnaryFktConstC(const_minus_carr,int idc=2*idx;c[idc]=br-a[idc];c[idc+1]=bi-a[idc+1];)
CUDA_UnaryFktConstC(arr_minus_Cconst,int idc=2*idx;c[idc]=a[idx]-br;c[idc+1]=a[idx]-bi;)
CUDA_UnaryFktConstC(Cconst_minus_arr,int idc=2*idx;c[idc]=br-a[idx];c[idc+1]=bi;)

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
CUDA_3DFkt(arr_diag_set,  int iddt=ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1]); c[iddt]=a[ids];)
CUDA_3DFkt(carr_diag_set,  int idcdt=2*(ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1])); c[idcdt]=a[2*ids];c[idcdt+1]=a[2*ids*1];)
CUDA_3DFkt(arr_diag_get,  int iddt=ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1]); a[ids]=c[iddt];)
CUDA_3DFkt(carr_diag_get,  int idcdt=2*(ids+dOffs.s[0]+dSize.s[0]*(ids+dOffs.s[1])); a[2*ids]=c[idcdt];a[2*ids*1]=c[idcdt+1];)

// referencing and assignment with index vectors.No Index checking performed
CUDA_BinaryFkt(arr_subsref_vec,{c[idx]=a[(int) b[idx]];})
CUDA_BinaryFkt(carr_subsref_vec,{c[2*idx]=a[2*((int) b[idx])];c[2*idx+1]=a[2*((int) b[idx])+1];})

CUDA_BinaryFkt(arr_subsasg_vec,{c[(int) b[idx]]=a[idx];})
CUDA_BinaryFkt(carr_subsasg_vec,{c[2*((int) b[idx])]=a[2*idx];c[2*((int) b[idx])+1]=a[2*idx+1];})

// binary logical operations

CUDA_BinaryFkt(arr_or_arr,{c[idx]=(float) (a[idx]!=0) || (b[idx]!=0);})
CUDA_UnaryFktConst(arr_or_const,{c[idx]=(float) (a[idx]!=0) || (b!=0);})
CUDA_UnaryFktConst(const_or_arr,{c[idx]=(float) (b!=0) || (a[idx]!=0);})

CUDA_BinaryFkt(arr_and_arr,{c[idx]=(float) (a[idx]!=0) && (b[idx]!=0);})
CUDA_UnaryFktConst(arr_and_const,{c[idx]=(float) (a[idx]!=0) && (b!=0);})
CUDA_UnaryFktConst(const_and_arr,{c[idx]=(float) (b!=0) && (a[idx]!=0);})

// Unary logical operations
CUDA_UnaryFkt(not_arr,c[idx]=(a[idx] == 0);)

// Comparison
CUDA_BinaryFkt(arr_smaller_arr,c[idx]=a[idx]<b[idx];)
CUDA_UnaryFktConst(arr_smaller_const,c[idx]=a[idx]<b;)
CUDA_UnaryFktConst(const_smaller_arr,c[idx]=b<a[idx];)

CUDA_BinaryFkt(arr_larger_arr,c[idx]=a[idx]>b[idx];)
CUDA_UnaryFktConst(arr_larger_const,c[idx]=a[idx]>b;)
CUDA_UnaryFktConst(const_larger_arr,c[idx]=b>a[idx];)

CUDA_BinaryFkt(arr_smallerequal_arr,c[idx]=a[idx]<=b[idx];)
CUDA_UnaryFktConst(arr_smallerequal_const,c[idx]=a[idx]<=b;)
CUDA_UnaryFktConst(const_smallerequal_arr,c[idx]=b<=a[idx];)

CUDA_BinaryFkt(arr_largerequal_arr,c[idx]=a[idx]>=b[idx];)
CUDA_UnaryFktConst(arr_largerequal_const,c[idx]=a[idx]>=b;)
CUDA_UnaryFktConst(const_largerequal_arr,c[idx]=b>=a[idx];)

// equals will always output a real valued array
CUDA_BinaryFkt(arr_equals_arr,c[idx]=(a[idx]==b[idx]);)
CUDA_BinaryFkt(carr_equals_carr, int idc=2*idx; c[idx]=(a[idc]==b[idc]) && (a[idc+1]==b[idc+1]);)
CUDA_BinaryFkt(carr_equals_arr,int idc=2*idx; c[idx]=(a[idc]==b[idx]) && (a[idc+1] == 0);)
CUDA_BinaryFkt(arr_equals_carr,int idc=2*idx; c[idx]=(a[idx]==b[idc]) && (b[idc+1] == 0);)
CUDA_UnaryFktConst(arr_equals_const,c[idx]=(a[idx]==b);)
CUDA_UnaryFktConst(const_equals_arr,c[idx]=(b==a[idx]);)
CUDA_UnaryFktConstC(carr_equals_const,int idc=2*idx; c[idx]=(a[idc]==br) && (a[idc+1]==bi);)
CUDA_UnaryFktConstC(const_equals_carr,int idc=2*idx; c[idx]=(br==a[idc]) && (bi==a[idc+1]);)
CUDA_UnaryFktConstC(arr_equals_Cconst,c[idx]=(a[idx]==br) && (bi==0);)
CUDA_UnaryFktConstC(Cconst_equals_arr,c[idx]=(br==a[idx]) && (bi==0);)

// not equals will always output a real valued array
CUDA_BinaryFkt(arr_unequals_arr,c[idx]=(a[idx]!=b[idx]);)
CUDA_BinaryFkt(carr_unequals_carr, int idc=2*idx; c[idx]=(a[idc]!=b[idc]) || (a[idc+1]!=b[idc+1]);)
CUDA_BinaryFkt(carr_unequals_arr,c[idx]=(a[2*idx]!=b[idx]) || (a[2*idx+1] != 0);)
CUDA_BinaryFkt(arr_unequals_carr,c[idx]=(a[idx]!=b[2*idx]) || (b[2*idx+1] != 0);)
CUDA_UnaryFktConst(arr_unequals_const,c[idx]=(a[idx]!=b);)
CUDA_UnaryFktConst(const_unequals_arr,c[idx]=(b!=a[idx]);)
CUDA_UnaryFktConstC(carr_unequals_const,c[idx]=(a[2*idx]!=br) || (a[2*idx+1]!=bi);)
CUDA_UnaryFktConstC(const_unequals_carr,c[idx]=(br!=a[2*idx]) || (bi!=a[2*idx+1]);)
CUDA_UnaryFktConstC(arr_unequals_Cconst,c[idx]=(a[idx]!=br) || (bi!=0);)
CUDA_UnaryFktConstC(Cconst_unequals_arr,c[idx]=(br!=a[idx]) || (bi!=0);)

// other Unary oparations
CUDA_UnaryFkt(uminus_arr,c[idx]=-a[idx];)
CUDA_UnaryFkt(uminus_carr,int idc=2*idx; c[idc]=-a[idc];c[idc+1]=-a[idc+1];)   // negates real and imaginary part

CUDA_UnaryFkt(exp_arr,c[idx]= exp(a[idx]);)
CUDA_UnaryFkt(exp_carr,int idc=2*idx; float len=exp(a[idc]);c[idc]=len*cos(a[idc+1]);c[idc+1]=len*sin(a[idc+1]);)

CUDA_UnaryFkt(sin_arr,c[idx]= sin(a[idx]);)
CUDA_UnaryFkt(sin_carr,int idc=2*idx; c[idc]=sin(a[idc])*cosh(a[idc+1]);c[idc+1]=cos(a[idc])*sinh(a[idc+1]);)

CUDA_UnaryFkt(cos_arr,c[idx]= cos(a[idx]);)
CUDA_UnaryFkt(cos_carr,int idc=2*idx; c[idc]=cos(a[idc])*cosh(a[idc+1]);c[idc+1]=sin(a[idc])*sinh(a[idc+1]);)

CUDA_UnaryFkt(sinh_arr,c[idx]= sinh(a[idx]);)
CUDA_UnaryFkt(sinh_carr,int idc=2*idx; c[idc]=sinh(a[idc])*cos(a[idc+1]);c[idc+1]=cosh(a[idc])*sin(a[idc+1]);)

CUDA_UnaryFkt(cosh_arr,c[idx]= cosh(a[idx]);)
CUDA_UnaryFkt(cosh_carr,int idc=2*idx; c[idc]=cosh(a[idc])*cos(a[idc+1]);c[idc+1]=sinh(a[idc])*sin(a[idc+1]);)

CUDA_UnaryFkt(sinc_arr, c[idx]= (a[idx] != 0) ? sin(a[idx])/a[idx] : 1.0;)
CUDA_UnaryFkt(sinc_carr,int idc=2*idx; c[idc]=0;c[idc+1]=0;) 
// c[idc]= (a[idc] == 0) ? sin(a[idc])*cosh(a[idc+1])/a[idc] : cosh(a[idc+1]);c[idc+1]= (a[idc] == 0) ? cos(a[idc])*sinh(a[idc+1])/a[idc] : sinh(a[idc+1]);)

CUDA_UnaryFkt(log_arr,c[idx]=log(a[idx]);)
CUDA_UnaryFkt(log_carr,c[2*idx]=log(a[2*idx]);c[2*idx+1]=0;)   //  not implemented

CUDA_UnaryFkt(abs_arr,c[idx]= (a[idx] > 0) ? a[idx] : -a[idx];)
CUDA_UnaryFkt(abs_carr,int idc=2*idx; c[idx]=sqrt(a[idc]*a[idc]+a[idc+1]*a[idc+1]);)

CUDA_UnaryFkt(conj_arr,c[idx]=a[idx];)
CUDA_UnaryFkt(conj_carr,int idc=2*idx; c[idc]=a[idc];c[idc+1]=-a[idc+1];)  // only affects the imaginary part

CUDA_UnaryFkt(sqrt_arr,c[idx]= sqrt(a[idx]);)
// funny expression below is the sign function ((x>0)-(x<0))
CUDA_UnaryFkt(sqrt_carr,int idc=2*idx; float L=sqrt(a[idc]*a[idc]+a[idc+1]*a[idc+1]); c[idc]=sqrt((L+a[idc])/2);c[idc+1]=((a[idc+1]>0)-(a[idc+1])<0)*sqrt((L-a[idc])/2);)

// Unary functions resulting in just a single value
CUDA_UnaryFkt(isIllegal_arr,if (isnan(a[idx]) || isinf(a[idx]) ) c[0]=1;)
CUDA_UnaryFkt(isIllegal_carr,if (a[2*idx+1]!=0 || isnan(a[2*idx]) || isnan(a[2*idx+1]) || isinf(a[2*idx]) || isinf(a[2*idx+1]) ) c[0]=1;)

CUDA_UnaryFkt(any_arr,if (a[idx]!=0) c[0]=1;)
CUDA_UnaryFkt(any_carr,if (a[2*idx]!=0 || a[2*idx+1]!=0) c[0]=1;)

// Binary functions with real valued input returning always complex arrays
CUDA_BinaryFkt(arr_complex_arr,c[2*idx]=a[idx];c[2*idx+1]=b[idx];)
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


CUDA_UnaryFktIntVec(arr_circshift_vec,CoordsNDFromIdx(idx,sSize,pos);for(int _d=0;_d<CUDA_MAXDIM;_d++){pos.s[_d]-=b.s[_d];}int ids=0;IdxNDFromCoords(pos,sSize,ids);c[idx]=a[ids];)  // a[idx]
CUDA_UnaryFktIntVec(carr_circshift_vec,CoordsNDFromIdx(idx,sSize,pos);for(int _d=0;_d<CUDA_MAXDIM;_d++){pos.s[_d]-=b.s[_d];}int ids=0;IdxNDFromCoords(pos,sSize,ids);c[2*idx]=a[2*ids];c[2*idx+1]=a[2*ids+1];)

// In code below, the loop runs over the source dimensions. The array sizes are still set to the source sizes and will be (again) adjusted later
CUDA_UnaryFktIntVec(arr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos);
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=0;}
        for(_d=0;_d<CUDA_MAXDIM;_d++){
                if ((b.s[_d]<CUDA_MAXDIM) && (b.s[_d]>=0)) {
                        dSize.s[_d]=sSize.s[b.s[_d]]; posnew.s[_d] = pos.s[b.s[_d]];}
                } 
        int idd=0;IdxNDFromCoords(posnew,dSize,idd);c[idd]=a[idx];}) // a[idx]

CUDA_UnaryFktIntVec(carr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos);
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=0;}
        for(_d=0;_d<CUDA_MAXDIM;_d++){
                if ((b.s[_d]<CUDA_MAXDIM) && (b.s[_d]>=0)) {
                        dSize.s[_d]=sSize.s[b.s[_d]]; posnew.s[_d] = pos.s[b.s[_d]];}
                }
        int idd=0;IdxNDFromCoords(posnew,dSize,idd);c[2*idd]=a[2*idx];c[2*idd+1]=a[2*idx+1];}) 
/*
CUDA_UnaryFktIntVec(arr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos); \
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=pos.s[_d];} \
        for(_d=0;_d<CUDA_MAXDIM;_d++){ \
                if (b.s[_d]<CUDA_MAXDIM && b.s[_d]>=0) { \
                        dSize.s[b.s[_d]]=sSize.s[_d]; posnew.s[b.s[_d]] = pos.s[_d];} \
                } \
        int idd=0;IdxNDFromCoords(posnew,dSize,idd);c[idd]=a[idx];}) 

CUDA_UnaryFktIntVec(carr_permute_vec,{int _d;SizeND posnew; SizeND dSize; CoordsNDFromIdx(idx,sSize,pos); \
        for(_d=0;_d<CUDA_MAXDIM;_d++) {dSize.s[_d]=1;posnew.s[_d]=pos.s[_d];} \
        for(_d=0;_d<CUDA_MAXDIM;_d++){ \
                if (b.s[_d]<CUDA_MAXDIM && b.s[_d]>=0) { \
                        dSize.s[b.s[_d]]=sSize.s[_d]; posnew.s[b.s[_d]] = pos.s[_d];} \
                } \
        int idd=0;IdxNDFromCoords(posnew,dSize,idd);c[2*idd]=a[2*idx];c[2*idd+1]=a[2*idx+1];}) 
*/
        
CUDA_Fkt2Vec(arr_xyz_2vec,CoordsNDFromIdx(idx,sSize,pos);float val=0;for(int _d=0;_d<CUDA_MAXDIM;_d++){val += vec1.s[_d]+pos.s[_d]*(vec2.s[_d]-vec1.s[_d])/sSize.s[_d];} c[idx]=val;)  // a[idx]
CUDA_Fkt2Vec(arr_rr_2vec,CoordsNDFromIdx(idx,sSize,pos);float val=0;for(int _d=0;_d<CUDA_MAXDIM;_d++){val += Sqr(vec1.s[_d]+pos.s[_d]*(vec2.s[_d]-vec1.s[_d])/sSize.s[_d]);} c[idx]=sqrt(val);)  // a[idx]
CUDA_Fkt2Vec(arr_phiphi_2vec,CoordsNDFromIdx(idx,sSize,pos); c[idx]=atan2(vec1.s[0]+pos.s[0]*(vec2.s[0]-vec1.s[0])/sSize.s[0],vec1.s[1]+pos.s[1]*(vec2.s[1]-vec1.s[1])/sSize.s[1]);)  // phiphi

// Now include all the user-defined functions
// #include "user/user_cu_code.inc"
#include "user_cu_code.inc"


__global__ void set_arr(float b, float * c, int N)                          
{                                                                   
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   
	c[idx]=b;
}                                                                   
extern "C" const char * CUDAset_arr(float b, float * c, int N)  
{                                                                       
    cudaError_t myerr;                                                \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          
	set_arr<<<nBlocks,blockSize>>>(b,c,N);                            
  myerr=cudaGetLastError();                                             
  if (myerr != cudaSuccess)                                             
      return cudaGetErrorString(myerr);                                 
  return 0;                                                                   
}                                                                       

__global__ void set_carr(float br, float bi, float * c, int N)               
{                                                                   
	int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=N) return;   
    int idc=idx*2;                                                  
	c[idc]=br;c[idc+1]=bi;
}

extern "C" const char * CUDAset_carr(float br, float bi, float * c, int N)
{                                                                       
    cudaError_t myerr;                                                \
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);          
	set_carr<<<nBlocks,blockSize>>>(br,bi,c,N);                        
  myerr=cudaGetLastError();                                             
  if (myerr != cudaSuccess)                                             
      return cudaGetErrorString(myerr);                                 
  return 0;                                                             
}                                                                       


extern "C" unsigned long CUDAmaxSize() {
    int dev=0;
    cudaGetDevice(&dev);
    struct cudaDeviceProp prop;
    int status=cudaGetDeviceProperties(&prop,dev);

    // return prop.maxThreadsPerBlock;  // 512
    // return prop.multiProcessorCount;   // 30
    // return prop.warpSize;   // 32
    // return prop.maxThreadsDim[0];   // 512  = max blocksize
    // return prop.maxGridSize[0];   // 65535  = max GridSize = max nBlocks?
    return ((long)prop.maxGridSize[0])*((long)prop.maxThreadsDim[0]);   // 65535  = max GridSize = max nBlocks?
}


__global__ void
arr_times_const_checkerboard(float*a,float b, float * c, int N, int sx,int sy,int sz)
{
	int ids=blockIdx.x*blockDim.x+threadIdx.x;   // which source array element do I have to deal with?
	if(ids>=N) return;  // not in range ... quit

	int px=(ids/2)%sx;   // my x pos
	int py=(ids/2)/sx;   // my y pos
    float minus1=(1-2*((px+py)%2));
	c[ids]=a[ids]*b*minus1;
}

extern "C" int CUDAarr_times_const_checkerboard(float * a, float b, float * c, int * sizes, int dims)  // multiplies with a constand and scrambles the array
{
    int sx=sizes[0],sy=1,sz=1;
    if (dims>1)
        sy=sizes[1];
    if (dims>2)
        sz=sizes[2];
    int N=sx*sy*sz*2;  // every pair will be processed exactly once
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);	// add extra block if N can't be divided by blockSize
	arr_times_const_checkerboard<<<nBlocks,blockSize>>>(a,b,c,N,sx,sy,sz);
	return 0;
}


/// cyclicly rotates datastack cyclic into positive direction in all coordinates by (dx,dy,dz) voxels
/// simple version with all processors dealing with exactly one element
__global__ void
rotate2(float*a,float b, float * c, int sx,int sy,int sz, int dx, int dy, int dz)
{
  int ids=(blockIdx.x*blockDim.x+threadIdx.x); // id of this processor
  int x=(ids + dx)%sx;  // advance by the offset steps along the chain
  int y=(ids/sx + dy)%sy;
  int z=(ids/(sx*sy) + dz)%sz;
  int idd=x+sx*y+sx*sy*z;
  if(ids>=sx*sy*sz) return;
  // float tmp = a[ids];
  // __syncthreads();             // nice try but does not work !
  c[idd] = b*a[ids];
}

/// cyclicly rotates datastack cyclic into positive direction in all coordinates by (dx,dy,dz) voxels
__global__ void
rotate(float*a,float b, float * c, int sx,int sy,int sz, int dx, int dy, int dz, int ux, int uy, int uz)
{
  int id=(blockIdx.x*blockDim.x+threadIdx.x); // id of this processor

  int Processes=blockDim.x * gridDim.x;
  int chains=ux*uy*uz; // total number of independent chains
  int N=sx*sy*sz;  // total size of array, has to be chains*length_of_chain
  int length=N/chains;  // chain length
  int steps=N/Processes;  // this is how many steps each processor has to do

  int step,nl,nx,ny,nz,x,y,z,i,idd;
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

int gcd(int a, int b) // greatest commod divisor by recursion
{ 
   return ( b == 0 ? a : gcd(b, a % b) ); 
}

extern "C" int CUDAarr_times_const_rotate(float * a, float b, float * c, int * sizes, int dims, int complex,int direction)  // multiplies with a constand and cyclilcally rotates the array using the chain algorithm
{
    int sx=1,sy=1,sz=1;
    if (dims>0)
        sx=sizes[0];
    if (dims>1)
        {sx=sizes[0];sy=sizes[1];}
    if (dims>2)
        sz=sizes[2]; 

    int dx=(sx+direction*sx/2)%sx,dy=(sy+direction*sy/2)%sy,dz=(sz+direction*sz/2)%sz;  // how much to cyclically rotate
    if (complex) {sx=sx*2;dx=dx*2;}
    //printf("sx %d sy %d dx %d dy %d\n",sx,sy,dx,dy);

    // calculate the length of each swapping chain
    int ux=gcd(sx,dx);  // unit cell in x. Any repeat along y directions will be also a repeat in x. Chain length is sx/ux
    // int lx=sx/ux; // how many accesses to get one round in x
    int uy=gcd(((sx/ux)*dy%sy),sy); // how many times must the first chain be repeated to form a longer chain. This defines unit cell y
    int uz=gcd(((sy/uy)*dz%sz),sz); // similar for z
    int length=sx*sy*sz/(ux*uy*uz);  // chain length

    // in one dimension the gcd=greatest common divisor, would mean that one has to start task at position 0 ... gcd-1
    // in several dimensions even completing one round leaving a spacing at gcd does not mean that this is a complete loop
    // however it could be a complete loop. The number of steps that where performed in the lower dimension are s/gcd before reaching the beginning again
    // with the size of the dimension s. If we are at the same startingpoint in the next dimension the chain is complete.
    // So the number of times a super chain (in 2D) must be executed is sy/gcd(sy,s/gcd(sx,dx))
    int dev=0;
    cudaGetDevice(&dev);
    struct cudaDeviceProp prop;
    int status=cudaGetDeviceProperties(&prop,dev);

    int m=1;
    if (ux>uy)
        m=ux;
    else
        m=uy;
    if (uz>m)
        m=uz;
    if (length>m)
        m=length;

    //int blockSize=1; // prop.warpSize; // ux*uy*uz;
    //int nBlocks=m;	// add extra block if N can't be divided by blockSize
    
    //    rotate<<<nBlocks,blockSize>>>(a,b,c,sx,sy,sz,dx,dy,dz,ux,uy,uz);  // get unit cell sizes

    int N=sx*sy*sz; // includes the space for coomplex numbers
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);	// add extra block if N can't be divided by blockSize
                                                                //    printf("BlockSize %d, ux %d, uy %d, uz %d\n",blockSize,ux,uy,uz);
    // unfortunately we have to do it out of place.
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
arr_times_const_scramble(float*a,float b, float * c, int sx,int sy,int sz, int ox, int oy, int oz)
{
	int pnum=blockIdx.x*blockDim.x+threadIdx.x;   // which source array element do I have to deal with?

	int px=pnum%(sx/2);   // my x pos of a complex number in the subarray
	int py=pnum/(sx/2);   // my y pos of a complex number
	if(px>=(sx/2) || py >= (sy/2)) return;  // not in range ... quit
    int ids=2*(px+py*sx);  /// offset to array start in floats
    int idd=2*((ox+px)+(oy+py)*sx);

    // echange two values using a tmp
    float tmpR = c[idd];
    float tmpI = c[idd+1];
    c[idd]=a[ids]; // (float)(ox+px); // 
    c[idd+1]=a[ids+1]; // (float)(oy+py); // 
    a[ids]=tmpR;
    a[ids+1]=tmpI;
}

__global__ void
array_copy(float*a, float * c, int mx, int my, int mz, int sx,int sy,int sz, int ox, int oy, int oz)  // copies between two memories with different strides
{
	int pnum=blockIdx.x*blockDim.x+threadIdx.x;   // which source array element do I have to deal with?

	int px=pnum%(sx/2);   // my x pos of a complex number in the subarray
	int py=pnum/(sx/2);   // my y pos of a complex number
	if(px>=sx || py >= (sy/2)) return;  // not in range ... quit
    int ids=2*(px+py*sx);  /// offset to array start in floats
    int idd=2*((ox+px)+(oy+py)*sx);

    // echange two values using a tmp
    float tmpR = c[idd];
    float tmpI = c[idd+1];
    c[idd]=a[ids]; // (float)(ox+px); // 
    c[idd+1]=a[ids+1]; // (float)(oy+py); // 
    a[ids]=tmpR;
    a[ids+1]=tmpI;
}


extern "C" int CUDAarr_times_const_scramble(float * a, float b, float * c, int * sizes, int dims)  // multiplies with a constand and scrambles the array
{
    int sx=sizes[0],sy=1,sz=1, iseven=1;
    if (sx%2 == 1) iseven=0;
    if (dims>1) {
        sy=sizes[1];
        if (sy%2 == 1) iseven=0;
        }

    if (dims>2) {
        sz=sizes[2];
        if (sz%2 == 1) iseven=0;
        }
    int N=sx*sy*sz*2;  // every pair will be processed exactly once
	int blockSize=BLOCKSIZE; int nBlocks=NBLOCKS(N,blockSize);	// add extra block if N can't be divided by blockSize

    if (! iseven)
        {
            float * tmp=0;
            cudaMalloc((void **) &tmp,sizeof(tmp[0])*(1+sx/2)*(1+sy/2));
        }
	arr_times_const_scramble<<<nBlocks,blockSize>>>(a,b,c,sx,sy,sz,sx/2,sy/2,0);
	arr_times_const_scramble<<<nBlocks,blockSize>>>(a+2*(sx/2),b,c+2*(sx/2),sx,sy,sz,-sx/2,sy/2,0);
	return 0;
}


