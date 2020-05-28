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
*/
#ifndef externC
#define externC extern "C"
#endif

#define CUDA_MAXDIM 5   // is needed for transferring parameters, CAREFUL! This uses for some function this number of variables on each processor
#define CUDA_MAXPROJ 5  // needed for projections
#define ACCUTYPE double   // is used in cudaArith.cu and cuda_cuda.c  to define the type in which accumulation operations are computed

// THESE STRUCT DEFINITIONS ARE NEEDED, AS CUDA CANNOT DEAL CORRECTLY WITH FIXED LENGTH ARRAYS IN THE ARGUMENT
// ACCESING THEM WILL CAUSE A CRASH!
// HOWEVER, STRUCTS WITH THE ARRAY INSIDE ARE OK
// Also inside a Cuda function one has to use the structure rather than an array of fixed size.
typedef struct {
    size_t s[CUDA_MAXDIM];
} SizeND ;

typedef struct {
    long long s[CUDA_MAXDIM];
} IntND ;  // for integer shifts that can be positive or negative

typedef struct {
    unsigned char s[CUDA_MAXDIM];
} BoolND ;

typedef struct {
    float s[CUDA_MAXDIM];
} VecND ;

typedef struct {
    size_t s[5];
} Size5D ;

typedef struct {
    size_t s[3];
} Size3D ;

static const Size5D Size5DOnes={1,1,1,1,1};
static const Size5D Size5DZeros={0,0,0,0,0};

externC size_t CUDAmaxSize(void);   // returns the maximal total number of threads
externC size_t GetCurrentRedSize(void);  // returns the current size of the allocated ReduceArray(s).
externC int GetMaxThreads(void);  // returns the maximal number of threads per block
externC int ReduceThreadsDef(void);
externC long GetMaxBlocksX(void);  // returns the maxmimal number of blocks along X
externC struct cudaDeviceProp GetDeviceProp(void);  // returns the device properties of the active cuda device
externC const char * SetDeviceProperties(void);  // initializes a static variable with the device properties

externC const char * CUDAsumpos_arr(float * a, size_t N, ACCUTYPE * resp);
externC const char * CUDAsum_arr(float * a, size_t N, ACCUTYPE * resp);
externC const char * CUDAsum_carr(float * a, size_t N, ACCUTYPE * resp);  // N refers to nuhmber of complex entries, resp needs to point to two floats
externC const char * CUDAmax_arr(float * a, size_t N, ACCUTYPE * resp);   // returns max and index of  maximum, resp needs to point to two floats
externC const char * CUDAmin_arr(float * a, size_t N, ACCUTYPE * resp);   // returns max and index of  maximum, resp needs to point to two floats

// partial reductions
externC const char * CUDApsum_arr(float * a, float * mask,float * c, size_t sSize[5], int ProjDir);  // partial reductions (projections)
externC const char * CUDApsum_carr(float * a, float * mask,float * c, size_t sSize[5], int ProjDir);  // partial reductions (projections)
externC const char * CUDApmax_arr(float * a, float * mask,float * c, float * cIdx, size_t sSize[5], int ProjDir); 
externC const char * CUDApmin_arr(float * a, float * mask,float * c, float * cIdx, size_t sSize[5], int ProjDir); 

// Assigning with modulo interpretation in source (for repmat)
externC const char * CUDAarr_3drepcpy_arr(float *a, float * c, size_t sSize[3], size_t dSize[3]);
externC const char * CUDAcarr_3drepcpy_carr(float *a, float * c, size_t sSize[3], size_t dSize[3]);

externC const char * CUDAarr_3dsubcpy_arr(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAcarr_3dsubcpy_carr(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAarr_3dsubcpy_carr(float * a, float * c, size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);

externC const char * CUDAconst_3dsubcpy_arr(float * c, float br, float bi, size_t dSize[3], size_t dROI[3], size_t dOffs[3]);
externC const char * CUDAcconst_3dsubcpy_carr(float * c, float br, float bi, size_t dSize[3], size_t dROI[3], size_t dOffs[3]);

externC const char * CUDAarr_3dsubcpyT_arr(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAcarr_3dsubcpyT_carr(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAcarr_3dsubcpyCT_carr(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);

// 5d versions of the same
externC const char * CUDAarr_5drepcpy_arr(float * a, float * c,size_t sSize[5], size_t dSize[5]);
externC const char * CUDAcarr_5drepcpy_carr(float * a, float * c, size_t sSize[5], size_t dSize[5]);

externC const char * CUDAarr_5dsubcpy_arr(float * a, float * c, Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep);
externC const char * CUDAcarr_5dsubcpy_carr(float * a, float * c,Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep);
externC const char * CUDAarr_5dsubcpy_carr(float * a, float * c, Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep);

externC const char * CUDAconst_5dsubcpy_arr(float * c, float br, float bi, Size5D dSize, Size5D dROI, Size5D dOffs, Size5D dStep);
externC const char * CUDAcconst_5dsubcpy_carr(float * c, float br, float bi,Size5D dSize, Size5D dROI, Size5D dOffs, Size5D dStep);

externC const char * CUDAarr_5dsubcpyT_arr(float * a, float * c,Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep);
externC const char * CUDAcarr_5dsubcpyT_carr(float * a, float * c,Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep);
externC const char * CUDAcarr_5dsubcpyCT_carr(float * a, float * c,Size5D sSize, Size5D dSize, Size5D sOffs, Size5D sROI, Size5D dOffs, Size5D sStep, Size5D dStep);

// 
externC const char * CUDAarr_boolassign_const(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_boolassign_const(float * a, float br, float bi,  float * c, size_t N);

externC const char * CUDAarr_complex_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);  // real in complex out
externC const char * CUDAarr_complex_const(float * a, float b, float * c, size_t N);  // real in complex out
externC const char * CUDAconst_complex_arr(float * a, float b, float * c, size_t N);  // real in complex out
// Power
externC const char * CUDAarr_power_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_power_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_power_arr(float * a, float b, float * c, size_t N);
externC const char * CUDACconst_power_arr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_power_carr(float * a, float br, float bi, float * c, size_t N);

// Unary functions with one return value
externC const char * CUDAisIllegal_arr(float * a, float * c, size_t N);  // Size of c has to only be one
externC const char * CUDAany_arr(float * a, float * c, size_t N);  // Size of c has to only be one
externC const char * CUDAisIllegal_carr(float * a, float * c, size_t N);  // Size of c has to only be one
externC const char * CUDAany_carr(float * a, float * c, size_t N);  // Size of c has to only be one


// Multiplication
externC const char * CUDAarr_times_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_times_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_times_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_times_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_times_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_times_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_times_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_times_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_times_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_times_arr(float * a, float br, float bi, float * c, size_t N);

// Division
externC const char * CUDAarr_divide_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_divide_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_divide_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_divide_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_divide_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_divide_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_divide_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_divide_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_divide_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_divide_arr(float * a, float br, float bi, float * c, size_t N);

// Element-wise maximum operation
externC const char * CUDAarr_max_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_max_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_max_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_max_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_max_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_max_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_max_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_max_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_max_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_max_arr(float * a, float br, float bi, float * c, size_t N);

// Element-wise minimum operation
externC const char * CUDAarr_min_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_min_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_min_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_min_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_min_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_min_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_min_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_min_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_min_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_min_arr(float * a, float br, float bi, float * c, size_t N);

// Addition
externC const char * CUDAarr_plus_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_plus_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_plus_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_plus_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_plus_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_plus_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_plus_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_plus_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_plus_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_plus_arr(float * a, float br, float bi, float * c, size_t N);

// Subtraction
externC const char * CUDAarr_minus_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_minus_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_minus_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_minus_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_minus_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_minus_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_minus_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_minus_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_minus_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_minus_arr(float * a, float br, float bi, float * c, size_t N);


// Referencing with another real-valued array (interpretet as boolean)
externC const char * CUDAarr_subsref_arr(float * in, float * mask, float *  out, size_t N, size_t * pM);
externC const char * CUDAcarr_subsref_arr(float * in, float * mask, float *  out, size_t N, size_t * pM);

externC const char * CUDAarr_subsref_ind(float * a, float * b, float * c, size_t N, size_t M);
externC const char * CUDAcarr_subsref_ind(float * a, float * b, float * c, size_t N, size_t M);
externC const char * CUDAarr_subsasgn_ind(float * a, float * b, float * c, size_t N, size_t M);
externC const char * CUDAcarr_subsasgn_ind(float * a, float * b, float * c, size_t N, size_t M);

externC const char * CUDAarr_conv_arr(float * a, float * b, float *  c, SizeND SA, SizeND SB, size_t SIDX, size_t DA);
externC const char * CUDAcarr_conv_arr(float * a, float * b, float *  c, SizeND SA, SizeND SB, size_t SIDX, size_t DA);

externC const char * CUDAarr_subsrefND_ind(float * a, float * b, float * c, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
externC const char * CUDAcarr_subsrefND_ind(float * a, float * b, float * c, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
externC const char * CUDAarr_subsasgnND_ind(float * a, float * b, float * c, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
externC const char * CUDAcarr_subsasgnND_ind(float * a, float * b, float * c, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
externC const char * CUDAarr_subsasgnND_const_ind(float * a, float * b, float c, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
externC const char * CUDAcarr_subsasgnND_const_ind(float * a, float * b, float cr, float ci, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
externC const char * CUDAarr_subsasgnND_Cconst_ind(float * a, float * b, float cr, float ci, SizeND SA, SizeND SC, size_t SIDX, size_t DA);
// externC const char * CUDAcarr_subsasgnND_const_ind(float * a, float * b, float cr, float ci, SizeND SA, SizeND SC, size_t SIDX, size_t DA);

// Line below: Wrapping does not make sense, so this is just for compatibility
externC const char * CUDAarr_subsasgn_const(float * a, float b, float * c, size_t N);
externC const char * CUDAarr_subsasgn_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAcarr_subsasgn_const(float * a, float br, float bi, float * c, size_t N);

// diagonal matrix generation
externC const char * CUDAarr_diag_set(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAcarr_diag_set(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAarr_diag_get(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);
externC const char * CUDAcarr_diag_get(float * a, float * c,size_t sSize[3], size_t dSize[3], size_t sOffs[3], size_t sROI[3], long long dOffs[3]);

// referencing and assignment using an index vector (b), N always refers to the output size. No index checking!
externC const char * CUDAarr_subsref_vec(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_subsref_vec(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_subsasgn_vec(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_subsasgn_vec(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

// Assigning to by another real-valued array (interpretet as boolean)
externC const char * CUDAarr_subsasgn_arr(float * in, float * mask, float *  out, size_t N, size_t * pM);
externC const char * CUDAcarr_subsasgn_arr(float * in, float * mask, float *  out, size_t N, size_t * pM);

// or operation
externC const char * CUDAarr_or_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_or_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_or_arr(float * a, float b, float * c, size_t N);

// and operation
externC const char * CUDAarr_and_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_and_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_and_arr(float * a, float b, float * c, size_t N);

// not operation
externC const char * CUDAnot_arr(float * a, float * c, size_t N);

// sign operation
externC const char * CUDAsign_arr(float * a, float * c, size_t N);
externC const char * CUDAsign_carr(float * a, float * c, size_t N);

// smaller than
externC const char * CUDAarr_smaller_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_smaller_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_smaller_arr(float * a, float b, float * c, size_t N);

// larger than
externC const char * CUDAarr_larger_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_larger_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_larger_arr(float * a, float b, float * c, size_t N);

// smaller or equal
externC const char * CUDAarr_smallerequal_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_smallerequal_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_smallerequal_arr(float * a, float b, float * c, size_t N);

// larger or equal
externC const char * CUDAarr_largerequal_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_largerequal_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_largerequal_arr(float * a, float b, float * c, size_t N);

// ==
externC const char * CUDAarr_equals_arr(float * a, float * b, float * c, size_t N, int numdims,SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_equals_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_equals_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_equals_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_equals_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_equals_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_equals_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_equals_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_equals_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_equals_arr(float * a, float br, float bi, float * c, size_t N);

externC const char * CUDAdiff_carrSizeConst(float * a, float * c, float b, SizeND SA, SizeND SC, size_t N);  // b is the stride. (offset)
externC const char * CUDAdiff_arrSizeConst(float * a, float * c, float b, SizeND SA, SizeND SC, size_t N);

// !=   ~=
externC const char * CUDAarr_unequals_arr(float * a, float * b, float * c, size_t N, int numdims,SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_unequals_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAcarr_unequals_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_unequals_carr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);

externC const char * CUDAarr_unequals_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_unequals_arr(float * a, float b, float * c, size_t N);
externC const char * CUDAcarr_unequals_const(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAconst_unequals_carr(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDAarr_unequals_Cconst(float * a, float br, float bi, float * c, size_t N);
externC const char * CUDACconst_unequals_arr(float * a, float br, float bi, float * c, size_t N);

// other Unary operations
externC const char * CUDAset_arr(float b, float * c, size_t N);
externC const char * CUDAset_carr(float br, float bi, float * c, size_t N);

externC const char * CUDAuminus_arr(float * a, float * c, size_t N);
externC const char * CUDAuminus_carr(float * a, float * c, size_t N);

externC const char * CUDAround_arr(float * a, float * c, size_t N);
externC const char * CUDAround_carr(float * a, float * c, size_t N);

externC const char * CUDAfloor_arr(float * a, float * c, size_t N);
externC const char * CUDAfloor_carr(float * a, float * c, size_t N);

externC const char * CUDAceil_arr(float * a, float * c, size_t N);
externC const char * CUDAceil_carr(float * a, float * c, size_t N);

externC const char * CUDAexp_arr(float * a, float * c, size_t N);
externC const char * CUDAexp_carr(float * a, float * c, size_t N);

externC const char * CUDAsin_arr(float * a, float * c, size_t N);
externC const char * CUDAsin_carr(float * a, float * c, size_t N);

externC const char * CUDAcos_arr(float * a, float * c, size_t N);
externC const char * CUDAcos_carr(float * a, float * c, size_t N);

externC const char * CUDAtan_arr(float * a, float * c, size_t N);

// externC const char * CUDAtan_carr(float * a, float * c, size_t N);

externC const char * CUDAsinc_arr(float * a, float * c, size_t N);
externC const char * CUDAsinc_carr(float * a, float * c, size_t N);

externC const char * CUDAsinh_arr(float * a, float * c, size_t N);
externC const char * CUDAsinh_carr(float * a, float * c, size_t N);

externC const char * CUDAcosh_arr(float * a, float * c, size_t N);
externC const char * CUDAcosh_carr(float * a, float * c, size_t N);

externC const char * CUDAabs_arr(float * a, float * c, size_t N);
externC const char * CUDAabs_carr(float * a, float * c, size_t N);

externC const char * CUDAarr_besselj_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_besselj_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_besselj_arr(float * a, float b, float * c, size_t N);

externC const char * CUDAarr_atan2_arr(float * a, float * b, float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingletonA, BoolND isSingletonB);
externC const char * CUDAarr_atan2_const(float * a, float b, float * c, size_t N);
externC const char * CUDAconst_atan2_arr(float * a, float b, float * c, size_t N);

externC const char * CUDAlog_arr(float * a, float * c, size_t N);
externC const char * CUDAlog_carr(float * a, float * c, size_t N);

externC const char * CUDAlog10_arr(float * a, float * c, size_t N);
externC const char * CUDAlog10_carr(float * a, float * c, size_t N);

externC const char * CUDAconj_arr(float*a,float * c, size_t N);
externC const char * CUDAconj_carr(float*a,float * c, size_t N);

externC const char * CUDAsqrt_arr(float*a,float * c, size_t N);
externC const char * CUDAsqrt_carr(float*a,float * c, size_t N);

externC const char * CUDAreal_arr(float * a, float * c, size_t N);
externC const char * CUDAreal_carr(float * a, float * c, size_t N);

externC const char * CUDAimag_arr(float * a, float * c, size_t N);
externC const char * CUDAimag_carr(float * a, float * c, size_t N);

externC const char * CUDAphase_arr(float * a, float * c, size_t N);
externC const char * CUDAphase_carr(float * a, float * c, size_t N);

externC const char * CUDAisnan_arr(float * a, float * c, size_t N);
externC const char * CUDAisnan_carr(float * a, float * c, size_t N);

externC const char * CUDAisinf_arr(float * a, float * c, size_t N);
externC const char * CUDAisinf_carr(float * a, float * c, size_t N);

externC int CUDAarr_times_const_checkerboard(float * a, float b, float * c, size_t * sizes, int dims);  // multiplies with a constand and scrambles the array
externC int CUDAarr_times_const_scramble(float * a, float b, float * c, size_t * sizes, int dims);  // multiplies with a constand and scrambles the array
externC int CUDAarr_times_const_rotate(float * a, float b, float * c, SizeND mySize, SizeND DirYes, int dims, int complex,int direction);  // rotates an array in place or out of place by the distance [dx,dy,dz]

externC const char * CUDAarr_circshift_vec(float * a,size_t nshifts[CUDA_MAXDIM],float * c, size_t mySize[CUDA_MAXDIM],size_t N);    // cyclic rotation
externC const char * CUDAcarr_circshift_vec(float * a,size_t nshifts[CUDA_MAXDIM],float * c, size_t mySize[CUDA_MAXDIM],size_t N);   // cyclic rotation by the vector nshifts

externC const char * CUDAarr_permute_vec(float * a,size_t nshifts[CUDA_MAXDIM],float * c, size_t mySize[CUDA_MAXDIM],size_t N);    // cyclic rotation
externC const char * CUDAcarr_permute_vec(float * a,size_t nshifts[CUDA_MAXDIM],float * c, size_t mySize[CUDA_MAXDIM],size_t N);   // cyclic rotation by the vector nshifts

externC const char * CUDAarr_xyz_2vec(float * c, VecND vec1, VecND vec2, SizeND sSize, size_t N);    // generates xx, yy and zz
externC const char * CUDAarr_rr_2vec(float * c, VecND vec1, VecND vec2, SizeND sSize, size_t N);    // generates rr, first vec for center second for scaling
externC const char * CUDAarr_phiphi_2vec(float * c, VecND vec1, VecND vec2, SizeND sSize, size_t N);    // generates rr, first vec for center second for scaling

externC const char * CUDAsvd3D_last(float *X, float *Ye, float * Yv, size_t N);  // singular value decomposition along the last array dimension
externC const char * CUDAsvd3D_recomp(float *Y, float *E, float * V, size_t N);  // N is NOT the total size, but only the size excluding the last dimension (of size 3)

externC const char * CUDAsvd2D_last(float *X, float *Ye, float * Yv, size_t N);  // singular value decomposition along the last array dimension
externC const char * CUDAsvd2D_recomp(float *Y, float *E, float * V, size_t N);  // N is NOT the total size, but only the size excluding the last dimension (of size 2)

externC const char * CUDA3DConv(float *deviceInputImageData, float *deviceMaskData, float * deviceOutputImageData, SizeND ImgSize);  // some code for fast convolution with a small kernel  

externC const char * CUDAHessianCyc(float * a, float * c, SizeND SA, SizeND SC, size_t N, size_t ndims); // cyclic Hessian for n dims

// Now include all the user-defined functions
#include "user_h_code.inc"
