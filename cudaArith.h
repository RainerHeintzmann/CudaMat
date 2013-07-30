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

// THESE STRUCT DEVINITION ARE NEEDED, AS CUDA CANNOT DEAL CORRECTLY WITH FIXED LENGTH ARRAYS IN THE ARGUMENT
// ACCESING THEM WILL CAUSE A CRASH!
// HOWEVER, STRUCTS WITH THE ARRAY INSIDE ARE OK
// Also inside a Cuda function one has to use the structure rather than an array of fixed size.
typedef struct {
    int s[CUDA_MAXDIM];
} SizeND ;

typedef struct {
    float s[CUDA_MAXDIM];
} VecND ;

externC unsigned long CUDAmaxSize(void);   // returns the maximal total number of threads
externC int GetCurrentRedSize(void);  // returns the current size of the allocated ReduceArray(s).
        
externC const char * CUDAsum_arr(float * a, int N, float * resp);
externC const char * CUDAsum_carr(float * a, int N, float * resp);  // N refers to nuhmber of complex entries, resp needs to point to two floats
externC const char * CUDAmax_arr(float * a, int N, float * resp);   // returns max and index of  maximum, resp needs to point to two floats
externC const char * CUDAmin_arr(float * a, int N, float * resp);   // returns max and index of  maximum, resp needs to point to two floats

// partial reductions
externC const char * CUDApsum_arr(float * a, float * mask,float * c, int sSize[5], int ProjDir);  // partial reductions (projections)
externC const char * CUDApsum_carr(float * a, float * mask,float * c, int sSize[5], int ProjDir);  // partial reductions (projections)
externC const char * CUDApmax_arr(float * a, float * mask,float * c, float * cIdx, int sSize[5], int ProjDir); 
externC const char * CUDApmin_arr(float * a, float * mask,float * c, float * cIdx, int sSize[5], int ProjDir); 

// Assigning with modulo interpretation in source (for repmat)
externC const char * CUDAarr_3drepcpy_arr(float * a, float * c,int sSize[3], int dSize[3]);
externC const char * CUDAcarr_3drepcpy_carr(float * a, float * c, int sSize[3], int dSize[3]);

externC const char * CUDAarr_3dsubcpy_arr(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAcarr_3dsubcpy_carr(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAarr_3dsubcpy_carr(float * a, float * c, int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);

externC const char * CUDAconst_3dsubcpy_arr(float * c, float br, float bi, int dSize[3], int dROI[3], int dOffs[3]);
externC const char * CUDAcconst_3dsubcpy_carr(float * c, float br, float bi, int dSize[3], int dROI[3], int dOffs[3]);

externC const char * CUDAarr_3dsubcpyT_arr(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAcarr_3dsubcpyT_carr(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAcarr_3dsubcpyCT_carr(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);

// 5d versions of the same
externC const char * CUDAarr_5drepcpy_arr(float * a, float * c,int sSize[5], int dSize[5]);
externC const char * CUDAcarr_5drepcpy_carr(float * a, float * c, int sSize[5], int dSize[5]);

externC const char * CUDAarr_5dsubcpy_arr(float * a, float * c,int sSize[3], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5]);
externC const char * CUDAcarr_5dsubcpy_carr(float * a, float * c,int sSize[3], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5]);
externC const char * CUDAarr_5dsubcpy_carr(float * a, float * c, int sSize[3], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5]);

externC const char * CUDAconst_5dsubcpy_arr(float * c, float br, float bi, int dSize[5], int dROI[5], int dOffs[5]);
externC const char * CUDAcconst_5dsubcpy_carr(float * c, float br, float bi, int dSize[5], int dROI[5], int dOffs[5]);

externC const char * CUDAarr_5dsubcpyT_arr(float * a, float * c,int sSize[5], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5]);
externC const char * CUDAcarr_5dsubcpyT_carr(float * a, float * c,int sSize[5], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5]);
externC const char * CUDAcarr_5dsubcpyCT_carr(float * a, float * c,int sSize[5], int dSize[5], int sOffs[5], int sROI[5], int dOffs[5]);

// 
externC const char * CUDAarr_boolassign_const(float * a, float b, float * c, int N);
externC const char * CUDAcarr_boolassign_const(float * a, float br, float bi,  float * c, int N);

externC const char * CUDAarr_complex_arr(float * a, float * b, float * c, int N);  // real in complex out
externC const char * CUDAarr_complex_const(float * a, float b, float * c, int N);  // real in complex out
externC const char * CUDAconst_complex_arr(float * a, float b, float * c, int N);  // real in complex out
// Power
externC const char * CUDAarr_power_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_power_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_power_arr(float * a, float b, float * c, int N);

// Unary functions with one return value
externC const char * CUDAisIllegal_arr(float * a, float * c, int N);  // Size of c has to only be one
externC const char * CUDAany_arr(float * a, float * c, int N);  // Size of c has to only be one
externC const char * CUDAisIllegal_carr(float * a, float * c, int N);  // Size of c has to only be one
externC const char * CUDAany_carr(float * a, float * c, int N);  // Size of c has to only be one


// Multiplication
externC const char * CUDAarr_times_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_times_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_times_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_times_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_times_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_times_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_times_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_times_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_times_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_times_arr(float * a, float br, float bi, float * c, int N);

// Division
externC const char * CUDAarr_divide_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_divide_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_divide_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_divide_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_divide_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_divide_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_divide_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_divide_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_divide_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_divide_arr(float * a, float br, float bi, float * c, int N);

// Element-wise maximum operation
externC const char * CUDAarr_max_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_max_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_max_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_max_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_max_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_max_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_max_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_max_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_max_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_max_arr(float * a, float br, float bi, float * c, int N);

// Element-wise minimum operation
externC const char * CUDAarr_min_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_min_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_min_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_min_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_min_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_min_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_min_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_min_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_min_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_min_arr(float * a, float br, float bi, float * c, int N);

// Addition
externC const char * CUDAarr_plus_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_plus_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_plus_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_plus_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_plus_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_plus_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_plus_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_plus_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_plus_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_plus_arr(float * a, float br, float bi, float * c, int N);

// Subtraction
externC const char * CUDAarr_minus_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_minus_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_minus_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_minus_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_minus_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_minus_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_minus_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_minus_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_minus_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_minus_arr(float * a, float br, float bi, float * c, int N);

// Referencing with another real-valued array (interpretet as boolean)
externC const char * CUDAarr_subsref_arr(float * in, float * mask, float *  out, int N, int * pM);
externC const char * CUDAcarr_subsref_arr(float * in, float * mask, float *  out, int N, int * pM);

// diagonal matrix generation
externC const char * CUDAarr_diag_set(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAcarr_diag_set(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAarr_diag_get(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);
externC const char * CUDAcarr_diag_get(float * a, float * c,int sSize[3], int dSize[3], int sOffs[3], int sROI[3], int dOffs[3]);

// referencing and assignment using an index vector (b), N always refers to the output size. No index checking!
externC const char * CUDAarr_subsref_vec(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_subsref_vec(float * a, float * b, float * c, int N);
externC const char * CUDAarr_subsasgn_vec(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_subsasgn_vec(float * a, float * b, float * c, int N);

// Assigning to by another real-valued array (interpretet as boolean)
externC const char * CUDAarr_subsasgn_arr(float * in, float * mask, float *  out, int N, int * pM);
externC const char * CUDAcarr_subsasgn_arr(float * in, float * mask, float *  out, int N, int * pM);

// or operation
externC const char * CUDAarr_or_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_or_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_or_arr(float * a, float b, float * c, int N);

// and operation
externC const char * CUDAarr_and_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_and_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_and_arr(float * a, float b, float * c, int N);

// not operation
externC const char * CUDAnot_arr(float * a, float * c, int N);

// smaller than
externC const char * CUDAarr_smaller_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_smaller_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_smaller_arr(float * a, float b, float * c, int N);

// larger than
externC const char * CUDAarr_larger_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_larger_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_larger_arr(float * a, float b, float * c, int N);

// smaller or equal
externC const char * CUDAarr_smallerequal_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_smallerequal_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_smallerequal_arr(float * a, float b, float * c, int N);

// larger or equal
externC const char * CUDAarr_largerequal_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_largerequal_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_largerequal_arr(float * a, float b, float * c, int N);

// ==
externC const char * CUDAarr_equals_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_equals_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_equals_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_equals_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_equals_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_equals_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_equals_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_equals_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_equals_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_equals_arr(float * a, float br, float bi, float * c, int N);

// !=   ~=
externC const char * CUDAarr_unequals_arr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_unequals_carr(float * a, float * b, float * c, int N);
externC const char * CUDAcarr_unequals_arr(float * a, float * b, float * c, int N);
externC const char * CUDAarr_unequals_carr(float * a, float * b, float * c, int N);

externC const char * CUDAarr_unequals_const(float * a, float b, float * c, int N);
externC const char * CUDAconst_unequals_arr(float * a, float b, float * c, int N);
externC const char * CUDAcarr_unequals_const(float * a, float br, float bi, float * c, int N);
externC const char * CUDAconst_unequals_carr(float * a, float br, float bi, float * c, int N);
externC const char * CUDAarr_unequals_Cconst(float * a, float br, float bi, float * c, int N);
externC const char * CUDACconst_unequals_arr(float * a, float br, float bi, float * c, int N);

// other Unary operations
externC const char * CUDAset_arr(float b, float * c, int N);
externC const char * CUDAset_carr(float br, float bi, float * c, int N);

externC const char * CUDAuminus_arr(float * a, float * c, int N);
externC const char * CUDAuminus_carr(float * a, float * c, int N);

externC const char * CUDAexp_arr(float * a, float * c, int N);
externC const char * CUDAexp_carr(float * a, float * c, int N);

externC const char * CUDAsin_arr(float * a, float * c, int N);
externC const char * CUDAsin_carr(float * a, float * c, int N);

externC const char * CUDAcos_arr(float * a, float * c, int N);
externC const char * CUDAcos_carr(float * a, float * c, int N);

externC const char * CUDAsinc_arr(float * a, float * c, int N);
externC const char * CUDAsinc_carr(float * a, float * c, int N);

externC const char * CUDAsinh_arr(float * a, float * c, int N);
externC const char * CUDAsinh_carr(float * a, float * c, int N);

externC const char * CUDAcosh_arr(float * a, float * c, int N);
externC const char * CUDAcosh_carr(float * a, float * c, int N);

externC const char * CUDAabs_arr(float * a, float * c, int N);
externC const char * CUDAabs_carr(float * a, float * c, int N);

externC const char * CUDAlog_arr(float * a, float * c, int N);
externC const char * CUDAlog_carr(float * a, float * c, int N);

externC const char * CUDAconj_arr(float*a,float * c, int N);
externC const char * CUDAconj_carr(float*a,float * c, int N);

externC const char * CUDAsqrt_arr(float*a,float * c, int N);
externC const char * CUDAsqrt_carr(float*a,float * c, int N);

externC const char * CUDAreal_arr(float * a, float * c, int N);
externC const char * CUDAreal_carr(float * a, float * c, int N);

externC const char * CUDAimag_arr(float * a, float * c, int N);
externC const char * CUDAimag_carr(float * a, float * c, int N);

externC const char * CUDAphase_arr(float * a, float * c, int N);
externC const char * CUDAphase_carr(float * a, float * c, int N);

externC const char * CUDAisnan_arr(float * a, float * c, int N);
externC const char * CUDAisnan_carr(float * a, float * c, int N);

externC const char * CUDAisinf_arr(float * a, float * c, int N);
externC const char * CUDAisinf_carr(float * a, float * c, int N);

externC int CUDAarr_times_const_checkerboard(float * a, float b, float * c, int * sizes, int dims);  // multiplies with a constand and scrambles the array
externC int CUDAarr_times_const_scramble(float * a, float b, float * c, int * sizes, int dims);  // multiplies with a constand and scrambles the array
externC int CUDAarr_times_const_rotate(float * a, float b, float * c, int * sizes, int dims, int complex,int direction);  // rotates an array in place or out of place by the distance [dx,dy,dz]

externC const char * CUDAarr_circshift_vec(float * a,int nshifts[CUDA_MAXDIM],float * c, int mySize[CUDA_MAXDIM],int N);    // cyclic rotation
externC const char * CUDAcarr_circshift_vec(float * a,int nshifts[CUDA_MAXDIM],float * c, int mySize[CUDA_MAXDIM],int N);   // cyclic rotation by the vector nshifts

externC const char * CUDAarr_permute_vec(float * a,int nshifts[CUDA_MAXDIM],float * c, int mySize[CUDA_MAXDIM],int N);    // cyclic rotation
externC const char * CUDAcarr_permute_vec(float * a,int nshifts[CUDA_MAXDIM],float * c, int mySize[CUDA_MAXDIM],int N);   // cyclic rotation by the vector nshifts

externC const char * CUDAarr_xyz_2vec(float * c, VecND vec1, VecND vec2, SizeND sSize, int N);    // generates xx, yy and zz
externC const char * CUDAarr_rr_2vec(float * c, VecND vec1, VecND vec2, SizeND sSize, int N);    // generates rr, first vec for center second for scaling
externC const char * CUDAarr_phiphi_2vec(float * c, VecND vec1, VecND vec2, SizeND sSize, int N);    // generates rr, first vec for center second for scaling


// Now include all the user-defined functions
#include "user_h_code.inc"
