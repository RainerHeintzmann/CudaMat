% function disableCuda()   : temporarily disables the use of cuda. Thus all cuda generator functions are set back to zero (no cuda object will be generated).
% The meaning of a double case also changes automatically to force_double().

function disableCuda()
global cuda_enabled; 

global use_zeros_cuda; 
global use_ones_cuda; 
global use_rand_cuda; 
global use_newim_cuda; 
global use_newimar_cuda; 
global use_xyz_cuda; 
global use_double_cuda; 

global remember_use_zeros_cuda; 
global remember_use_ones_cuda; 
global remember_use_rand_cuda; 
global remember_use_newim_cuda; 
global remember_use_newimar_cuda; 
global remember_use_xyz_cuda; 
global remember_use_double_cuda; 

remember_use_zeros_cuda=use_zeros_cuda;
remember_use_ones_cuda=use_ones_cuda;
remember_use_rand_cuda=use_rand_cuda;
remember_use_newim_cuda=use_newim_cuda;
remember_use_newimar_cuda=use_newimar_cuda;
remember_use_xyz_cuda=use_xyz_cuda;
remember_use_double_cuda=use_double_cuda; 

use_zeros_cuda=0;
use_ones_cuda=0;
use_rand_cuda=0;
use_newim_cuda=0;
use_newimar_cuda=0;
use_xyz_cuda=0;
use_double_cuda=0;

cuda_enabled=0; 
