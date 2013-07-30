% function enableCuda()   : enables the use of cuda (after disableCuda() was called). Thus all cuda generator functions are set back to their original values.
% The meaning of a double cast also goes back to remaining a cuda datatype.

function enableCuda()
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

if (cuda_enabled == 0)
    use_zeros_cuda=remember_use_zeros_cuda;
    use_ones_cuda=remember_use_ones_cuda;
    use_rand_cuda=remember_use_rand_cuda;
    use_newim_cuda=remember_use_newim_cuda;
    use_newimar_cuda=remember_use_newimar_cuda;
    use_xyz_cuda=remember_use_xyz_cuda;
    use_double_cuda=remember_use_double_cuda;
end

clear remember_use_zeros_cuda; 
clear remember_use_ones_cuda; 
clear remember_use_rand_cuda; 
clear remember_use_newim_cuda; 
clear remember_use_newimar_cuda; 
clear remember_use_xyz_cuda; 
clear remember_use_double_cuda; 

cuda_enabled=1; 

