% initCuda(useGenerators,useDoubleCuda, useCula)  : initializes the CudaMat toolbox
% useGenerators : a flag describing wether all the generator functions (xxm,yy,zz,rr,ramp,phiphi,rand) should be producing cuda output. This can be individually changed. See set_rr_cuda().
% useDoubleCuda : a flag describing 
% useCula : a flag describing whether the Cula toolbox (with eigenvector routine) should be used and is installed.
%
% see also:
% setCudaSynchronize, installDebug, recompile, setCudaDevice
%
function initCuda(useGenerators,useDoubleCuda, useCula)
global nocula; nocula=1;
global use_zeros_cuda; use_zeros_cuda=0;
global use_ones_cuda; use_ones_cuda=0;
global use_rand_cuda; use_rand_cuda=0;
global use_newim_cuda; use_newim_cuda=0;
global use_newimar_cuda; use_newimar_cuda=0;
global use_xyz_cuda; use_xyz_cuda=0;
global use_ramp_cuda; use_ramp_cuda=0;
global use_double_cuda; 
global cuda_enabled; cuda_enabled=1;
global UserBase;

if nargin < 3
    useCula=0;
end
if nargin < 2
    useDoubleCuda=1;
end
if nargin < 1
    useGenerators=1;
end

mp=userpath();
if isempty(mp)
    userpath('reset');
    mp = userpath();
end
if mp(end)==';' || mp(end)==':'    % Windows and Linux
    mp=mp(1:end-1);
end
if isempty(UserBase)
    pathstr=fileparts(mfilename('fullpath'));  % ignores name and extension
    if isdeployed()==0
        addpath(pathstr);
    end
    UserBase=[mp filesep 'LocalCudaMatSrc' filesep];
    % UserBase=[tempdir() 'user' filesep];
    [SUCCESS,MESSAGE,MESSAGEID] =mkdir(UserBase); % Just in case it does not exist. Ignor unsuccessful attempts
    if isdeployed()==0
        addpath(UserBase);
    end
end

if exist('dip_image','file')
    createDipHandles();
end

if useCula
    nocula=0;
end

if useGenerators
    use_zeros_cuda=1;
    use_ones_cuda=1;
    use_rand_cuda=1;
    use_newim_cuda=1;
    use_newimar_cuda=1;
    use_xyz_cuda=1;
    use_ramp_cuda=1;
end

use_double_cuda=useDoubleCuda;

try
    cuda(10);
catch
    fprintf('WARNING! No cuda compiled cuda installation could be detected. Try to type "recompile". For now the precomplied version is installed.\n');
    
    CudaBase= which('cuda');
    CudaBase=CudaBase(1:end-12);
    % UserBase=[tempdir() 'user' filesep];  % This causes problems in Linux with multiple users.
    mp=userpath();
    if mp(end)==';' || mp(end)==':'    % Windows and Linux
        mp=mp(1:end-1);
    end
    UserBase=[CudaBase filesep 'bin' filesep];
    if isdeployed()==0
        addpath(UserBase);
    end
    UserBase=[mp filesep 'LocalCudaMatSrc' filesep];
%%% THE CODE BELOW IS NOT NEEDED SINCE THE USER PATH IS SIMPLY CHANGED TO THE BIN FOLDER. No need to copy!
%     mkdir(UserBase); % Just in case it does not exist
%     if ~(exist([UserBase filesep 'cuda_cuda.mexw64'], 'file') == 2)
%         global CudaVERSION
%         if isempty(CudaVERSION)
%             CudaVERSION=90;
%         end
%         if size(CudaVERSION,2) == 1
%             CV=[sprintf('%d',CudaVERSION) '0'];
%         else
%             CV = CudaVERSION;
%         end
%         copyfile([CudaBase filesep 'bin' filesep 'cublas64_' CV '.dll'],[UserBase filesep]);
%         copyfile([CudaBase filesep 'bin' filesep 'cudart64_' CV '.dll'],[UserBase filesep]);
%         copyfile([CudaBase filesep 'bin' filesep 'cuda_cuda.mexw64'],[UserBase filesep 'cuda_cuda.mexw64']);
%         copyfile([CudaBase filesep 'bin' filesep 'cufft64_' CV '.dll'],[UserBase filesep]);
%     end      
end