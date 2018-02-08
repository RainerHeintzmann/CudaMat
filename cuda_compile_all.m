% cuda_compile_all(forceRecompile) : compiles all defined cuda functions into the system of cuda commands that can be executed on NVidia Graphic cards
% forceRecompile: optional flag to force recompilation independent of the global flag cuda_to_compile.needsRecompile. Default: 0 (no recompilation forced)
%
% globals are used:
% cuda_to_compile  : contains all the collected snippets to compile into a cuda toolbox
% NVCCFLAGS
% CVERSION : 9,10,11  :Version of the Visual C compiler to use
% CudaVERSION: 4,5,6 :Verson of cuda to use
% example : 
%
% cuda_define('myreal','unary','c[idx]=a[idx];','c[idx]=a[2*idx];');
% cuda_compile_all();
% myreal(cuda(readim + i*readim('orka')))
%

function cuda_compile_all(forceRecompile) 
if nargin < 1
    forceRecompile=0;
end
global cuda_to_compile;  % contains all the collected snippets to compile into a cuda toolbox
global NVCCFLAGS;
global CVERSION;
global CudaVERSION;

if ~exist('cuda_to_compile') || isempty(cuda_to_compile)
    error('Cannot compile any user defined cuda functions. None defined. Use cuda_define()');
end

if ~cuda_to_compile.needsRecompile && ~forceRecompile
    fprintf('cuda_compile_all: Nothing to recompile. Code has not changed\n')
    return;
end
c_text = '';
cu_text = '';
h_text = '';

CudaBase= which('cuda');
CudaBase=CudaBase(1:end-12);
% UserBase=[tempdir() 'user' filesep];  % This causes problems in Linux with multiple users.
mp=userpath();
if mp(end)==';' || mp(end)==':'    % Windows and Linux
    mp=mp(1:end-1);
end
UserBase=[mp filesep 'LocalCudaMatSrc' filesep];
mkdir(UserBase); % Just in case it does not exist

if isfield(cuda_to_compile,'name')
    for n=1:length(cuda_to_compile.name)
        c_text = sprintf('%s \n\n%s',c_text,cuda_to_compile.c_snippet{n});
        cu_text = sprintf('%s \n\n%s',cu_text,cuda_to_compile.cu_snippet{n});
        h_text = sprintf('%s \n\n%s',h_text,cuda_to_compile.h_snippet{n});
        
        m_text=sprintf('%s',cuda_to_compile.m_snippet{n});
        Filename = sprintf('%s.m',cuda_to_compile.name{n});
        fd=fopen([UserBase Filename],'w');
        fprintf(fd,['%' m_text]);  % otherwise the first character causes trouble
        fclose(fd);
    end
end

fd=fopen([UserBase 'user_c_code.inc'],'w');
if (fd < 0) 
    error('Error defining user-defined cuda code: Cannot open user unclude file for writing');
end
fprintf(fd,c_text);
fclose(fd);

fd=fopen([UserBase 'user_cu_code.inc'],'w');
if (fd < 0) 
    error('Error defining user-defined cuda code: Cannot open user unclude file for writing');
end
fprintf(fd,cu_text);
fclose(fd);

fd=fopen([UserBase 'user_h_code.inc'],'w');
if (fd < 0) 
    error('Error defining user-defined cuda code: Cannot open user unclude file for writing');
end
fprintf(fd,h_text);
fclose(fd);

addpath(UserBase);
% try
%     cuda_shutdown;
% catch
% end
CurrentPath=pwd;
cd(UserBase)
if (0)
    MEXFLAGS='-g';
else
    global MEXFLAGS;
    if isempty(MEXFLAGS)
        MEXFLAGS='';
    end
end
if ispc
    if CVERSION==15        % correct line 133 in the file   c:/program files/nvidia gpu computing toolkit/cuda/v9.0/include/crt/host_config.h  to #if _MSC_VER < 1600 || _MSC_VER > 1911
        CudaComp=' --cl-version 2017 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.11.25503\bin\HostX64\x64" "-IC:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Tools\MSVC\14.11.25503\include" -I./ -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\/include" -I../../common/inc -I"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.11.25503\include" ';
    elseif CVERSION==14        
        CudaComp=' --cl-version 2015 -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64" "-Ic:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include" -I./ -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\/include" -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" ';
        %CudaComp=' --cl-version 2015 -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64" "-Ic:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include" -I./ -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\/include" -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" ';
        % CudaComp=' --cl-version 2015 -ccbin "d:\Programme (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64" "-ID:\Programme (x86)\Microsoft Visual Studio 14.0\VC\include" -I./ -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\/include" -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" ';
    elseif CVERSION==12
        global CPATH;
        if isempty(CPATH)
            CudaComp=' --cl-version 2013 -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64" "-Ic:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include" -I./ -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\/include" -I../../common/inc ';
            % -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" 
        else
            CudaComp=[' --cl-version 2013 -ccbin "' CPATH '\VC\bin\x86_amd64" "-I' CPATH '\VC\include" -I./ -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\/include" -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" '];
        end
    elseif CVERSION==11
        CudaComp=' -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\include" -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" ';
    elseif CVERSION==10
        CudaComp=' -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include" ';
    else
        warning('No global CVERSION flag was set. Assuming version 11.0. Choices are 9, 10 or 11.')
        CudaComp=' -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\include" ';
    end
    if CudaVERSION==9
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64" ';
        NVCC_BIN = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\nvcc"';
    elseif CudaVERSION==8
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64" ';
        NVCC_BIN = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc"';
    elseif CudaVERSION==7
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64" ';
        NVCC_BIN = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\nvcc"';
    elseif CudaVERSION==6
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\x64" ';
        NVCC_BIN = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\bin\nvcc"';
    elseif CudaVERSION==5
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64" ';
        NVCC_BIN = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\bin\nvcc"';
    elseif CudaVERSION==4
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\lib\x64" ';
        NVCC_BIN = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\bin\nvcc"';
    else
        warning('No global CudaVERSION flag was set. Assuming version 6. Choices are 6, 5 or 4.')
        MexComp=' "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\x64" ';
    end

    global nocula;
    % system('"c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"')
    % system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin')
    % system('"c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\vcvars32.bat"')
    % vcvars64.bat has to be present at C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\amd64
    % status=system(['nvcc -c ' MEXFLAGS ' ' NVCCFLAGS ' ' CudaBase 'cudaArith.cu  -I.']);
    status=system([NVCC_BIN ' -O3 -c ' NVCCFLAGS ' '  CudaComp CudaBase 'cudaArith.cu  -I. -Xcudafe "--diag_suppress=divide_by_zero"']);
    if status ~= 0
        error('nvcc command failed. Try defining the global variables CudaVERSION={4,5,6} and CVERSION={10,11} in the startup file.');
    end
    % system(['nvcc -c ' CudaBase 'cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include\:' UserBase '"'])
    % system(['nvcc -c ' CudaBase 'cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include"'])
    if ~isempty(nocula)
        % mex cuda_cuda.c cudaArith.obj -DNOCULA -Ic:\\CUDA\include\ -Lc:\\CUDA\lib64\ -lcublas -lcufft -lcudart
        % mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" -lcublas -lcufft -lcudart
        % eval(['mex ' MEXFLAGS ' ' CudaBase 'cuda_cuda.c cudaArith.obj -DNOCULA "-I' UserBase '" "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64" -lcublas -lcufft -lcudart']);
        eval(['mex ' MEXFLAGS ' ' CudaBase 'cuda_cuda.c cudaArith.obj -DNOCULA "-I' UserBase  '" ' MexComp '-lcublas -lcufft -lcudart']);
    else
        % mex cuda_cuda.c cudaArith.obj -Ic:\\CUDA\include\ -Ic:\\CULA\include\ -Lc:\\CUDA\lib64\ -Lc:\\CULA\lib64\ -lcublas -lcufft -lcudart -lcula
        % mex cuda_cuda.c cudaArith.obj "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" "-LC:\Program Files\CULA\R14\lib64" -lcublas -lcufft -lcudart -lcula_core -lcula_lapack
        eval(['mex ' MEXFLAGS ' ' CudaBase 'cuda_cuda.c cudaArith.obj "-I' UserBase '" ' MexComp '"-IC:\Program Files\CULA\R14\include -LC:\Program Files\CULA\R14\lib64" -lcublas -lcufft -lcudart -lcula_core -lcula_lapack']);
    end
else
    global nocula;
    bv=[];
    if ~ispc
        bv = 'CFLAGS="\$CFLAGS -std=gnu99"';
    end
    if ~isempty(nocula)
        status=system(['nvcc -O3 -c ' NVCCFLAGS ' -Xcompiler -fPIC ' CudaBase  'cudaArith.cu -I/usr/local/cuda/include/ -I.']);
        if status ~= 0
            error('nvcc command failed');
        end
        eval(['mex ' bv ' ' MEXFLAGS ' ' CudaBase 'cuda_cuda.c cudaArith.o -DNOCULA "-I' UserBase '" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart']);
    else
        status=system(['nvcc -c  ' NVCCFLAGS  ' -Xcompiler -fPIC ' CudaBase ' cudaArith.cu -I/usr/local/cula/include/  -I/usr/local/cuda/include/ -I.']);
        if status ~= 0
            error('nvcc command failed');
        end
        eval(['mex ' bv ' ' MEXFLAGS ' ' CudaBase 'cuda_cuda.c cudaArith.o "-I' UserBase '" -I/usr/local/cula/include -I/usr/local/cuda/include -L/usr/local/cula/lib64 -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart -lcula']);
    end
end
cd(CurrentPath)
cuda_to_compile.needsRecompile = 0;   % everything is up to date now.

