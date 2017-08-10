% cuda_compile_all : compiles all defined cuda functions into the system of cuda commands that can be executed on NVidia Graphic cards
% 
% example : 
%
% cuda_define('myreal','unary','c[idx]=a[idx];','c[idx]=a[2*idx];');
% cuda_compile_all();
% myreal(cuda(readim + i*readim('orka')))


function cuda_compile_all() 
global cuda_to_compile;  % contains all the collected snippets to compile into a cuda toolbox

if ~exist('cuda_to_compile') || isempty(cuda_to_compile)
    error('Cannot compile any user defined cuda functions. None defined. Use cuda_define()');
end

if ~cuda_to_compile.needsRecompile
    fprintf('cuda_compile_all: Nothing to recompile. Code has not changed\n')
    return;
end
c_text = '';
cu_text = '';
h_text = '';

CudaBase= which('cuda');
CudaBase=CudaBase(1:end-12);

for n=1:length(cuda_to_compile.name)
    c_text = sprintf('%s \n\n%s',c_text,cuda_to_compile.c_snippet{n});
    cu_text = sprintf('%s \n\n%s',cu_text,cuda_to_compile.cu_snippet{n});
    h_text = sprintf('%s \n\n%s',h_text,cuda_to_compile.h_snippet{n});

    m_text=sprintf('%s',cuda_to_compile.m_snippet{n});
    Filename = sprintf('%s.m',cuda_to_compile.name{n});
    fd=fopen([CudaBase 'user/' Filename],'w');
    fprintf(fd,['%' m_text]);  % otherwise the first character causes trouble
    fclose(fd);
end

fd=fopen([CudaBase 'user/user_c_code.inc'],'w');
fprintf(fd,c_text);
fclose(fd);

fd=fopen([CudaBase 'user/user_cu_code.inc'],'w');
fprintf(fd,cu_text);
fclose(fd);

fd=fopen([CudaBase 'user/user_h_code.inc'],'w');
fprintf(fd,h_text);
fclose(fd);

addpath([CudaBase '/user']);
try
    cuda_shutdown;
catch
end
if ispc
    global nocula;
    % system('"c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"')
    % system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin')
    % system('"c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\vcvars32.bat"')
    % vcvars64.bat has to be present at C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\amd64
    system('nvcc -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include"')
    if ~isempty(nocula)
        % mex cuda_cuda.c cudaArith.obj -DNOCULA -Ic:\\CUDA\include\ -Lc:\\CUDA\lib64\ -lcublas -lcufft -lcudart
        mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" -lcublas -lcufft -lcudart
    else
        % mex cuda_cuda.c cudaArith.obj -Ic:\\CUDA\include\ -Ic:\\CULA\include\ -Lc:\\CUDA\lib64\ -Lc:\\CULA\lib64\ -lcublas -lcufft -lcudart -lcula
        mex cuda_cuda.c cudaArith.obj "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" "-LC:\Program Files\CULA\R14\lib64" -lcublas -lcufft -lcudart -lcula_core -lcula_lapack
    end
else
    global nocula;
    if ~isempty(nocula)
        system('nvcc -c cudaArith.cu -I/usr/local/cuda/include/')
        mex cuda_cuda.c cudaArith.o -DNOCULA -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart
    else
        system('nvcc -c cudaArith.cu -I/usr/local/cula/include/')
        mex cuda_cuda.c cudaArith.o -I/usr/local/cula/include -I/usr/local/cuda/include -L/usr/local/cula/lib64 -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart -lcula
    end
end
cuda_to_compile.needsRecompile = 0;   % everything is up to date now.

