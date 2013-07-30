% recompile : recompiles the cuda functions and clears the cuda variables before doing so.
mywho=whos;
for n=1:size(mywho,1)
    if strcmp(mywho(n).class,'cuda')
        clear(mywho(n).name);
    end
end

try
    cuda_shutdown;
catch
end
%system('nvcc -maxrregcount=30 -Xptxas=-v -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include"')
system('nvcc -maxrregcount=30 -c cudaArith.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin" "-Ic:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include"')
%mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\lib\x64" -lcublas -lcufft -lcudart
mex cuda_cuda.c cudaArith.obj -DNOCULA "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include" "-IC:\Program Files\CULA\R14\include" "-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\x64" -lcublas -lcufft -lcudart
