% InvalidateCuda()  : invalidates the cuda sources, forcing a recompile on next check

function InvalidateCuda()  % cuda_to_compile.needsRecompile=1;

global cuda_to_compile
cuda_to_compile.needsRecompile=1;
