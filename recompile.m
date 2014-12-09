% recompile : recompiles the cuda functions and clears the cuda variables before doing so.


mywho=whos;   % delete all existing cuda variables
for n=1:size(mywho,1)
    if strcmp(mywho(n).class,'cuda')
        clear(mywho(n).name);
    end
end

try
    cuda_shutdown;
catch
end

global cuda_to_compile
cuda_to_compile.needsRecompile=1;
cuda_compile_all;
