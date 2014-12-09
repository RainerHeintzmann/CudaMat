% function getVal(in,idx) : retrieves one value

function res=getVal(in,idx) 
    res=cuda_cuda('getVal',in.ref,idx); % -1 for index means end
end