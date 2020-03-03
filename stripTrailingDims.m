% res=stripTrailingDims(in) : removes the trainling dimensions (size 1). Matlab does this automatically.
function res=stripTrailingDims(in)    
    sz=size(in);
    lastDim=find(sz > 1,1,'last');
    if lastDim < 2
        lastDim = 2;
    end
    if ndims(in) > 2
        newsize = sz(1:lastDim);
        res=reshape(in,newsize);
    end