function out=TestRefs(myvar,n)
if nargin < 2
    n=10;
end
if n>0
    xyz=myvar;
    out=TestRefs(myvar,n-1);
else
    out=cuda_cuda('getLinkedVarNum',myvar);
end
