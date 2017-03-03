% y = flipdim(x,dim) : Cuda version of the Matlab or DipImage flipdim

function res = flipdim(toflip,dim)

if nargin~=2
   error('Requires two arguments.')
end
if ~isnumeric(dim) | length(dim)~=1 | fix(dim)~=dim | dim<1
   error('DIM must be a positive integer.')
end
if ndims(toflip) < dim
   error('Cannot flip along non-existent dimension.')
end

dimsize = size(toflip,dim);
if (dimsize <= 1)
    % No-op.
    res = toflip;
else
    v(1:ndims(toflip)) = {':'};
    if ~isa(toflip,'cuda') || (isa(toflip,'cuda') && ~isDip(toflip))
        v{dim} = dimsize:-1:1;
    else
        v{dim} = dimsize-1:-1:0;
    end
    res = toflip(v{:});
end

