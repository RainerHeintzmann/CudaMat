% res=convn(A,B,shape) : convolve with a 3x3x3 kernel
% 
function out=convn(A,Kernel,shape)
error('Implementation of convn has not finnished yet. (It currently performs circular convolutions instead ...)')

if ~isa(A,'cuda')
    A=cuda(A);
end
if ~isa(Kernel,'cuda')
    Kernel=cuda(Kernel);
end
if (ndims(A) > ndims(Kernel))
    Kernel = expanddim(Kernel,ndims(A));
end

if (ndims(A) ~= ndims(Kernel))
    error('convn: Numberof dimensions have to be equal between array and kernel');
end

myref=cuda_cuda('convND',Kernel.ref,A.ref); % figures out from the datasize whether to call the 3D or 2D routine
out=cuda();
out.ref=myref;
out.fromDip = A.fromDip ;   % If input was dipimage, result will be

