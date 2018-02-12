% res=convn(A,B,shape) : convolve with a 3x3x3 kernel
% 
function out=convn(A,B,shape)
error('Implementation of convn has not finnished yet. (It is not working!)')

if norm(size(B) - [3 3 3])~=0
    error('convn is only implemented in 3d for a 3x3x3 kernel');    
end
if ~isa(A,'cuda')
    A=cuda(A);
end
if ~isa(B,'cuda')
    B=cuda(B);
end

myref=cuda_cuda('Conv3DMask',A.ref,B.ref); % figures out from the datasize whether to call the 3D or 2D routine
out=cuda();
out.ref=myref;
out.fromDip = A.fromDip ;   % If input was dipimage, result will be

