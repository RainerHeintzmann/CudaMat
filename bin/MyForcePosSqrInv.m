% [fwd,grad] = MyForcePosSqrInv(f1)   : User defined cuda function. Wraps up commuting cuda code
 function [fwd,grad] = MyForcePosSqrInv(f1)
 allcuda=1;anyDip=0;
 if prod(size(f1)) > 1 && ~isa(f1,'cuda')  
 error('MyForcePosSqrInv: Automatic conversion to cuda not allowed due to in-place operations.');f1=cuda(f1); 
 end
 allcuda= allcuda && isa(f1,'cuda');
 anyDip= anyDip || f1.fromDip;
 if allcuda 
    ref=cuda_cuda('MyForcePosSqrInv',getReference(f1)); 
 else 
    error('MyForcePosSqrInv: Needs arrays to be cuda as input'); 
 end 
 grad=cuda(ref,anyDip);   
 fwd=f1;  