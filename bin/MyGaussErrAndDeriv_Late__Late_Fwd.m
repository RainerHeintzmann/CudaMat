% [fwd,grad] = MyGaussErrAndDeriv_Late__Late_Fwd(f1,f2)   : User defined cuda function. Wraps up cuda commuting code
 function [fwd,grad] = MyGaussErrAndDeriv_Late__Late_Fwd(f1,f2)
 allcuda=1;anyDip=0;
 if prod(size(f1)) > 1 && ~isa(f1,'cuda')  
 error('MyGaussErrAndDeriv_Late__Late_Fwd: Automatic conversion to cuda not allowed due to in-place operations.');f1=cuda(f1); 
 end
 allcuda= allcuda && isa(f1,'cuda');
 anyDip= anyDip || f1.fromDip;
 if prod(size(f2)) > 1 && ~isa(f2,'cuda')  
 error('MyGaussErrAndDeriv_Late__Late_Fwd: Automatic conversion to cuda not allowed due to in-place operations.');f2=cuda(f2); 
 end
 allcuda= allcuda && isa(f2,'cuda');
 anyDip= anyDip || f2.fromDip;
 if allcuda 
    ref=cuda_cuda('MyGaussErrAndDeriv_Late__Late_Fwd',getReference(f1),getReference(f2)); 
 else 
    error('MyGaussErrAndDeriv_Late__Late_Fwd: Needs arrays to be cuda as input'); 
 end 
 grad=cuda(ref,anyDip);   
 fwd=f1;  