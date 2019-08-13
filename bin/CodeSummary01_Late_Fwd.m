% [fwd,grad] = CodeSummary01_Late_Fwd(f1,f2,f3)   : User defined cuda function. Wraps up cuda commuting code
 function [fwd,grad] = CodeSummary01_Late_Fwd(f1,f2,f3)
 allcuda=1;anyDip=0;
 if prod(size(f1)) > 1 && ~isa(f1,'cuda')  
 error('CodeSummary01_Late_Fwd: Automatic conversion to cuda not allowed due to in-place operations.');f1=cuda(f1); 
 end
 allcuda= allcuda && isa(f1,'cuda');
 anyDip= anyDip || f1.fromDip;
 if prod(size(f2)) > 1 && ~isa(f2,'cuda')  
 error('CodeSummary01_Late_Fwd: Automatic conversion to cuda not allowed due to in-place operations.');f2=cuda(f2); 
 end
 allcuda= allcuda && isa(f2,'cuda');
 anyDip= anyDip || f2.fromDip;
 if prod(size(f3)) > 1 && ~isa(f3,'cuda')  
 error('CodeSummary01_Late_Fwd: Automatic conversion to cuda not allowed due to in-place operations.');f3=cuda(f3); 
 end
 allcuda= allcuda && isa(f3,'cuda');
 anyDip= anyDip || f3.fromDip;
 if allcuda 
    ref=cuda_cuda('CodeSummary01_Late_Fwd',getReference(f1),getReference(f2),getReference(f3)); 
 else 
    error('CodeSummary01_Late_Fwd: Needs arrays to be cuda as input'); 
 end 
 grad=cuda(ref,anyDip);   
 fwd=f1;  