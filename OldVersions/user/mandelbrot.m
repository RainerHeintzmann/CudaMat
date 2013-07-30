% out = mandelbrot(in1)   : User defined cuda function. Computes the Mandelbrot set
 function out = mandelbrot(in1)
 if isa(in1,'cuda')  
    ref=cuda_cuda('mandelbrot',getReference(in1)); 
 else 
 error('mandelbrot: Unknown datatype'); 
 end 
 out=cuda(ref,getFromDip(in1));   
