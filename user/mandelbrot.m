% out = mandelbrot(in1)   : User defined cuda function. Computes the Mandelbrot set
 function out = mandelbrot(in1)
 if isa(in1,'cuda')  
    ref=cuda_cuda('mandelbrot',in1.ref); 
 else 
 error('mandelbrot: Unknown datatype'); 
 end 
 out=cuda(ref,in1.fromDip);   
