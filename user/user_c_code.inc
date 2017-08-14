 

else if (strcmp(command,"mandelbrot")==0) 
{ if (nrhs != 2) mexErrMsgTxt("cuda: mandelbrot needs two arguments"); 
  if (isComplexType(getCudaRefNum(prhs[1]))) 
     CUDAmandelbrot_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1]));  
 else 
     CUDAmandelbrot_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1]));  
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: mandelbrot");
} 
