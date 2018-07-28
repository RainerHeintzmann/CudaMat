 

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
 

else if (strcmp(command,"MyForcePosSqr_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyForcePosSqr_Late_Fwd,cudaAlloc,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosSqr_Late_Fwd");
} 
 

else if (strcmp(command,"MyForcePosSqr_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyForcePosSqr_Late_Inv,cudaAlloc,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosSqr_Late_Inv");
} 
 

else if (strcmp(command,"MyForcePosPiecewise2nd_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyForcePosPiecewise2nd_Late_Fwd,cudaAlloc,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosPiecewise2nd_Late_Fwd");
} 
 

else if (strcmp(command,"MyForcePosPiecewise2nd_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyForcePosPiecewise2nd_Late_Inv,cudaAlloc,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosPiecewise2nd_Late_Inv");
} 
 

else if (strcmp(command,"MyPoissonErrAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyPoissonErrAndDeriv_Late_Fwd,cudaAlloc,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyPoissonErrAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"MyPoissonErrAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyPoissonErrAndDeriv_Late_Inv,cudaAlloc,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyPoissonErrAndDeriv_Late_Inv");
} 
 

else if (strcmp(command,"MyGaussErrAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyGaussErrAndDeriv_Late_Fwd,cudaAlloc,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyGaussErrAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"MyGaussErrAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyGaussErrAndDeriv_Late_Inv,cudaAlloc,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyGaussErrAndDeriv_Late_Inv");
} 
 

else if (strcmp(command,"ScaGaussReadNoiseErrorAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(ScaGaussReadNoiseErrorAndDeriv_Late_Fwd,cudaAlloc,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: ScaGaussReadNoiseErrorAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"ScaGaussReadNoiseErrorAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(ScaGaussReadNoiseErrorAndDeriv_Late_Inv,cudaAlloc,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: ScaGaussReadNoiseErrorAndDeriv_Late_Inv");
} 
 

else if (strcmp(command,"FixGaussErrorAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(FixGaussErrorAndDeriv_Late_Fwd,cudaAlloc,3)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: FixGaussErrorAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"FixGaussErrorAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(FixGaussErrorAndDeriv_Late_Inv,cudaAlloc,3)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: FixGaussErrorAndDeriv_Late_Inv");
} 
