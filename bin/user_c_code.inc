 

else if (strcmp(command,"MyForcePosSqrFwd")==0) {
 CallCUDA_NArgsFkt(MyForcePosSqrFwd,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosSqrFwd");
} 
 

else if (strcmp(command,"MyForcePosSqrInv")==0) {
 CallCUDA_NArgsFkt(MyForcePosSqrInv,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosSqrInv");
} 
 

else if (strcmp(command,"MyForcePosPiecewise2ndFwd")==0) {
 CallCUDA_NArgsFkt(MyForcePosPiecewise2ndFwd,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosPiecewise2ndFwd");
} 
 

else if (strcmp(command,"MyForcePosPiecewise2ndInv")==0) {
 CallCUDA_NArgsFkt(MyForcePosPiecewise2ndInv,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosPiecewise2ndInv");
} 
 

else if (strcmp(command,"MyPoissonErrAndDerivFwd")==0) {
 CallCUDA_NArgsFkt(MyPoissonErrAndDerivFwd,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyPoissonErrAndDerivFwd");
} 
 

else if (strcmp(command,"MyPoissonErrAndDerivInv")==0) {
 CallCUDA_NArgsFkt(MyPoissonErrAndDerivInv,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyPoissonErrAndDerivInv");
} 
 

else if (strcmp(command,"MyGaussErrAndDerivFwd")==0) {
 CallCUDA_NArgsFkt(MyGaussErrAndDerivFwd,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyGaussErrAndDerivFwd");
} 
 

else if (strcmp(command,"MyGaussErrAndDerivInv")==0) {
 CallCUDA_NArgsFkt(MyGaussErrAndDerivInv,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyGaussErrAndDerivInv");
} 
 

else if (strcmp(command,"ScaGaussReadNoiseErrorAndDerivFwd")==0) {
 CallCUDA_NArgsFkt(ScaGaussReadNoiseErrorAndDerivFwd,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: ScaGaussReadNoiseErrorAndDerivFwd");
} 
 

else if (strcmp(command,"ScaGaussReadNoiseErrorAndDerivInv")==0) {
 CallCUDA_NArgsFkt(ScaGaussReadNoiseErrorAndDerivInv,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: ScaGaussReadNoiseErrorAndDerivInv");
} 
 

else if (strcmp(command,"FixGaussErrorAndDerivFwd")==0) {
 CallCUDA_NArgsFkt(FixGaussErrorAndDerivFwd,cudaAllocSized,3)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: FixGaussErrorAndDerivFwd");
} 
 

else if (strcmp(command,"FixGaussErrorAndDerivInv")==0) {
 CallCUDA_NArgsFkt(FixGaussErrorAndDerivInv,cudaAllocSized,3)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: FixGaussErrorAndDerivInv");
} 
