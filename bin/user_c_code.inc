 

else if (strcmp(command,"MyForcePosSqr_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyForcePosSqr_Late_Fwd,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosSqr_Late_Fwd");
} 
 

else if (strcmp(command,"MyForcePosSqr_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyForcePosSqr_Late_Inv,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosSqr_Late_Inv");
} 
 

else if (strcmp(command,"MyForcePosPiecewise2nd_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyForcePosPiecewise2nd_Late_Fwd,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosPiecewise2nd_Late_Fwd");
} 
 

else if (strcmp(command,"MyForcePosPiecewise2nd_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyForcePosPiecewise2nd_Late_Inv,cudaAllocSized,1)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyForcePosPiecewise2nd_Late_Inv");
} 
 

else if (strcmp(command,"MyPoissonErrAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyPoissonErrAndDeriv_Late_Fwd,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyPoissonErrAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"MyPoissonErrAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyPoissonErrAndDeriv_Late_Inv,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyPoissonErrAndDeriv_Late_Inv");
} 
 

else if (strcmp(command,"MyGaussErrAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(MyGaussErrAndDeriv_Late_Fwd,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyGaussErrAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"MyGaussErrAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(MyGaussErrAndDeriv_Late_Inv,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: MyGaussErrAndDeriv_Late_Inv");
} 
 

else if (strcmp(command,"ScaGaussReadNoiseErrorAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(ScaGaussReadNoiseErrorAndDeriv_Late_Fwd,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: ScaGaussReadNoiseErrorAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"ScaGaussReadNoiseErrorAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(ScaGaussReadNoiseErrorAndDeriv_Late_Inv,cudaAllocSized,2)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: ScaGaussReadNoiseErrorAndDeriv_Late_Inv");
} 
 

else if (strcmp(command,"FixGaussErrorAndDeriv_Late_Fwd")==0) {
 CallCUDA_NArgsFkt(FixGaussErrorAndDeriv_Late_Fwd,cudaAllocSized,3)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: FixGaussErrorAndDeriv_Late_Fwd");
} 
 

else if (strcmp(command,"FixGaussErrorAndDeriv_Late_Inv")==0) {
 CallCUDA_NArgsFkt(FixGaussErrorAndDeriv_Late_Inv,cudaAllocSized,3)
 if (nlhs > 0) 
     plhs[0] =  mxCreateDoubleScalar((double)free_array); 
 Dbg_printf("cuda: FixGaussErrorAndDeriv_Late_Inv");
} 
