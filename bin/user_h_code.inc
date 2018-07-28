 

externC const char * CUDAmandelbrot_arr(float * a, float * c, size_t N);
externC const char * CUDAmandelbrot_carr(float * a, float * c, size_t N);

 

externC const char * CUDAarr_MyForcePosSqr_Late_Fwd_NArgs(float * f[1], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[1]);
 

externC const char * CUDAarr_MyForcePosSqr_Late_Inv_NArgs(float * f[1], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[1]);
 

externC const char * CUDAarr_MyForcePosPiecewise2nd_Late_Fwd_NArgs(float * f[1], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[1]);
 

externC const char * CUDAarr_MyForcePosPiecewise2nd_Late_Inv_NArgs(float * f[1], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[1]);
 

externC const char * CUDAarr_MyPoissonErrAndDeriv_Late_Fwd_NArgs(float * f[2], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[2]);
 

externC const char * CUDAarr_MyPoissonErrAndDeriv_Late_Inv_NArgs(float * f[2], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[2]);
 

externC const char * CUDAarr_MyGaussErrAndDeriv_Late_Fwd_NArgs(float * f[2], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[2]);
 

externC const char * CUDAarr_MyGaussErrAndDeriv_Late_Inv_NArgs(float * f[2], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[2]);
 

externC const char * CUDAarr_ScaGaussReadNoiseErrorAndDeriv_Late_Fwd_NArgs(float * f[2], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[2]);
 

externC const char * CUDAarr_ScaGaussReadNoiseErrorAndDeriv_Late_Inv_NArgs(float * f[2], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[2]);
 

externC const char * CUDAarr_FixGaussErrorAndDeriv_Late_Fwd_NArgs(float * f[3], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[3]);
 

externC const char * CUDAarr_FixGaussErrorAndDeriv_Late_Inv_NArgs(float * f[3], float * c, size_t N, int numdims, SizeND sizesC, BoolND isSingleton[3]);
