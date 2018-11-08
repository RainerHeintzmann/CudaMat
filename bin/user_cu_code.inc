 

CUDA_NArgsFkt(arr_MyForcePosSqrFwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosSqr_Coder start
Grad = 2*Grad*Fwd; Fwd = Fwd * Fwd;
// ForcePosSqr_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyForcePosSqrInv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosSqr_Coder start
Fwd = sqrt(Fwd);

// ForcePosSqr_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyForcePosPiecewise2ndFwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosPiecewise2nd_Coder start
{float tmp=Fwd;Grad=(tmp<0)? Grad / ((1.0 - Fwd)*(1.0 - Fwd)) : Grad*(1.0+2.0*Fwd);Fwd = (tmp>=0)? 1.0 + tmp + tmp*tmp: 1/(1.0 - tmp);}
// ForcePosPiecewise2nd_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyForcePosPiecewise2ndInv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosPiecewise2nd_Coder start
Fwd = (Fwd < 1.0)? 1.0-1.0/Fwd : sqrt(Fwd-0.75)-0.5;

// ForcePosPiecewise2nd_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyPoissonErrAndDerivFwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// PoissonErrorAndDeriv_Coder start
{Grad=Grad*(1.0-Data0/(Fwd+1.e-7)); Fwd = (Fwd-Data0) - Data0*log((Fwd+1e-7)/(Data0+1e-7));}

// PoissonErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_MyPoissonErrAndDerivInv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// PoissonErrorAndDeriv_Coder start


// PoissonErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_MyGaussErrAndDerivFwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// GaussErrorAndDeriv_Coder start
float tmp = (Fwd-Data0);Fwd=tmp*tmp; Grad=Grad*2.0*tmp;
// GaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_MyGaussErrAndDerivInv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// GaussErrorAndDeriv_Coder start


// GaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_ScaGaussReadNoiseErrorAndDerivFwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// ScaGaussReadNoiseErrorAndDeriv_Coder start
float XMinusMu=Data0-Fwd;float muPlusC=Fwd+1;Fwd=log(muPlusC)+XMinusMu*XMinusMu / muPlusC;
Grad=Grad*(1.0-2.0*XMinusMu-XMinusMu*XMinusMu / muPlusC) / muPlusC;
// ScaGaussReadNoiseErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_ScaGaussReadNoiseErrorAndDerivInv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// ScaGaussReadNoiseErrorAndDeriv_Coder start


// ScaGaussReadNoiseErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_FixGaussErrorAndDerivFwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];float Data1=f.s[2][idx];

// FixGaussErrorAndDeriv_Coder start
float tmp=(Fwd-Data0);Fwd=(tmp*tmp)/Data1;Grad=Grad*2.0*tmp/Data1;

// FixGaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},3) 
 

CUDA_NArgsFkt(arr_FixGaussErrorAndDerivInv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];float Data1=f.s[2][idx];

// FixGaussErrorAndDeriv_Coder start


// FixGaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},3) 
