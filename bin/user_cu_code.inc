 

CUDA_UnaryFkt(mandelbrot_arr,c[idx]=a[idx];) 
CUDA_UnaryFkt(mandelbrot_carr,int n;
 float myrs=a[2*idx];
float myis=a[2*idx+1];
float myr=myrs;
float myi=myis;
 float res=-1.0;
float tmp;
for (n=0;n<1000;n++)
    {tmp=myr*myr-myi*myi+myrs;     myi=2*myr*myi+myis;      myr=tmp;
     if ((res < 0) && (myr*myr+myi*myi)>16.0) {res=(float) n;}    }
c[idx]= res;)
 

CUDA_NArgsFkt(arr_MyForcePosSqr_Late_Fwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosSqr_Coder start
Grad = 2*Grad*Fwd; Fwd = Fwd * Fwd;
// ForcePosSqr_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyForcePosSqr_Late_Inv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosSqr_Coder start
Fwd = sqrt(Fwd);

// ForcePosSqr_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyForcePosPiecewise2nd_Late_Fwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosPiecewise2nd_Coder start
{float tmp=Fwd;Grad=(tmp<0)? Grad / ((1.0 - Fwd)*(1.0 - Fwd)) : Grad*(1.0+2.0*Fwd);Fwd = (tmp>=0)? 1.0 + tmp + tmp*tmp: 1/(1.0 - tmp);}
// ForcePosPiecewise2nd_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyForcePosPiecewise2nd_Late_Inv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{

// ForcePosPiecewise2nd_Coder start
Fwd = (Fwd < 1.0)? 1.0-1.0/Fwd : sqrt(Fwd-0.75)-0.5;

// ForcePosPiecewise2nd_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},1) 
 

CUDA_NArgsFkt(arr_MyPoissonErrAndDeriv_Late_Fwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// PoissonErrorAndDeriv_Coder start
{Grad=Grad*(1.0-Data0/(Fwd+1.e-7)); Fwd = (Fwd-Data0) - Data0*log((Fwd+1e-7)/(Data0+1e-7));}

// PoissonErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_MyPoissonErrAndDeriv_Late_Inv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// PoissonErrorAndDeriv_Coder start


// PoissonErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_MyGaussErrAndDeriv_Late_Fwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// GaussErrorAndDeriv_Coder start
float tmp = (Fwd-Data0);Fwd=tmp*tmp; Grad=Grad*2.0*tmp;
// GaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_MyGaussErrAndDeriv_Late_Inv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// GaussErrorAndDeriv_Coder start


// GaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_ScaGaussReadNoiseErrorAndDeriv_Late_Fwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// ScaGaussReadNoiseErrorAndDeriv_Coder start
float XMinusMu=Data0-Fwd;float muPlusC=Fwd+1;Fwd=log(muPlusC)+XMinusMu*XMinusMu / muPlusC;
Grad=Grad*(1.0-2.0*XMinusMu-XMinusMu*XMinusMu / muPlusC) / muPlusC;
// ScaGaussReadNoiseErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_ScaGaussReadNoiseErrorAndDeriv_Late_Inv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];

// ScaGaussReadNoiseErrorAndDeriv_Coder start


// ScaGaussReadNoiseErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},2) 
 

CUDA_NArgsFkt(arr_FixGaussErrorAndDeriv_Late_Fwd_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];float Data1=f.s[2][idx];

// FixGaussErrorAndDeriv_Coder start
float tmp=(Fwd-Data0);Fwd=(tmp*tmp)/Data1;Grad=Grad*2.0*tmp/Data1;

// FixGaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},3) 
 

CUDA_NArgsFkt(arr_FixGaussErrorAndDeriv_Late_Inv_NArgs,{float Grad=1.0;float Fwd=f.s[0][idxArg[0]];{float Data0=f.s[1][idx];float Data1=f.s[2][idx];

// FixGaussErrorAndDeriv_Coder start


// FixGaussErrorAndDeriv_Coder end
} 
c[idxArg[0]]=Grad;f.s[0][idxArg[0]]=Fwd;
},3) 
