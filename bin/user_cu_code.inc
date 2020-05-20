 

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
