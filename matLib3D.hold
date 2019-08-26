/* =========================================================================
 * %
 * %  Author: stamatis.lefkimmiatis@epfl.ch
 * %
 * % =========================================================================*/


__device__ inline double hypot2(double x, double y) {
  return sqrt(x*x+y*y);
}

// Symmetric Householder reduction to tridiagonal form.

__device__ inline void tred2(double V[9], double d[3], double e[3]) {
  
//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.
  
  int i, j, k;
  double f, g, h, hh;
  for (j = 0; j < 3; j++) {
    d[j] = V[2+3*j];
  }
  
  // Householder reduction to tridiagonal form.
  
  for (i = 2; i > 0; i--) {
    
    // Scale to avoid under/overflow.
    
    double scale = 0.0;
    double h = 0.0;
    for (k = 0; k < i; k++) {
      scale = scale + fabs(d[k]);
    }
    if (scale == 0.0) {
      e[i] = d[i-1];
      for (j = 0; j < i; j++) {
        d[j] = V[i-1+3*j];
        V[i+3*j] = 0.0;
        V[j+3*i] = 0.0;
      }
    } else {
      
      // Generate Householder vector.
      
      for (k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      f = d[i-1];
      g = sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h = h - f * g;
      d[i-1] = f - g;
      for (j = 0; j < i; j++) {
        e[j] = 0.0;
      }
      
      // Apply similarity transformation to remaining columns.
      
      for (j = 0; j < i; j++) {
        f = d[j];
        V[j+3*i] = f;
        g = e[j] + V[j+3*j] * f;
        for (k = j+1; k <= i-1; k++) {
          g += V[k+3*j] * d[k];
          e[k] += V[k+3*j] * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for (j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      hh = f / (h + h);
      for (j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (k = j; k <= i-1; k++) {
          V[k+3*j] -= (f * e[k] + g * d[k]);
        }
        d[j] = V[i-1+3*j];
        V[i+3*j] = 0.0;
      }
    }
    d[i] = h;
  }
  
  // Accumulate transformations.
  
  for (i = 0; i < 2; i++) {
    V[2+3*i] = V[4*i];
    V[4*i] = 1.0;
    h = d[i+1];
    if (h != 0.0) {
      for (k = 0; k <= i; k++) {
        d[k] = V[k+3*(i+1)] / h;
      }
      for (j = 0; j <= i; j++) {
        g = 0.0;
        for (k = 0; k <= i; k++) {
          g += V[k+3*(i+1)] * V[k+3*j];
        }
        for (k = 0; k <= i; k++) {
          V[k+3*j] -= g * d[k];
        }
      }
    }
    for (k = 0; k <= i; k++) {
      V[k+3*(i+1)] = 0.0;
    }
  }
  for (j = 0; j < 3; j++) {
    d[j] = V[2+3*j];
    V[2+3*j] = 0.0;
  }
  V[8] = 1.0;
  e[0] = 0.0;
}

// Symmetric tridiagonal QL algorithm.

__device__ inline void tql2(double V[9], double d[3], double e[3]) {
  
//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.
  
  int i, j, m, l, k;
  double g, p, r, dl1, h, f, tst1, eps;
  double c, c2, c3, el1, s, s2;
  
  for (i = 1; i < 3; i++) {
    e[i-1] = e[i];
  }
  e[2] = 0.0;
  
  f = 0.0;
  tst1 = 0.0;
  eps = pow(2.0, -52.0);
  for (l = 0; l < 3; l++) {
    
    // Find small subdiagonal element
    
    tst1 = max(tst1, fabs(d[l]) + fabs(e[l]));
    m = l;
    while (m < 3) {
      if (fabs(e[m]) <= eps*tst1) {
        break;
      }
      m++;
    }
    
    // If m == l, d[l] is an eigenvalue,
    // otherwise, iterate.
    
    if (m > l) {
      int iter = 0;
      do {
        iter = iter + 1;  // (Could check iteration count here.)
        
        // Compute implicit shift
        
        g = d[l];
        p = (d[l+1] - g) / (2.0 * e[l]);
        r = hypot2(p, 1.0);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l+1] = e[l] * (p + r);
        dl1 = d[l+1];
        h = g - d[l];
        for (i = l+2; i < 3; i++) {
          d[i] -= h;
        }
        f = f + h;
        
        // Implicit QL transformation.
        
        p = d[m];
        c = 1.0;
        c2 = c;
        c3 = c;
        el1 = e[l+1];
        s = 0.0;
        s2 = 0.0;
        for (i = m-1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p, e[i]);
          e[i+1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i+1] = h + s * (c * g + s * d[i]);
          
          // Accumulate transformation.
          
          for (k = 0; k < 3; k++) {
            h = V[k+3*(i+1)];
            V[k+3*(i+1)] = s * V[k+3*i] + c * h;
            V[k+3*i] = c * V[k+3*i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        
        // Check for convergence.
        
      } while (fabs(e[l]) > eps*tst1);
    }
    d[l] = d[l] + f;
    e[l] = 0.0;
  }
  
  // Sort eigenvalues and corresponding vectors.
  
  for (i = 0; i < 2; i++) {
    k = i;
    p = d[i];
    for (j = i+1; j < 3; j++) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = p;
      for (j = 0; j < 3; j++) {
        p = V[j+3*i];
        V[j+3*i] = V[j+3*k];
        V[j+3*k] = p;
      }
    }
  }
}

__device__ inline void eigen3x3SymRec(double X[6], double V[9], double E[3]){
  X[0]=V[0]*V[0]*E[0]+V[3]*V[3]*E[1]+V[6]*V[6]*E[2];
  X[1]=V[0]*V[1]*E[0]+V[3]*V[4]*E[1]+V[6]*V[7]*E[2];
  X[2]=V[0]*V[2]*E[0]+V[3]*V[5]*E[1]+V[6]*V[8]*E[2];
  X[3]=V[1]*V[1]*E[0]+V[4]*V[4]*E[1]+V[7]*V[7]*E[2];
  X[4]=V[1]*V[2]*E[0]+V[4]*V[5]*E[1]+V[7]*V[8]*E[2];
  X[5]=V[2]*V[2]*E[0]+V[5]*V[5]*E[1]+V[8]*V[8]*E[2];
}
