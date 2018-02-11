% function [E,V]=svd2D_decomp(X)
%
%   Let X be a NxMx3 matrix such that:
%   
%   P_mn = [X(n,m,1) X(n,m,2)
%           X(n,m,2) X(n,m,3)]
%           
%   is a symmetric matrix. Then the present function computes the eigenvalues
%   E(n,m,1) E(n,m,2) and the first eigenvector [V(n,m,1) V(n,m,2)] (the second 
%   one being [V(n,m,2) -V(n,m,1)]). Hence the function outputs two matrices E
%   and V of size NxMx2.
%   
%   Compilation:
%      -linux: mex -v svd2D_decomp.cpp CFLAGS="\$CFLAGS -openmp" LDFLAGS="\$LDFLAGS -openmp" -largeArrayDims
%      -mac  : mex svd2D_decomp.cpp -DUSE_BLAS_LIB -DNEW_MATLAB_BLAS -DINT_64BITS -largeArrayDims CXX=/usr/local/Cellar/gcc/6.3.0_1/bin/g++-6 CXXOPTIMFLAGS="-O3
%                    -mtune=native -fomit-frame-pointer -fopenmp" LDOPTIMFLAGS=" -O " LINKLIBS="$LINKLIBS -lmwblas -lmwlapack -L"/usr/local/Cellar/gcc/6.3.0_1/lib/gcc/6" -L/ -fopenmp"
%  
%   Copyright (C) 2017 E. Soubies emmanuel.soubies@epfl.ch
% 
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.%
%%

function [E,V]=svd2D_decomp(in1)
    if nargout > 1
       [Eref,Vref]=cuda_cuda('svd_last',in1.ref);  % figures out from the datasize whether to call the 3D or 2D routine
       E=cuda();
       V=cuda();
       E.ref=Eref;
       V.ref=Vref;
       E.fromDip = in1.fromDip ;   % If input was dipimage, result will be
       V.fromDip = in1.fromDip ;   % If input was dipimage, result will be
    end
end
