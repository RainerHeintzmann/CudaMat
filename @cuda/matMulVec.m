% [myerr,myder] = matMulVec(aMatrix, aVector) : Multiplies a Vector (anImg) with a Matrix (Multfoectors). AnImg is always interpreted as a vector. The outermost dimension of MulFactors forms the result axis direction.
% In Einstein notation: result_k = aMatrix_nk * aVector_n
% It also works for the following case: result_k = aMatrix_nmk * aVector_nm, in the case Matlab direction 2 (!) defines the size of the result vector whereas size1*size3*... are interpreted as a vector
% This cuda-version is optimized to avoid any copy operations

%************************** CudaMat ****************************************
%   Copyright (C) 2008-2009 by Rainer Heintzmann                          *
%   heintzmann@gmail.com                                                  *
%                                                                         *
%   This program is free software; you can redistribute it and/or modify  *
%   it under the terms of the GNU General Public License as published by  *
%   the Free Software Foundation; Version 2 of the License.               *
%                                                                         *
%   This program is distributed in the hope that it will be useful,       *
%   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
%   GNU General Public License for more details.                          *
%                                                                         *
%   You should have received a copy of the GNU General Public License     *
%   along with this program; if not, write to the                         *
%   Free Software Foundation, Inc.,                                       *
%   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
%**************************************************************************
%
function out=matMulVec(in1, in2)
if ~isa(in1,'cuda') 
    in1=cuda(in1);
end
if ~isa(in2,'cuda') 
    in2=cuda(in2);
end

sz1 = cuda_cuda('getSize',in1.ref);   % this is a modification of the size of the now existing array. Use with care!    
sz2 = cuda_cuda('getSize',in2.ref);   % this is a modification of the size of the now existing array. Use with care!    
% sz1 = size(in1);
% sz2 = size(in2);
if numel(sz1) ~= numel(sz2)+1
    if numel(sz1) ~= 2 || numel(sz2) ~=2 || sz2(2) ~= 1
        error('Matrix needs to have exctly one more dimension than vector input')
    end
end

if numel(sz1) > 2
    ns1 = [sz1(2) sz1(1)*prod(sz1(3:end))];
else
    ns1 = sz1;
end
ns2 = [prod(sz2) 1];
if ns2(1) ~= ns1(2)
    error('matMulVec: sizes are not matching')
end
res_size = [ns1(1) 1]; 
% if isDipImage(in1) and numel(sz1) > 2
%     res_size(end)=sz1(end);
% else
%     res_size(end)=sz1(1);
% end
% new_sz2 = [prod(sz2)] not needed?
cuda_cuda('setSize',in1.ref,ns1);   % this is a modification of the size of the now existing array. Use with care!    
cuda_cuda('setSize',in2.ref,ns2);   % this is a modification of the size of the now existing array. Use with care!    
ref=cuda_cuda('mvtimes',in1.ref,in2.ref);
cuda_cuda('setSize',in1.ref,sz1);   % this is a modification of the size of the now existing array. Use with care!    
cuda_cuda('setSize',in2.ref,sz2);   % this is a modification of the size of the now existing array. Use with care!    

cuda_cuda('setSize',ref,res_size);   % this is a modification of the size of the now existing array. Use with care!    

out=cuda(ref,in1.fromDip);
