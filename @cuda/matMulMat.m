% [myerr,myder] = matMulMat(aMatrix, aVector) : Multiplies a Vector (anImg) with a Matrix (Multfoectors). AnImg is always interpreted as a vector. The outermost dimension of MulFactors forms the result axis direction.
% In Einstein notation: result_nm = aMatrix_nk * aMatrix_km
% The first matrix is interpreted as size 1k, if it really is k1, to avoid a copy transpose for speed and memory reasons.
% If all other indices except index 2 (n or m) are interpreted as part of index k. The result will have the size indicated by all (remaining) indices in m, if n is 1.
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
function out=matMulMat(in1, in2)
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

ns1=sz1(1:2);
% ns2=sz2(1:2);
if numel(sz1) > 2
    ns1(1) = sz1(1)*prod(sz1(3:end));
end
if numel(sz2) > 2
    ns2 = [sz2(2) sz2(1)*prod(sz2(3:end))];
%    ns2(2) = prod(sz2(2:end));
else
    ns2 = sz2;
end
if ns1(2) == 1  % automatically transpose first argument, if needed
    ns1(2) = ns1(1);
    ns1(1) = 1;
end

if ns1(2) ~= ns2(1)
    error('matMulMat: sizes are not matching')
end

% rs1=sz2; rs1(2)=[];
% rs2=sz2; rs2(1)=[];
res_size = sz2; % [rs1,rs2]; 
res_size(1)=[];   % THIS PROBABLY NEEDS TO BE FIXED FOR MATRIX * MATRIX MULTIPLICATIONS

cuda_cuda('setSize',in1.ref,ns1);   % this is a modification of the size of the now existing array. Use with care!    
cuda_cuda('setSize',in2.ref,ns2);   % this is a modification of the size of the now existing array. Use with care!    

ref=cuda_cuda('mtimes',in1.ref,in2.ref); % matrix product

cuda_cuda('setSize',in1.ref,sz1);   % this is a modification of the size of the now existing array. Use with care!    
cuda_cuda('setSize',in2.ref,sz2);   % this is a modification of the size of the now existing array. Use with care!    
cuda_cuda('setSize',ref,res_size);   % this is a modification of the size of the now existing array. Use with care!    

out=cuda(ref,in1.fromDip);
if prod(size(out)) ~= prod(res_size)
    error("internal error: Size mismatch. FIX THIS")
end
