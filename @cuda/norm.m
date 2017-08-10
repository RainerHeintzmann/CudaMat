% norm(in1, method) : norm of the vector. This includes taking the absolute values for complex numbers

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

function out = norm(in1,method)
if nargin < 2
    method=2;
end

if method == 2
    %tmpSize=cuda_cuda('getSize',in1.ref);
    %cuda_cuda('setSize',in1.ref,prod(tmpSize));  % flatten it (in case of matrices or images). NOT  CORRECT. Norm of matrix is not equal to norm of flat vector
    out=sqrt(dot(in1,in1));
    %cuda_cuda('setSize',in1.ref,tmpSize);  % back to normal size
elseif method == 1
    out=max(abs(in1));
else
    error('cuda norm: method not implemented');
end
