% svd(in) : singular value decomposition using CULA

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
function [U,S,V] = svd(in1)
if isa(in1,'cuda') 
    if nargout > 1
        [refU,refS,refV]=cuda_cuda('svd',in1.ref); 
        U=cuda(refU,in1.fromDip);
        S=cuda(refS,in1.fromDip);
        V=cuda(refV,in1.fromDip);
        V=V';
        S=diag(S);
    else
        refS=cuda_cuda('svd',in1.ref); 
        S=cuda(refS,in1.fromDip);
        U=S;V=S;
    end
else
    error('svd: Unknown datatype');
end
