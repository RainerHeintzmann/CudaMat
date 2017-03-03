% plus(in1,in2) : adds two cuda objects on the card

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
function out = plus(in1,in2)
out=cuda();
if prod(size(in1)) > 1 && ~isa(in1,'cuda')
    in1=cuda(in1);
end
if prod(size(in2)) > 1 && ~isa(in2,'cuda')
    in2=cuda(in2);
end

if isa(in1,'cuda') && prod(size(in2)) == 1
    if isa(in2,'cuda')
        in2=double_force(in2);
    end
    out.ref=cuda_cuda('plus_alpha',in1.ref,double(in2));
    out.fromDip = in1.fromDip;   % If either was dipimage, result will be
elseif prod(size(in1)) == 1 && isa(in2,'cuda')
    if isa(in1,'cuda')
        in1=double_force(in1);
    end
    out.ref=cuda_cuda('plus_alpha',in2.ref,double(in1));
    out.fromDip = in2.fromDip;   % If either was dipimage, result will be
elseif isa(in1,'cuda') && isa(in2,'cuda')
    if (~in1.fromDip && any(size(in1) - size(in2)))
        error('cuda:plus of Matlab type: Matrix dimensions must agree.')
    end
    
    didSwap1=0;didSwap2=0;
    if in1.fromDip == 1 && ndims(in1) == 1 && ndims(in2) > 1
        cuda_cuda('swapSizeForceDim2',in1.ref);didSwap1=1;
    end
    if in2.fromDip == 1 && ndims(in2) == 1 && ndims(in1) > 1
        cuda_cuda('swapSizeForceDim2',in2.ref);didSwap2=1;
    end
    
    out.ref=cuda_cuda('plus',in1.ref,in2.ref);

    if didSwap1
        cuda_cuda('swapSizeForceDim1',in1.ref);
    end        
    if didSwap2
        cuda_cuda('swapSizeForceDim1',in2.ref);
    end        
    out.fromDip = in1.fromDip ;   % If either was dipimage, result will be
end
