% rft(in) DipImage style Fourier transforms cuda dat (up to 3D), but only half-comlex and not unscambled. For speed reasons.

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
function out=rft(in)  % Attention! The size must be even. If not, the back transform will yield a different size
out=cuda();
if isa(in,'cuda') 
    if strcmp(datatype(in),'sfloat')
        %isodd_biggerone=mod(size(in),2) & (size(in)>1);
        %if numel(isodd_biggerone)>3
        %    isodd_biggerone=isodd_biggerone(1:3);
        %end
        %if any(isodd_biggerone)
        %   error('Cuda rft function only accepts even sizes, as the ift of the result would yield a different size');
        if in.fromDip && (mod(size(in,2),2) || size(in,2)==1) || ((~ in.fromDip) && (mod(size(in,1),2) || size(in,1)==1))
            error('Cuda rft function only accepts even size along dim 1/2 (matlab/dipImage), as the rift of the result would yield a different size');
        end
        out.ref=cuda_cuda('rfft3d',in.ref);
    else
        error('Error using rft. Datatype needs to be sfloat');
    end
else
    error('rft: Unsupported datatype');
end
out.fromDip=in.fromDip;
