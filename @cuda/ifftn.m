% ifftn(in) : n-dimensional inverse Fourier transforms cuda dat (up to 3D)

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

function out=ifftn(in)
out=cuda();
if isa(in,'cuda') 
    if in.fromDip
        error('ifftn: Not used for datatype DipImage. Use ft or dip_fouriertransform instead!');
    end
    insize=size(in);
    dims=length(insize);
    if dims>3 && prod(insize(4:end)) > 1
        error('ifftn is only implemented up to 3D in CudaMat');
    else
        out.ref=cuda_cuda('fft3d',in.ref,-1);
    end
else
    error('ifftn: Unsupported datatype');
end
out.fromDip=in.fromDip;

out = out / prod(insize);   % This is a special Matlab feature...
