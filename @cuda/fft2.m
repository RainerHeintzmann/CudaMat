% fft2(in) : Fourier transforms (2d) cuda data (matlab style)

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

function out=fft2(in,sx,sy)
if nargin==3
    in2=in;
    in=zeros_cuda(sx,sy);
    S=struct('type','()','subs',{{1:size(in2,1),1:size(in2,2)}});
    in=subsasgn(in,S,in2);
    clear in2;
end
out=cuda();
if isa(in,'cuda') 
    if in.fromDip
        error('fft: Not used for datatype DipImage. Use ft instead!');
    end
    insize=size(in);
    dims=length(insize);
    if dims>2
        transformdir = zeros(1,dims);
        transformdir(1:2) = 1;
        out.ref=cuda_cuda('fftnd',in.ref,1,double(transformdir));  % double cast is very important here. Otherwise datatype does not match
        % error('fft2 for higher dimension data not implemented. Did you want to use ifftn for 3d ffts?');
    else
        out.ref=cuda_cuda('fftnd',in.ref,1);
    end
else
    error('fft: Unsupported datatype');
end
out.fromDip=in.fromDip;
