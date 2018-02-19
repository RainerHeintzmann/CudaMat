% fftn_part(in, transformDirs) : n-dimensional Fourier transform along selected dimensions
% see also: 
% dip_fouriertransform

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

function out=fftn_part(in,transformDirs)
out=cuda();
if isa(in,'cuda') 
    if in.fromDip
        error('fftn_part: Not used for datatype DipImage. Use ft or dip_fouriertransform instead!');
    end
    insize=size(in);
    if numel(insize) ~= numel(transformDirs)
        error('fftn_part needs a transformDirection of equal size as dimensions of the input')
    end
    out.ref=cuda_cuda('fftnd',in.ref,1,transformDirs);
else
    error('fftn: Unsupported datatype');
end
out.fromDip=in.fromDip;
