% fft(in) : Fourier transforms cuda dat (up to 3D)

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

function out=fft(in)
out=cuda();
if isa(in,'cuda') 
    if in.fromDip
        error('fft: Not used for datatype DipImage. Use ft instead!');
    end
    myDirYes = zeros(1,ndims(in));
    myDirYes(firstNonSingleton(in))=1;
    out.ref=cuda_cuda('fftnd',in.ref,1,myDirYes);
else
    error('fft: Unsupported datatype');
end
out.fromDip=in.fromDip;
