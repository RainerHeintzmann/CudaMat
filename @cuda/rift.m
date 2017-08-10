% rift(in) DipImage style inverse Fourier transforms cuda dat (up to 3D), but only half-comlex and not unscambled. For speed reasons.

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
function out=rift(in)
out=cuda();
if isa(in,'cuda') 
    if strcmp(datatype(in),'scomplex')
      out.ref=cuda_cuda('rifft3d',in.ref);
    else
      error('Error using rift. Datatype needs to be scomplex');
    end
else
    error('rift: Unsupported datatype');
end
out.fromDip=in.fromDip;
