% circshift(in,shifts) : circularly shifts (rotates) and array by the vector 

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
function out = circshift(in,shifts)
out=cuda();
if isa(in,'cuda') 
    if (~in.fromDip && size(shifts,2)>1)
        tmp=shifts(1);shifts(1)=shifts(2);shifts(2)=tmp;
    end
    if (size(in,1) == 1 && size(shifts,2)==1)  % This is some funny special case that Matlab implements
        shifts=[0 shifts];  % Interprete in only this case as a shift along the vector
    else
    if (size(shifts,2)>1)
        tmp=shifts(1);shifts(1)=shifts(2);shifts(2)=tmp;
    end
    end
     out.ref=cuda_cuda('circshift',in.ref,shifts);
else
    error('cuda cirshift: Unknown datatype');
end
out.fromDip = in.fromDip;  
