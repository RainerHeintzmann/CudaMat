% transpose(in) : tranposing of a complex cuda array

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
function out = transpose(in1)
if isa(in1,'cuda') 
    if (in1.fromDip)
        ref=cuda_cuda('copy',in1.ref);  % no transpose for DipImages for compatability reasons
    else
        if ndims(in1) > 2
            s=size(in1);
            if prod(s(3:end)) > 1
                error('CudaMat: Transpose only defined for less than 3 dimensions (as also in Matlab).');
            end
        end
        ref=cuda_cuda('transpose',in1.ref,0);  % no conjugation
    end
    out=cuda(ref,in1.fromDip);
else
    error('transpose: Unknown datatype');
end
