% diff(in,N,DIM) : calculates the difference with a user-defined shift. The datasize is decrease.
% in : array to use for difference calculation
% N : number of recursions
% DIM : dimension along wich to shift (default is 1)

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

function out = diff(in1,N,mydim)
if nargin < 2
    N=1.0;
end
if nargin < 3
    mydim=1.0;
end

if N < 1
    error('Difference recurseion level N must be a positive integer scalar.')
end

if (mydim > ndims(in1))
    out=[];
    return;
end

if in1.fromDip
    if mydim==0
        mydim=1;
    elseif mydim==1;
        mydim=1;
    end
end

out=cuda();
if isa(in1,'cuda') 
    mySize=size(in1);
    newref=in1.ref;
    for r=1:N
        mystride = prod(mySize(1:mydim-1));  % Always do difference of size 1 !  Otherwise *stepsize
        mySize(mydim) = mySize(mydim) - 1;  % - abs(stepsize)
        oldref=newref;
        newref=cuda_cuda('diff',newref,mystride,mySize);  % mystride is in destination array. MySize is what to allocate
        if r>1
            cuda_cuda('delete',oldref);
        end
    end
    out.ref=newref;
else
    error('not: Unknown datatype');
end
out.fromDip = in1.fromDip;  
