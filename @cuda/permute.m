% permute(in, vargin) :  alters the shape by permuting the dimensions. See Matlab documentation for details.

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
function out = permute(in,varargin)
% error('cuda: permute. NOT finnished implementing.');
out=cuda();
permvec=varargin{1};
if ~isnumeric(permvec) | sum(size(permvec)>1)>1 | prod(size(permvec))<1 | fix(permvec)~=permvec
   error('cuda permute: permvec must be an integer vector.')
elseif length(permvec) < ndims(in)
        error('cuda permute: permvec must have at least NDIMS elements.')
elseif any(permvec<1)
   error('cuda permute: permvec contains an invalid permutation index.')
elseif length(permvec)~=length(unique(permvec))
   error('cuda permute: permvec cannot contain repeated permutation indices.')
end
if in.fromDip
	if length(permvec)~= ndims(in)
        error('cuda permute: DipImage style permvec must have NDIMS elements.')
    elseif any(permvec>ndims(in))
        error('cuda permute: DipImage style permvec must not refer to larger indices than NDIMS.')
    end
    if length(permvec)>1
        tmp=permvec(1);permvec(1)=permvec(2);permvec(2)=tmp;
        tmp=find(permvec==1);permvec(permvec==2)=1;permvec(tmp)=2;clear tmp;
    end
    out.ref=cuda_cuda('permute',in.ref,permvec-1);   % this is a modification of the size of the now existing array. Use with care!
else
    out.ref=cuda_cuda('permute',in.ref,permvec-1);
end

out.fromDip = in.fromDip; 
