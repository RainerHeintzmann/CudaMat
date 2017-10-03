% reshape(in, vargin) :  alters the shape without changing the data

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
function out = reshape(in,varargin)
s=size(in);
if length(varargin{1}) > 1  % this is a multidimensional size vector
    ns=varargin{1};
else
    ns=cell2mat(varargin);
end
if prod(ns) ~= prod(s)
    error('cuda: reshape Totalsize is not compatible.')
end
if in.fromDip
      if (length(s) > 1 && ((length(ns)<2) || any(s(1:2) ~= ns(1:2))))  % only in this condition we really need to go down this ugly road and copy data around
          out=permute(in,[2 1 3:ndims(in)]);  % Very ugly ! Why is this necessary?
          cuda_cuda('setSize',out.ref,ns);   % this is a modification of the size of the now existing array. Use with care!    
          if (length(ns) > 1)
              out=permute(out,[2 1 3:ndims(out)]);  % Very ugly ! Why is this necessary?
          end
      else
        out=in; % copy(in); % will create a copy
        if (length(ns) > 1)
            tmp=ns(1);ns(1)=ns(2);ns(2)=tmp;
        end
        cuda_cuda('setSize',out.ref,ns);   % this is a modification of the size of the now existing array. Use with care!    
      end
else
    if length(ns) < 2
        error('cuda: sizevector must have at least two elements (matlab style)');
    end
    out=in; % copy(in); % will create a copy
    cuda_cuda('setSize',out.ref,ns);
end

