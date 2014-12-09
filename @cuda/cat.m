% cat(direction,in1,in2): appends datasets

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

function out = cat(direction,varargin)
if size(varargin,2)==1  % only copy (transport) the input to output
    if direction > ndims(varargin{1})
        out=copy(varargin{1});  % unfortunately a real copy has to be done here as the input is still lower dimensional in size and cannot be modified
        mysize=size(out);
        mysize(length(mysize)+1:direction)=1;  % expands the dimensions
        cuda_cuda('setSize',out.ref,mysize);
        if out.fromDip
            if (length(mysize)>1)
                cuda_cuda('swapSize',out.ref); % to deal with strange size order in DipImage
            end
        end
    else
        out=varargin{1};
    end
else
out=cuda();
for pos=1:length(varargin)
        varargin{pos}=cuda(varargin{pos});
end
out.fromDip = varargin{1}.fromDip;
out.isBinary = varargin{1}.isBinary; % mark this as a binary result (needed for subsasgn)
out.ref= varargin{1}.ref;

if varargin{1}.fromDip
    if (direction == 1)
        direction =2;
    elseif (direction ==2)
        direction =1;
    end
end
for m=2:size(varargin,2)
    if ~isa(varargin{m},'cuda')
        varargin{m}=cuda(varargin{m});  % convert to cuda
    end
    refold=out.ref;
    if isa(varargin{m},'cuda')        
         out.ref=cuda_cuda('cat',out.ref,varargin{m}.ref,direction);
    else
        error('unary minus: Unknown datatype');
    end
    if m>2
        cuda_cuda('delete',refold);
    end
    out.fromDip = out.fromDip || varargin{m}.fromDip; % If eiter was dipimage, result will be
    out.isBinary = out.isBinary && varargin{m}.isBinary;
end
end