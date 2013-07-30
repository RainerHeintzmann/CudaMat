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
out=cuda();
for pos=1:length(varargin)
        varargin{pos}=cuda(varargin{pos});
end
out.fromDip = varargin{1}.fromDip;
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
end
