% dip_image(in,varargin) : conversion from cuda to dip_image


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

function out = dip_image(in,varargin)

if length(varargin) > 0
    %    if strcmp(varargin,'single') || strcmp(varargin,'double') || strcmp(varargin{,'int')
    if strcmp(varargin{1},'scomplex')  || strcmp(varargin{1},'dcomplex')
        if ~cuda_cuda('isCpx',in)
            out=scomplex(in);
        else
            out=copy(in);
        end
    else
        out=copy(in);
    end
else
    out=copy(in);
end
if in.fromDip == 0  % This was a matlab object
    s=size(in);
    if length(s) == 2 & s(2) == 1
        cuda_cuda('setSize',out.ref,s(1));
    end
end
out.fromDip=1;
