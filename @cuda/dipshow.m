% dipshow(in,vararg) : displays a cuda dipimage object by converting to dipimage and using the original dipshow function

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
function res=dipshow(varargin)  % casts the cuda object to a normal dipimage or double and calls dipshow
if isa(varargin{1},'cuda')
    nin=1;
elseif ~isa(varargin{1},'cuda')
    nin=2;
else 
    error('Wrong usage for dipshow, cuda version.');
end
if varargin{nin}.fromDip
    varargin{nin}=dip_image_force(varargin{nin});
else
    varargin{nin}=double_force(varargin{nin});
end
res=dipshow(varargin{:});



