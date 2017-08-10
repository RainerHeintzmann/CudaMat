% superiorfloat(varargin) : downcasts the datatype (single if one argument si single).

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
function out=superiorfloat(varargin)
out='double';
for d=1:length(varargin)
    if strcmp(class(varargin{d}),'single')
        out='single';
    elseif strcmp(class(varargin{d}),'cuda')
        out='single';
    elseif strcmp(class(varargin{d}),'double') ~= 1
        error(['superiorfloat cuda: inputs must be floats, single or double (or cuda version of them). Found: ' class(varargin{d}) ]);
    end
end
