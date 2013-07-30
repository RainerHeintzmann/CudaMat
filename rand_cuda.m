% rand_zeros : overloaded function from the cudaMat package. 
% Decides whether the standard matlab function or the cuda version is used, depending on the state of the global use_rand_cuda variable, 
% which can be set via the function set_rand_cuda()

%***************************************************************************
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

function res=rand_cuda(varargin)
global use_rand_cuda;
if (use_rand_cuda)
    res= cuda(rand(varargin{:}));
else
    res=builtin('rand',varargin{:});  % call the standart matlab zeros function
end
