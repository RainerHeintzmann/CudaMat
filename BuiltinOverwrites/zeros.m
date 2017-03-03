% zeros : overloaded function from the cudaMat package. 
% Decides whether the standard matlab function or the cuda version is used, depending on the state of the global use_ones_cuda variable, 
% which can be set via the function set_ones_cuda()

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

function varargout=zeros(varargin)
global use_ones_cuda;
if (use_ones_cuda)
    tmp=cuda(1.0);
    [varargout{1:nargout}] = zeros_cuda2(tmp,varargin{:});        
else
    [varargout{1:nargout}] =builtin('zeros',varargin{:});  % call the standart matlab zeros function
end
