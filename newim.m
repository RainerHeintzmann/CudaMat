% newim : overloaded function from the cudaMat package. 
% Decides whether the standard dipimage function or the cuda version is used, depending on the state of the global use_newim_cuda variable, 
% which can be set via the function set_newim_cuda()

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

function res=newim(varargin)
global use_newim_cuda;
global diphandle_newim;
if (use_newim_cuda)
    tmp=cuda();  % this serves to call the cuda version
    res= newim_cuda2(tmp,varargin{:});        
else
    if nargin >= 1 && isa(varargin{1},'cuda')
        error('A cuda object was supplied to the dipimage version of newim (see setting of use_newim_cuda');
    end
    res=feval(diphandle_newim,varargin{:});  % call the standart matlab zeros function
end
