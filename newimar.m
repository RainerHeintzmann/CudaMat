% newimar : overloaded function from the cudaMat package which shadows the newimar function of DIPImage. 
% Decides whether the standard dipimage function or the cuda version is used, depending on the state of the global use_newimar_cuda variable, 
% which can be set via the function set_newimar_cuda()

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

function res=newimar(varargin)
global use_newimar_cuda;
global diphandle_newimar;
if (use_newimar_cuda)
    if nargin > 0 && isnumeric(varargin{1})
        % asize=cell2mat(varargin);
        if nargin < 2
            res=cell([varargin{1},1]);
        else
            res=cell(varargin{:});
        end
    elseif nargin > 0
        asize=numel(varargin);
        res=cell([asize,1]);
        for n=1:nargin
            res{n}=varargin{n};
        end
    else
        res=cell();
    end
else
    res=feval(diphandle_newimar,varargin{:});  % call the standart matlab zeros function
end
