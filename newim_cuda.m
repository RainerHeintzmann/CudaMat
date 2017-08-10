% newim_cuda : overloaded function from the cudaMat package. 
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

function res=newim_cuda(varargin)
global use_newim_cuda;
if (use_newim_cuda)
    tmp=cuda(0);
    res= newim_cuda2(tmp,varargin{:});        
else
    dip_type = 'sfloat';
    n = [256,256];    
    if nargin > 0 && isstr(varargin{end})
        dip_type=varargin{end};
        args={varargin{1:end-1}};  % delete that last entry
    else
        args={varargin{1:end}};  
    end
    for p=1:length(args)
        if strcmp(class(args{p}),'cuda') || strcmp(class(args{p}),'dip_image')
            args{p}=size(args{p});
        end
    end
    if nargin > 0
        n=cell2mat(args); % size, vectors get converted correctlya as well
    end
    res=dip_image('zeros',n,dip_type); % call the standart dipimage function
end
