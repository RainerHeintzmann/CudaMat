% ones_cuda2(in,varargin) :  mimics the matlab ones function but generates a cuda array

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
function out=ones_cuda2(in,varargin)
    myconst=1;
    if isstr(varargin{end})
        if (strcmp(varargin{end},'scomplex') || strcmp(varargin{end},'dcomplex'))
            myconst=complex(1,0);
        end
        varargin{end}=[];  % delete that last entry
    end
    ms=cell2mat(varargin); % size
    if prod(ms) == 1
        out=myconst;
    else
        if length(ms)==1
            ms(2)=ms(1);
        end
        out=in;
        cuda_cuda('delete',out.ref);
        out.ref=cuda_cuda('newarr',ms,myconst);
        out.fromDip=0;
    end
