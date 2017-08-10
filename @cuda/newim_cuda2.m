% newim_cuda2(in,varargin) :  mimics the matlab zeros function but generates a cuda array

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
function out=newim_cuda2(in,varargin)
    myconst=0;
    if isstr(varargin{end})
        if (strcmp(varargin{end},'scomplex') || strcmp(varargin{end},'dcomplex'))
            myconst=complex(0,0);
        elseif (~strcmp(varargin{end},'single') && ~strcmp(varargin{end},'double') && ~strcmp(varargin{end},'sfloat'))
            error('newim_cuda : trying to construct an unsupported datatype');
        end
        args={varargin{1:end-1}}; % size, vectors get converted correctlya as well
    else
        args={varargin{1:end}};
    end
    for p=1:length(args)
        if isa(args{p},'cuda') || isa(args{p},'dip_image')
            args{p}=size(args{p});
        end
    end
    ms=cell2mat(args); % size, vectors get converted correctlya as well
    if prod(ms) == 1
        out=myconst;
    else
        if length(ms)>1   % Swap dimensions to be in accordance with the dipimage nomenclature
            tmp=ms(2);
            ms(2)=ms(1);
            ms(1)=tmp;
        end
        out=in;
        cuda_cuda('delete',out.ref);
        out.ref=cuda_cuda('newarr',ms,myconst);
        out.fromDip=1;
    end
