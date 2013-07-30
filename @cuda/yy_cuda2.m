% yy_cuda2(in,varargin) :  mimics the dipimage yy function in cuda. See the dipimage function for documentation.

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
function out=yy_cuda2(in,varargin)
    pos1=1;
    mysize=[];
    if numel(varargin) > 0  && isnumeric(varargin{pos1})
        while numel(varargin) >= pos1 && isnumeric(varargin{pos1})
            mysize=[mysize varargin{pos1}];
            pos1=pos1+1;
        end
    elseif numel(varargin) > 0
        mysize=size(varargin{1});
        pos1=2;
    else
        mysize=[256 256];
        pos1=1;
    end
    if length(mysize)>1   % Swap dimensions to be in accordance with the dipimage nomenclature
       tmp=mysize(2);mysize(2)=mysize(1);mysize(1)=tmp;
    end
    if numel(varargin) >= pos1
        location=varargin{pos1}; % options are 'left', 'right', 'true', 'corner'
    else
        location='right';  % default value
    end
    
    if numel(mysize)> 1
        dsize=mysize(1);
    else
        dsize=0;
    end
    %if location(1)=='m'
    %     dsize=-(dsize-2);
    % end
    switch(location)
        case {'right','mleft'}
            startVec=[-floor(dsize/2) 0];
            endVec=[floor((dsize+1)/2) 0];
        case {'corner','mcorner'}
            startVec=[0 0];
            endVec=[dsize 0];
        case {'true','mtrue'}
            startVec=[-dsize/2+0.5 0];
            endVec=[dsize/2+0.5 0];
        case {'left','mright','math'}
            startVec=[-floor((dsize-1)/2) 0];
            endVec=[floor(dsize/2)+1 0];
        case 'freq'
            startVec=[-floor(dsize/2)/(2*floor(dsize/2+0.1)) 0];
            endVec=[floor((dsize+1)/2)/(2*floor(dsize/2+0.1)) 0];
        case 'mfreq'
            startVec=[-floor(-(dsize-2)/2)/(2*ceil(-(dsize+0.1)/2)) 0];
            endVec=[floor((-dsize-1)/2)/(2*ceil(-(dsize+0.1)/2)) 0];
        case 'radfreq'
            startVec=[-pi*floor(dsize/2)/(floor(dsize/2+0.1)) ];
            endVec=[pi*floor((dsize+1)/2)/(floor(dsize/2+0.1)) 0];
        case 'mradfreq'
            startVec=[-pi*floor(-(dsize-2)/2)/(ceil(-(dsize+0.1)/2)) 0];
            endVec=[pi*floor((-dsize-1)/2)/(ceil(-(dsize+0.1)/2)) 0];
        otherwise
            fprintf('Wrong location method: %s. ',location);
            error('Location method not implemented');
    end
      if location(1)=='m'  % If it starts with m, this axis has to be exchanged
          tmp=startVec(1);
          sv=endVec(1)-startVec(1);
          startVec(1)=startVec(1)+sv*(dsize-1)/dsize;
          endVec(1)=startVec(1)-sv;
      end
    startVec(isnan(startVec)|isinf(startVec))=0;
    endVec(isnan(endVec)|isinf(endVec))=0;
    out=in;
    cuda_cuda('delete',out.ref);
    out.ref=cuda_cuda('xyz',mysize,startVec,endVec);
    out.fromDip=1;
end
