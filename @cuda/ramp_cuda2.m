% ramp_cuda2(in,varargin) :  mimics the dipimage ramp function. See the dipimage function for documentation.

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
function out=ramp_cuda2(in,mysize,adim,varargin)
    pos1=1;
    if nargin < 2
        mysize=[];
    end
    if nargin < 3
        adim = 1;
    end
    if isempty(mysize)
        mysize=[256 256];
    end
    if ~isnumeric(mysize)
        mysize=size(mysize{1});
    end
    if length(mysize)>1   % Swap dimensions to be in accordance with the dipimage nomenclature
       tmp=mysize(2);mysize(2)=mysize(1);mysize(1)=tmp;
       if adim == 2
           adim = 1;
       elseif adim == 1
           adim =2;
       end
    end
    if numel(varargin) >= pos1
        location=varargin{pos1}; % options are 'left', 'right', 'true', 'corner'
    else
        location='right';  % default value
    end
    
    if numel(mysize)> 1
        dsize=mysize(adim);
    else
        dsize=mysize(1);
    end

    switch(location)
        case {'right','math','mright'}
            startPos=-floor(dsize/2);
            endPos=floor((dsize+1)/2);
        case {'corner','mcorner'}
            startPos=0;
            endPos=dsize;
        case {'true','mtrue'}
            startPos=-dsize/2+0.5;
            endPos=dsize/2+0.5;
        case {'left','mleft'}
            startPos=-floor((dsize-1)/2);
            endPos=floor(dsize/2)+1;
        case {'freq','mfreq'}
            startPos=-floor(dsize/2)/(2*floor(dsize/2));
            endPos=floor((dsize+1)/2)/(2*floor(dsize/2));
        case {'radfreq','mradfreq'}
            startPos=-pi*floor(dsize/2)/(floor(dsize/2));
            endPos=pi*floor((dsize+1)/2)/(floor(dsize/2));
        otherwise
            fprintf('Wrong location method: %s. ',location);
            error('Location method not implemented');
    end
    startVec=mysize * 0;
    startVec(adim)=startPos;
    endVec=mysize * 0;
    endVec(adim)=endPos;
            
    if numel(mysize)==1
        startVec=startVec(2);
        endVec=endVec(2);
    end
    startVec(isnan(startVec)|isinf(startVec))=0;
    endVec(isnan(endVec)|isinf(endVec))=0;
    out=in;
    cuda_cuda('delete',out.ref);
    out.ref=cuda_cuda('xyz',mysize,startVec,endVec);
    out.fromDip=1;
end
