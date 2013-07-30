% repmat(in,varargin) :  replicates a matrix or 3d array of cuda type.

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
function out=repmat(in,varargin)
    ms=cell2mat(varargin); % size
    %if prod(ms) == 1   % This is not OK, as repmat may want to change the dimensionality anyway
    %    out=in;
    %else
        if length(ms)==1
            ms(2)=ms(1);
        end
        if in.fromDip
            tmp=ms(2);ms(2)=ms(1);ms(1)=tmp;
        end
        out=cuda();
        out.ref=cuda_cuda('repmat',in.ref,ms);
        out.fromDip=in.fromDip;
    %end
