% max(in1) : finds the minimum of a cuda array

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

function [val,pos] = min(in1,mask,projdir)
if isa(in1,'cuda') 
    if in1.fromDip
        if (nargin < 3)  % min over all pixels
            [val,pos]=cuda_cuda('min',in1.ref);
            pos=Idx2IdxVec(pos,cuda_cuda('getSize',in1.ref));
            if (length(pos)>1)
                tmp=pos(1);pos(1)=pos(2);pos(2)=tmp;
            end
        else
            inref=in1.ref;
            val=cuda();
            val.fromDip = 1;
            if nargout > 1
                pos=cuda();
                pos.fromDip = 1;
            end
            for p=1:length(projdir)
            if projdir(p) == 1
                projdir(p) = 2;
            elseif projdir(p) == 2
                projdir(p) = 1
            end
                if nargout > 1
                    if length(projdir) > 1
                        error('cuda/min: Can only compute position for one dimension at a time');
                    end
                    [valref,posref]=cuda_cuda('part_min',inref,mask,projdir(p));
                    val.ref=valref;pos.ref=posref;
                else
                    val.ref=cuda_cuda('part_min',inref,mask,projdir(p));
                end
                if p > 1
                    cuda_cuda('delete',inref);
                end
                inref=val.ref;
            end
        end
    else
        if (nargin < 2)  % sum over all pixels
            projdir=1;
        else
            projdir=mask;
        end
        if length(projdir) > 1
            error('cuda/min: Matlab type arguments can only be projected over a maximum of one dimension at a time.');
        end
        si=size(in1);
        if prod(si) ~= max(si)
            val=cuda();
            val.ref=cuda_cuda('part_min',in1.ref,[],projdir);
            val.fromDip = 0;
            % error('cuda/sum: Patial sum of multidemensional arrays in Matlab style not yet supported.');
        else
            val=cuda_cuda('min',in1.ref);
        end
    end
else
    error('cuda/min: Unknown datatype');
end
