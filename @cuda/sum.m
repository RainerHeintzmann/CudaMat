% sum(in) : computes the sum of a cuda array. Partial sums not implemented yet)

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
function val = sum(in1,mask,projdir)
if isa(in1,'cuda') 
    if ((nargin>1) && (~isempty(mask)))
        if ~isa(mask,'cuda')
            mask=cuda(mask);
        end
        maskref=mask.ref;
    else
        maskref=[];
    end

    if in1.fromDip
        if (nargin < 2)  % sum over all pixels
             val=cuda_cuda('sum',in1.ref);
        else
            if nargin < 3  % sum over all pixels with mask. For now: Have to use the part_sum function
                projdir=1:ndims(in1);
            end
            inref=in1.ref;
            val=cuda();
            val.fromDip = 1;
            for p=1:length(projdir)
                if projdir(p) == 1
                    projdir(p) = 2;
                elseif projdir(p) == 2
                    projdir(p) = 1;
                end
                val.ref=cuda_cuda('part_sum',inref,maskref,projdir(p));
                if p > 1
                    cuda_cuda('delete',inref);
                else
                    maskref=[];
                end
                inref=val.ref;
            end
        end
        if nargin < 3  % sum over all pixels with mask. Result should be converted to single value
            val=double_force(val);
        end
    else  % for matlab type, mask has the meaning of projdir, but only one direction allowed
        if (nargin < 2)  % sum over all pixels
            projdir=1;
        else
            projdir=mask;
        end
        if length(projdir) > 1
            error('cuda sum: Matlab type arguments can only be projected over a maximum of one dimension at a time.');
        end
        si=size(in1);
        if prod(si) ~= max(si)
            val=cuda();
            val.ref=cuda_cuda('part_sum',in1.ref,[],projdir);
            val.fromDip = 0;
            % error('cuda/sum: Patial sum of multidemensional arrays in Matlab style not yet supported.');
        else
            val=cuda_cuda('sum',in1.ref);
        end
    end
else
    error('cuda/sum: Unknown datatype');
end

