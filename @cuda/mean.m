% mean(in) : computes the mean of a cuda array

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

function [val,pos] = mean(in1,mask,projdir)
if ~isa(in1,'cuda')
    in1=cuda(in1);
end
if nargin >1 && ~isempty(mask) && ~isa(mask,'cuda')
    mask=cuda(mask);
end
if nargin >2 && isa(projdir,'cuda') 
    projdir=double_force(projdir);
end
if isa(in1,'cuda') 
    if ((nargin>1) && (~isempty(mask)))
        if ~isa(mask,'cuda')
            mask=cuda(mask);
        end
        maskref=mask.ref;
        if nargin < 3
            msum=sum(mask,[]);
        else
            msum=sum(mask,[],projdir);
        end
    else
        maskref=[];
    end
    
    if in1.fromDip
        if isempty(maskref) && (nargin < 3 || isempty(projdir)) % mean over all pixels
            val=cuda_cuda('sum',in1.ref);
            val=val/prod(size(in1));
        else
            if nargin < 3  % sum over all pixels with mask. For now: Have to use the part_sum function
                projdir=1:ndims(in1);
            end
            inref=in1.ref;
            val=cuda();
            val.fromDip = 1;
           for p=1:length(projdir)
               origprojdir=projdir(p);
                if origprojdir == 1
                    projdir(p) = 2;
                elseif origprojdir == 2
                    projdir(p) = 1;
                end
                val.ref=cuda_cuda('part_sum',inref,maskref,projdir(p));
                if isempty(mask)
                    val=val/size(in1,origprojdir);
                end

                if p > 1
                    cuda_cuda('delete',inref);
                else
                    maskref=[];  % mask only needs to be applied once
                end
                inref=val.ref;
            end
        end
        if nargin < 3  % sum over all pixels with mask. Result should be converted to single value
            val=double_force(val);
        end
        
        if nargin > 1 && ~isempty(mask)  % the division was omitted so far
            val=val ./ msum;
        end

    else  % for matlab type, mask has the meaning of projdir, but only one direction allowed
        si=size(in1);
        if (nargin < 2)  % sum over all pixels
            projdir=find(si~=1,1);  % This is the first non-singleton dimension, which MATLAB chooses to project over
            if isempty(projdir)
                val = cuda_cuda('getVal',in1.ref,0);
                return;
            end
        else
            projdir=mask;
            if isa(projdir,'cuda') 
                projdir=double_force(projdir);
            end
            mask=[];
        end
        if length(projdir) > 1
            error('cuda mean: Matlab type arguments can only be projected over a maximum of one dimension at a time.');
        end
        if prod(si) ~= max(si)
            val=cuda();
            val.ref=cuda_cuda('part_sum',in1.ref,[],projdir);
            if isempty(mask)
                val=val/size(in1,projdir);
                % val=val/size(in1,p);
            end
            val.fromDip = 0;
            % error('cuda/sum: Patial sum of multidemensional arrays in Matlab style not yet supported.');
        else
            val=cuda_cuda('sum',in1.ref);
            val=val/prod(size(in1));
        end
    end
else
    error('cuda/mean: Unknown datatype');
end
            
if isa(val,'cuda') && ~val.fromDip
    val=removeTrailingDims(val);  % This is a crazy Matlab thing: Trailing empty dimensions are removed. Not so in DipImage.
end
