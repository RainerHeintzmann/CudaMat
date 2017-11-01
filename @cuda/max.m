% max(in1) : finds the maximum of a cuda array

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

function [val,pos] = max(in1,mask,projdir)
if ~isa(in1,'cuda')
    in1=cuda(in1);
end
if nargin > 1 && ~isempty(mask) && ~isa(mask,'cuda')
    mask=cuda(mask);
end
if nargin >2 && isa(projdir,'cuda') 
    projdir=double_force(projdir);
end
    si=size(in1);
    if ~in1.fromDip && nargin < 2
        projdir=find(si~=1,1);  % This is the first non-singleton dimension, which MATLAB chooses to project over
        if isempty(projdir)
            val = cuda_cuda('getVal',in1.ref,0);
            return;
        else
            [val,pos] = max(in1,[],projdir);
            return;
        end
    end

    if nargin ~= 2 % This should be changed later as it is incorrect: A check for the datatype being binary needs to be made but cuda does not remember to be binary
        if (nargin < 3) || prod(si) == max(si) % max over all pixels
            [val,pos]=cuda_cuda('max',in1.ref);
            pos=Idx2IdxVec(pos,cuda_cuda('getSize',in1.ref));
            if (length(pos)>1)
                tmp=pos(1);pos(1)=pos(2);pos(2)=tmp;
            end
        else
            inref=in1.ref;
            val=cuda();
            val.fromDip = in1.fromDip;
            if nargout > 1
                pos=cuda();
                pos.fromDip = 1;
            end
            if ~isempty(mask) && ~isa(mask,'cuda')
                mask=cuda(mask);
            end
            for p=1:length(projdir)
            if projdir(p) == 1 && in1.fromDip
                projdir(p) = 2;
            elseif projdir(p) == 2 && in1.fromDip
                projdir(p) = 1 
            end
            if ~isempty(mask)
                maskref=mask.ref;
            else
                maskref=[];
            end
                if nargout > 1
                    if length(projdir) > 1
                        error('cuda/max: Can only compute position for one dimension at a time');
                    end
                    [valref,posref]=cuda_cuda('part_max',inref,maskref,projdir(p));
                    val.ref=valref;pos.ref=posref;
                else
                    val.ref=cuda_cuda('part_max',inref,maskref,projdir(p));
                end
                if p > 1
                    cuda_cuda('delete',inref);
                end
                inref=val.ref;
            end
        end
    else  % Compares arr with a max number or array
        in2=mask;
        val=cuda();
        if prod(size(in1)) > 1 && ~isa(in1,'cuda')
            in1=cuda(in1);
        end
        if prod(size(in2)) > 1 && ~isa(in2,'cuda')
            in2=cuda(in2);
        end
        if isa(in1,'cuda') && prod(size(in2)) == 1
            if isa(in2,'cuda')
                in2=double_force(in2);
            end
            val.ref=cuda_cuda('max_alpha',in1.ref,double(in2));
            val.fromDip = in1.fromDip;   % If eiter was dipimage, result will be
        elseif prod(size(in1)) == 1 && isa(in2,'cuda')
            if isa(in1,'cuda')
                in1=double_force(in1);
            end
            val.ref=cuda_cuda('max_alpha',in2.ref,double(in1));
            val.fromDip = in2.fromDip;   % If eiter was dipimage, result will be
        elseif isa(in1,'cuda') && isa(in2,'cuda')
            if (~in1.fromDip && any(size(in1) - size(in2)))
                error('cuda:max of Matlab array type: Matrix dimensions must agree.')
            end
            
            didSwap1=0;didSwap2=0;
            if in1.fromDip == 1 && ndims(in1) == 1 && ndims(in2) > 1
                cuda_cuda('swapSizeForceDim2',in1.ref);didSwap1=1;
            end
            if in2.fromDip == 1 && ndims(in2) == 1 && ndims(in1) > 1
                cuda_cuda('swapSizeForceDim2',in2.ref);didSwap2=1;
            end
            
            val.ref=cuda_cuda('max_arr',in1.ref,in2.ref);

            if didSwap1
                cuda_cuda('swapSizeForceDim1',in1.ref);
            end
            if didSwap2
                cuda_cuda('swapSizeForceDim1',in2.ref);
            end
            val.fromDip = in1.fromDip ;   % If eiter was dipimage, result will be
        end
    end
end
