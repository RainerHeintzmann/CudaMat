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
    if nargin ~= 2 % This should be changed later as it is incorrect: A check for the datatype being binary needs to be made but cuda does not remember to be binary
        if (nargin < 3)  % max over all pixels
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
            if ~isempty(mask) && ~isa(mask,'cuda')
                mask=cuda(mask);
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
                    [valref,posref]=cuda_cuda('part_min',inref,mask.ref,projdir(p));
                    val.ref=valref;pos.ref=posref;
                else
                    val.ref=cuda_cuda('part_min',inref,mask.ref,projdir(p));
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
            val.ref=cuda_cuda('min_alpha',in1.ref,double(in2));
            val.fromDip = in1.fromDip;   % If eiter was dipimage, result will be
        elseif prod(size(in1)) == 1 && isa(in2,'cuda')
            if isa(in1,'cuda')
                in1=double_force(in1);
            end
            val.ref=cuda_cuda('min_alpha',in2.ref,double(in1));
            val.fromDip = in2.fromDip;   % If eiter was dipimage, result will be
        elseif isa(in1,'cuda') && isa(in2,'cuda')
            if (~in1.fromDip && any(size(in1) - size(in2)))
                error('cuda:min of Matlab array type: Matrix dimensions must agree.')
            end
            val.ref=cuda_cuda('min_arr',in1.ref,in2.ref);
            val.fromDip = in1.fromDip ;   % If eiter was dipimage, result will be
        end
    end
end
