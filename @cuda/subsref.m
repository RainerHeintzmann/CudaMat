% subsref(in, index) : referencing of subarrays 

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
function varargout=subsref(in,index)
switch index.type
    case '()'
        varargout{1}=cuda();
        if isa(index.subs{1},'cuda') && index.subs{1}.fromDip % referencing with a dip_image type mask image
            if ndims(in)~=ndims(index.subs{1})
                error('cuda subreferencing: Arrays have to have equal number of dimensions');
            elseif norm(size(in)-size(index.subs{1}))==0
                varargout{1}.ref=cuda_cuda('subsref_cuda',in.ref,index.subs{1}.ref);
            else
                error('cuda subreferencing: Array sizes have to be equal');
            end
        else
        oldsize=size(in);
            if length(index.subs) ~= 1 && length(index.subs) ~= length(oldsize)
                error('cuda subreferencing: Wrong number of dimensions');
            end
            if length(index.subs) == 1 && length(oldsize) > 1  % trying to access multidimensional array with one index
               cuda_cuda('setSize',in.ref,prod(oldsize));
            end
            isblock=1;
            for d=1:length(index.subs)
                if isempty(index.subs{1})
                    varargout{1} = [];
                    return;
                end
                if (size(index.subs{d}(1),2) == 1) && ischar(index.subs{d}) && index.subs{d}(1) ==':'    % User really want the whole range
                    if in.fromDip == 0
                        moffs(d)= 1;   % matlab style
                    else
                        moffs(d)= 0;   % DipImage style
                    end
                    msize(d)=size(in,d);
                else
                    moffs(d)=index.subs{d}(1);
                    msize(d)=index.subs{d}(end)-moffs(d)+1;
                    if length(index.subs{d}) ~= msize(d) || sum(abs(index.subs{d}-[moffs(d):moffs(d)+msize(d)-1])) ~= 0
                        isblock=0;
                    end
                end
            end
            if in.fromDip && length(moffs) > 1
                tmp=moffs(1);moffs(1)=moffs(2);moffs(2)=tmp;
                tmp=msize(1);msize(1)=msize(2);msize(2)=tmp;
            end
            if in.fromDip == 0
                moffs=moffs-1;
            end
            if isblock
                varargout{1}.ref=cuda_cuda('subsref_block',in.ref,moffs,msize);
            else
                error('cuda subreferencing: subreferencing with non-block vectors not yet implemented');
                varargout{1}.ref=cuda_cuda('subsref_vecs',in.ref,index.subs);
            end
            if length(index.subs) == 1 && length(oldsize) > 1 % restore the old size (if changed)
                    cuda_cuda('setSize',in.ref,oldsize);
                    cuda_cuda('setSize',varargout{1}.ref,[1 msize]);
            end

        end
        varargout{1}.fromDip = in.fromDip;
        if prod(size(varargout{1})) == 1
            varargout{1} = double_force(varargout{1});  % make sure this is just a matlab variable now
        end
    case '.'
%        fprintf('subsref .\n');
        varargout{1}=in.(index.subs);
    case '{}'
        if index.subs{1}==1   % only this access is legal to a cuda object
            varargout{1}=in;
        else
            fprintf('subsref {}\n');
            error('subsref {} called for cuda object with an index unequal to one. This object is no cell array.');
        end
end
