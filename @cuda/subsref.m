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
        if isa(index.subs{1},'cuda') && ((~isempty(index.subs{1}.isBinary) && index.subs{1}.isBinary) || (ndims(index.subs{1})>1 && size(index.subs{1},1) > 1 && size(index.subs{1},2) > 1))  % && index.subs{1}.fromDip % referencing with a cuda image 
            if ndims(in)~=ndims(index.subs{1})
                error('cuda subreferencing: Arrays have to have equal number of dimensions');
            elseif norm(size(in)-size(index.subs{1}))==0
                varargout{1}.ref=cuda_cuda('subsref_cuda',in.ref,index.subs{1}.ref);
               	if varargout{1}.ref < 0  % empty sub-reference. Did not allocate.
                    varargout{1}=[];
                    return;
                end
            else
                error('cuda subreferencing: Array sizes have to be equal');
            end
            varargout{1}.fromDip = in.fromDip;
            if prod(size(varargout{1})) == 1
                varargout{1} = double_force(varargout{1});  % make sure this is just a matlab variable now
            end
            return
        else
            % isblock=0;
            % if length(index.subs)>1 || ndims(in) == 1 || prod(size(index.subs{1})) == 1 || ischar(index.subs{1})  % the latter is needed as the index can be ':'
            isblock=1;
            oldsize=size(in);
            if length(index.subs) ~= 1 && length(index.subs) ~= length(oldsize)
                error('cuda subreferencing: Wrong number of dimensions');
            end
            if length(index.subs) == 1 && length(oldsize) > 1  % trying to access multidimensional array with one index
               cuda_cuda('setSize',in.ref,prod(oldsize));
            end
                for d=1:length(index.subs)
                    if isempty(index.subs{1})
                        varargout{1} = [];
                        return;
                    end
                    if (size(index.subs{d}(1),2) == 1) && ischar(index.subs{d}) && index.subs{d}(1) ==':'    % User really wants the whole range along this dimension d
                        if in.fromDip == 0
                            moffs(d)= 1;   % matlab style
                        else
                            moffs(d)= 0;   % DipImage style
                        end
                        mstep(d)=1;
                        if length(index.subs)==1
                            msize=size(in);  % assign all the sizes here, as d will only iterate over 1 direction
                        else
                            msize(d)=size(in,d);  % only for this direction (which has a ':')
                        end
                        % msize(d)=size(in,d);
                    else
                        if isa(index.subs{d},'cuda')  % due to some funny Matlab bug which prevents calling subsref within subsref for type cuda
                            moffs(d)=getVal(index.subs{d},0);
                            % isblock=0;
                            if (getVal(index.subs{d},-1) == moffs(d))
                                mstep(d)=1;
                            else
                                mstep(d)=(getVal(index.subs{d},-1)-moffs(d)) / (length(index.subs{d})-1);
                            end
                            msize(d)=floor(abs(getVal(index.subs{d},-1)-moffs(d))/abs(mstep(d)))+1;
                        else
                            moffs(d)=index.subs{d}(1);
                            msize(d)=abs(index.subs{d}(end)-moffs(d))+1;
                            if (index.subs{d}(end) == moffs(d))
                                mstep(d)=1;
                            else
                                mstep(d)=(index.subs{d}(end)-moffs(d)) / (length(index.subs{d})-1);
                            end
                            msize(d)=floor(abs(index.subs{d}(end)-moffs(d))/abs(mstep(d)))+1;
                        end
                        if length(index.subs{d}) ~= abs(msize(d)) || sum(abs(index.subs{d}-[moffs(d):mstep(d):moffs(d)+mstep(d)*(length(index.subs{d})-1)])) ~= 0
                            isblock=0;
                        end
                    end
                end
                if in.fromDip && length(moffs) > 1
                    tmp=moffs(1);moffs(1)=moffs(2);moffs(2)=tmp;
                    tmp=msize(1);msize(1)=msize(2);msize(2)=tmp;
                    tmp=mstep(1);mstep(1)=mstep(2);mstep(2)=tmp;
                end
                if in.fromDip == 0
                    moffs=moffs-1;
                end
            end
            if isblock
                varargout{1}.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
            else
                if length(index.subs) > 1
                	error('cuda subreferencing: subreferencing with multidimensional non-block vectors not yet implemented');
                end
                if in.fromDip == 0
                    index.subs{1}=cuda(index.subs{1})-1;
                else
                    index.subs{1}=cuda(index.subs{1});
                end

                varargout{1}.ref=cuda_cuda('subsref_1Didx',in.ref,index.subs{1}.ref);
                msize=size(index.subs{1},2);
                varargout{1}.fromDip = in.fromDip;
            end
            if length(index.subs) == 1 && length(oldsize) > 1 % restore the old size (if changed)
                    if in.fromDip
                        tmp=oldsize(1);oldsize(1)=oldsize(2);oldsize(2)=tmp;
                    end
                    cuda_cuda('setSize',in.ref,oldsize);
                    % cuda_cuda('setSize',varargout{1}.ref,[1 msize]);
            end

        % end
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
