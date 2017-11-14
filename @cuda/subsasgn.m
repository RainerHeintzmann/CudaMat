% subasgn(in,index, val) : assignment of subarrays. 

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
function out=subsasgn(in,index,val)        % DANGER !!! This still does not do a copy of the object. a=b; b(13)=17; will change "a" as well!
if (isempty(val) && (isa(index.subs{1},'cuda') && ((~isempty(index.subs{1}.isBinary) && index.subs{1}.isBinary) || (ndims(index.subs{1})>1 && size(index.subs{1},1) > 1 && size(index.subs{1},2) > 1)))  || isempty(index.subs{1}))
    out=in;
   return;
end
if ~isa(in,'cuda')
    if isa(val,'cuda')
        out=subsasgn(in,index,single_force(val));
        return;
    else
        error('Why is this the cuda subsasgn function called for non-cuda variables?');
    end
end

valsize=size(val);
tvalsize=prod(valsize);
if tvalsize > 1
    if isa(val,'cuda')
        valref=val.ref;
    else
        valref=cuda_cuda('put',single(val));
    end
end

% To deal with a possibly existing copy of the variable to assign to, we
% will have to make an explicit copy now (if needed) by exploiting a Matlab
% hack: The first pointer in the structure points to a possible linked
% variable
linkedNum=cuda_cuda('getLinkedVarNum',in);
if linkedNum~=0
    % linkedNum
    if (1)  % if this is disabled there is a big danger of unwantedly changing variables, but a (small?) speed and memory advantage.
        in=copy(in);  % make a copy. Unfortunately this is done always, as the argument is on the function stack
    end
end

insize=size(in); % tinsize=prod(insize);
oldsize=insize;
sizechanged=0;
switch index.type
    case '()'
        out=cuda(); 
        % referencing using a binary image
        if isa(index.subs{1},'cuda') && ((~isempty(index.subs{1}.isBinary) && index.subs{1}.isBinary) || (ndims(index.subs{1})>1 && size(index.subs{1},1) > 1 && size(index.subs{1},2) > 1))  % && index.subs{1}.fromDip % referencing with a cuda image
            mybool=index.subs{1};
            mVecSize=tvalsize;
            if mVecSize ~= 1
                sumind=cuda_cuda('sumpos',index.subs{1}.ref);  % Bad. This should be avoided for speed reasons
                if mVecSize ~= sumind 
                    error('subsasgn_cuda_vec : mask size does not correspond to size of vector to assign');
                else
                    out.ref=cuda_cuda('subsasgn_cuda_vec',in.ref,index.subs{1}.ref,valref,1);
                end
                % error('Cuda subassigning of vectors to subarrays indexed with logical images not yet implemented');
            else
                out.ref=cuda_cuda('subsasgn_cuda_const',in.ref,mybool.ref,val,1);  % no delete irgnored
            end
            out.fromDip = in.fromDip;
            if tvalsize > 1
                if ~isa(val,'cuda')
                    cuda_cuda('forceDelete',valref);
                    valref=0;
                end
            end
            return
            % cuda_cuda('subsasgn_cuda',in.ref,index.subs{1});
        else
            isblock=0;
            if length(index.subs)>1 || ndims(in) == 1 || prod(size(index.subs{1})) == 1 || ischar(index.subs{1})  % the latter is needed as the index can be ':'
                isblock=1;

                if length(index.subs) ~= 1 && length(index.subs) ~= length(oldsize)
                    error('cuda subreferencing: Wrong number of dimensions');
                end
                if length(index.subs) == 1 && length(oldsize) > 1  % trying to access multidimensional array with one index
                    insize=prod(oldsize);
                    cuda_cuda('setSize',in.ref,insize);
                    sizechanged=1;
                end
                for d=1:length(index.subs)
                    if ischar(index.subs{d}) && index.subs{d}(1) ==':' && (size(index.subs{d}(1),2) == 1)     % User really want the whole range
                        if in.fromDip == 0
                            moffs(d)= 1;   % matlab style
                        else
                            moffs(d)= 0;   % DipImage style
                        end
                        mstep(d)=1;
                        if length(index.subs)==1
                            msize=insize;  % assign all the sizes here, as d will only iterate over 1 direction
                        else
                            msize(d)=insize(d);  % only for this direction (which has a ':')
                        end
                    else
                        if isempty(index.subs{d})
                            out=in;
                            return; % do nothing
                            % error('Trying to assign to an empty index')
                        end
                        if isa(index.subs{d},'cuda')  % due to some funny Matlab bug which prevents calling subsref within subsref for type cuda
                            moffs(d)=getVal(index.subs{d},0);
                            if (getVal(index.subs{d},-1) == moffs(d))
                                mstep(d)=1;
                            else
                                mstep(d)=(getVal(index.subs{d},-1)-moffs(d)) / (length(index.subs{d})-1);
                            end
                            msize(d)=floor(abs(getVal(index.subs{d},-1)-moffs(d))/abs(mstep(d)))+1;
                            % isblock=0;
                        else
                            moffs(d)=index.subs{d}(1);
                            if (index.subs{d}(end) == moffs(d))
                                mstep(d)=1;
                            else
                                mstep(d)=(index.subs{d}(end)-moffs(d)) / (length(index.subs{d})-1);
                            end
                            msize(d)=floor(abs(index.subs{d}(end)-moffs(d))/abs(mstep(d)))+1;
                        end
                        if length(index.subs{d}) ~= abs(msize(d)) || sum(abs(index.subs{d}-[moffs(d):mstep(d):moffs(d)+mstep(d)*(length(index.subs{d})-1)])) ~= 0
                        % if length(index.subs{d}) ~= length([moffs(d):mystep:moffs(d)+msize(d)-sign(mystep)]) || sum(abs(index.subs{d}-[moffs(d):mystep:moffs(d)+msize(d)-sign(mystep)])) ~= 0
                            isblock=0;
                        end
                    end
                end
                
                if in.fromDip && length(moffs) > 1
                    tmp=moffs(1);moffs(1)=moffs(2);moffs(2)=tmp;
                    tmp=msize(1);msize(1)=msize(2);msize(2)=tmp;
                    tmp=mstep(1);mstep(1)=mstep(2);mstep(2)=tmp;
                    tmp=valsize(1);valsize(1)=valsize(2);valsize(2)=tmp;
                    tmp=insize(1);insize(1)=insize(2);insize(2)=tmp;
                end
                if in.fromDip == 0
                    moffs=moffs-1;
                    if length(moffs) <= 1  % single argument assignment addressing
                        if (insize(1) > 1)
                            moffs=[moffs 0];
                            msize=[msize 1];
                        else
                            moffs=[0 moffs];
                            msize=[1 msize];
                        end
                        % squeeze the valsize as well, as it is allowed to assign a matrix using one-D adressing
                        valsize=[tvalsize 1];
                    end
                end
            end
            
            if isblock
                if any((moffs + msize.*mstep) > insize) || (isreal(in) && ~isreal(val))  % dataset needs expansion or change of datatype
                    ns=max((moffs + msize.*mstep), insize); % new size
                    if isreal(in) && isreal(val)
                        tmp=zeros_cuda2(cuda(0),ns);  % makes a new object
                    else
                        tmp=zeros_cuda2(cuda(0),ns,'scomplex');  % makes a new object
                    end
                    tmp.fromDip=in.fromDip;
                    % the line below will update the tmp array without deleting
                    cuda_cuda('subsasgn_block',in.ref,tmp.ref,ns*0,insize,-1,mstep);  % no delete ignored
                    in=tmp;
                end
                ddiff=length(msize)-length(valsize);
                if ddiff > 0
                    valsize=[valsize ones(1,ddiff)];
                elseif ddiff < 0
                    msize=[msize ones(1,-ddiff)];
                end
                if sum(abs(msize-valsize)) ~= 0 && prod(valsize) > 1
                    if (prod(msize) == prod(valsize))  % a(:) = 2Dimage;
                        tmp=msize(1);msize(1)=msize(2);msize(2)=tmp;
                    else
                        error('subsassgn: Sizes of source and destination region not matching');
                    end
                end
                if prod(valsize) == 1   % this is a single value to assign
                    out.ref=cuda_cuda('subsasgn_block',double(val),in.ref,moffs,msize,1,mstep);  % is a constant to assing, next delete ignored
                else
                    out.ref=cuda_cuda('subsasgn_block',valref,in.ref,moffs,msize,0,mstep);  % is an array to assign, next delete is ignored
                end
            else
                mybool=index.subs{1};
                mVecSize=tvalsize;
                if isa(index.subs{1},'cuda')
                    subsrefnum=index.subs{1}.ref;
                else
                    subsrefnum=cuda_cuda('put',single(index.subs{1}));
                end
                if ~in.fromDip
                    % index.subs{1}=index.subs{1}-1;  % adjust for matlab indexing starting with 1
                    subsrefnum2=cuda_cuda('plus_alpha',subsrefnum,double(-1.0));
                else
                    subsrefnum2=subsrefnum;
                end
                if mVecSize ~= 1
                    %if mVecSize ~= size(index.subs{1},2) && mVecSize ~= size(index.subs{1},1) % sum(index.subs{1} ~= 0)
                    %    error('subsref_cuda_vec : mask size does not correspond to size of vector to assign');
                    %else
                        % out.ref=cuda_cuda('subsasgn_cuda_vec',in.ref,index.subs{1}.ref,valref,1);
                        out.ref=cuda_cuda('subsasgn_1Didx',valref,subsrefnum2,in.ref); % vector of 1D indices
                    %end
                    % error('Cuda subassigning of vectors to subarrays indexed with logical images not yet implemented');
                else
                    % out.ref=cuda_cuda('subsasgn_cuda_const',in.ref,mybool.ref,val,1);  % no delete irgnored
                    
                    out.ref=cuda_cuda('subsasgn_1Didx_const',in.ref,subsrefnum2,val);% vector of 1D indices
                end
                if ~in.fromDip
                    cuda_cuda('forceDelete',subsrefnum2);
                    subsrefnum2=0;
                end
                if ~isa(index.subs{1},'cuda')
                    cuda_cuda('forceDelete',subsrefnum);
                    subsrefnum=0;
                end                
            end
        end
    case '.'
        fprintf('subsasgn .\n');
        out=in.(index.subs);
    case '{}'
        fprintf('subsasgn {}\n');
end
out.fromDip=in.fromDip;
if tvalsize > 1
    if ~isa(val,'cuda')
        cuda_cuda('forceDelete',valref);
        valref=0;
    end
end

if sizechanged % (ndims(insize)~=ndims(oldsize) || norm(insize-oldsize)~=0) && (length(index.subs) == 1 && length(oldsize) > 1) % restore the old size (if changed)
    insize=oldsize;
    if in.fromDip
        tmp=oldsize(1);oldsize(1)=oldsize(2);oldsize(2)=tmp;
    end
    cuda_cuda('setSize',in.ref,oldsize);
    % cuda_cuda('setSize',varargout{1}.ref,[1 msize]);
end

