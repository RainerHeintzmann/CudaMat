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
if ~isa(in,'cuda')
    if isa(val,'cuda')
        out=subsasgn(in,index,single_force(val));
        return;
    else
        error('Why is this the cuda subsasgn function called for non-cuda variables?');
    end
end
if ~isa(val,'cuda') && numel(val) > 1
    val=cuda(val);
end

valsize=size(val);tvalsize=prod(valsize);
insize=size(in);tinsize=prod(insize);
switch index.type
    case '()'
        out=cuda();
        if isa(index.subs{1},'cuda') % referencing with a cuda image
            mybool=index.subs{1};
            mVecSize=tvalsize;
            if mVecSize ~= 1
                if mVecSize ~= sum(index.subs{1} ~= 0)
                    error('subsref_cuda_vec : mask size does not correspond to size of vector to assign');
                else
                    out.ref=cuda_cuda('subsasgn_cuda_vec',in.ref,index.subs{1}.ref,val.ref,1);
                end
                % error('Cuda subassigning of vectors to subarrays indexed with logical images not yet implemented');
            else
                out.ref=cuda_cuda('subsasgn_cuda_const',in.ref,mybool.ref,val,1);  % no delete irgnored
            end

            % cuda_cuda('subsasgn_cuda',in.ref,index.subs{1});
        else
            isblock=1;
            for d=1:length(index.subs)
                if ischar(index.subs{d}) && index.subs{d}(1) ==':' && (size(index.subs{d}(1),2) == 1)     % User really want the whole range
                    if in.fromDip == 0
                        moffs(d)= 1;   % matlab style
                    else
                        moffs(d)= 0;   % DipImage style
                    end
                    msize(d)=insize(d);
                else
                    if isempty(index.subs{d})
                        error('Trying to assing to an empty index')
                    end
                    moffs(d)=index.subs{d}(1);
                    msize(d)=index.subs{d}(end)-moffs(d)+1;
                    if sum(abs(index.subs{d}-[moffs(d):moffs(d)+msize(d)-1])) ~= 0
                        isblock=0;
                end
                end
            end
            
            if in.fromDip && length(moffs) > 1
                tmp=moffs(1);moffs(1)=moffs(2);moffs(2)=tmp;
                tmp=msize(1);msize(1)=msize(2);msize(2)=tmp;
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
             if any((moffs + msize) > insize) || (isreal(in) && ~isreal(val))  % dataset needs expansion or change of datatype
                 ns=max((moffs + msize), insize); % new size
                 if isreal(in) && isreal(val)
                     tmp=zeros_cuda2(cuda(0),ns);  % makes a new object
                 else
                     tmp=zeros_cuda2(cuda(0),ns,'scomplex');  % makes a new object
                 end
                 tmp.fromDip=in.fromDip;
                 % the line below will update the tmp array without deleting
                 cuda_cuda('subsasgn_block',in.ref,tmp.ref,ns*0,insize,-1);  % no delete ignored
                 in=tmp;
             end
            if isblock
                ddiff=length(msize)-length(valsize);
                if ddiff > 0
                    valsize=[valsize ones(1,ddiff)];
                elseif ddiff < 0
                    msize=[msize ones(1,-ddiff)];
                end
                if sum(abs(msize-valsize)) ~= 0 && prod(valsize) > 1
                   error('subsassgn: Sizes of source and destination region not matching');
                end
                if prod(valsize) == 1   % this is a single value to assign
                    out.ref=cuda_cuda('subsasgn_block',double(val),in.ref,moffs,msize,1);  % is a constant to assing, next delete ignored
                else
                    out.ref=cuda_cuda('subsasgn_block',val.ref,in.ref,moffs,msize,0);  % is an array to assing, next delete ignored
                end
            else
                out.ref=cuda_cuda('subsasgn_vecs',in.ref,index.subs);
            end
        end
    case '.'
        fprintf('subsasgn .\n');
        out=in.(index.subs);
    case '{}'
        fprintf('subsasgn {}\n');
end
out.fromDip=in.fromDip;

