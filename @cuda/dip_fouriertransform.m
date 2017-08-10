% dip_fouriertransform(in,direction,transformdir) : detailed Fouriertransformation routine. See dipimage/dip_fouriertransform for more details
% in : array to transform
% direction : either 'forward' or 'inverse'
% transformdir : boolean vector of directions in which to transform

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
function out=dip_fouriertransform(in,direction,transformdir)
if nargin<2 || isempty(direction)
    direction='forward';
end

switch direction
    case 'forward',
        mode =2;  % transform with correct scaling
    case 'inverse',
        mode =-2;  % transform with correct scaling
    otherwise,
        error('cuda: dip_fouriertransform. Unknown transform direction. Use forward or inverse.')
end

if in.fromDip ~= 1
    in=dip_image(in);
end

if sum(transformdir>0) > 3 || (length(transformdir)>3 && (sum(transformdir(4:end)) ~= 0))
    error('cuda dip_fouriertransform only supported up to 3 dimensions');
end

if length(transformdir)>3 && (transformdir(4) == 0)
    myres=cell(1,size(in,4));
    for e=0:size(in,4)-1
        mysize=size(in);
        mysize(4)=[];
        myin = reshape(SubSlice(in,4,e),mysize);
        myres{e+1}=dip_fouriertransform(myin,direction,transformdir(1:3));    
    end
    out=cat(4,myres{:});
    clear myres;
    return
end

if length(transformdir<3)
    transformdir = [transformdir zeros(1,3-length(transformdir))];  % append zeros
end
transformdir=(transformdir > 0);

tmp=transformdir(2);transformdir(2)=transformdir(1);transformdir(1)=tmp;  % To deal with the fact that CudaMat has the sizes 1 and 2 the other way than DipImage.

if (1)  % new code
    out=cuda();
    if isa(in,'cuda')
        out.ref=cuda_cuda('fft3d',in.ref,mode,double(transformdir));  % double cast is very important here. Otherwise datatype does not match
    else
        error('fft: Unsupported datatype');
    end
    out.fromDip=in.fromDip;
    
    return;
end


mz=cuda_cuda('getSize',in.ref); % size(in);
if length(mz<3)
    mz = [mz builtin('ones',1, 3-length(mz))];  % make size 3d
end

% tmp=mz(2);mz(2)=mz(1);mz(1)=tmp;  % To deal with the fact that CudaMat has the sizes 1 and 2 the other way than DipImage.

if strcmp(direction,'forward')
    fwd=1;
elseif strcmp(direction,'inverse')
    fwd=0;
else
    error('cuda dip_fouriertransform : unknown direction of transform');
end

%oldsize=cuda_cuda('getSize',in.ref);
%cuda_cuda('setSize',in.ref,mz);  % This actually switches sizes as the DipImage answer was flipped
newsize=size(in);  % to be used for new image
% tmp=cuda();
%tmp.fromDip=1;
if all(transformdir == [0 0 0])
    out=in; % no transformation
elseif all(transformdir == [1 1 1])
    if fwd
        out=ft(in); % transfomr all coordinates
    else
        out=ift(in); % transfomr all coordinates
    end
elseif all(transformdir == [1 1 0])
    out=newim_cuda2(cuda(0),newsize,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),mz(3));
    for p=0:mz(3)-1
        moffs=[0 0 p];
        msize=[mz(1) mz(2) 1];
        mstep=[1 1 1];
        cuda_cuda('delete',tmp.ref);
        tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
        if fwd
            tmp2=ft(tmp);
        else
            tmp2=ift(tmp);
        end
        out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1,mstep);  % is an array to assing, next delete ignored
    end
elseif all(transformdir == [1 0 1])
    out=newim_cuda2(cuda(0),newsize,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),mz(3));
    for p=0:mz(2)-1
        moffs=[0 p 0];
        msize=[mz(1) 1 mz(3)];
        mstep=[1 1 1];
        cuda_cuda('delete',tmp.ref);
        tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
        if fwd
            tmp2=ft(tmp);
        else
            tmp2=ift(tmp);
        end
        out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1,mstep);  % is an array to assing, next delete ignored
    end
elseif all(transformdir == [0 1 1])
    out=newim_cuda2(cuda(0),newsize,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(2),mz(3));
    for p=0:mz(1)-1
        moffs=[p 0 0];
        msize=[1 mz(2) mz(3)];
        mstep=[1 1 1];
        cuda_cuda('delete',tmp.ref);
        tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
        if fwd
            tmp2=ft(tmp);
        else
            tmp2=ift(tmp);
        end
        out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1,mstep);  % is an array to assing, next delete ignored
    end
elseif all(transformdir == [1 0 0])
    out=newim_cuda2(cuda(0),newsize,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),1); 
    for q=0:mz(3)-1
        for p=0:mz(2)-1  % because direction 1 was switched with 2
            moffs=[0 p q];
            msize=[mz(1) 1 1];
            mstep=[1 1 1];
            cuda_cuda('delete',tmp.ref);
            tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
            if fwd
                tmp2=ft(tmp);
            else
                tmp2=ift(tmp);
            end
            out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1,mstep);  % is an array to assing, next delete ignored
        end
    end
elseif all(transformdir == [0 1 0])
    out=newim_cuda2(cuda(0),newsize,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),1); 
    for q=0:mz(3)-1
        for p=0:mz(1)-1  % because direction 1 was switched with 2
            moffs=[p 0 q];
            msize=[1 mz(2) 1];
            mstep=[1 1 1];
            cuda_cuda('delete',tmp.ref);
            tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
            if fwd
                tmp2=ft(tmp);
            else
                tmp2=ift(tmp);
            end
            out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1,mstep);  % is an array to assing, next delete ignored
        end
    end
elseif all(transformdir == [0 0 1])
    out=newim_cuda2(cuda(0),newsize,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),1); 
    for q=0:mz(2)-1
        for p=0:mz(1)-1  % because direction 1 was switched with 2
            moffs=[p q 0];
            msize=[1 1 mz(3)];
            mstep=[1 1 1];
            cuda_cuda('delete',tmp.ref);
            tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize,mstep);
            if fwd
                tmp2=ft(tmp);
            else
                tmp2=ift(tmp);
            end
            out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1,mstep);  % is an array to assing, next delete ignored
        end
    end
else
    error('yet unsupported single directional transform');
end

%cuda_cuda('setSize',in.ref,oldsize);   % back to original size
%cuda_cuda('setSize',out.ref,oldsize);   % back to original size


    
