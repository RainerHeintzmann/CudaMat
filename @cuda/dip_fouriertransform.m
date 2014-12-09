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
if nargin<2
    direction='forward';
end

if in.fromDip ~= 1
    in=dip_image(in);
end
if length(transformdir)>3
    error('cuda dip_fouriertransform only supported up to 3 dimensions');
end

if length(transformdir<3)
    transformdir = [transformdir zeros(1,3-length(transformdir))];  % append zeros
end
transformdir=(transformdir > 0);
mz=size(in);
if length(mz<3)
    mz = [mz builtin('ones',1, 3-length(mz))];  % make size 3d
end

if strcmp(direction,'forward')
    fwd=1;
elseif strcmp(direction,'inverse')
    fwd=0;
else
    error('cuda dip_fouriertransform : unknown direction of transform');
end

oldsize=cuda_cuda('getSize',in.ref);
cuda_cuda('setSize',in.ref,[mz(2) mz(1) mz(3)]);

tmp=cuda();
tmp.fromDip=1;
if all(transformdir == [0 0 0])
    out=in; % no transformation
elseif all(transformdir == [1 1 1])
    if fwd
        out=ft(in); % transfomr all coordinates
    else
        out=ift(in); % transfomr all coordinates
    end
elseif all(transformdir == [1 1 0])
    out=newim_cuda2(cuda(0),mz,'scomplex');
    for p=0:mz(3)-1
        moffs=[0 0 p];
        msize=[mz(1) mz(2) 1];
        tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize);
        if fwd
            tmp2=ft(tmp);
        else
            tmp2=ift(tmp);
        end
        out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1);  % is an array to assing, next delete ignored
        cuda_cuda('delete',tmp.ref);
    end
elseif all(transformdir == [1 0 1])
    out=newim_cuda2(cuda(0),mz,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),mz(3));
    for p=0:mz(2)-1
        moffs=[0 p 0];
        msize=[mz(1) 1 mz(3)];
        tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize);
        if fwd
            tmp2=ft(tmp);
        else
            tmp2=ift(tmp);
        end
        out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1);  % is an array to assing, next delete ignored
        cuda_cuda('delete',tmp.ref);
    end
elseif all(transformdir == [0 1 1])
    out=newim_cuda2(cuda(0),mz,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(2),mz(3));
    for p=0:mz(1)-1
        moffs=[0 p q];
        msize=[1 mz(2) mz(3)];
        tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize);
        if fwd
            tmp2=ft(tmp);
        else
            tmp2=ift(tmp);
        end
        out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1);  % is an array to assing, next delete ignored
        cuda_cuda('delete',tmp.ref);
    end
elseif all(transformdir == [1 0 0])
    out=newim_cuda2(cuda(0),mz,'scomplex');
    tmp=newim_cuda2(cuda(0),mz(1),1); 
    for q=0:mz(3)-1
        for p=0:mz(2)-1  % because direction 1 was switched with 2
            moffs=[p 0 q];
            msize=[1 mz(1) 1];
            tmp.ref=cuda_cuda('subsref_block',in.ref,moffs,msize);
            if fwd
                tmp2=ft(tmp);
            else
                tmp2=ift(tmp);
            end
            out.ref=cuda_cuda('subsasgn_block',tmp2.ref,out.ref,moffs,msize,-1);  % is an array to assing, next delete ignored
            cuda_cuda('delete',tmp.ref);
        end
    end
else
    error('yet unsupported single directional transform');
end

cuda_cuda('setSize',in.ref,oldsize);   % back to original size
cuda_cuda('setSize',out.ref,oldsize);   % back to original size


    
