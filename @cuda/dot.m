% dot(in1,in2) : calculates the scalar product (dot product)


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

function out=dot(in1,in2)
s1=size(in1);
s2=size(in2);

if ~isa(in1,'cuda')
    in1=cuda(in1);
end
if ~isa(in2,'cuda') 
    in2=cuda(in2);
end

if any(s1 ~= s2)  % sizes need to be equal
    tmp=s2(2);s2(2)=s2(1);s2(1)=tmp;
    if ~in1.fromDip && any(s1 ~= s2)  % sizes need to be equal
        error('cuda dot: Sizes need to be equal');
    end
end

myref1=in1.ref;
myref2=in2.ref;
if cuda_cuda('isCpx',in2.ref) && ~cuda_cuda('isCpx',in1.ref)
    myref1=cuda_cuda('complex',in1.ref);
end
if cuda_cuda('isCpx',in1.ref) && ~cuda_cuda('isCpx',in2.ref)
    myref2=cuda_cuda('complex',in2.ref);
end

%fprintf('Dot Product of size %d\n',size(in1,2));

if in1.fromDip
     out=cuda_cuda('sprod',in1.ref,in2.ref,1);   % somehow slow!
    %ref=cuda_cuda('times',myref1,myref2);
    %out=cuda_cuda('sum',ref);
    %    cuda_cuda('delete',ref);
else
    if (prod(size(in1)) == max(size(in1)))
        out=cuda_cuda('sprod',myref1,myref2,1);   % with complex conjugation
    %ref=cuda_cuda('times',myref1,myref2);
    %out=cuda_cuda('sum',ref);
    %    cuda_cuda('delete',ref);

    else
        error('cuda dot: scalar product of matrices not implemented yet.');
    end
end

if (myref1 ~= in1.ref)
    cuda_cuda('delete',myref1);
end
if (myref2 ~= in2.ref)
    cuda_cuda('delete',myref2);
end
