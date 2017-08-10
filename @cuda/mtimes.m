% mtimes(in1,in2) : pointwise or array multiplication or matrix multiplication in dependence of datatype

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
function out=mtimes(in1,in2)
if isa(in1,'cuda') 
    if in1.fromDip
        out=times(in1,in2);
        return;
    end
end
if isa(in2,'cuda') 
    if in2.fromDip
        out=times(in1,in2);
        return
    end
end
if (~isnumeric(in1) && prod(size(in1))==1)
    in1=single_force(in1);   % convert into numbers
end
if (~isnumeric(in2) && prod(size(in2))==1)
    in2=single_force(in2);   % convert into numbers
end
if (isnumeric(in1) && length(in1)==1) || (isnumeric(in2)&& length(in2)==1)
    out=times(in1,in2);
else  % this needs to be a proper matrix product
    if ~isa(in1,'cuda')
        in1=cuda(in1);
    end
    if ~isa(in2,'cuda')
        in2=cuda(in2);
    end
    myref1=in1.ref;
    myref2=in2.ref;
    if cuda_cuda('isCpx',in2.ref) && ~cuda_cuda('isCpx',in1.ref)
        myref1=cuda_cuda('complex',in1.ref);
    end
    if cuda_cuda('isCpx',in1.ref) && ~cuda_cuda('isCpx',in2.ref)
        myref2=cuda_cuda('complex',in2.ref);
    end
    
    if (size(in1,1) == 1) && (size(in2,2) == 1)  % This is a scalar product
        % out=cuda_cuda('sprod',myref1,myref2,0);  % without complex conjugation
        % the above line is super slow
        out=dot(in1,in2);
    elseif size(in2,2) ==1  % matrix times vector
        ref=cuda_cuda('mvtimes',myref1,myref2);
        out=cuda(ref,in1.fromDip);
    else  % Two proper matrices
        ref=cuda_cuda('mtimes',myref1,myref2);
        out=cuda(ref,in1.fromDip);
    end
    if (myref1 ~= in1.ref)
        cuda_cuda('delete',myref1);
    end
    if (myref2 ~= in2.ref)
        cuda_cuda('delete',myref2);
    end
    %if prod(size(out)) == 1
    %    out=single_force(out);   % convert into a scalar if necessary
    %end
end
end
