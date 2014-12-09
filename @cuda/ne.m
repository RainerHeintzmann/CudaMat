% ne(in1, in2) : not equal

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
function out=ne(in1,in2)
out=cuda();
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
    out.ref=cuda_cuda('unequals_alpha',in1.ref,double(in2));
    out.fromDip = in1.fromDip;  
elseif prod(size(in1)) == 1 && isa(in2,'cuda')
    if isa(in1,'cuda')
        in1=double_force(in1);
    end
    out.ref=cuda_cuda('unequals_alpha',in2.ref,double(in1));
    out.fromDip = in2.fromDip;  
elseif isa(in1,'cuda') && isa(in2,'cuda')
    out.ref=cuda_cuda('unequals',in2.ref,in1.ref);
    out.fromDip = (in1.fromDip || in2.fromDip);   % If eiter was dipimage, result will be
end
out.isBinary = 1; % mark this as a binary result (needed for subsasgn)
