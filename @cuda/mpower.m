% mpower(in1, in2) : depending on whether the object is a dip_image cuda array or not, either the power function is executed or the matrix is risen to the in2 power

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
function out = mpower(in1,in2)
out=cuda();
if prod(size(in1)) > 1 && ~isa(in1,'cuda')
    in1=cuda(in1);
end
if prod(size(in2)) > 1 && ~isa(in2,'cuda')
    in2=cuda(in2);
end

if in1.fromDip || in2.fromDip
    out=power(in1,in2);
else
    error('Taken a matlab-type matrix to the nth power is not yet implemented.');
end

end
