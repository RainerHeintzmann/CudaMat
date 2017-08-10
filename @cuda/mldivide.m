%mldivide(in1,in2) : solving the equation system A x = b.  Unfortunately at the moment NOT done on the card

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
function out=mldivide(in1,in2)
if (prod(size(in2))==1) || (isa(in1,'cuda') && in1.fromDip) || (isa(in2,'cuda') && in2.fromDip)  || (isa(in1,'double') && length(in1)==1) || (isa(in2,'double')&& length(in2)==1)
    out=ldivide(in1,in2);
else  % Equation solving
    % error('cuda mldivide: Equation solving not yet implemented');
    %out=cuda(double_force(in1) \ double_force(in2));  % This is horrible, but the only workaround at the moment
    ref=cuda_cuda('mldivide',in1.ref,in2.ref);  % A \ b solves the A x = b equation
    out=cuda(ref,(in1.fromDip || in2.fromDip));
    % out.fromDip = (in1.fromDip || in2.fromDip);   % If eiter was dipimage, result will be
end
