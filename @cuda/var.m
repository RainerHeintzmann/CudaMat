% var(in) : computes the variance of a cuda array

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

function val = var(in1,mask,projdir)
if ~isa(in1,'cuda')
    in1=cuda(in1);
end
if nargin >1 && ~isempty(mask) && ~isa(mask,'cuda')
    mask=cuda(mask);
end
if nargin >2 && isa(projdir,'cuda') 
    projdir=double_force(projdir);
end

if nargin ==3
    mymean = mean(in1,mask,projdir);
    mymean2 = mean(in1.*conj(in1),mask,projdir);
elseif nargin ==2
    mymean = mean(in1,mask);
    mymean2 = mean(in1.*conj(in1),mask);
else
    mymean = mean(in1);
    mymean2 = mean(in1.*conj(in1));
end
val = mymean2-mymean*mymean;
