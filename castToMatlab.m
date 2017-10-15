% castToMatlab(in): conversion from cuda to double or dip_image depending on the type
%
% see also: double_force
%

%***************************************************************************
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
function out = castToMatlab(in)
if isa(in,'cuda')
    % out = double(cuda_cuda('get',in.ref));
    if in.fromDip
        out = dip_image(cuda_cuda('get',getReference(in)));
    else
        out = double(cuda_cuda('get',getReference(in)));
    end
else
    out=double(in);
end


