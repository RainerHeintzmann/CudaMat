% end(a,k,n) :   Overloaded function for indexing expressions.
%   END(A,K,N) is called for indexing expressions involving the object
%   A when END is part of the K-th index out of N indices.


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

function ii = end(a,k,n)
    s=size(a);
    if (n ~= length(size(a)))
        error('cuda end: Indexing expressing not matching with dimensionality of array')
    end
    if (k > length(size(a)))
        error('cuda end: Dimension to determine end of not inside dimensionality of array')
    end
if a.fromDip
    ii=s(k)-1;   % 0-based indexing
else
    ii=s(k);    % 1-based indexing
end

