% roots(in) : calculates the roots of a polynomial (currently not implemented in cuda).

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
function out=roots(in)
global use_ones_cuda; global use_zeros_cuda;
tmp1=use_ones_cuda; tmp2=use_zeros_cuda;  % Just ot avoid confusions with the zeros function used in the matlab root function
use_ones_cuda=0;use_zeros_cuda=0;
out=cuda(roots(double_force(in)));  % This is cheating but the only possiblility right now
use_ones_cuda=tmp1;
use_zeros_cuda=tmp2;
