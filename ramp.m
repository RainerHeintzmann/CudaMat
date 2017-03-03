% ramp(size,dimension,varargin) :  mimics the dipimage ramp function. See its documentation.
% first argument can be a size vector (or [256 256] is assumed),
% dimension is the direction into which to generate the ramp,
% and third argument one of
% 'left','right','true','corner','freq','radfreq','math',
% 'mleft','mright','mtrue','mcorner','mfreq','mradfreq'
%
% the global variable use_ramp_cuda decides whether to generate a dip_image or a cuda dip_image.

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
function res=ramp(varargin)
global use_ramp_cuda;
global diphandle_ramp;
if (use_ramp_cuda)
    tmp=cuda(1.0);
    res= ramp_cuda2(tmp,varargin{:});
else
    res=feval(diphandle_ramp,varargin{:});  % call the standart matlab zeros function
end