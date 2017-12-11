% shift(in1,vec) : shifts an image by the amount given in vec using FFTs.

%************************** CudaMat ****************************************
%   Copyright (C) 2008-2017 by Rainer Heintzmann                          *
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
function out = shift(in1,myvec,killNq)
if nargin < 3
    killNq=1;
end
if prod(size(in1)) > 1 && ~isa(in1,'cuda')
    in1=cuda(in1);
end
% out=cuda();
myphase=1;
for d=1:length(myvec)
    mysize=ones(length(myvec),1);
    mysize(d)=size(in1,d);
    myramp=ramp(mysize,d,'radfreq');
    myphase = myphase .* exp(-1i.*myvec(d).*myramp);
end
in1=ft(in1) .* myphase;

if killNq 
    sx=size(in1,1);
    if floor(sx/2)*2==sx
        idx=cell(1,ndims(in1));
        idx{1}=0;
        for d=2:ndims(in1)
            idx{d}=':';
        end
        myidx=struct('type','()','subs',{idx});
        in1=subsasgn(in1,myidx,0);
    end
end
out=real(ift(in1));
end
