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

obj=a;
myim=mcconv;  % noise free
%myVec=double(reshape(obj,prod(size(obj))));
myVec=double(repmat(mean(myim),[1 prod(size(obj))]));
[err,grad]=MyIdivErrorAndDeriv(myVec);
mygrad=grad*0;
UnitD=mygrad;

opteps = max(myim)/max(grad)
eps = 50;
MINTEST=200;
MAXTEST=size(myVec,2);
MAXTEST=MINTEST+40;
for d=MINTEST:MAXTEST % 
    UnitD = myVec*0;
    UnitD(d) = 1;
    mygrad(d) = (MyIdivErrorAndDeriv(myVec+eps * UnitD) - err) / eps;
end
g=reshape(dip_image(grad','single'),size(myim));
myg=reshape(dip_image(mygrad','single'),size(myim));
r = (myg(MINTEST:MAXTEST) - g(MINTEST:MAXTEST)) ./ max(abs(g(MINTEST:MAXTEST)));

relerror = (mygrad(MINTEST:MAXTEST) - (grad(MINTEST:MAXTEST))') ./ max(abs((grad(MINTEST:MAXTEST))'));
fprintf('Max Error :%g\n',max(abs(relerror)))
