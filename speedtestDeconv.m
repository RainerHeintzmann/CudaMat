% mytime=speedtestDeconv(useCuda) : performs a simple 3D deconvolution of the cromo3d dataset from DipImage
% DipImage and Mark Schmid's minFunc (http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) need to be installed.
% In the file polyinterp.m line 103 needs to be cahnged to:
% for qq = 1:length(cp); xCP = cp(qq);
% If everything is prepared run:
% speedtestDeconv(0)
% e.g. result could be 29.4 sec and
% speedtestDeconv(1)
% e.g. result could be 3.11 sec

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

function mytime=speedtestDeconv(useCuda)
set_ones_cuda(0);
set_zeros_cuda(0);

a=readim('chromo3d');
%h=kSimPSF({'sX',size(a,1);'sY',size(a,2);'sZ',size(a,3);'scaleX',20;'scaleY',20;'scaleZ',100;'confocal',1});
h=readim('psf.ics');

mcconv=convolve(a,h);
%mcconv=norm3d*real(ift(ft(a) .* ft(h)));
%img=noise(100*mcconv/max(mcconv),'poisson');  % put some extra noise on the image
img=100*mcconv/max(mcconv);  % no extra noise

global myim;
global otfrep;
global lambdaPenalty;
global Factor

% unix('touch cudaArith.cu; touch cuda_cuda.c; make'); clear classes  % compile command, just to copy and paste it

Scale=1e-3;

myim=img*Scale;
otfrep=ft(h);
lambdaPenalty=0.0;
Factor=1.0;


NumIter=20;
%options=struct('DerivativeCheck','off','Method','cg','Display','on','notify',1,'TolX',1e-29,'TolFun',10^-29,'MaxIter',NumIter); 

%options=struct('DerivativeCheck','off','Method','lbfgs','Display','on','notify',1,'TolX',1e-29,'TolFun',10^-29,'MaxIter',NumIter); 
options=struct('DerivativeCheck','off','Method','newton0','Display','on','notify',1,'TolX',1e-29,'TolFun',10^-29,'MaxIter',NumIter); 
%options=struct('DerivativeCheck','off','Display','on','notify',1,'TolX',1e-29,'TolFun',10^-29,'MaxIter',NumIter); 
%startVec=double(mcconv);  % Starting value for iteration
%startVec=img;  % Starting value for iteration

startVec=(double(repmat(mean(myim),[1 prod(size(img))])))'; 

if useCuda
    startVec=cuda(startVec);myim=cuda(myim);otfrep=cuda(otfrep);  % convert the important images to cuda type
    set_ones_cuda(useCuda); set_zeros_cuda(useCuda);   % make sure that the zeros and ones function from cuda is used.
end

tic
Factor = 1;
lambdaPenalty=0; % 1e6;
[err,grad]=MyIdivErrorAndDeriv(startVec);
%Factor = max(grad)/max(myim)
%Factor = max(myim)/max(abs(grad))
Factor = mean(myim)/mean(abs(grad))/20000;
lambdaPenalty=Factor*7e-4; %0.2; % 1e6;
%lambdaPenalty=0; % 1e6;
%Factor = mean(abs(grad))/mean(myim)
[myRes,msevalue,moreinfo]=minFunc(@MyIdivErrorAndDeriv,startVec,options); % @ means: 'Function handle creation' 
set_ones_cuda(0);
set_zeros_cuda(0);
mytime=toc
aRecon=reshape(dip_image(myRes','single'),size(myim));
cat(1,myim,aRecon)

% startVec=(double(reshape(img,prod(size(a)))))'; 
% startVec=a; 
% myim=a;
% [myRes,msevalue,moreinfo]=minFunc(@MyIdivErrorAndDeriv,startVec,options); % @ means: 'Function handle creation' 
% aRecon=reshape(dip_image(myRes','single'),size(myim));

