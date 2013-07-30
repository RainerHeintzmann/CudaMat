% applemantest(use_cuda) : tests cudaMat performance

%************************** CudaMat ****************************************
%   Copyright (C) 2008-2012 by Rainer Heintzmann                          *
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

function applemantest(usecuda)
%minx=-2.5;maxx=1.5;miny=-2;maxy=2;
%minx=-0.2;maxx=-0.1;miny=-1.1;maxy=-0.98;
%minx=-0.131;maxx=-0.128;miny=-1.022;maxy=-1.0185;
%minx=-0.76;maxx=-0.72;miny=-0.22;maxy=-0.16;

minx=-0.74;maxx=-0.731;miny=-0.22;maxy=-0.205;
%minx=-0.733;maxx=-0.7315;miny=-0.209;maxy=-0.207;
res=2048;iter=300;
%res=4098;iter=4000;

rg1=[minx:(maxx-minx)/res:maxx];
rg2=[miny:(maxy-miny)/res:maxy];
% if usecuda
%     rg1=cuda(rg1);
%     rg2=cuda(rg2);
% end
% Repmat not yet implemented!

if usecuda == 2  % define a precompiled cuda function
    % The C-code below executes the mandelbrot calculation on each cuda-core. That is for each pixel of the image
    % This yields the impressive overall speedup of 54000
    ProgramText = sprintf( [ ...
    'int n;\n' ...
    'float myrs=a[2*idx];\n ' ...
    'float myis=a[2*idx+1];\n' ...
    'float myr=myrs;\nfloat myi=myis;\n' ...
    'float res=-1.0;\nfloat tmp;\n' ...
    'for (n=0;n<%d;n++)\n' ...
    '    {tmp=myr*myr-myi*myi+myrs; ' ...
    '     myi=2*myr*myi+myis;' ...
    '     myr=tmp;\n' ...
    '     if ((res < 0) && (myr*myr+myi*myi)>16.0) {res=(float) n;}' ...
    '    }\n' ...
    'c[idx]= res;' ... 
    ],iter);
    % Even fast is :  '     if ((myr*myr+myi*myi)>16.0) {res=(float) n;break;}' ...
    % But we keep this for a fair comparison

    cuda_define('mandelbrot','CUDA_UnaryFkt','Computes the Mandelbrot set','c[idx]=a[idx];',ProgramText);
    cuda_compile_all;
    if isempty(which('mandelbrot'))
        error('Mandelbrot program failed to compile');
    end
end
c=repmat(rg1,[length(rg2) 1]) + i*repmat(rg2',[1 length(rg1)]);
if usecuda
    c=cuda(c);
end
if usecuda == 2  % define a precompiled cuda function
tic
    res=mandelbrot(c);
else
tic
    z=c;
    res=0*z;
    for p=1:iter   % This is the loop calculating the mandelbrot set
        z=z.*z + c;
        res((res == 0) & (abs(z)>4)) = p;
        % res = abs(z).^2;
    end
end
fprintf('Am I done?\n');
toc

%dip_image(res)
image(rg1,rg2,100*res/iter)
fprintf('Now! Overhead is 0.2 sec\n');
toc