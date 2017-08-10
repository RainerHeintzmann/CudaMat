% MyIdivErrorAndDeriv(aRecon) : Error measure for deconvolution algorithm

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

function [err,grad]=MyIdivErrorAndDeriv(aRecon)
global myim;
global otfrep;
global lambdaPenalty;
global Factor
%Factor= 1.239e6; % max(myim)/max(grad)
% Recast the matlab data into the dip_image datastructure
%aRecon=dip_image(aRecon);
norm3D = sqrt(prod(size(myim)));
aRecon=reshape(dip_image(aRecon','single'),size(myim));

%aRecon=cuda(aRecon);
% Forward model
%tic
ftRecons=ft(aRecon);
Recons=norm3D*real(ift(ftRecons .* otfrep));  % convolve with PSF, still in Fourier space
% Residual: (here Cezar's i-divergence)
fidivval=sum(Recons-myim .* log(Recons));  % fast version omitting constants
% Total error term:
err = Factor*double(fidivval + lambdaPenalty * sum(aRecon.^2.*(aRecon<0)))/prod(size(otfrep));  % /myMeasSum
%fprintf(',  %g ',err);
% Backward model (Transpose and inverse order of operations)
ratio = myim ./ Recons -1;
Penalty = aRecon;
%Penalty(aRecon > 0) = 0;

grad=ft(ratio);
grad=-Factor*norm3D*real(ift(grad .* conj(otfrep)))  + 2*lambdaPenalty *Penalty;
%toc
%err
grad=(double(reshape(grad,[prod(size(myim)) 1])))'/prod(size(otfrep)); 
%grad=single_force(grad);
