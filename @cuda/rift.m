% rift(in) DipImage style inverse Fourier transforms cuda dat (up to 3D), but only half-comlex and not unscambled. For speed reasons.

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
function out=rift(in, transformDirs)
out=cuda();
if isa(in,'cuda') 
    isDip=in.fromDip;
    %if ~isDip
    %    in=dip_image(in);
    %end
    if strcmp(datatype(in),'scomplex')
        if isDip 
            if nargin < 2
                out.ref=cuda_cuda('rifftnd',in.ref,1.0);  % 1.0: sqrt scaling.
            else
                transformDirs2=transformDirs;
                if numel(transformDirs) > 1
                    tmp=transformDirs2(1);transformDirs2(1)=transformDirs2(2);transformDirs2(2)=tmp;
                end
                out.ref=cuda_cuda('rifftnd',in.ref,1.0,transformDirs2);  % 1.0: sqrt scaling.
            end
       else
            if nargin < 2
                out.ref=cuda_cuda('rifftnd',in.ref,0.0);  % 0.0: no scaling.
            else
                out.ref=cuda_cuda('rifftnd',in.ref,0.0,transformDirs);  % 1.0: sqrt scaling.                
            end
        end
    else
      error('Error using rift. Datatype needs to be scomplex');
    end
    out.fromDip=in.fromDip;
    %if ~isDip
    %    out=double(out);
    %end
else
    error('rift: Unsupported datatype');
end
