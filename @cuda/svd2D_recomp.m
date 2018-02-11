% function [X]=svd2D_recomp(E,V)
%
%  Reconstruct X from E and V obtained by svd2D_decomp
%  
%  Copyright (C) 2017 E. Soubies emmanuel.soubies@epfl.ch
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.%
%%

function [X]=svd2D_recomp(E,V)
    if nargout > 0
       Xref=cuda_cuda('svd_recomp',E.ref,V.ref); % figures out from the datasize whether to call the 3D or 2D routine
       X=cuda();
       X.ref=Xref;
       X.fromDip = E.fromDip ;   % If input was dipimage, result will be
    end
end
