% function [X]=svd3D_recomp(E,V)
%
%  Reconstruct X from E and V obtained by svd3D_decomp
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

function [X]=svd3D_recomp(E,V)
    if nargout > 0
       Xref=cuda_cuda('svd3D_recomp',E.ref,V.ref);
       X=cuda();
       X.ref=Xref;
       X.fromDip = E.fromDip ;   % If input was dipimage, result will be
    end
end
