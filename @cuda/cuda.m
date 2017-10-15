% cuda : the basic class description file for the cuda class in CudaMat

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

classdef (InferiorClasses = {?dip_image,?double}) cuda < handle % takes the lead over dip_image or double
    properties
        ref
        fromDip    % was this variable generated from a DipImage (or from a matlab double object)?
        isBinary    % is this a binary vector (or image), in matlab called 'logical'
    end
    methods
        function out=cuda(rhs,rhs2)
            if nargin == 0
                %fprintf('constructing empty cuda object\n');
                out.ref=-1;
                out.isBinary=0;
            elseif nargin == 1
                in=rhs;
                %fprintf('constructing cuda obj\n');
                if isa(in,'cuda')  % unfortunately not allowed:  || (prod(size(in)) < 2)
                    % fprintf('cuda copy constructor\n');  % jsut ignore this
                    out=in;
                elseif isa(in,'double') || isa(in,'single')
                    if ~isempty(in)
                    out.ref=cuda_cuda('put',single(in));
                    out.fromDip=0;
                    cuda_cuda('setSize',out.ref,size(in)); % just because matlab can eliminate sizes during cast to single
                    else
                        error('Cannot convert an empty object to cuda');
                    end
                elseif isa(in,'dip_image')
                    if strcmp(datatype(in), 'scomplex') || strcmp(datatype(in), 'dcomplex')
                        out.ref=cuda_cuda('put',single(in));
                    else
                        out.ref=cuda_cuda('put',single(in));
                    end
                    cuda_cuda('setSize',out.ref,size(in)); % just because matlab can eliminate sizes during cast to single
                    if (length(size(in))>1)
                        cuda_cuda('swapSize',out.ref); % to deal with strange size order in DipImage
                    end
                    out.fromDip=1;
                end
            elseif nargin == 2 % reference and DipStatus are already known
                out.ref=rhs;
                out.fromDip=rhs2;
            end
        end
        function delete(in)
            if in.ref > 0
                cuda_cuda('delete',in.ref);
            end
        end
        function display(in)
            % global use_zeros_cuda;
            % global use_ones_cuda;
            % tmpz=use_zeros_cuda;      % since the display functions use the zeros command
            % tmpo=use_zeros_cuda;      % since the display functions use the zeros command
            % set_zeros_cuda(0);
            % set_ones_cuda(0);
            if in.fromDip
                dip_image_force(in)
            else
                double_force(in)
            end
            % set_zeros_cuda(tmpz);
            % set_ones_cuda(tmpo);
            % disp(in);  % shows some object details
        end
        function [s,varargout]=size(rhs,which)  
            lhs=cuda_cuda('getSize',rhs.ref);
            if length(lhs) > 1 && rhs.fromDip
                tmp=lhs(2);lhs(2)=lhs(1);lhs(1)=tmp;
            end
                
            if nargin>1
                if which > length(lhs) 
                    lhs(end+1:which)=1;   % expand with one sizes
                    % error('Trying to access size beyond dimensionality of array');
                end
                if which < 1
                    error('Trying to access size dimension below one');
                end
                lhs=lhs(which);
            end
            if nargout<=1
                s=lhs;
            else
                if nargout > length(lhs)
                    lhs(end+1:nargout)=1;   % expand with one sizes
                end
                s=lhs(1);
                varargout=num2cell(lhs(2:nargout));
            end
            % lhs=size(dip_image(rhs));
        end

        function lhs=length(in) % length equivalent to the matlab length function for cuda matlab arrays. returns 1 for cuda dip arrays
            if isempty(in)
                lhs=0;
            elseif in.fromDip
                lhs=1;
            else
                lhs=max(size(in));
            end
        end

    end
end
