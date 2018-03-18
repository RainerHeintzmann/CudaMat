% y=doHessian(x,bc,numdims)  : calculates the Hessian matrix in a compressed form, stored as the trailing dimension of size 3 (2D) or 6 (3D)
function y=doHessian(x,bc,numdims)
if nargin < 2
    bc='circular';
end
if nargin < 2
    numdims=ndims(x);
end

% switch according to the boundary condition
switch(bc)
    case('circular')
        % switch according to the number of dimension of the input
        y=cuda();
        y.ref=cuda_cuda('HessianCirc',x.ref,numdims);
        y.fromDip=x.fromDip;
    case('mirror')
        % switch according to the number of dimension of the input
        y = zeros_([size(x),numdims*(numdims+1)/2]);
        switch(numdims)
            % 2 dimension
            case(2)
                % xx
                y(:,:,1)=x([3:end,end,end-1],:) -2*x([2:end,end],:) + x;
                % xy
                y(:,:,2)=x([2:end,end],[2:end,end]) -x([2:end,end],:) - x(:,[2:end,end]) +x;
                % yy
                y(:,:,3)=x(:,[3:end,end,end-1]) -2*x(:,[2:end,end]) + x;
                % 3 dimensions
            case(3)
                % xx
                y(:,:,:,1)=x([3:end,end,end-1],:,:) -2*x([2:end,end],:,:) + x;
                % xy
                y(:,:,:,2)=x([2:end,end],[2:end,end],:) -x([2:end,end],:,:) - x(:,[2:end,end],:) +x;
                % xz
                y(:,:,:,3)=x([2:end,end],:,[2:end,end]) -x([2:end,end],:,:) - x(:,:,[2:end,end]) +x;
                % yy
                y(:,:,:,4)=x(:,[3:end,end,end-1],:) -2*x(:,[2:end,end],:) + x;
                % yz
                y(:,:,:,5)=x(:,[2:end,end],[2:end,end]) -x(:,[2:end,end],:) - x(:,:,[2:end,end]) +x;
                % zz
                y(:,:,:,6)=x(:,:,[3:end,end,end-1]) -2*x(:,:,[2:end,end]) + x;
        end
    case('zeros')
        % switch according to the number of dimension of the input
        y = zeros_([size(x),numdims*(numdims+1)/2]);
        switch(numdims)
            % 2 dimension
            case(2)
                % xx
                y(1:end-2,:,1)=x(3:end,:)-2*x(2:end-1,:)+x(1:end-2,:);
                y(end-1,:,1)=x(end-1,:)-2*x(end,:);
                y(end,:,1)=x(end,:);
                % xy
                y(1:end-1,1:end-1,2)=x(2:end,2:end) - x(2:end,1:end-1)-x(1:end-1,2:end) + x(1:end-1,1:end-1);
                y(end,1:end-1,2)=x(end,1:end-1)-x(end,2:end);
                y(1:end-1,end,2)=x(1:end-1,end)-x(2:end,end);
                y(end,end,2)=x(end,end);
                % yy
                y(:,1:end-2,3)=x(:,3:end)-2*x(:,2:end-1)+x(:,1:end-2);
                y(:,end-1,3)=x(:,end-1)-2*x(:,end);
                y(:,end,3)=x(:,end);
                % 3 dimensions
            case(3)
                % xx
                y(1:end-2,:,:,1)=x(3:end,:,:)-2*x(2:end-1,:,:)+x(1:end-2,:,:);
                y(end-1,:,:,1)=x(end-1,:,:)-2*x(end,:,:);
                y(end,:,:,1)=x(end,:,:);
                % xy
                y(1:end-1,1:end-1,:,2)=x(2:end,2:end,:) - x(2:end,1:end-1,:)-x(1:end-1,2:end,:) + x(1:end-1,1:end-1,:);
                y(end,1:end-1,:,2)=x(end,1:end-1,:)-x(end,2:end,:);
                y(1:end-1,end,:,2)=x(1:end-1,end,:)-x(2:end,end,:);
                y(end,end,:,2)=x(end,end,:);
                % xz
                y(1:end-1,:,1:end-1,3)=x(2:end,:,2:end) - x(2:end,:,1:end-1)-x(1:end-1,:,2:end) + x(1:end-1,:,1:end-1);
                y(end,:,1:end-1,3)=x(end,:,1:end-1)-x(end,:,2:end);
                y(1:end-1,:,end,3)=x(1:end-1,:,end)-x(2:end,:,end);
                y(end,:,end,3)=x(end,:,end);
                % yy
                y(:,1:end-2,:,4)=x(:,3:end,:)-2*x(:,2:end-1,:)+x(:,1:end-2,:);
                y(:,end-1,:,4)=x(:,end-1,:)-2*x(:,end,:);
                y(:,end,:,4)=x(:,end,:);
                % yz
                y(:,1:end-1,1:end-1,5)=x(:,2:end,2:end) - x(:,2:end,1:end-1)-x(:,1:end-1,2:end) + x(:,1:end-1,1:end-1);
                y(:,end,1:end-1,5)=x(:,end,1:end-1)-x(:,end,2:end);
                y(:,1:end-1,end,5)=x(:,1:end-1,end)-x(:,2:end,end);
                y(:,end,end,5)=x(:,end,end);
                % zz
                y(:,:,1:end-2,6)=x(:,:,3:end)-2*x(:,:,2:end-1)+x(:,:,1:end-2);
                y(:,:,end-1,6)=x(:,:,end-1)-2*x(:,:,end);
                y(:,:,end,6)=x(:,:,end);
        end
end
