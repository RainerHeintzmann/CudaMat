% bsxfun(myFun, A,B) : element-wise application of the function myFun to vectors using broadcasting. This implementation simply replaces it by calling myFun(A,B)
% myFunc : handle to the function to apply
% A : input A to the funciton myFunc
% B : input B to the funciton myFunc
%
% Example:
% x = 1:10;
% y = [1:11]';
% z = bsxfun(@(x, y) x.*sin(y), x, y);
%
% This function is not needed any more since as of Matlab 2016a you can write:
% z = x.*sin(y);
%
function C=bsxfun(myFun,A,B)
C=myFun(A,B);
