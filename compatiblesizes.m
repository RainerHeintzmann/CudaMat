% isequal=equasizes(size1,size2) : Determines whether the two sizes are equal exept for singleton dimensions. This is in accordance with the new (Matlab 2016a)
% size1, size2 : size vectors like obtained by using the size() function
function iscompatible=compatiblesizes(size1,size2)
l1=length(size1);l2=length(size2);
if l1 > l2
    size2=[size2 ones(l1-l2,1)];
elseif l1 < l2
    size1=[size1 ones(l2-l1,1)];
end
allones=(size1==1) | (size2==1);
ds=size1 - size2;
iscompatible=norm(ds(~allones)) == 0;
