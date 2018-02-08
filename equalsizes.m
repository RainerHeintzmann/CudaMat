% isequal=equasizes(size1,size2) : Determines whether the two sizes are equal exept for trailing singleton dimensions
function isequal=equalsizes(size1,size2)
l1=length(size1);l2=length(size2);
if l1 > l2
    size2=[size2 ones(l1-l2,1)];
elseif l1 < l2
    size1=[size1 ones(l2-l1,1)];
end
isequal=norm(size1 - size2) == 0;
