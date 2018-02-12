a=cuda(readim('chromo3d'));
b=cuda(newim([3 3 3])+1);
%%
q=convn(a,b)
mean(q)
