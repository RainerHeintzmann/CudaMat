% expendeddata = expanddim(inputdata,dimensions) :  Fills in dimensions with size 1, not altering the data (opposite of squeeze)
% This version is meant to subsitute for the dip-image specific version
% inputdata : data to expand
% dimensions : number of required dimensions
%
% authors: Rainer Heintzmann, Ondrej Mandula
% see also: squeeze

function outputdata=expanddim(inputdata,dimensions)
sizevec=size(inputdata);
newsvec=squeeze(ones(1,dimensions));
newsvec(1:size(sizevec,2))=sizevec(:);
outputdata=reshape(inputdata,newsvec);
