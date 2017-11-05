% [resCuda]=debugCheck(fkt,varargin)  : calls a function twice with and without cuda acceleration to see if the results are the same
% This function is used a lot in installDebug.m
%
function [resCuda]=debugCheck(fkt,varargin)
argMatlab=cell(1,nargin-1);
hasCuda=0;
Nout = nargout;
if Nout== 0
    Nout=1;
end
inargs=varargin{1};

global remMax;
global remPos;
global rem;

if isempty(remMax)
    remMax=20;
    remPos=0;
end
if isempty(rem)
    rem={};
end

if remMax > 0
    remPos = remPos+1;
end

for n=1:numel(inargs)
    if (isa(inargs{n},'cuda'))
        argMatlab{n}=castToMatlab(inargs{n});  % convert to double or dip_image
        hasCuda=1;
    else
        argMatlab{n}=inargs{n};  
    end
end
resMatlab=cell(1,Nout);
eval(['[resMatlab{:}]=' fkt '(argMatlab{:});']);
if ~hasCuda
    resCuda=resMatlab;
    if Nout>0
        resCuda=resCuda{:};
    end
    if exist('remMax','var') && remMax>0
        rem{remPos}={resCuda};  % Store this in the Matlab-only run
    end
    if exist('remMax','var') && remMax>0 && remPos >= remMax
        error('End of Debugging run reached.')
    end
    return;
end
%% Compare Matlab to CudaMat result
resCuda=cell(1,Nout);
eval(['[resCuda{:}]=' fkt '(inargs{:});']);

for n=1:Nout
    D=double(resMatlab{n}- castToMatlab(resCuda{n}));
    D=norm(D(:));
    M1=max(abs(resMatlab{n}(:)));
    M2=max(abs(resCuda{n}(:)));
    M=max(M1,M2);
    if (D/M > 0.01)  % one percent error leads to scream
        error('Function %s: detected disagreement of results!',fkt);
    end    
end

fprintf('checked function %s .. relative error is %d percent\n',fkt,D/M*100);

if exist('remMax','var') && remMax>0 && remPos < remMax
    if length(rem) >= remPos
        for n=1:Nout
            D=double(castToMatlab(rem{remPos}{n}) - castToMatlab(resCuda{n}));
            D=norm(D(:));
            M1=max(abs(resMatlab{n}(:)));
            M2=max(abs(resCuda{n}(:)));
            M=max(M1,M2);
            if (D/M > 0.01)  % one percent error leads to scream
                error('Function %s: detected disagreement of results!',fkt);
            end
        end
        fprintf('checked function %s .. relative error to previous run is %d percent\n',fkt,D/M*100);
    end
end

if exist('remMax','var') && remMax>0 && remPos >= remMax
    error('End of Debugging run reached.')
end

if Nout>0
    resCuda=resCuda{:};
end
