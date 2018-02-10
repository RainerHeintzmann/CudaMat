function [IndexMatrix,DestSize]=GenIndexMatrix(in,index,inFromDip)
maxsize=1;
if inFromDip && numel(index.subs) > 1
    tmp=index.subs{1};index.subs{1}=index.subs{2};index.subs{2}=tmp;
end
for d=1:ndims(in)  % Generates a list of indexing vectors (whos outer product describes whree to access)
    if d<=numel(index.subs) && ~isempty(index.subs{d}) && ~ischar(index.subs{d})
        if inFromDip == 0
            index.subs{d}=index.subs{d}-1; 
        end
    else
        index.subs{d}=[0:size(in,d)-1]; 
    end
    if numel(index.subs{d}) > maxsize
        maxsize=numel(index.subs{d});
    end
end
IndexMatrix=builtin('zeros',[maxsize ndims(in)]);  % zeros_cuda2(cuda(0),[maxsize ndims(in)]);
DestSize=[];
for d=1:ndims(in)  % writes all the index lists into a big 2D matrix and constructs a destination size vector
    IndexMatrix(1:numel(index.subs{d}),d)=index.subs{d};
    DestSize(d)=numel(index.subs{d});
end
%if inFromDip && numel(index.subs) > 1
%    tmp=DestSize(1);DestSize(1)=DestSize(2);DestSize(2)=tmp;
%end
IndexMatrix=cuda(IndexMatrix);



% old cuda code, which did not work ...

% maxsize=1;
% for d=1:ndims(in)  % Generates a list of indexing vectors (whos outer product describes whree to access)
%     if d<=numel(index.subs) && ~isempty(index.subs{d}) && ~ischar(index.subs{d})
%         if inFromDip == 0
%             index.subs{d}=cuda(index.subs{d})-1;
%         else
%             index.subs{d}=cuda(index.subs{d});
%         end
%     else
%         % index.subs{d}=[];
%         index.subs{d}=xx_cuda2(size(in,d),'corner');
%     end
%     if numel(index.subs{d}) > maxsize
%         maxsize=numel(index.subs{d});
%     end
% end
% IndexMatrix=zeros_cuda2(cuda(0),[maxsize ndims(in)]);
% DestSize=[];
% for d=1:ndims(in)  % writes all the index lists into a big 2D matrix and constructs a destination size vector
%     % IndexMatrix(1:numel(index.subs{d}),d)=index.subs{d};
%     idx={[1:numel(index.subs{d})],d};  S=struct('type','()','subs',{idx});
%     tmp=transpose(index.subs{d});
%     DestSize(d)=numel(index.subs{d});
%     ref=IndexMatrix.ref;
%     res=subsasgn(IndexMatrix,S,tmp); % res will not be used
%     cuda_cuda('clear_ignoreDelete',ref);  % This is just to make cuda_cuda ignore the delete. This reference is not deleted.
% end
% cuda_cuda('set_ignoreDelete',ref);  % This is just to make cuda_cuda ignore the delete. This reference is not deleted.
% 
