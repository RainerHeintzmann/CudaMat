% cuda_define(FktName, FktType, Comment, CoreCommandsReal, CoreCommandsCpxReal, CoreCommandsRealCpx, CoreCommandsCpxCpx)  : defines a function into the system of cuda commands that can be executed on NVidia Graphic cards
% FktName : Name of the function
% FktType : 
%           'CUDA_UnaryFkt': A cuda function that takes one array (a) as input and has one result array
%           'CUDA_UnaryFktConst': Takes one array (a) and an additional scalar real value (b) as input and has one result array.
%           'CUDA_UnaryFktConstC': Takes one array (a) and an additional complex value (real part:br complex part: bi) as input and has one result array.
%           'CUDA_BinaryFkt' : Takes two arrays (a and b) as input and has one (c) as output
%           'CUDA_3DAsgFkt' : Takes one scalar real or complex (br and bi) and has out output array (c).
%
% CoreCommandsReal : C/cuda - commands that are executed for every computing core for real valued input
% CoreCommandsRealCpx : C/cuda - commands that are executed for every computing core for complex valued input (one input) or a complex and real array
% CoreCommandsCpxReal : C/cuda - commands that are executed for every computing core for complex valued input
% CoreCommandsCpxCpx : C/cuda - commands that are executed for every computing core for complex valued input
%
% the following variables are available:
% a : first input array
% b : second input array  (binary only)
%
% c : output array
% idx : index running through the array
% idd : index in the destination array (only for functions with subarray indexing)
% x,y,z : pixel coordinates (only for FuntionType CUDA_3DFkt)
%
% Attention: The C-code must not contain any komma, as this upsets the macros with arguments and will thus not compile!
% 
% all defined function text is saved in a global variable "cuda_to_compile" which can be cleared by
% cuda_to_compile=[];
%
% see also: cuda_compile_all
% 
% example : 
%
% cuda_define('myadd','CUDA_BinaryFkt','Adds two arrays together.','c[idx]=a[idx]+b[idx];','c[2*idx]=a[2*idx]+b[idx];c[2*idx+1]=a[2*idx+1];','c[2*idx]=a[idx]+b[2*idx];c[2*idx+1]=b[2*idx+1];','c[2*idx]=a[2*idx]+b[2*idx];c[2*idx+1]=a[2*idx+1]+b[2*idx+1];');
% cuda_compile_all();
% mytst=myadd(readim,readim('orka'))
% whos
% 
% see also : the code of the appleman program for argument appleman(2)


function cuda_define(FktName, FktType, Comment, CoreCommandsReal, CoreCommandsCpxReal, CoreCommandsRealCpx, CoreCommandsCpxCpx) 
%           'CUDA_MaskIdx': Takes one array and a mask index vector as input. The latter can be used to modify the result inside the mask
%           'CUDA_PartRedMask' : Takes one array (a) as input and reduces it to a scalar
%           'CUDA_PartRedMaskCpx' : Takes a complex array (a) as input and reduces it to a complex scalar
%           'CUDA_PartRedMaskIdx' : Takes an array (a) and finds an index

%CUDA_PartRedMask(psum_arr,mysum)
%CUDA_PartRedMaskCpx(psum_carr,mysum)
%CUDA_PartRedMaskIdx(pmax_arr,maxCond)
%CUDA_3DAsgFkt(const_subcpy_arr,c[idd]=br;)
% CUDA_3DFkt(arr_subcpy_arr,c[idd]=a[ids];)
% CUDA_BinaryFkt(arr_power_arr,c[idx]=pow(a[idx],b[idx]);)
% CUDA_MaskIdx(arr_subsref_arr,c[mask_idx]=a[idx];)
% CUDA_UnaryFktConstC(arr_unequals_Cconst,c[idx]=(a[idx]!=br) || (bi!=0);)
% CUDA_UnaryFktConstC(Cconst_unequals_arr,c[idx]=(br!=a[idx]) || (bi!=0);)
% CUDA_BinaryFkt(arr_unequals_arr,c[idx]=(a[idx]!=b[idx]);)
if nargin < 5
    CoreCommandsCpxReal='';
end
if nargin < 6
    CoreCommandsRealCpx='';
end
if nargin < 7
    CoreCommandsCpxCpx='';
end
switch(FktType)
    case 'CUDA_UnaryFkt'  % A cuda function that takes one array (a) as input and has one result array.
        m_snippet = sprintf('%% out = %s(in1)   : User defined cuda function. %s\n',FktName,Comment);
        m_snippet = sprintf('%s function out = %s(in1)\n',m_snippet,FktName);
        m_snippet = sprintf('%s if isa(in1,''cuda'')  \n',m_snippet);
        m_snippet = sprintf('%s    ref=cuda_cuda(''%s'',getReference(in1)); \n',m_snippet,FktName);
        m_snippet = sprintf('%s else \n',m_snippet);
        m_snippet = sprintf('%s error(''%s: Unknown datatype''); \n',m_snippet,FktName);
        m_snippet = sprintf('%s end \n',m_snippet);
        m_snippet = sprintf('%s out=cuda(ref,getFromDip(in1));   \n',m_snippet);

        c_snippet = sprintf('else if (strcmp(command,"%s")==0) \n{ if (nrhs != 2) mexErrMsgTxt("cuda: %s needs two arguments"); \n', FktName,FktName);
        c_snippet = sprintf('%s  if (isComplexType(getCudaRefNum(prhs[1]))) \n', c_snippet);
        c_snippet = sprintf('%s     CUDA%s_carr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1]));  \n',c_snippet,FktName);
        c_snippet = sprintf('%s else \n',c_snippet);
        c_snippet = sprintf('%s     CUDA%s_arr(getCudaRef(prhs[1]),cudaAllocReal(prhs[1]),getTotalSizeFromRef(prhs[1]));  \n',c_snippet,FktName);
        c_snippet = sprintf('%s if (nlhs > 0) \n',c_snippet);
        c_snippet = sprintf('%s     plhs[0] =  mxCreateDoubleScalar((double)free_array); \n',c_snippet);
        c_snippet = sprintf('%s Dbg_printf("cuda: %s\");\n} \n',c_snippet,FktName);

        cu_snippet = sprintf('%s(%s_arr,%s) \n%s(%s_carr,%s)\n',FktType,FktName,CoreCommandsReal,FktType,FktName,CoreCommandsCpxReal);
        h_snippet = sprintf('externC const char * CUDA%s_arr(float * a, float * c, int N);\nexternC const char * CUDA%s_carr(float * a, float * c, int N);\n\n',FktName,FktName);
    case 'CUDA_BinaryFkt'  % Takes two arrays (a and b) as input and has one (c) as output
        m_snippet = sprintf('%% out = %s(in1,in2)   : User defined cuda function. %s\n',FktName,Comment);
        m_snippet = sprintf('%s function out = %s(in1,in2)\n',m_snippet,FktName);
        m_snippet = sprintf('%s if prod(size(in1)) > 1 && ~isa(in1,''cuda'')  \n in1=cuda(in1); \nend\n',m_snippet);
        m_snippet = sprintf('%s if prod(size(in2)) > 1 && ~isa(in2,''cuda'')  \n in2=cuda(in2); \nend\n',m_snippet);
        m_snippet = sprintf('%s if isa(in1,''cuda'') && isa(in2,''cuda'')\n',m_snippet);
        m_snippet = sprintf('%s    ref=cuda_cuda(''%s'',in1.ref,in2.ref); \n',m_snippet,FktName);
        m_snippet = sprintf('%s else \n',m_snippet);
        m_snippet = sprintf('%s error(''%s: Needs two cuda arrays as input''); \n',m_snippet,FktName);
        m_snippet = sprintf('%s end \n',m_snippet);
        m_snippet = sprintf('%s out=cuda(ref,in1.fromDip || in2.fromDip);   \n',m_snippet);

        c_snippet = sprintf('else if (strcmp(command,"%s")==0) {\n CallCUDA_BinaryFkt(%s,cudaAlloc)\n', FktName,FktName);
        c_snippet = sprintf('%s if (nlhs > 0) \n',c_snippet);
        c_snippet = sprintf('%s     plhs[0] =  mxCreateDoubleScalar((double)free_array); \n',c_snippet);
        c_snippet = sprintf('%s Dbg_printf("cuda: %s\");\n} \n',c_snippet,FktName);

        cu_snippet = sprintf('%s(arr_%s_arr,%s) \n%s(arr_%s_carr,%s)\n%s(carr_%s_arr,%s)\n%s(carr_%s_carr,%s)\n',FktType,FktName,CoreCommandsReal,FktType,FktName,CoreCommandsRealCpx,FktType,FktName,CoreCommandsCpxReal,FktType,FktName,CoreCommandsCpxCpx);
        h_snippet = sprintf('externC const char * CUDAarr_%s_arr(float * a, float * b, float * c, int N);\nexternC const char * CUDAarr_%s_carr(float * a, float * b, float * c, int N);\nexternC const char * CUDAcarr_%s_arr(float * a, float * b, float * c, int N);\nexternC const char * CUDAcarr_%s_carr(float * a, float * b, float * c, int N);\n\n',FktName,FktName,FktName,FktName);

    otherwise
        error('cuda_define: Unknown Function Type!');
end
       
global cuda_to_compile;

if exist('cuda_to_compile') && ~isempty(cuda_to_compile)
    myindex=find(strcmp(cuda_to_compile.name,FktName));
    if isempty(myindex)   % This function has not already been defined
        myindex=length(cuda_to_compile.name)+1;  % append a new element
    else
        fprintf('WARNING: Redefining cuda function %s, which was already defined\n',FktName);
    end
else    
    myindex = 1;
end
    cuda_to_compile.name{myindex} = FktName;
    cuda_to_compile.m_snippet{myindex} = m_snippet;
    cuda_to_compile.c_snippet{myindex} = c_snippet;
    cuda_to_compile.cu_snippet{myindex} = cu_snippet;
    cuda_to_compile.h_snippet{myindex} = h_snippet;
    if ~isfield(cuda_to_compile,'needsRecompile')
        cuda_to_compile.needsRecompile=0;
    end
    if myindex==length(cuda_to_compile.name)+1 || ~isfield(cuda_to_compile,'coreCommandsReal') || ~strcmp(cuda_to_compile.coreCommandsReal{myindex},CoreCommandsReal)
        cuda_to_compile.coreCommandsReal{myindex} = CoreCommandsReal;
        cuda_to_compile.needsRecompile = 1;
    end
    if myindex==length(cuda_to_compile.name)+1 || ~isfield(cuda_to_compile,'coreCommandsCpxReal') || ~strcmp(cuda_to_compile.coreCommandsCpxReal{myindex},CoreCommandsCpxReal)
        cuda_to_compile.coreCommandsCpxReal{myindex} = CoreCommandsCpxReal;
        cuda_to_compile.needsRecompile = 1;
    end
    if myindex==length(cuda_to_compile.name)+1 || ~isfield(cuda_to_compile,'coreCommandsRealCpx') || ~strcmp(cuda_to_compile.coreCommandsRealCpx{myindex},CoreCommandsRealCpx)
        cuda_to_compile.coreCommandsRealCpx{myindex} = CoreCommandsRealCpx;
        cuda_to_compile.needsRecompile = 1;
    end
    if myindex==length(cuda_to_compile.name)+1 || ~isfield(cuda_to_compile,'coreCommandsCpxCpx') || ~strcmp(cuda_to_compile.coreCommandsCpxCpx{myindex},CoreCommandsCpxCpx)
        cuda_to_compile.coreCommandsCpxCpx{myindex} = CoreCommandsCpxCpx;
        cuda_to_compile.needsRecompile = 1;
    end
    
