% function createDipHandles  : creates a number of handles to overwritten dipimage functions
function createDipHandles()
global diphandle_xx;
global diphandle_yy;
global diphandle_zz;
global diphandle_rr;
global diphandle_ramp;
global diphandle_phiphi;
global diphandle_newim;
global diphandle_newimar;

if 1 % isdeployed()==0
    pathOld=pwd;  % remember where the current path is
    pathDIP=fileparts(which('dipimage/xx')); % get the directory path of the dipimage/xx function
    cd(pathDIP);  % go to the DipImage path to shadow the current cudaMat xx function.
    diphandle_xx=@xx;
    diphandle_yy=@yy;
    diphandle_zz=@zz;
    diphandle_rr=@rr;
    diphandle_ramp=@ramp;
    diphandle_phiphi=@phiphi;
    diphandle_newim=@newim;
    diphandle_newimar=@newimar;
    cd(pathOld);  % return to where the user was before
else
    diphandle_xx=@xx_dip;
    diphandle_yy=@yy_dip;
    diphandle_zz=@zz_dip;
    diphandle_rr=@rr_dip;
    diphandle_ramp=@ramp_dip;
    diphandle_phiphi=@phiphi_dip;
    diphandle_newim=@newim_dip;
    diphandle_newimar=@newimar_dip;
end
%[pathstr,name,ext]=fileparts(mfilename('fullpath'));
% pathadd=1;
% try
%     rmpath(pathstr);   % remove the current CudaMat path
% catch me
%     pathadd=0;
% end



%feval(diphandle_xx)
% if pathadd
%     addpath(pathstr);   % remove the current CudaMat path
% end
%feval(diphandle_xx)