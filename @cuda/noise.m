% noise :  This emulates the dipimage noise function by appying a forced cast from cuda to dipimage and back
% See documentation of dip_image/noise

function out =noise(varargin)

varargin{1}=dip_image_force(varargin{1});

out = noise(varargin{:});

out=cuda(out);