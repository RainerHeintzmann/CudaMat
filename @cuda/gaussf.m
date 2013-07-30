% gaussf :  This emulates the dipimage gaussf function by appying a forced cast from cuda to dipimage and back (SLOW!!)
% See documentation of dip_image/gaussf

function out =gaussf(varargin)

varargin{1}=dip_image_force(varargin{1});

out = gaussf(varargin{:});

out=cuda(out);