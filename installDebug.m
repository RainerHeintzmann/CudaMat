% This script shadows many CudaMat functions with a Debug version

max = @(varargin) debugCheck('max',varargin);
min = @(varargin) debugCheck('min',varargin);
sum = @(varargin) debugCheck('sum',varargin);
mean = @(varargin) debugCheck('mean',varargin);
norm = @(varargin) debugCheck('norm',varargin);

ft = @(varargin) debugCheck('ft',varargin);
ift = @(varargin) debugCheck('ift',varargin);
rft = @(varargin) debugCheck('rft',varargin);
rift = @(varargin) debugCheck('rift',varargin);
fft = @(varargin) debugCheck('fft',varargin);
ifft = @(varargin) debugCheck('ifft',varargin);
fftn = @(varargin) debugCheck('fftn',varargin);
ifftn = @(varargin) debugCheck('ifftn',varargin);

fftshift = @(varargin) debugCheck('fftshift',varargin);
ifftshift = @(varargin) debugCheck('ifftshift',varargin);

newim = @(varargin) debugCheck('newim',varargin);
flipdim = @(varargin) debugCheck('flipdim',varargin);
rr = @(varargin) debugCheck('rr',varargin);
phiphi = @(varargin) debugCheck('phiphi',varargin);
ramp = @(varargin) debugCheck('ramp',varargin);

repmat = @(varargin) debugCheck('repmat',varargin);

reshape = @(varargin) debugCheck('reshape',varargin);
cat = @(varargin) debugCheck('cat',varargin);
horzcat = @(varargin) debugCheck('horzcat',varargin);
vertcat = @(varargin) debugCheck('vertcat',varargin);
circshift = @(varargin) debugCheck('circhift',varargin);
transpose = @(varargin) debugCheck('transpose',varargin);
complex = @(varargin) debugCheck('complex',varargin);
besselj = @(varargin) debugCheck('besselj',varargin);
sin = @(varargin) debugCheck('sin',varargin);
cos = @(varargin) debugCheck('cos',varargin);
sinh = @(varargin) debugCheck('sinh',varargin);
cosh = @(varargin) debugCheck('cosh',varargin);
tan = @(varargin) debugCheck('tan',varargin);
atan2 = @(varargin) debugCheck('atan2',varargin);
log = @(varargin) debugCheck('log',varargin);
exp = @(varargin) debugCheck('exp',varargin);
abs = @(varargin) debugCheck('abs',varargin);
dot = @(varargin) debugCheck('dot',varargin);

conj = @(varargin) debugCheck('conj',varargin);
isreal = @(varargin) debugCheck('isreal',varargin);
real = @(varargin) debugCheck('real',varargin);
imag = @(varargin) debugCheck('imag',varargin);

any = @(varargin) debugCheck('any',varargin);
all = @(varargin) debugCheck('all',varargin);
le = @(varargin) debugCheck('le',varargin);
ge = @(varargin) debugCheck('ge',varargin);
lt = @(varargin) debugCheck('lt',varargin);
gt = @(varargin) debugCheck('gt',varargin);
eq = @(varargin) debugCheck('eq',varargin);
isinf = @(varargin) debugCheck('isinf',varargin);
isnan = @(varargin) debugCheck('isnan',varargin);
sign = @(varargin) debugCheck('sign',varargin);

isLegal = @(varargin) debugCheck('isLegal',varargin);
sqrt = @(varargin) debugCheck('sqrt',varargin);
phase = @(varargin) debugCheck('phase',varargin);
angle = @(varargin) debugCheck('angle',varargin);
ndims = @(varargin) debugCheck('ndims',varargin);

isfinite = @(varargin) debugCheck('isfinite',varargin);
logical = @(varargin) debugCheck('logical',varargin);
prod = @(varargin) debugCheck('prod',varargin);

floor = @(varargin) debugCheck('floor',varargin);
ceil = @(varargin) debugCheck('ceil',varargin);

%%
plus = @(varargin) debugCheck('plus',varargin);
minus = @(varargin) debugCheck('minus',varargin);
rdivide = @(varargin) debugCheck('rdivide',varargin);
mrdivide = @(varargin) debugCheck('mrdivide',varargin);
mldivide = @(varargin) debugCheck('mldivide',varargin);
mtimes = @(varargin) debugCheck('mtimes',varargin);
times = @(varargin) debugCheck('times',varargin);
power = @(varargin) debugCheck('power',varargin);

and = @(varargin) debugCheck('and',varargin);
or = @(varargin) debugCheck('or',varargin);
not = @(varargin) debugCheck('not',varargin);
