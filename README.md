# CudaMat 

CudaMat enables fast computing on graphics cards that supports the CUDA programming language. Currently such cards are available from NVidia. CudaMat is, as much as possible, invisible to the user. The idea is that the user can transform any existing Matlab code into a CudaMat code with minimal effort. E.g. with a single line like a=cuda(a) the Matlab object ‘a’ gets transformed into a CudaMat object ‘a’. This can be checked using the matlab command whos.
previous version (pre 2.0.0) were hosted elsewhere.

NOTE: To get a working version under Windows which does not need a pre-installed Cuda Development, you will have to obtain the file cufft64_90.dll and put it into the CudaMat/bin folder!

# Under which conditions will CudaMat be fast?

CudaMat will greately improve the speed of your code, when the main time of your Matlab code is spent in computing ‘expensive’ operations between large matrices and/or vectors, sums over them or Fourier transformations. However, when the problem consists of many operations on small matrices and vectors, CudaMat will probably not help you and might in fact turn out to be slower than standard matlab code. One way to think of this is that every start of a function execution in CudaMat has some overhead, but once it is running, it is quite fast.
It may be possible to adjust the performance a little bit by changing the two #define commands for BLOCKSIZE and CUIMAGE_REDUCE_THREADS  given at the top of the file cudaArith.cu.

# Is there a demo to quickly check the performance increase?

Yes. CudaMat comes with a two test programs ‘applemantest.m’ and  ‘speedtestDeconv.m’.

applemantest.m calculates the famous Mandelbrot set in a straight forward way. This test has the advantage that it does not require any toolboxes other than CudaMat and [NVidia’s cuda library](http://www.nvidia.com/object/cuda_get.html) to be installed. The speedup optained depend on the chosen datasize. On my Intel(R) Core(TM) i7 CPU @ 2,8 GHz, 64 bit processor, Windows 7 is about a factor of 30 (2.35 versus 75,5 seconds) for a 2048×2048 image with iteration depth 300.
The new (as of version 1.0.0.06 beta) on-the-fly compilation allows a further speedup by writing code snippets for the GPU. In this case the graphic card needs 0.088 second for the example above, yielding a total speedup bigger than 850! Type “edit applemantest” under matlab to get an example how to achieve such speed.

speedtestDeconv.m measures the performance for an example deconvolution of a 3D microscopy dataset (using the DipImage ‘chromo3d’ example image).
To run this demo, DipImage with the example images and CudaMat need to be installed, as well as the optimisation toolbox with the function minFunc() written by Mark Schmidt (line 103 in the file polyinterp.m needs to be changed to: for qq=1:length(cp);xCP=cp(qq); and the appearances of ones() and zeros() need to be changed to ones_cuda() and zeros_cuda()). A GeForce GTX 280 card gave about 10x speedup (3.3 versus 30.3 seconds) in comparison to a 2,4 GHz AMD Hammer 64 bit processor and gcc 4.3.2 run under OpenSuse11.1 .

# What is CUDA?

Cuda is a programming language extension to C which enables code to run in parallel on multi-processor graphics cards. Current graphics cards can have more than 200 processors running simultaneously. They all execute the same code (SIMD = single instruction, multiple data). If a branch point (e.g. initiated by an ‘if’) is reached, where some processors have to execute different code than others, these processes are temporarily suspended. The beauty of the hardware is that this switching between many thousands of processes is very efficient.
What changes may be necessary to existing Matlab code to run under CudaMat?

Note that CudaMat currently only supports the single floatingpoint datatype of matlab (4 bytes). Since Matlab usually computed with doubles, the results can differ depending on how sensitive the algorithm is to roud-off errors.
The general idea is that only large marix (image) input objects requiring time intensive conputations should converted to cuda before the existing Matlab code is run. Ideally no changes to the Matlab code should be necessary.
However, practically minor changes can be necessary, if CudaMat does not support the operation used in the Matlab code. This is especially the case for

* Additional datatypes defined by the Matlab code
* Using a standard Matlab operation that is not yet implemented in CudaMat
* If the Matlab code checks for the datatype with operations other than isreal() or isfloat(). E.g. if the operation isa() is used, the result is probably wrong.
* for loops iterating over the contense of a vector need a minor change (iterating over an access index and the assigning the component by indexing in the vector) to be compatible with CudaMat

Sometimes the system may perform an automatic conversion to a Matlab object, with the associated overhead involved in transferring from the graphics card.
In other cases the user will have to either force this conversion (e.g. using single_force(a)), find an alternative expression, which is supported in CudaMat or extend the CudaMat algorithms to support this additional feature (please send me an email with the new code, so I can put it up on the website).

In addition, there may be changes necessary inside the Matlab code, if new objects are generated, as these will be by default Matlab matrices.
Prominent examples are the Matlab commands zeros() and ones() , which by default generate Matlab objects. These function calls should be changed to zeros_cuda(), ones_cuda().
Global variables influencing the behaviour of zeros_cuda(), ones_cuda() but also overloaded DIPImage funcitons newim(), xx(), yy(), zz(), rr(), phiphi().
Whether they then generate a standard or a cuda object) can conveniently be set via the functions set_ones_cuda(state) and set_zeros_cuda(state) and alike.

Other command which generate Matlab objects are enumerations such as [1:N] or meshgrid().
In future versions, it will be possible to define by a set of global variables whether these functions should generate standard Matlab objects of cuda objects.
In addition it may (in rare cases) be necessary to convert standard Matlab matrices to cuda (e.g. using the command cuda(a)) within the Matlab code to run, as some CudaMat functions may not yet automatically do so.

# Why a separate datatype ‘cuda’?

To realize the idea of accessing the speed of the graphics cards from within the convenient programming environment of Matlab efficiently, one has to avoid memory transfer to and from the graphics card as much as possible. To this aim a datatype ‘cuda’ was introduced.
Whenever matlab needs to execute a function that involves a cuda object as one of it’s arguments, it checks for the presence of this function in the folder @cuda and executes the code given there. In this way it is ensured that code can efficiently be executed on the graphics card, without the cuda objects leaving the card.

# When will transfers be made to and from the graphics card?

If a cuda object is created (e.g. a=cuda(a)), the matlab object is transferred to the graphics card. This costs some time and should thus ideally not be performed within the inner loop of a calculation. With every output operation (e.g. printing the values on the screen or displaying an image) the data is transferred back from the graphics card to Matlab.
The commands double_force(a) and single_force(a) will force a conversion from a cuda object back to matlab (and not affect the object if it is already a standard matlab double or single).
In the event that a CudaMat operation results in a single value, the result will automatically transferred back to an ordinary Matlab object.

Why do ordinary conversion operations ‘single(a)’ and ‘double(a)’ not convert back to a Matlab matrix?
Currently these operations leave the objects on the graphics card, with the aim to require as little modification as possible to existing Matlab programs to be able to run under CudaMat. Currently these command are essentially ignored. To force a conversion use the command single_force(a) or double_force(a) with a cuda object ‘a’.

# How can I reset the graphics card when something went wrong?

If an error appeared during the execution of code on the graphics card, it is possible that cuda is in a state, where it needs a reset. In this case the first thing to try is the matlab command ‘clear classes’, which will reload the cuda class and force cuda to initialize on the next cuda call. If this does not work, one will have to quite Matlab and restart it.

# Supported Datatypes

Currently only the datatypes single and single complex are fully supported by CudaMat. This means that in the current version all computations in double are simply performed at single precision. This results in a loss in precisions, which is sometimes not acceptable in an application. Future versions will support more datatypes (e.g. int datatypes). Currently the cuda libraries (and in part the hardware) often also just supports single precision computations.

# How can I change the behavior of certain operations in CudaMat?

Currently there are very few possibilities to influence the behaviour of CudaMat. However, it is planned that the following can be influenced by global environment variables in the future:

* adjusting (optimizing) the threading parameters for the cuda code, by entering the number of processors that the code should assume. Also other optimisation parameters can be set.
* Defining whether the commands double() and single() will convert cuda objects back to Matlab objects or not.
* Defining the behavior subasgn should be executed (optimized or compatible)
* Control whether a warning should be printed when automatic conversions to cuda objects are performed.

# Interfacing with DipImage

CudaMat is designed to be compatible with standard Matlab objects as well as objects of the dipimage datatyp. This does not mean that DipImage needs to be installed. If no version of DipImage is installed, all objects are simply of Matlab origin (object.fromDip=false).
DipImage is an image processing toolbox from Delft university (see www.diplib.org) which can be obtained free of charge for the academic community.
This compatibility could be achieved by having the datatype cuda remember where each object came from using a tag ‘fromDip’ within each object. However, currently only very basic operations of DipImage are supported within CudaMat.

# Known incompatibilities

Matlab subassign operations such as ‘b=a;a(3:5,7:10)=10’ would change the variable b in the current version. The reason for this is that by simply changing the object ‘a’ the code currently avoids an extra copy and delete operation as it simply performs the subassign. However, if another identical copy of the object exists this object ‘b’ will be modified too (contrary to standard Matlab code), as Matlab is tricked in avoiding the extra copy operation.

# Additional CudaMat operations not present in standard Matlab

Many of the dipimage operations are implemented also for the cuda datatype when imported from a standard matlab object.
E.g. ft and ift perform fft and fft shift operations

# The really big speedup: Implementing your own Cuda function

If you type
edit applemantest.m
and look at the code, you get an idea, about how to really speed up the code. The essential bit is to write a small pice of C-style code which is automatically wrapped up by CudaMat into its own function that can then be called. This is possible for a number of standard functions.
The two essential commands which do the magic are:
“cuda_define” and “cuda_compile_all”. The former defines a new cuda function with its own name and a program code as given by a string. Then many such definitions can be collected and finally the cuda_compile_all command wraps them all up in the correct ways and compiles them such that they can be called from within matlab simply by their given name.
However, the programming of such new functions has to observe certain rule as described in the on-the-fly-programming-guide.

# Known errors / incompatibilities

* sum, min and max for arrays always sum over all elements in CudaMat. This has to be changed to be compatible with standard Matlab code (partial sums) and the possibility in DipImage to sum over arbitrary dimensions.
* for loops assigning vectors do not work (e.g. :  for q=cuda([1 2 3 4 5 4 3 2 1]);fprintf(‘Hello Wold\n’);end  would not produce the same result as standard matlab code)
* as CudaMat works always with floating point datatypes, certain kind of operations (integer division) and overflow errors (e.g. for byte datatype in dipimage) are not supported.

# The internal structure of CudaMat

CudaMat is based on the cuda datatype. All the methods operating on this datatype are stored in the @cuda folder and other methods (which also do something for other datatypes) are stored outside in the main CudaMat folder.
A cuda object stores a reference (myobject.ref) and the information whether it should be treated according to Matlab or DipImage conventions (myobject.fromDIP). The cuda functions are either taken direction from the Cuda fft and CuBlas libraries or are written in CUDA (all in the file cudaArith.cu). The mex file cuda_cuda.c is a frontend to cuda which supports all the functionalilty. The main mex function in this file is invoked always with a command string, telling it which command to execute. At the moment this sting is parsed simply by a daisy chain of strcmp operations. As the number of commands has grown, this might eventually present an unacceptable overhead, but I believe at the moment it should still not pose a problem.
This interface should make it comparably easy to adapt the code for working under Octave, Mathematica or in fact any other interpreter driven language.
How to obtain

Download the current version as a tar-gzip-file with all the necessary classes and an example html-file in it. Just place the CudaMat folder somewhere, add it to the Matlab path and call initCuda (see installation details below). Depending on the operation system it may be necessary to recompile the modules cudaArith.cu and cuda_cuda.c. A makefile for unix environment is provided.

For CudaMat you will need  NVidia’s cuda library  installed on your operating system and a graphics card which can run cuda programs (above GeForce 8800).
This software is released under the GPL2 license. It can be used for non-commercial purposes.

# Installation instructions

CudaMat can be installed in two different ways. The easy way is, if there is no need to modify any cuda code. You can simply download the newest version of CudaMat and unzip it. It will contain a folder called “user64bitCuda6VC11” or similar.
This folder has to be copied to the temp file location as obtained by typing “tempdir” in you Matlab installation and renamed to “user”. This directory will be user-specific.
Then only a Cuda Runtime library needs to be installed corresponding to the Cuda version in the filename and possibly C-runtime libraries corresponding to the C-version in the filename.

However, it should be noted, that this does not give you the capability of recompiling code or introducing user-defined cuda funtions. Thus you do not get the full benefit of CudaMat but should be able to run some fast code anyway.

# Installation instructions (64 Linux system)

Download the current version into a folder /usr/local/CudaMat/ and unpack it with tar -xzf CudaMat.tgz .
NVidia’s cuda driver and toolkit needs to be installed according to the manufacturer’s instruction. Make sure this is really the version corresponding to
the Cuda Toolkit.

To leave the X-window system under SuSe Linux, log off and the click on “menu” and select Console. The in the console (as superuser) you can run the driver installation program.

Edit the file “.profile” in your user home directory and add the lines:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/MATLAB/dip/Linuxa64/lib:/usr/local/cuda/lib64:/usr/local/cula/lib64:/usr/lib64:/usr/lib
export PATH=$PATH:/usr/local/cuda/bin

export CULA_ROOT=/usr/local/cula/
export CULA_INC_PATH=/usr/local/cula/include
export CULA_BIN_PATH_64=/usr/local/cula/
export CULA_LIB_PATH_64=/usr/local/cula/lib64
export CULA_BIN_PATH_32=/usr/local/cula/
export CULA_LIB_PATH_32=/usr/local/cula/lib

Install CULA (needs a free registration) from http://www.culatools.com/ to add support for the matlab “svd” and equation system solving commands.

To fix a problem with mex compilation in Matlab, modify the file
/usr/local/matlab2010a/bin/.matlab7rc.sh
and modify LDPATH_PREFIX to
LDPATH_PREFIX=’/usr/lib64′
in all theachitechure configurations.

Edit
~/.matlab/R2016b/mex_C_glnxa64.xml
(or in older versions: /usr/local/MATLAB/R2010a/bin/gccopts.sh)
and delete all occurances of “-ansi” to avoid compilation problems with C++ style comments.
Type
mex -setup
as a standart user in Matlab, to copy the above change into the local user directory

If compiling with mex inside matlab (after restart of matlab) still does not work, it might have to be done outside Matlab, since Matlab uses a wrong LD_LibraryPath the same mex command works also outside.

In some versions of Matlab the following links need to be created:

su
cd /usr/lib64
ln -s libGLU.so.1 libGLU.so
ln -s libX11.so.6 libX11.so
ln -s libXi.so.6 libXi.so
ln -s libXmu.so.6 libXmu.so
ln -s libglut.so.3 libglut.so
ln -s libcuda.so.1 libcuda.so
exit

In some Matlab versions it needs to know about the library. If matlab is installed in /usr/local/matlab type:
su

cd /usr/local/matlab/bin/glnxa64/
ln -s /usr/local/CudaMat/libcudaArith.so

exit

The commands for compilation under Matlab are
system(‘nvcc -c cudaArith.cu -I/usr/local/cuda/include/’)

and

mex cuda_cuda.c cudaArith.o -I/usr/local/cula/include -I/usr/local/cuda/include -L/usr/local/cula/lib64 -L/usr/local/cuda/lib64 -lcublas -lcufft -lcudart -lcula

with appropriately modified -I and -LC paths from the cuda and cula installation.

For more details on the setup and testing see Windows 64 bit installation below.

# Installation instructions (Windows 32 bit system)

add the path of the (visual studio) cl.exe comiler into PATH (windows -> home, or right click computer)
NVidia’s cuda library and SDK needs to be installed according to the manufacturer’s instruction.
compile under Matlab: Change to the directory where CudaMat was downloaded to, e.g.:
cd c:\Pro’gram Files’\dip\CudaMat\

Compile the cuda part of the program using NVidia’s nvcc compiler:
system(‘nvcc –compile cudaArith.cu’)
mex -setup
mex cuda_cuda.c cudaArith.obj -Ic:\CUDA\include\ -LC:\CUDA\lib -lcublas -lcufft -lcuda -lcudart
See if the installation was successful by typing in matlab:
applemantest(1);

For more details on the setup and testing see Windows 64 bit installation below.

# Installation instructions (Windows 64 bit system)

– Install VC++ Express and Windows SDK: Visual Studio does not come with 64-bit compiler (not quite sure) and 64-bit libraries (for sure). You have to obtain the windows SDK for your OS which provides the 64-bit libraries, headers, and the compiler. Ensure that 64-bit packages are selected when installing Windows SDK.
If installing on Visual C++ 2017 Cummunity edition, make sure that you select to also install the 
"Toolset für VC++ 2017, Version 15.4 v14.11", since cuda 9.1 is not compatible with any newer complier versions.

VC++ Express: http://www.microsoft.com/express/Downloads/#2010-Visual-CPP
Windows SDK: http://msdn.microsoft.com/en-us/windows/bb980924.aspx

– Install CUDA: There are three things to install, all available from http://www.nvidia.com/content/cuda/cuda-downloads.html
Download and install development version of NVIDIA drivers, CUDA Toolkit, CUDA SDK. Current version is 9.1

– Install CudaMat as described above. To be able to use cudamat one needs to compile the custom library cudaArithmatic.obj (with nvcc) and the mex file cuda_cuda.mexw64 (with mex). Precomiled version might possibly work, but not guaranteed (due to mismatch of systems).

Configuration of mex and MatLab:
> mex -setup
Works well iff VC++ and Windows SDK are installed and the 64-bit compiler (cl.exe) is visible on the system PATH.

If you have not installed and set up Cula you should add the following lines to your startup.m file:
addpath(‘/usr/local/CudaMat/’);
initCuda();

If you do not want to by default create ones (using “ones_cuda”), zeros (using “zeros_cuda”),  you should change these places in the code by replacing the matlab function “ones” with “ones_cuda” and “zeros” with “zeros_cuda”. See ones_cuda, zeros_cuda for more detail.

The dipimage generator functions “newimage”, “xx”,”yy”,”zz”, “rr”, “phiphi” are overwritten by CudaMat. By default they now generate cuda output. However this behaviour (and also of “ones_cuda” and “zeros_cuda”) can invidually be controlled by the global variables:
use_zeros_cuda=1; use_ones_cuda=1; use_newim_cuda=1; use_newimar_cuda=1; use_xyz_cuda=1;

Configuration of nvcc:
Trying to compile the cuda file (e.g. by going to the cuda directory and executing “applemantest(2)” you will get the error:
nvcc fatal : Visual Studio configuration file ‘(null)’ could not be found….”
This can be fixed by creating a file named
C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\vcvars64.bat
with the only text in it:
CALL setenv /x64
which you can also download here.see also
http://stackoverflow.com/questions/8900617/how-can-i-setup-nvcc-to-use-visual-c-express-2010-x64-from-windows-sdk-7-1

Testing the installation
You should go to the CudaMat installation directory and type
applemantest(1)
After about 6 seconds you should have a nice image in front of you.
If the compilation is installed all correctly you can type
applemantest(2)
which will first recompile but then yield a result in a few milliseconds. Running it again will make it even faster.

# Bug reports

If you find any bugs, please send them to me under heintzmannd at gmail dot com stating the system you were using as well as the version of CudaMat. Please put ‘CudaMat bug’ in the subject line.

# History of CudaMat and Acknowledgements

CudaMat started with the incentive to write faster deconvolution software for microscopy image processing. Using the fft code provided by NVidia, it quickly became clear that something more general would be useful and the idea of CudaMat was born. CudaMat was written by Rainer Heintzmann with discussions and contributions from Martin Kielhorn, Kai Wicker, Wouter Caarls, Bernd Rieger and Keith Lidke.

# Recent changes:

* The first version V 1.0.0beta was started around November 2008 and finished March 2009.
* V 1.0.1beta , bug fixes, added newim overload and complex function.
* V 1.0.2beta , bug fixes, added repmat and assignment and referencing with mask images (subsref and subsasgn) and dip_fouriertransform.
* V 1.0.3beta, bug fixes, partial reduction functions (such as [m,mm]=max(cuda(readim(‘chromo3d’)),[],3) ) fully supported now. Also sum, max and min have now correct performance for Matlab type arrays. Functions phase and angle were added. The functions zeros(), ones() and newim() were renamed to zeros_cuda(), ones_cuda() and newim_cuda() due to conflicts with the native code of dipimage and Matlab.
* V 1.0.4beta, made the file cuda_cuda.c compatible with older style ANSI C, as it would previously not compile under some compilers which require declarations at the beginning of a block.
* V 1.0.5beta, a few bug fixes. Introduced the first version of on-the-fly compilation (commands: ‘cuda_define’ and ‘cuda_compile_all’) for new cuda functions and included an impressive example (speedup 54000) by the command appleman(2)
* V 1.0.6beta, bug fixes. Added support for CULA, the cuda lapack library, which needs to be installed. svd and equation system solving (“\” and “/”, i.e. mldivide and mrdivide). Binary function on-the-fly compilation is now possible. Updated installation instructions and web page.
* V 1.0.7beta, bug fixes. Added functions (e.g. circshift). Improved the performance significantly by using an internal heap. Half-complex ffts are now available (“rft” and “rift”). They are fast and memory-efficient. Deconvolution toolbox now works with cudaMat. Now available as a zip file.
* V 1.0.8beta, bug fix.
* V 1.1.0beta, bug fixes (especially memory bug for reduce operations in older versions). New generator functions xx, yy, zz, rr and phiphi. These are now overloaded DIPImage functions. The same holds for newim and newimar, which are from now on (sorry for no backward compatibility here!) overloaded. Funktions “disableCuda()” and “enableCuda()” where introduced, which allow to easily switch off and on the use of cuda. New functions introduced (real and complex datatype): sin, cos, sinh, cosh. Also mpower (only partially implemented) was added. reshape bug was fixed and the function permute was implemented.
* V 1.1.1beta, bug fixes (plus a complex number was buggy adn the sum function had hickups). Introduced the rfftshift and rifftshift functions.
* V 1.1.2beta, bug fixes. Introduced “initCuda()” function, which should be started in the startup.m file. disableCuda() and enableCuda() allow easy turn on and turn off of CudaMat.
* V 1.1.3beta, bug fixes (especially the subassign function). The cuda_compile_all() function now uses the local temp directory to store the user-defined cuda sources and compiled results. This avoids clashes on multi-user systems. RFT (real valued fast Fourier transforms) support was added.
* V 1.1.4beta, bug fixes (the compilation in the temp directory did not work correctly). For speed reasons a “user” directory is created in the temp folder, in which all the additional user-defined compiled versions and .m files are placed.
* V 1.1.5beta, bug fixes (the ffts had a bug and the plans exhausted too quickly for some applications. Bugs in the multi-user capability using the temp folder to store the user-defined code and the executables were fixed. GitHub was introduced)
* V 1.1.6beta, bug fixes. The feature to avoid copy on write was now removed, as there were too many cases where this could cause trouble in nested function calls. Better handling of Cuda-Versions introduced in MatLab.
* V 1.1.7beta, bug fixes. xx and zz were updated.
* V 2.0.0beta, Major version change. CudaMat now supports python-style expansions for singleton dimension for binary functions of dip_image type input. Bug fixes. Mean projections of uneven sizes had a bug.
* V 2.1.0beta, Varous bug fixes and new functions. See changelog of GitHub for details.
* V 2.2.0beta, Now (close to) full support of the various matlab adressing schemes. Minor bug fixes

# Ongoing work / Future goals

More standard Matlab and DipImage functions should be supported. E.g.: Mean, var, rand, median, wavelet transformations, rfft and convolve
Where possible DipImage code which does not use DipLib can be used directly from CudaMat. A number of in-place operations are planned to allow more efficient programs to be written. E.g. ‘+=’ and alike should be implemented. These operations should also be implemented for standard Matlab and DipImage objects (for compatibility reasons). Even more special Multi-array operations could be useful.
A mechanism (via another datatype?) for parallelisation of Matlab loops could be introduced, which allows to profit from cuda even for operations involving small matrices).
Implement more Matlab and DipImage features in cuda. Most important are:

    Testing the software on different systems and GPUs.
    solving of equation systems. The current CUBLAS library unfortunately does not support this, as the Cholevski factorisation is not yet implemented. As soon as this changes equation system solving can be fully implemented in CudaMat. Currently the workaround is a conversion to standard matlab objects and back to cuda
    accessing elements via an index list should be implemented (e.g. a=[3 5 7];b=[1 2 3 4 5 6 7];b(a)). Not yet finnished, but subsref_vec function exists
    automatic decision to move small vectors and matrices back to standard matlab objects. The CudaMat overhead for smaller objects can be quite significant, so a global variable (max_cuda_size) might be useful to decide for automatic conversion back to standard matlab/DipImage.
    a faster implementation of the convolve operation and full support of half-complex transforms (implementation of this is started but not yet finnished).
    Implement more variations of on-the-fly cuda commands. Different types of functions and macros.

# Related Software

A package called Jacket by Accellereyes implements a CUDA based toolbox in the spirit of requiring minimal effort to change from standard Matlab code to code run on the GPU. The company XTech has developed GPULib.