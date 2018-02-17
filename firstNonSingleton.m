function mydir=firstNonSingleton(in)
    mydir=find(size(in) > 1,1,'first');
