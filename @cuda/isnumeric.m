function res=isnumeric(avariable)
    res=(~avariable.fromDip) && (isempty(avariable.isBinary) || ~avariable.isBinary);
end