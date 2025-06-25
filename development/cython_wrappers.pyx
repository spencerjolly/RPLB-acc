cimport scipy.special.cython_special as sp
cimport numpy as np

cdef api erfc(  double in_real,double in_imag,
                double* out_real,double* out_imag):
    
    cdef double complex in_
    in_.real=in_real
    in_.imag=in_imag
    
    cdef double complex output=sp.erfc(in_)
    
    out_real[0]=output.real
    out_imag[0]=output.imag