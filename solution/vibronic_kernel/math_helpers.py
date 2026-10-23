"""Harmonic matrix kernels and native thermal units."""
import numpy as np
from scipy import linalg
from pymatgen.analysis.defects.constants import KB

def log_partition(frequencies,beta):
    x=beta*np.asarray(frequencies)
    return float(np.sum(-x/2-np.log(-np.expm1(-x))))

def _mode_kernel(w,s):
    u=s*w/2
    tangent=np.tanh(u)
    plus=w*tangent
    minus=w/tangent
    log_pref=np.sum(.5*np.log(w/np.pi)-s*w/2-.5*np.log(-np.expm1(-2*s*w)))
    return plus,minus,log_pref

def _kernel_direction(w,s,dw,ds):
    u=s*w/2;tangent=np.tanh(u);du=(s*dw+w*ds)/2
    dp=dw*tangent+w*(1-tangent*tangent)*du
    dm=dw/tangent-w*(1-tangent*tangent)/(tangent*tangent)*du
    dl=np.sum(.5*dw/w-.5*(s*dw+w*ds)/np.tanh(s*w))
    return dp,dm,dl

def _positive_real_logdet(matrix):
    # Re(matrix)>0: each eigenvalue lies in the open right half-plane.
    # Sum of their analytic logarithms retains determinant winding.
    values=linalg.eigvals(matrix)
    if values.real.min()<=0:raise RuntimeError('Lost the positive-real Gaussian domain')
    return np.log(values).sum()

