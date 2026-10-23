"""Multimode quantum capture kernel and complete native-input derivative."""
import numpy as np
from .physical_response import prepare_relative
from .matrix_kernel import spectrum


def _array(value,shape,name):
    raw=np.asarray(value)
    if np.iscomplexobj(raw) or raw.shape!=shape:
        raise ValueError('Invalid real shape for '+name)
    array=np.asarray(raw,float)
    if not np.all(np.isfinite(array)):
        raise ValueError('Nonfinite '+name)
    return array


def capture_kernel(initial,final,force_initial,force_final,coupling,temperature,energies,width,masses=None,queries=()):
    try:
        return _capture(initial,final,force_initial,force_final,coupling,temperature,energies,width,masses,queries)
    except (TypeError,KeyError,IndexError) as error:
        raise ValueError('Malformed physical input') from error


def _capture(initial,final,force_initial,force_final,coupling,temperature,energies,width,masses,queries):
    if not isinstance(coupling,dict) or set(coupling)!={'constant','linear','quadratic'}:
        raise ValueError('Invalid coupling fields')
    n=len(initial);size=3*n;energies=np.asarray(energies,float)
    if energies.ndim!=1 or not len(energies) or not np.all(np.isfinite(energies)):
        raise ValueError('Invalid energy grid')
    temperature=float(temperature);width=float(width);w0=float(coupling['constant'])
    if not np.all(np.isfinite([temperature,width,w0])) or temperature<=0 or width<=0:
        raise ValueError('Invalid thermal/width/coupling scalar')
    force_initial=_array(force_initial,(size,size),'force_initial');force_final=_array(force_final,(size,size),'force_final')
    gradient=_array(coupling['linear'],(size,),'linear coupling');quadratic=_array(coupling['quadratic'],(size,size),'quadratic coupling')
    if masses is not None:masses=_array(masses,(n,),'masses')
    allowed={'force_initial','force_final','initial_cartesian','final_cartesian','masses','constant','linear','quadratic','temperature','energies','width'}
    shapes={'force_initial':(size,size),'force_final':(size,size),'initial_cartesian':(n,3),'final_cartesian':(n,3),'masses':(n,),'linear':(size,),'quadratic':(size,size),'energies':energies.shape,'constant':(),'temperature':(),'width':()}
    names={'constant':'w0','linear':'gradient'};directions=[];translation=np.tile(np.eye(3),(n,1))
    for query in queries:
        if not isinstance(query,dict) or not set(query)<=allowed:raise ValueError('Invalid query fields')
        clean={names.get(key,key):_array(value,shapes[key],key) for key,value in query.items()}
        for key in ('force_initial','force_final','quadratic'):
            matrix=clean.get(key)
            if matrix is not None and (np.max(np.abs(matrix-matrix.T))>1e-10 or np.max(np.abs(matrix@translation))>1e-9):
                raise ValueError('Query matrix violates symmetry/acoustic invariance')
        if 'gradient' in clean and np.max(np.abs(translation.T@clean['gradient']))>1e-10:
            raise ValueError('Query linear coupling violates translation invariance')
        directions.append(clean)
    parameters,changed,metadata=prepare_relative(initial,final,force_initial,force_final,w0,gradient,quadratic,masses=masses,queries=directions)
    answer=spectrum(parameters,energies,temperature,width,changed)
    return {'kernel':answer['spectrum'],'kernel_response':answer['response'],
            'omega_initial':metadata['initial_mode_energies'],'omega_final':metadata['final_mode_energies'],
            'log_Z_initial':answer['log_partition_initial'],'log_Z_final':answer['log_partition_final']}
