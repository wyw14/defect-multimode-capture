"""Native Cartesian / isotope response using an exact internal kinetic metric.

Relative coordinates remove acoustic translation without differentiating arbitrary
phonon eigenvectors. Full matrix harmonic kernels keep mode degeneracies regular.
This is still an author prototype, not a Harbor/model score.
"""
import numpy as np
from scipy import linalg
from .native_geometry import KAPPA,native_geometry



def prepare_relative(initial,final,force_initial,force_final,w0,gradient,quadratic,
                     masses=None,queries=(),anchor=None):
    # Native geometry validates chemistry, identity, periodic images and acoustic constraints.
    metadata=native_geometry(initial,final,force_initial,force_final,w0,gradient,quadratic,masses=masses)
    n=len(initial);anchor=n-1 if anchor is None else anchor
    others=np.array([i for i in range(n) if i!=anchor]);dim=3*(n-1)
    if len(others)!=n-1:raise ValueError('Invalid internal coordinate anchor')
    mass=np.array([float(x.specie.atomic_mass) for x in initial]) if masses is None else np.asarray(masses,float)
    indices=(3*others[:,None]+np.arange(3)).ravel()
    embed=np.eye(3*n)[:,indices]
    order=np.array(metadata['matched_final_indices']);axes=(3*order[:,None]+np.arange(3)).ravel()
    ki=np.asarray(force_initial,float);kf=np.asarray(force_final,float)[np.ix_(axes,axes)]
    gri=embed.T@np.asarray(gradient,float);bri=embed.T@np.asarray(quadratic,float)@embed
    kri=embed.T@ki@embed;krf=embed.T@kf@embed
    kinetic=np.kron(np.diag(1/mass[others])+np.ones((n-1,n-1))/mass[anchor],np.eye(3))
    values,vectors=linalg.eigh(kinetic);root=(vectors*np.sqrt(values))@vectors.T
    inverse=linalg.solve(root,np.eye(dim),assume_a='pos')
    delta=metadata['periodic_displacement'];relative=(delta[others]-delta[anchor]).ravel()
    p={'force_i':KAPPA*root@kri@root,'force_f':KAPPA*root@krf@root,
       'shift':inverse@relative/np.sqrt(KAPPA),'w0':w0,
       'linear':np.sqrt(KAPPA)*root@gri,'quadratic':KAPPA*root@bri@root}
    directions=[]
    for query in queries:
        dm=np.asarray(query.get('masses',np.zeros(n)))
        dgmatrix=np.kron(np.diag(-dm[others]/mass[others]**2)-np.ones((n-1,n-1))*dm[anchor]/mass[anchor]**2,np.eye(3))
        droot=linalg.solve_sylvester(root,root,dgmatrix);dinverse=-inverse@droot@inverse
        dki=embed.T@np.asarray(query.get('force_initial',np.zeros((3*n,3*n))))@embed
        rawfinal=np.asarray(query.get('force_final',np.zeros((3*n,3*n))))
        dkf=embed.T@rawfinal[np.ix_(axes,axes)]@embed
        dg=embed.T@np.asarray(query.get('gradient',np.zeros(3*n)))
        db=embed.T@np.asarray(query.get('quadratic',np.zeros((3*n,3*n))))@embed
        dpos=(np.asarray(query.get('final_cartesian',np.zeros((n,3))))[order]
              -np.asarray(query.get('initial_cartesian',np.zeros((n,3)))))
        dr=(dpos[others]-dpos[anchor]).ravel()
        directions.append({'force_i':KAPPA*(droot@kri@root+root@dki@root+root@kri@droot),
                           'force_f':KAPPA*(droot@krf@root+root@dkf@root+root@krf@droot),
                           'shift':(dinverse@relative+inverse@dr)/np.sqrt(KAPPA),
                           'linear':np.sqrt(KAPPA)*(droot@gri+root@dg),
                           'quadratic':KAPPA*(droot@bri@root+root@db@root+root@bri@droot),
                           'w0':query.get('w0',0.),'temperature':query.get('temperature',0.),
                           'energies':query.get('energies',0.),'width':query.get('width',0.)})
    metadata['initial_mode_energies']=np.sqrt(linalg.eigvalsh(p['force_i']))
    metadata['final_mode_energies']=np.sqrt(linalg.eigvalsh(p['force_f']))
    return p,directions,metadata

