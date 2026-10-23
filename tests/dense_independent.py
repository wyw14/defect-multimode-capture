"""Independent 15-mode spectral convolution with dense native force tensors.

Hidden intrinsic 1/2-mode Hamiltonians are integrated through explicit Hermite
wavefunctions. Energy bins are exactly commensurate, not a numerical histogram.
P/Q/R convolutions preserve interference of additive non-Condon block operators.
No Gaussian propagator, matrix heat kernel or candidate/reference import occurs.
"""
import itertools
import numpy as np
from scipy import linalg
from pymatgen.core import Lattice,Structure
from pymatgen.analysis.defects.constants import HBAR_EV,EV2J,AMU2KG,ANGS2M,KB
KAPPA=HBAR_EV**2*EV2J/(AMU2KG*ANGS2M**2)


def make_case():
    rng=np.random.default_rng(13007);atoms=6
    species=['Si','O','Si','O','Si','O'];cart=np.array([[.7+1.1*i,1.+.2*(i%3),1.4+.3*(i%2)] for i in range(atoms)])
    initial=Structure(Lattice.cubic(12.),species,cart,coords_are_cartesian=True,site_properties={'atom_id':['atom-'+str(i) for i in range(atoms)]})
    mass=np.array([float(s.specie.atomic_mass) for s in initial]);rm=np.repeat(np.sqrt(mass),3)
    free=linalg.null_space((rm[:,None]*np.tile(np.eye(3),(atoms,1))).T)
    orth,_=np.linalg.qr(rng.normal(size=(15,15)));physical=free@orth
    blocks=[]
    for j in range(5):
        theta=.31+.17*j;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        b=rng.normal(size=(2,2))*.00006;b=(b+b.T)/2
        blocks.append({'i':np.array([10+2*j,20+2*j]),'f':np.array([12+2*j,22+2*j]),'rotation':rotation,
                       'shift':rng.normal(size=2)*.4,'linear':rng.normal(size=2)*.001,'quadratic':b})
    for j in range(5):
        blocks.append({'i':np.array([14+2*j]),'f':np.array([16+2*j]),'rotation':np.eye(1),'shift':rng.normal(size=1)*.4,'linear':rng.normal(size=1)*.001,'quadratic':np.array([[rng.normal()*.00006]])})
    wi=np.concatenate([b['i'] for b in blocks])*.005;wf=np.concatenate([b['f'] for b in blocks])*.005
    rotate=linalg.block_diag(*[b['rotation'] for b in blocks]);shift=np.concatenate([b['shift'] for b in blocks]);v=np.concatenate([b['linear'] for b in blocks]);b=linalg.block_diag(*[b['quadratic'] for b in blocks])
    ki=rm[:,None]*(physical@np.diag(wi*wi)@physical.T)*rm[None,:]/KAPPA
    kf=rm[:,None]*(physical@rotate.T@np.diag(wf*wf)@rotate@physical.T)*rm[None,:]/KAPPA
    delta=np.sqrt(KAPPA)/rm*(physical@rotate.T@shift)
    final=Structure(initial.lattice,species,cart+delta.reshape((atoms,3)),coords_are_cartesian=True,site_properties=initial.site_properties)
    gradient=rm*(physical@v)/np.sqrt(KAPPA);quadratic=rm[:,None]*(physical@b@physical.T)*rm[None,:]/KAPPA
    return {'initial':initial,'final':final,'ki':ki,'kf':kf,'coupling':{'constant':.004,'linear':gradient,'quadratic':quadratic},
            'temperature':250.,'energies':np.array([-.12,-.065,-.01,.035,.075,.12,.18,.25]),'width':.018,'masses':None,'queries':[],
            'intrinsic_blocks':blocks,'base_masses':mass,'cartesian_shift':delta.reshape((atoms,3)),
            'scales':{'mass':1.,'force':1.,'geometry':1.,'linear':1.,'quadratic':1.}}


def local_moments(block,temperature,cut,scale):
    n=len(block['i']);unit=.005*np.sqrt(scale['force']/scale['mass']);wi=block['i']*unit;wf=block['f']*unit
    j=block['rotation'];d=block['shift']*np.sqrt(scale['mass'])*scale['geometry']
    v=block['linear']*scale['linear']/np.sqrt(scale['mass']);b=block['quadratic']*scale['quadratic']/scale['mass']
    nodes=n*(cut-1)+2;x,w=np.polynomial.hermite.hermgauss(nodes)
    stiffness=np.diag(wi)+j.T@np.diag(wf)@j;source=j.T@(wf*d);center=np.linalg.solve(stiffness,source)
    vals,vecs=np.linalg.eigh(stiffness);transform=(vecs/np.sqrt(vals))@vecs.T*np.sqrt(2)
    grid=np.array(list(itertools.product(range(nodes),repeat=n)));points=center+x[grid]@transform.T;weights=np.prod(w[grid],axis=1);final=points@j.T-d
    def wave(arguments):
        h=np.empty((n,cut,len(points)));h[:,0]=1.;h[:,1]=np.sqrt(2)*arguments.T
        for k in range(1,cut-1):h[:,k+1]=np.sqrt(2/(k+1))*arguments.T*h[:,k]-np.sqrt(k/(k+1))*h[:,k-1]
        states=np.array(list(itertools.product(range(cut),repeat=n)));values=np.ones((len(states),len(points)))
        for axis in range(n):values*=h[axis,states[:,axis]]
        return states,values
    ni,ai=wave(points*np.sqrt(wi));nf,af=wave(final*np.sqrt(wf))
    pref=np.prod((wi*wf)**.25/np.sqrt(np.pi))*np.sqrt(2)**n/np.sqrt(np.linalg.det(stiffness))*np.exp(-.5*(d@(wf*d)-source@center))
    operator=points@v+.5*np.einsum('pi,ij,pj->p',points,b,points)
    overlap=pref*(af*weights[None,:])@ai.T;amplitude=pref*(af*(weights*operator)[None,:])@ai.T
    beta=1/(KB*temperature);prob=np.prod(1-np.exp(-beta*wi))*np.exp(-beta*(ni@wi))
    origin=int((np.sum(block['f'])-np.sum(block['i']))//2);lo=-int(np.sum(block['i']))*(cut-1)+origin;hi=int(np.sum(block['f']))*(cut-1)+origin
    indices=(nf@block['f'])[:,None]-(ni@block['i'])[None,:]+origin-lo
    p=np.bincount(indices.ravel(),weights=(prob[None,:]*overlap*overlap).ravel(),minlength=hi-lo+1)
    q=np.bincount(indices.ravel(),weights=(prob[None,:]*overlap*amplitude).ravel(),minlength=hi-lo+1)
    r=np.bincount(indices.ravel(),weights=(prob[None,:]*amplitude*amplitude).ravel(),minlength=hi-lo+1)
    return p,q,r,lo


def convolved_kernel(case,cut=12):
    p=np.array([1.]);q=np.array([case['coupling']['constant']]);r=q*q;origin=0
    for block in case['intrinsic_blocks']:
        pb,qb,rb,lo=local_moments(block,case['temperature'],cut,case['scales'])
        pnew=np.convolve(p,pb)
        qnew=np.convolve(q,pb)+np.convolve(p,qb)
        rnew=np.convolve(r,pb)+np.convolve(p,rb)+2*np.convolve(q,qb)
        p,q,r=pnew,qnew,rnew;origin+=lo
    unit=.005*np.sqrt(case['scales']['force']/case['scales']['mass']);lines=unit*(np.arange(len(r))+origin);width=case['width']
    kernel=np.array([np.sum(r*np.exp(-.5*((e-lines)/width)**2))/(np.sqrt(2*np.pi)*width) for e in case['energies']])
    wi=np.concatenate([b['i'] for b in case['intrinsic_blocks']])*unit;wf=np.concatenate([b['f'] for b in case['intrinsic_blocks']])*unit;beta=1/(KB*case['temperature'])
    return {'kernel':kernel,'probability_mass':float(p.sum()),'operator_second_moment':float(r.sum()),'omega_initial':np.sort(wi),'omega_final':np.sort(wf),
            'log_Z_initial':float(-np.log(2*np.sinh(beta*wi/2)).sum()),'log_Z_final':float(-np.log(2*np.sinh(beta*wf/2)).sum())}


def queries(case):
    first={'masses':.01*case['base_masses'],'force_initial':.003*case['ki'],'force_final':.003*case['kf'],'final_cartesian':.02*case['cartesian_shift']}
    second={'temperature':.8,'energies':np.linspace(-.003,.004,len(case['energies'])),'width':.0008,'constant':.0003,'linear':.015*case['coupling']['linear'],'quadratic':.02*case['coupling']['quadratic']}
    return [first,second]


def shifted_case(case,direction,step):
    changed={**case,'coupling':dict(case['coupling']),'scales':dict(case['scales'])}
    if 'masses' in direction:changed['scales'].update(mass=1.+.01*step,force=1.+.003*step,geometry=1.+.02*step)
    if 'temperature' in direction:
        changed['temperature']+=direction['temperature']*step;changed['energies']=case['energies']+step*direction['energies'];changed['width']+=step*direction['width'];changed['coupling']['constant']+=step*direction['constant'];changed['scales'].update(linear=1.+.015*step,quadratic=1.+.02*step)
    return changed
