"""Full matrix harmonic quantum trace with analytic Cartesian/mass-ready responses."""
import numpy as np
from scipy import linalg,integrate,special
from .math_helpers import KB,log_partition,_mode_kernel,_kernel_direction,_positive_real_logdet


def matrix_kernel(force,s,directions):
    eigenvalues,vectors=linalg.eigh(force)
    if eigenvalues.min()<=0:raise ValueError('Internal harmonic force matrix must be positive')
    frequencies=np.sqrt(eigenvalues)
    plus,minus,pref=_mode_kernel(frequencies,s)
    kp=(vectors*plus)@vectors.T;km=(vectors*minus)@vectors.T
    derivatives=[]
    for df,ds in directions:
        localized=vectors.T@df@vectors
        dw=np.diag(localized)/(2*frequencies)
        # Trace-log derivatives only require the diagonal in the current eigenbasis.
        _,_,dlog=_kernel_direction(frequencies,s,dw,ds)
        gap=eigenvalues[:,None]-eigenvalues[None,:]
        close=np.abs(gap)<=1e-7*np.maximum(np.maximum(eigenvalues[:,None],eigenvalues[None,:]),1e-12)
        mid=np.sqrt((eigenvalues[:,None]+eigenvalues[None,:])/2)
        fp,fm,_=_kernel_direction(mid,s,1/(2*mid),0.)
        lp=np.empty_like(gap,dtype=complex);lm=np.empty_like(gap,dtype=complex)
        np.divide(plus[:,None]-plus[None,:],gap,out=lp,where=~close)
        np.divide(minus[:,None]-minus[None,:],gap,out=lm,where=~close)
        lp[close]=fp[close];lm[close]=fm[close]
        dp,dm,_=_kernel_direction(frequencies,s,np.zeros_like(frequencies),ds)
        derivatives.append(((vectors*(dp))@vectors.T+vectors@(lp*localized)@vectors.T,
                            (vectors*(dm))@vectors.T+vectors@(lm*localized)@vectors.T,dlog,dw))
    return kp,km,pref,frequencies,derivatives


def correlation(parameters,time,temperature,directions=(),tilt=None):
    p=parameters;n=len(p['shift']);d=np.asarray(p['shift']);v=np.asarray(p['linear']);b=np.asarray(p['quadratic']);w0=p['w0']
    beta=1/(KB*temperature);alpha=beta/2 if tilt is None else tilt
    dbetas=[-beta*x.get('temperature',0.)/temperature for x in directions]
    initial=[(np.asarray(x.get('force_i',np.zeros((n,n)))),db/2 if tilt is None else db) for x,db in zip(directions,dbetas)]
    final=[(np.asarray(x.get('force_f',np.zeros((n,n)))),db/2 if tilt is None else 0.) for x,db in zip(directions,dbetas)]
    ip,im,il,wi,di=matrix_kernel(np.asarray(p['force_i']),beta-alpha-1j*time,initial)
    fp,fm,fl,wf,df=matrix_kernel(np.asarray(p['force_f']),alpha+1j*time,final)
    mp=ip+fp;mm=im+fm
    invp=linalg.solve(mp,np.eye(n),assume_a='sym');invm=linalg.solve(mm,np.eye(n),assume_a='sym')
    source=fp@d;mean=invp@source;same=(invp+invm)/2;cross=(invp-invm)/2
    exponent=(il+fl+n*np.log(2*np.pi)-.5*(_positive_real_logdet(mp)+_positive_real_logdet(mm))
              -d@fp@d+source@mean-log_partition(wi,beta))
    trace=np.exp(exponent)
    shifted=w0+v@mean+.5*mean@b@mean+.5*np.trace(b@same);slope=v+b@mean
    moment=shifted**2+slope@cross@slope+.5*np.trace(b@cross@b@cross)
    value=trace*moment;response=[]
    for x,dbeta,dx,dy in zip(directions,dbetas,di,df):
        dip,dim,dil,dwi=dx;dfp,dfm,dfl,_=dy
        dd=np.asarray(x.get('shift',np.zeros(n)));dv=np.asarray(x.get('linear',np.zeros(n)));db=np.asarray(x.get('quadratic',np.zeros((n,n))));dw0=x.get('w0',0.)
        dmp=dip+dfp;dmm=dim+dfm;dsource=dfp@d+fp@dd
        dmean=invp@(dsource-dmp@mean);dinvp=-invp@dmp@invp;dinvm=-invm@dmm@invm
        dsame=(dinvp+dinvm)/2;dcross=(dinvp-dinvm)/2
        dlogz=np.sum(-.5*(beta*dwi+wi*dbeta)/np.tanh(beta*wi/2))
        dexponent=(dil+dfl-.5*np.trace(invp@dmp+invm@dmm)-2*dd@fp@d-d@dfp@d
                   +2*dsource@mean-mean@dmp@mean-dlogz)
        dshift=(dw0+dv@mean+v@dmean+mean@b@dmean+.5*mean@db@mean+.5*np.trace(db@same+b@dsame))
        dslope=dv+db@mean+b@dmean
        dmoment=2*shifted*dshift+2*dslope@cross@slope+slope@dcross@slope+np.trace(db@cross@b@cross)+np.trace(b@dcross@b@cross)
        response.append(trace*(dmoment+moment*dexponent))
    return value,np.asarray(response,dtype=complex)

def spectrum(parameters,energies,temperature,width,directions=(),atol=2e-11):
    """Infinite quantum-state sum with normalized Gaussian energy broadening."""
    parameters={**parameters,'wi':np.sqrt(linalg.eigvalsh(parameters['force_i'])),'wf':np.sqrt(linalg.eigvalsh(parameters['force_f']))}
    energies=np.asarray(energies,float);q=len(directions);beta=1/(KB*temperature)
    alpha=0.5
    if not beta>alpha:raise ValueError('Thermal contour outside the supported positive domain')
    tilt=alpha*energies+.5*alpha*alpha*width*width
    shifted=energies+alpha*width*width
    pref=np.exp(tilt)
    dtilt=[];dshift=[]
    for direction in directions:
        dbeta=-beta*direction.get('temperature',0.)/temperature
        de=np.broadcast_to(np.asarray(direction.get('energies',np.zeros_like(energies))),energies.shape);dg=direction.get('width',0.)
        dtilt.append(alpha*de+alpha*alpha*width*dg)
        dshift.append(de+2*alpha*width*dg)
    dtilt=np.asarray(dtilt).reshape((q,len(energies)));dshift=np.asarray(dshift).reshape((q,len(energies)))
    def integrand(t):
        c,dc=correlation(parameters,t,temperature,directions,tilt=alpha)
        kernel=np.exp(-.5*(width*t)**2+1j*shifted*t)
        rows=[kernel*c]
        for k,direction in enumerate(directions):
            rows.append(kernel*(dc[k]+c*(-width*t*t*direction.get('width',0.)+1j*t*dshift[k])))
        return np.asarray(rows).real
    # Absolute tail estimate uses |C_alpha(t)| <= C_alpha(0), a positive spectral sum.
    c0=abs(correlation(parameters,0.,temperature,tilt=alpha)[0]);largest=max(float(pref.max(initial=1.)),1.)
    exponent=max(45.,np.log(max(c0,1.)*largest/max(atol,1e-300))+8)
    stop=np.sqrt(2*exponent)/width
    integral,error=integrate.quad_vec(integrand,0.,stop,epsabs=atol/(10*largest),epsrel=2e-10,limit=4000)
    values=pref*integral[0]/np.pi
    response=pref[None,:]*integral[1:]/np.pi+values[None,:]*dtilt
    return {'spectrum':values,'response':response,'quadrature_error_bound':float(error*largest/np.pi),
            'tail_bound':float(c0*largest*special.erfc(width*stop/np.sqrt(2))/(np.sqrt(2*np.pi)*width)),
            'log_partition_initial':log_partition(parameters['wi'],beta),
            'log_partition_final':log_partition(parameters['wf'],beta)}
