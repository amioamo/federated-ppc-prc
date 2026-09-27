"""Bounded public-feature releases and background-aware private composition.

The caller owns data access and the privacy ledger. Random generators passed to
release_summary must be secret for a real private release; experiments here use
public data and reproducible seeds. No evaluator or private model is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, softmax


def orbit_frequencies(tokens: np.ndarray, k: int) -> np.ndarray:
    """Normalized masses for reverse-complement orbits; odd k only."""
    x = np.asarray(tokens)
    if x.ndim != 2 or x.dtype.kind not in 'iu' or np.any((x < 0) | (x > 3)):
        raise ValueError('expected integer DNA tokens [n,length]')
    if k < 1 or k % 2 != 1 or k > x.shape[1]:
        raise ValueError('k must be odd and no longer than the sequence')
    codes = np.arange(4**k); work = codes.copy(); rc = np.zeros_like(codes)
    for _ in range(k):
        rc = 4*rc + 3-work % 4; work //= 4
    _, lookup = np.unique(np.minimum(codes, rc), return_inverse=True)
    windows = x.shape[1]-k+1
    out = np.zeros((len(x), 4**k//2), dtype=np.float64)
    # Bound temporary storage for full training/public banks.
    for start in range(0, len(x), 512):
        batch = x[start:start+512]; ids = np.zeros((len(batch), windows), dtype=np.int32)
        for j in range(k):
            ids = 4*ids + batch[:,j:j+windows]
        rows = np.repeat(np.arange(len(batch)), windows)
        np.add.at(out[start:start+len(batch)], (rows, lookup[ids.ravel()]), 1/windows)
    return out


def gc_fraction(tokens: np.ndarray) -> np.ndarray:
    return ((tokens == 1) | (tokens == 2)).mean(axis=1)


def orbit_gc_direction(k: int) -> np.ndarray:
    codes=np.arange(4**k);work=codes.copy();rc=np.zeros_like(codes);gc=np.zeros_like(codes)
    for _ in range(k):
        a=work%4;gc+=(a==1)|(a==2);rc=4*rc+3-a;work//=4
    return gc[codes < rc]/k


@dataclass
class PublicFeatureMap:
    """An uncentered contraction; projection/bins must come from public data."""
    projection5: np.ndarray
    gc_edges: np.ndarray

    def __post_init__(self):
        if self.projection5.shape!=(512,32) or not np.isfinite(self.projection5).all():
            raise ValueError('invalid public 5-mer projection')
        if not np.allclose(self.projection5.T@self.projection5,np.eye(32),atol=1e-10):
            raise ValueError('the public projection must be orthonormal')
        if self.gc_edges.shape!=(3,) or not np.all(np.diff(self.gc_edges)>0) or not np.all((self.gc_edges>0)&(self.gc_edges<1)):
            raise ValueError('invalid public GC boundaries')

    @classmethod
    def fit_public(cls, tokens: np.ndarray) -> 'PublicFeatureMap':
        h=orbit_frequencies(tokens,5)
        covariance=h.T@h/len(h)-np.outer(h.mean(0),h.mean(0))
        _,v=np.linalg.eigh(covariance)
        projection=v[:,-32:]
        # Canonical signs make stored bases easier to compare across LAPACK runs.
        signs=np.sign(projection[np.argmax(abs(projection),axis=0),np.arange(32)])
        projection*=signs
        edges=np.quantile(gc_fraction(tokens),[.25,.5,.75])
        if len(np.unique(edges))!=3: raise ValueError('public GC quartiles coincide')
        return cls(projection,edges)

    def transform(self, tokens: np.ndarray, family: str) -> np.ndarray:
        h3=orbit_frequencies(tokens,3)
        if family=='k3': return h3/math.sqrt(2)
        if family!='multiscale': raise ValueError(f'unknown feature family {family}')
        return np.concatenate((.5*h3,.5*(orbit_frequencies(tokens,5)@self.projection5)),axis=1)

    def strata(self, tokens: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.gc_edges,gc_fraction(tokens),side='right')

    def gc_direction(self, family: str) -> np.ndarray:
        d=orbit_gc_direction(3)
        return math.sqrt(2)*d if family=='k3' else np.r_[2*d,np.zeros(32)]


def privacy_rho(epsilon: float, delta: float) -> float:
    if epsilon <= 0 or not 0 < delta < 1: raise ValueError('invalid privacy parameters')
    a=math.log(1/delta)
    return (epsilon/(math.sqrt(a+epsilon)+math.sqrt(a)))**2


def simplex_projection(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('invalid simplex input')
    u=np.sort(x)[::-1];css=np.cumsum(u)-1
    j=np.flatnonzero(u-css/np.arange(1,len(u)+1)>0)[-1]
    return np.maximum(x-css[j]/(j+1),0)


def release_summary(features: np.ndarray, strata: np.ndarray, *, conditional: bool,
                    epsilon: float | None, delta: float, rng: np.random.Generator) -> dict:
    """Release PublicFeatureMap outputs, whose proven diameter is at most one.

    The norm check below is necessary but does not certify the pairwise diameter
    of arbitrary externally supplied vectors. This API's calibration requires
    the uncentered feature maps defined in this module (or the same proved bound).
    """
    f=np.asarray(features,dtype=float);n,d=f.shape
    if n<1 or not np.isfinite(f).all() or np.linalg.norm(f,axis=1).max()>1/math.sqrt(2)+1e-9:
        raise ValueError('features violate the declared norm bound')
    rho=privacy_rho(epsilon,delta) if epsilon is not None else None
    if conditional:
        if strata.shape!=(n,) or not np.isin(strata,np.arange(4)).all(): raise ValueError('invalid strata')
        joint=np.stack([f[strata==g].sum(0)/n for g in range(4)])
        counts=np.bincount(strata,minlength=4)/n
        feature_sd=0. if rho is None else 1/n/math.sqrt(2*.8*rho)
        count_sd=0. if rho is None else math.sqrt(2)/n/math.sqrt(2*.2*rho)
        return {'mean':joint+rng.normal(0,feature_sd,joint.shape),
                'proportions':counts+rng.normal(0,count_sd,4),'n':n,'rho':rho,
                'feature_sd':feature_sd,'count_sd':count_sd,'conditional':True}
    sd=0. if rho is None else 1/n/math.sqrt(2*rho)
    return {'mean':f.mean(0)+rng.normal(0,sd,d),'n':n,'rho':rho,
            'feature_sd':sd,'count_sd':0.,'conditional':False}


def fit_energy(features: np.ndarray, mean: np.ndarray, reg: float = 1e-5,
               center: np.ndarray | None = None) -> tuple[np.ndarray,dict]:
    f=np.asarray(features,dtype=float);mean=np.asarray(mean,dtype=float)
    center=np.zeros(f.shape[1]) if center is None else np.asarray(center,dtype=float)
    def objective(theta):
        z=f@theta;p=softmax(z)
        delta=theta-center
        return logsumexp(z)-math.log(len(f))-theta@mean+.5*reg*(delta@delta), f.T@p-mean+reg*delta
    result=minimize(objective,center.copy(),jac=True,method='L-BFGS-B',
                    options={'gtol':1e-9,'ftol':1e-13,'maxiter':3000})
    grad=float(np.linalg.norm(objective(result.x)[1]))
    if not np.isfinite(result.x).all() or grad>1e-6:
        raise RuntimeError(f'energy fit failed: {result.message}; gradient {grad}')
    return result.x,{'gradient_norm':grad,'iterations':int(result.nit),'optimizer_success':bool(result.success),'regularization':float(reg)}


def conditional_weights(score: np.ndarray, strata: np.ndarray, mass: np.ndarray) -> np.ndarray:
    w=np.zeros(len(score))
    for g in range(len(mass)):
        mask=strata==g
        if not mask.any():
            if mass[g]>1e-12: raise ValueError('positive target mass on empty public stratum')
            continue
        w[mask]=mass[g]*softmax(score[mask])
    return w


def effective_count(w: np.ndarray) -> float:
    return float(1/(w@w))


def temper_within_strata(score: np.ndarray, strata: np.ndarray, mass: np.ndarray,
                         fraction: float) -> tuple[np.ndarray,float,bool]:
    if not 0 < fraction <= 1: raise ValueError('invalid effective-count floor')
    raw=conditional_weights(score,strata,mass);floor=fraction*len(raw)
    if effective_count(raw)>=floor: return raw,1.,True
    flat=conditional_weights(np.zeros_like(score),strata,mass)
    if effective_count(flat)<floor: return flat,0.,False
    beta=brentq(lambda b:effective_count(conditional_weights(b*score,strata,mass))-floor,0,1,xtol=1e-12)
    return conditional_weights(beta*score,strata,mass),float(beta),True


def correct_gc(score: np.ndarray, features: np.ndarray, means: list[np.ndarray],
               direction: np.ndarray) -> tuple[np.ndarray,float,float]:
    g=features@direction;w=softmax(score)
    components=np.clip([m@direction for m in means],0,1)
    target=float(np.clip(w@g,min(components),max(components)))
    lo,hi=float(g.min()),float(g.max())
    if hi-lo<1e-10: return w,0.,lo
    target=float(np.clip(target,lo+1e-9,hi-1e-9))
    def objective(gamma): return float(softmax(score+gamma*g)@g-target)
    a,b=-1.,1.
    while objective(a)>0:a*=2
    while objective(b)<0:b*=2
    gamma=brentq(objective,a,b,xtol=1e-11)
    return softmax(score+gamma*g),float(gamma),target


def fit_conditional(features: np.ndarray, strata: np.ndarray, released: list[dict],
                    prior_records: float = 50., reg: float = 1e-5,
                    noise_aware: bool = False) -> dict:
    """Fit conditional and global-score ablation with the same protected input."""
    f=features;public_mass=np.bincount(strata,minlength=4)/len(f)
    if (public_mass<=0).any(): raise ValueError('all public strata must be represented')
    public_mean=np.stack([f[strata==g].mean(0) for g in range(4)])
    site_theta=[];site_global=[];site_mass=[];diagnostics=[]
    for release in released:
        p=simplex_projection(release['proportions']);a=prior_records/release['n']
        stable_mass=(p+a*public_mass)/(1+a);site_mass.append(stable_mass)
        joint=(release['mean']+a*public_mass[:,None]*public_mean)/(1+a)
        global_t,global_diag=fit_energy(f,joint.sum(0),reg)
        theta=[]
        for g in range(4):
            penalty=reg;center=None
            if noise_aware:
                reference_sd=1/release['n']/math.sqrt(2*privacy_rho(3.,1e-6))
                penalty=reg*(1+(release['feature_sd']/(stable_mass[g]*reference_sd))**2)
                center=global_t
            t,diag=fit_energy(f[strata==g],joint[g]/stable_mass[g],penalty,center=center)
            theta.append(t);diagnostics.append(diag)
        site_theta.append(np.stack(theta))
        site_global.append(global_t);diagnostics.append(global_diag)
    theta=sum(site_theta);global_theta=sum(site_global);mass=np.mean(site_mass,axis=0)
    score=np.einsum('nd,nd->n',f,theta[strata]);global_score=f@global_theta
    offsets=np.array([math.log(mass[g]/public_mass[g])-logsumexp(score[strata==g])+math.log((strata==g).sum()) for g in range(4)])
    return {'theta':theta,'global_theta':global_theta,'mass':mass,'score':score,
            'global_score':global_score,'offsets':offsets,'diagnostics':diagnostics,
            'weights':conditional_weights(score,strata,mass),
            'global_weights':conditional_weights(global_score,strata,mass)}


def independent_mh(energy: np.ndarray, *, chains: int, steps: int, burn: int,
                   draws: int, rng: np.random.Generator) -> dict:
    """Consume one independently generated public proposal for each transition."""
    e=np.asarray(energy,dtype=float)
    if len(e)!=chains*(steps+1) or not 0 <= burn < steps or draws>chains*(steps-burn):
        raise ValueError('incompatible proposal/chain sizes')
    if not np.isfinite(e).all():raise ValueError('nonfinite energies')
    current=np.arange(chains);history=[];accepted=[]
    for step in range(steps):
        proposal=(step+1)*chains+np.arange(chains)
        accept=np.log(rng.random(chains))<np.minimum(0,e[proposal]-e[current])
        current=np.where(accept,proposal,current);history.append(current.copy());accepted.append(accept)
    history=np.asarray(history);accepted=np.asarray(accepted)
    return {'indices':history[burn:].ravel()[:draws], 'history':history,
            'accepted':accepted,'acceptance':float(accepted.mean()),
            'retained_acceptance':float(accepted[burn:].mean())}
