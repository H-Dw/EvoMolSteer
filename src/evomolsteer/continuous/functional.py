"""Global polynomial curves and whole-batch uncertainty; never time-bin fits."""
import numpy as np
from numpy.polynomial import Legendre, Polynomial
from numpy.polynomial.legendre import legvander
from ..statistics import _draws


def quadrature(times):
    t = np.asarray(times,float)
    if len(t)<2 or np.any(np.diff(t)<=0): raise ValueError('Strictly increasing time grid required')
    dt = np.diff(t)
    w = np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]
    return w / (t[-1]-t[0])


def global_fit(times, values, cfg, degree=None):
    """Values are independent batch curves, [batch,time], on the WHOLE window.

    Complexities 0..max_degree are compared by leave-one-BATCH-out error on all
    times; the one-standard-error rule chooses the lowest degree. Conditional
    coefficient/curve uncertainty resamples complete independent batch curves.
    """
    t = np.asarray(times,float); y = np.asarray(values,float)
    if y.ndim!=2 or y.shape[1]!=len(t) or not np.isfinite(y).all():
        raise ValueError('Fitting requires complete finite independent batch curves')
    n = len(y); w = quadrature(t); u = 2*(t-t[0])/(t[-1]-t[0])-1
    max_degree = min(int(cfg.get('max_degree',5)),len(t)-2)
    vander = legvander(u,max_degree)
    allcoef, losses = [], []
    for d in range(max_degree+1):
        basis = vander[:,:d+1]
        projection = np.linalg.pinv(basis*np.sqrt(w[:,None]))*np.sqrt(w)[None,:]
        coefficients = y @ projection.T
        allcoef.append(coefficients)
        if n>1:
            loo = (coefficients.sum(0)[None,:]-coefficients)/(n-1)
            losses.append(np.sum((y-loo@basis.T)**2*w,axis=1))
        else: losses.append(np.full(n,np.nan))
    loss = np.asarray(losses); cv = loss.mean(1)
    if degree is None:
        best = int(np.argmin(cv))
        se = float(loss[best].std(ddof=1)/np.sqrt(n)) if n>1 else 0.
        chosen = int(np.flatnonzero(cv<=cv[best]+se+1e-15)[0])
    else:
        chosen = int(degree)
        if chosen<0 or chosen>max_degree: raise ValueError('Invalid frozen degree')
    coefficients = allcoef[chosen]; coef = coefficients.mean(0)
    model = Legendre(coef,domain=[t[0],t[-1]])
    fit = model(t); derivative=model.deriv()(t); second=model.deriv(2)(t)
    mean = y.mean(0)
    denom = np.sum((mean-np.sum(mean*w))**2*w)
    residual = np.sum((mean-fit)**2*w)
    r2 = float(1-residual/denom) if denom>1e-20 else (1. if residual<1e-20 else None)
    signs, bootstrap, exact = _draws(n,cfg['seed'],cfg['permutations'])
    # Global test uses the UNFITTED full curve; cancellation of positive and
    # negative effects in the window integral cannot hide a sign reversal.
    gram = (y*w) @ y.T
    null = np.sum((signs@gram)*signs,axis=1)
    observed = float(np.sum(mean*mean*w))
    hits = int((null>=observed-1e-14*max(observed,1e-14)).sum())
    p = hits/len(signs) if exact else (hits+1)/(len(signs)+1)
    draws = bootstrap@coefficients
    basis=vander[:,:chosen+1]; fitted_draws=draws@basis.T
    derivative_basis=np.column_stack([Legendre.basis(i,domain=[t[0],t[-1]]).deriv()(t) for i in range(chosen+1)])
    derivative_draws=draws@derivative_basis.T
    ci=np.quantile(fitted_draws,[.025,.975],axis=0)
    dci=np.quantile(derivative_draws,[.025,.975],axis=0)
    half=float(np.quantile(np.max(np.abs(fitted_draws-fit),axis=1),.95))
    roots=model.deriv().roots()
    roots=[float(r.real) for r in roots if abs(r.imag)<1e-10 and t[0]<r.real<t[-1]]
    integral=float(np.sum(mean*w)); change=float(mean[-1]-mean[0])
    coef_ci=np.quantile(draws,[.025,.975],axis=0)
    return {
        'degree':chosen,'n_batches':n,'n_times':len(t),'time_start':float(t[0]),'time_end':float(t[-1]),
        'legendre_coefficients':coef.tolist(),'power_coefficients_t':model.convert(kind=Polynomial).coef.tolist(),
        'coefficient_ci_low':coef_ci[0].tolist(),'coefficient_ci_high':coef_ci[1].tolist(),
        'cv_mse_by_degree':cv.tolist(),'cv_chosen_mse':float(cv[chosen]),
        'cv_constant_mse':float(cv[0]),'mean_curve_r2':r2,'mean_curve_rmse':float(np.sqrt(residual)),
        'global_curve_p':float(p),'window_mean_effect':integral,'end_minus_start':change,
        'turning_times':roots,'uncertainty':'whole-batch bootstrap conditional on discovery-selected degree',
        'curves': {'time':t,'observed_mean':mean,'fitted':fit,'derivative':derivative,'second_derivative':second,
            'ci_low':ci[0],'ci_high':ci[1],'simultaneous_low':fit-half,'simultaneous_high':fit+half,
            'derivative_ci_low':dci[0],'derivative_ci_high':dci[1]},
        'batch_coefficients':coefficients}


def evaluate_frozen(model, times):
    t=np.asarray(times,float)
    if t.min()<model['time_start']-1e-7 or t.max()>model['time_end']+1e-7:
        raise ValueError('No extrapolation outside observed selection window')
    return Legendre(model['legendre_coefficients'],domain=[model['time_start'],model['time_end']])(t)
