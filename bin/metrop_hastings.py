import numpy as np

def prune(xs, L, E):
    """
    If the largest length scale of the space of probable states is L,
        a metropolis-hastings method whose proposal distribution generates a random walk with step size E
        must be run for at least:
            T >= (L/E)^2
        iterations to obtain an independent sample.
    So for N independent samples you first need N*T samples
    """
    T = int(L**2/E**2)
    return xs[np.arange(T, len(xs), T)]

def prune_2(xs, nburn, nskip):
    """
    burns the first `nburn` samples, then skips every `nskip` samples after that
    """
    return xs[np.arange(nburn, len(xs), nskip)]

def mh_sa_base(x0, n, p_pdf_fcn, q_rand_fcn, a_fcn):
    """
    x0 is starting sample from posterior
    n is int, the number of iterations of sampling
    p_pdf_fcn(x) is probability of x occurring; can be evaluated can for any x
    q_rand_fcn(x) picks the next sample from a latent q_pdf_fcn, given current sample position x
    """
    x = x0
    xs = [x]
    px = p_pdf_fcn(x)
    c = 0
    for i, r in enumerate(np.random.uniform(size=n-1)):
        xstar = q_rand_fcn(x)
        pxstar = p_pdf_fcn(xstar)
        if r < a_fcn(i, x, px, xstar, pxstar):
            x = xstar
            px = pxstar
            c += 1
        xs.append(x)
    if c == 0:
        raise Exception("Never moved! Initial guess of {0} is probably too unlikely. Or, try logging your p_pdf_fcn.".format(x0))
    return np.array(xs)

def make_mh_rule(q_pdf_fcn, p_logged):
    def a_fcn(i, x1, px1, x2, px2):
        post = np.exp(px2 - px1) if p_logged else ((px2 / px1) if px1 != 0.0 else px2)
        rtrn = (q_pdf_fcn(x1, x2) / q_pdf_fcn(x2, x1)) if q_pdf_fcn is not None else 1.0
        return post*rtrn
    return a_fcn

def metropolis_hastings(x0, n, p_pdf_fcn, q_rand_fcn, q_pdf_fcn=None, p_logged=True):
    """
    q_pdf_fcn is proposal density
        q_pdf_fcn(x, xstar) is probability of returning to x given current position at xstar
        if q_pdf_fcn is None, this sampler assumes q_pdf_fcn (from which q_rand_fcn samples) is symmetric, e.g. gaussian
    p_logged is bool, specifying whether p_pdf_fcn has been logged

    [see mh_sa_base() for remaining parameters]

    n.b. must prune the returned samples to obtain independent sequence!
    n.b. this is random-walk hastings if q_pdf_fcn is dependent upon current position
    """
    a_fcn = make_mh_rule(q_pdf_fcn, p_logged)
    return mh_sa_base(x0, n, p_pdf_fcn, q_rand_fcn, a_fcn)

def make_sa_rule(p_logged, T_fcn):
    def a_fcn(i, x1, px1, x2, px2):
        return np.exp((px2 - px1)/T_fcn(i)) if p_logged else ((px2**T_fcn(i) / px1**T_fcn(i)) if px1**T_fcn(i) != 0.0 else px2)
    return a_fcn

def simulated_annealing(x0, n, p_pdf_fcn, q_rand_fcn, T_fcn, p_logged=True):
    """
    T_fcn is a cooling function, non-increasing on the domain of range(n), s.t. T_fcn(i) -> 0 as i -> inf
    p_logged is bool, specifying whether p_pdf_fcn has been logged

    [see mh_sa_base() for remaining parameters]
    """
    a_fcn = make_sa_rule(p_logged, T_fcn)
    return mh_sa_base(x0, n, p_pdf_fcn, q_rand_fcn, a_fcn)
