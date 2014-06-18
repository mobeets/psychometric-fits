import numpy as np
from scipy.optimize import minimize

def params(theta, unfold=False):
    """
    a is scale
    b is shape
    minV is lower asymptote [default = 0.0 or 0.5]
    maxV is upper asymptote [default = 1.0]
    """
    if len(theta) == 2:
        a, b = theta
        minV = 0.0 if unfold else 0.5
        maxV = 1.0
    elif len(theta) == 3:
        a, b, maxV = theta
        minV = 0.0 if unfold else 0.5
    elif len(theta) == 4:
        a, b, minV, maxV = theta
    else:
        raise Exception("params must be length 2, 3, or 4: {0}".format(theta))
    return a, b, minV, maxV

def weibull(x, theta, unfold=False):
    """
    """
    a, b, minV, maxV = params(theta, unfold)
    return maxV - (maxV-minV) * np.exp(-pow(x/a, b))

def inv_weibull(theta, y, unfold=False):
    """
    the function calculates the inverse of a weibull function
    with given parameters (theta) for a given y value
    returns the x value
    """
    a, b, minV, maxV = params(theta, unfold)
    return a * pow(np.log((maxV-minV)/(maxV-y)), 1.0/b)

def weibull_mle(theta, xs, ys, unfold=False):
    yh = weibull(xs, theta, unfold)
    logL = np.sum(ys*np.log(yh) + (1-ys)*np.log(1-yh))
    if np.isnan(logL):
        yh = yh*0.99 + 0.005
        logL = np.sum(ys*np.log(yh) + (1-ys)*np.log(1-yh))
    return -logL

def solve(xs, ys, unfold=False, guess=(0.3, 1.0), ntries=20, quick=True):
    neg_log_likelihood_fcn = lambda theta: weibull_mle(theta, xs, ys, unfold)
    sol = None
    ymin = 100000
    for i in xrange(ntries):
        if i > 0:
            guess = [g + i/10.0 for g in guess]
        bounds = [(0, None), (0, None)] + ([(0.0, 1.0)] * (len(guess) - 2))
        soln = minimize(neg_log_likelihood_fcn, guess, method='TNC', bounds=bounds, constraints=[])
        if soln['success']:
            theta_hat = soln['x']
            if not quick and soln['fun'] < ymin:
                sol = theta_hat
            else:
                return theta_hat
    return sol
