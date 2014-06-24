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

def rand_weibull(x, theta):
    return np.random.binomial(1, weibull(x, theta))

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
