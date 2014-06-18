import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from weibull import weibull, weibull_mle

## FREQUENTISTS

def frequentist_guess(guess, i):
    return [g + i/10.0 for g in guess]

def frequentist_solve(xs, ys, log_prior_fcn, bounds=None, guess=(0.3, 1.0), ntries=20, quick=True):
    neg_log_posterior_fcn = lambda theta: -log_prior_fcn(theta) + weibull_mle(theta, xs, ys, unfold=False)
    sol = None
    ymin = 100000
    bounds = [(0, None), (0, None)] + ([(0.0, 1.0)] * (len(guess) - 2)) if bounds is None else bounds
    guesses = [frequentist_guess(guess, i) for i in xrange(ntries)]
    for guess in guesses:
        soln = minimize(neg_log_posterior_fcn, guess, method='TNC', bounds=bounds, constraints=[])
        if soln['success']:
            theta_hat = soln['x']
            if not quick and soln['fun'] < ymin:
                sol = theta_hat
            else:
                return theta_hat
    return sol

## BAYESIANS

def bayesian_solve(xs, ys, log_prior_fcn, guess=(0.3, 1.0)):
    return

## SIMULATION

def data_gen(xs, theta):
    return np.array([np.random.binomial(1, weibull(x, theta)) for x in xs])

def solve(xs, theta, log_prior_fcn, bounds):
    ys = data_gen(xs, theta)
    return frequentist_solve(xs, ys, log_prior_fcn, bounds), bayesian_solve(xs, ys, log_prior_fcn)

def log_uniform_prior(bounds):
    """
    n.b. frequentists need to bound the search space wherever prior is zero, to avoid infs
    """
    def pf(theta):
        return np.log(np.array([float(l <= t <= b) for t, (l, b) in zip(theta, bounds)])).sum()
    return pf

def simulate(N=5, n_per_condition=100):
    """
    Goal: simulate the results of fitting weibull, given priors on parameters, between two methods:
        * minimize unnormalized posterior
        * metropolis-hastings
    n.b. no bootstrapping!
    """
    xs0 = range(100)
    xs = np.array(xs0*n_per_condition)
    pts = []
    bounds = [(1.0, 4.0), (0.0, 2.0)]
    log_prior_fcn = log_uniform_prior(bounds)
    for i in xrange(N):
        if i % 10 == 0:
            print i
        theta = (3, 1)
        fth, bth = solve(xs, theta, log_prior_fcn, bounds)
        pts.append((theta, fth, bth))
    return np.array(pts)

def plot(pts, i=0):
    error_fcn = lambda th, t: th - t
    fths = zip(pts[:, 0], pts[:, 1])
    bths = zip(pts[:, 0], pts[:, 2])

    zs = [(x[i], error_fcn(y[i], x[i])) for x, y in fths if y is not None]
    if len(zs) > 0:
        plt.scatter(*zip(*zs), color='c', label='frequentists')

    zs = [(x[i], error_fcn(y[i], x[i])) for x, y in bths if y is not None]
    if len(zs) > 0:
        plt.scatter(*zip(*zs), color='r', label='bayesians')

    plt.show()

def main():
    pts = simulate()
    plot(pts)

if __name__ == '__main__':
    main()
