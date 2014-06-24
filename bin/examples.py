import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import exponweib, norm

from tools import color_list
from metrop_hastings import prune, metropolis_hastings, simulated_annealing, minimizer
from weibull import weibull, rand_weibull

def example_1():
    a, c = 3, 1
    xs = np.linspace(exponweib.ppf(0.01, a, c), exponweib.ppf(0.99, a, c), 100)
    l = round(max(xs) - min(xs))
    pf = lambda x: exponweib.pdf(x, a, c)

    e = 1 # step size of random-walk
    qrf = lambda x: norm.rvs(x, e)
    x0 = 2
    xhs0 = metropolis_hastings(x0, 100000, pf, qrf, None, None, False)
    xhs = prune(xhs0, l, e)

    plt.plot(xs, pf(xs), color='b', label='actual posterior')
    plt.hist(xhs, 100, color='c', normed=True, label='pruned m-h samples')
    plt.xlabel('x')
    plt.ylabel('normalized count')
    plt.legend()
    # plt.savefig('../img/example-1.png')
    plt.show()

def example_2():
    a, c = 3, 1
    data = exponweib.rvs(a, c, size=1000)
    l = round(max(data) - min(data))
    pf = lambda ah, ch=c, d=data: np.sum(np.log(exponweib.pdf(data, ah, ch)))

    e = 1 # step size of random-walk
    qrf = lambda x, e=e: norm.rvs(x, e)
    x0 = 2
    ahs0 = metropolis_hastings(x0, 100000, pf, qrf)
    # ahs = ahs0
    ahs = prune(ahs0, l, e)
    print 'Generated {0} samples after pruning from {1}.'.format(len(ahs), len(ahs0))
    print min(ahs), max(ahs)

    plt.axvline(a, label='theta', color='b', linestyle='--')
    plt.hist(ahs, 100, normed=True, label='pruned m-h samples')
    plt.legend()
    # plt.savefig('../img/example-2.png')
    plt.show()

def example_3():
    a, c = 3, 1
    data = exponweib.rvs(a, c, size=1000)
    l = round(max(data) - min(data))
    pf = lambda ah, ch=c, d=data: np.sum(np.log(exponweib.pdf(data, ah, ch)))

    d = 1.0
    Tf = lambda i: d/np.log(i+2) # cooling function

    e = 1 # step size of random-walk
    qrf = lambda x, e=e: norm.rvs(x, e)
    x0 = 2
    ahs0 = simulated_annealing(x0, 100000, pf, qrf, Tf)
    print 'Generated {0} samples. MAP estimate is {1}'.format(len(ahs0), ahs0[-1])

    plt.hist(ahs0, 100, normed=True, label='samples')
    plt.axvline(a, label='theta', color='b', linestyle='--')
    plt.axvline(ahs0[-1], label='theta-hat', color='c', linestyle='--')
    plt.xlim([2.8, 3.4])
    plt.legend()
    # plt.savefig('../img/example-3.png')
    plt.show()

def example_4():
    a, c = 3, 1
    data = exponweib.rvs(a, c, size=1000)
    pf = lambda ah, ch=c, d=data: -np.sum(np.log(exponweib.pdf(data, ah, ch)))
    x0 = 2
    sols = {}
    methods = ["nelder-mead", "powell", "Anneal", "BFGS", "TNC", "L-BFGS-B", "SLSQP"]
    for method in sorted(methods):
        th = {'success': False}
        while not th['success']:
            th = minimizer(x0, pf, method)
            print '{0} ({1}): {2}'.format(method, x0, th['message'])
            x0 = np.random.uniform(0, 10)
        sols[method] = th['x']
    for i, (method, sol) in enumerate(sols.iteritems()):
        print '{0}: {1}'.format(method, sol)

def log_likelihood(data, fcn, thetas):
    """
    data is array of [(x0, y0), (x1, y1), ...], where each yi in {0, 1}
    fcn if function, and will be applied to each xi
    thetas is tuple, a set of parameters passed to fcn along with each xi

    calculates the sum of the log-likelihood of data
        = sum_i fcn(xi, *thetas)^(yi) * (1 - fcn(xi, *thetas))^(1-yi)
    """
    # likelihood = lambda row: (1 - row[1]) + (2*row[1] - 1)*fcn(row[0], thetas)
    likelihood = lambda row: fcn(row[0], thetas) if row[1] else 1-fcn(row[0], thetas)
    log_likeli = lambda row: np.log(likelihood(row))
    val = sum(map(log_likeli, data))
    return val

def example_5_inner(method, theta, maxtries, pf, bounds):
    th = {'success': False}
    th0 = np.array(theta)
    i = 0
    while i < maxtries and not th['success']:
        th = minimizer(th0, pf, method, bounds, None, {'maxiter': 400})
        th0 = th0*np.random.uniform(0.95, 1.05)
        i += 1
    return th['x'] if th['success'] else None

def example_5_plot(sols, methods, xs0, theta, cmap):
    # plt.scatter(xs0, [np.sum([yc for xc, yc in data if xc == x])*1.0/nblocks for x in xs0])
    for i, method in enumerate(methods):
        color = cmap[method]
        for sol in sols[method]:
            plt.plot(xs0, weibull(xs0, sol), color=color, label=method)
            plt.axhline(sol[2], color=color, linestyle='--')
            plt.axhline(sol[3], color=color, linestyle='--')
    plt.plot(xs0, weibull(xs0, theta), linewidth=2, label='simulated')
    plt.axhline(theta[2], linestyle='--')
    plt.axhline(theta[3], linestyle='--')
    plt.xscale('log')
    # plt.legend()
    plt.show()

    for i, method in enumerate(methods):
        color = cmap[method]
        ss = zip(*sols[method])
        for i, s in enumerate(ss):
            plt.subplot(2, 2, i+1)
            plt.title('{0}, {1}'.format(method, ['scale', 'shape', 'lower-bound', 'upper-bound'][i]))
            plt.hist(s, color=color, label=method)
            plt.axvline(theta[i], linestyle='--')
        plt.tight_layout()
        plt.show()

def example_5():
    theta = [0.3, 0.9, 0.045, 0.96]
    # theta = [0.1, 0.5, 0.03, 0.98]
    xs0 = np.logspace(np.log10(0.001), np.log10(1.0), 20)
    nblocks = 50
    xs = xs0.repeat(nblocks)

    th0 = theta # [0.5, 0.2, 0.0, 1.0]
    APPROX_ZERO, APPROX_ONE = 0.00001, 0.99999
    bounds = [(APPROX_ZERO, None), (APPROX_ZERO, None), (APPROX_ZERO, APPROX_ONE), (APPROX_ZERO, APPROX_ONE)]
    methods = ["L-BFGS-B", "SLSQP"] # ["SLSQP", "TNC", "L-BFGS-B"]
    maxtries = 2 # 20
    nboots = 1000
    sols = dict((method, []) for method in methods)

    for i in xrange(nboots):
        if i % 10 == 0:
            print i
        data = zip(xs, rand_weibull(xs, theta))
        pf = lambda th, d=data: -log_likelihood(d, weibull, th)
        for method in methods:
            # print method
            sol = example_5_inner(method, theta, maxtries, pf, bounds)
            sols[method].append(sol)
    cmap = dict(zip(methods, color_list(len(sols)+2, 'Dark2')))
    example_5_plot(sols, methods, xs0, theta, cmap)

if __name__ == '__main__':
    # example_1()
    # example_2()
    # example_3()
    # example_4()
    example_5()
