import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import exponweib, norm

from metrop_hastings import prune, metropolis_hastings

def example_1():
    a, c = 3, 1
    xs = np.linspace(exponweib.ppf(0.01, a, c), exponweib.ppf(0.99, a, c), 100)
    l = round(max(xs) - min(xs))
    pf = lambda x: exponweib.pdf(x, a, c)

    e = 1 # step size of random-walk
    qrf = lambda x: norm.rvs(x, e)
    xhs0 = metropolis_hastings(3, 100000, pf, qrf, None, None, False)
    xhs = prune(xhs0, l, e)

    plt.plot(xs, pf(xs), color='b', label='actual posterior')
    plt.hist(xhs, 100, color='c', normed=True, label='pruned m-h samples')
    plt.xlabel('x')
    plt.ylabel('normalized count')
    plt.legend()
    # plt.savefig('img/example-1.png')
    plt.show()

def example_2():
    a, c = 3, 1
    data = exponweib.rvs(a, c, size=1000)
    l = round(max(data) - min(data))
    pf = lambda ah, ch=c, d=data: np.sum(np.log(exponweib.pdf(data, ah, ch)))

    e = 1 # step size of random-walk
    qrf = lambda x, e=e: norm.rvs(x, e)
    ahs0 = metropolis_hastings(2, 100000, pf, qrf)
    # ahs = ahs0
    ahs = prune(ahs0, l, e)
    print 'Generated {0} samples after pruning from {1}.'.format(len(ahs), len(ahs0))
    print min(ahs), max(ahs)

    plt.hist(ahs, 100, normed=True, label='theta-hat')
    plt.axvline(a, label='theta', color='b', linestyle='--')
    plt.legend()
    # plt.savefig('img/example-2.png')
    plt.show()

if __name__ == '__main__':
    example_1()
    # example_2()
