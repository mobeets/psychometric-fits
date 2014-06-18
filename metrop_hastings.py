from scipy.stats import exponweib, norm
import matplotlib.pyplot as plt
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

def make_mh_rule(q_pdf_fcn, p_logged):
    def a_fcn(x1, px1, x2, px2):
        post = np.exp(px2 - px1) if p_logged else ((px2 / px1) if px1 != 0.0 else px2)
        rtrn = (q_pdf_fcn(x1, x2) / q_pdf_fcn(x2, x1)) if q_pdf_fcn is not None else 1.0
        return post*rtrn
    return a_fcn

def metropolis_hastings(x0, n, p_pdf_fcn, q_rand_fcn, q_pdf_fcn=None, p_logged=True):
    """
    p_pdf_fcn(x) is probability of x occurring
        can be evaluated can for any x
    q_rand_fcn samples from q_pdf_fcn given x
        q_rand_fcn(x) provides the next guess, given current position at x
    q_pdf_fcn is proposal density
        q_pdf_fcn(x, xstar) is probability of returning to x given current position at xstar
        if q_pdf_fcn is None
            assumes pdf from which q_rand_fcn samples is symmetric, e.g. gaussian
    p_logged is bool, specifying whether p_pdf_fcn has been logged

    n.b. must prune the returned samples to obtain independent sequence!
    n.b. this is random-walk hastings since q_pdf_fcn is dependent upon current position
    """
    x = x0
    xs = [x]
    px = p_pdf_fcn(x)
    c = 0
    a_fcn = make_mh_rule(q_pdf_fcn, p_logged)
    for r in np.random.uniform(size=n):
        xstar = q_rand_fcn(x)
        pxstar = p_pdf_fcn(xstar)
        if r < a_fcn(x, px, xstar, pxstar):
            x = xstar
            px = pxstar
            c += 1
        xs.append(x)
    if c == 0:
        raise Exception("Never moved! Initial guess of {0} is probably too unlikely. Or, try logging your p_pdf_fcn.".format(x0))
    return np.array(xs)

def example_1():
    a, c = 3, 1
    xs = np.linspace(exponweib.ppf(0.01, a, c), exponweib.ppf(0.99, a, c), 100)
    l = round(max(xs) - min(xs))
    pf = lambda x: exponweib.pdf(x, a, c)

    e = 1 # step size of random-walk
    qrf = lambda x: norm.rvs(x, e)
    xhs0 = metropolis_hastings(3, 100000, pf, qrf, None, False)
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
