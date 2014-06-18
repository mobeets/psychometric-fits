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
    return xs[np.arange(0, len(xs), T)]

def prune_2(xs, nburn, nskip):
    """
    burns the first `nburn` samples, then skips every `nskip` samples after that
    """
    return xs[np.arange(nburn, len(xs), nskip)]

def metropolis_hastings(x0, n, p_pdf_fcn, q_rand_fcn, q_pdf_fcn=None):
    """
    p_pdf_fcn(x) is probability of x occurring
        can be evaluated can for any x
    q_pdf_fcn is proposal density
        q_pdf_fcn(x, xstar) is probability of returning to x given current position at xstar
        if q_pdf_fcn is None
            assumes pdf from which q_rand_fcn samples is symmetric, e.g. gaussian
    q_rand_fcn samples from q_pdf_fcn given x
        q_rand_fcn(x) provides the next guess, given current position at x
    n.b. must prune the returned samples to obtain independent sequence!
    """
    x = x0
    xs = [x]
    p_old = p_pdf_fcn(x)
    if q_pdf_fcn is None:
        a_fcn = lambda x, psx, xstar, psxs: (psxs / psx)
    else:
        a_fcn = lambda x, psx, xstar, psxs: (psxs / psx) * (q_fcn(x, xstar) / q_fcn(xstar, x))
    rnds = np.random.uniform(size=n)
    for r in rnds:
        xstar = q_rand_fcn(x)
        pstar = p_pdf_fcn(xstar)
        if r < a_fcn(x, p_old, xstar, pstar):
            x = xstar
            p_old = pstar
        xs.append(x)
    return np.array(xs)

def example():
    a, c = 3, 1
    xs = np.linspace(exponweib.ppf(0.01, a, c), exponweib.ppf(0.99, a, c), 100)
    l = round(max(xs) - min(xs))
    pf = lambda x: exponweib.pdf(x, a, c)

    e = 1 # step size of random-walk
    qrf = lambda x: norm.rvs(x, e)
    xhs = metropolis_hastings(3, 100000, pf, qrf)
    xhs = prune(xhs, l, e)

    plt.plot(xs, pf(xs))
    plt.hist(xhs, 100, normed=True)
    plt.show()

def main():
    a, c = 3, 1
    data = exponweib.rvs(a, c, size=100)
    l = round(max(data) - min(data))
    pf = lambda ah, ch=c, d=data: np.sum([np.log(exponweib.pdf(d, ah, ch)) for d in data])

    e = 1 # step size of random-walk
    qrf = lambda x: norm.rvs(x, e)
    ahs = metropolis_hastings(3, 10000, pf, qrf)
    ahs = prune(ahs, l, e)

    plt.hist(ahs, 100, normed=True)
    plt.show()

if __name__ == '__main__':
    example()
    # main()
