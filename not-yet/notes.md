Good overview # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.7133&rep=rep1&type=pdf

MCMC
    In particular, it is constructed so that the samples x(i) mimic samples drawn from the target distribution p(x).

M-H
    Two simple instances of the MH algorithm:
        * independent sampler: proposal is independent of current state
        * Metropolis algorithm: random-walk is symmetric

    If the proposal is too narrow, only one mode of p(x) might be visited. On the other hand, if it is too wide, the rejection rate can be very high, resulting in high correlations.

Simulated annealing # http://stuff.mit.edu/~dbertsim/papers/Optimization/Simulated%20annealing.pdf
    Let us assume that instead of wanting to approximate p(x), we want to ﬁnd its global maximum

    If using MCMC to do this, this method is inefﬁcient because the random samples only rarely come from the vicinity of the mode. Unless the distribution has large probability mass around the mode, computing resources will be wasted exploring areas of no interest.

    This technique involves simulating a non-homogeneous Markov chain whose invariant distribution at iteration i is no longer equal to p(x), but to:
        p_i(x) ∝ p^(1/T_i(x))

    Just like M-H only your rule is changed:
        * T_0 = 1, and T_i+1 is set "according to a chosen cooling schedule" s.t. T_∞ = 0.
            * Ti = (C ln(i + T_0)) − 1 for some C
        * instead of p(x*)/p(x), you have p(x)^(1/T_i)/p(x*)^(1/T_i)
            * or, instead of exp(p(x*) - p(x)), you have exp((p(x*) - p(x))/T_i)
