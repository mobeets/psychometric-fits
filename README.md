### Metropolis-Hastings sampling

_Metropolis-Hastings_ sampling (M-H) draws samples from a posterior given a _posterior-ish function_ (e.g. the unnormalized posterior given some measured data) and a _proposal function_. M-H is provided with an initial draw from the posterior and aims to generate a series of samples by moving from that initial draw to the next, following a certain rule. (More details can be found [here](http://www.journalofvision.org/content/5/5/8.short).)

The proposal function describes how to generate the next potential sample. (If the proposal function is symmetric this sampling procedure is sometimes called _Metropolis samping_.)

The posterior-ish function is used to determine whether or not the next potential sample is in a region of higher posterior probability compared witht the last sample.

#### Example 1

The function `example_1()` function in `metrop_hastings.py` currently uses a gaussian proposal function to draw samples from a weibull pdf with given shape and scale parameters. As you can see, the samples generated by `metropolis_hastings()` (after pruning) are a close match to the actual weibull pdf:

![Example of posterior samples from M-H](/img/example-1.png?raw=true "Example of posterior samples from M-H")

#### Example 2

But `example_1()` is more like a sanity-check, isn't it? If we hand M-H the pdf, it can generate samples from that pdf--not that impressive.

On the other hand, `example_2()` is a little closer to what we'd want M-H to do. It simulates data from a weibull pdf given shape and scale parameters, and M-H tries to generate samples of (i.e. fit) the shape parameter, by calculating the log-likelihood of the simulated data.

So now, the generated samples are all estimates of the shape parameter. The shape parameter used to simulate the data was 3. The estimates cluster around 3, which is good, though this is not always the case given multiple simulations.

![Example of posterior samples from M-H](/img/example-2.png?raw=true "Example of posterior samples from M-H")
