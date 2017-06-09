import numpy as np
import emcee
from scipy.stats import norm
from scipy.special import erfc
from joblib import Parallel, delayed
from chainconsumer import ChainConsumer

# Our model parameters
mux, sigmax, alpha = 100, 10, 92
muy, sigmay, beta = 30, 5, 0.2
ndim, nwalkers = 3, 10
epsilon, num_obs = 4, 100
num_realisations, num_cores = 16, 4
s2 = np.sqrt(2)  # So I dont have to type this out a lot

# Set num_realisations to 100 to produce the paper plot, and increase MC samples to 100k
# To speed up the code I also fix sigma_y instead of fitting it


def lnprob_no_correction(theta, xs, ys):
    mux, sigmax, muy = theta
    if sigmax < 0:
        return -np.inf
    return np.sum(norm.logpdf(xs, mux, sigmax) + norm.logpdf(ys, muy, sigmay))


def lnprob_approx_correction(theta, xs, ys):
    mux, sigmax, muy = theta
    if sigmax < 0:
        return -np.inf
    if mux < alpha:
        return -np.inf
    return np.sum(norm.logpdf(xs, mux, sigmax) + norm.logpdf(ys, muy, sigmay) - np.log(0.5 * erfc((alpha - mux - epsilon)/(s2 * sigmax))))


def get_data(mux, sigmax, muy, n=1000):
    x = np.random.normal(loc=mux, scale=sigmax, size=n)
    y = np.random.normal(loc=muy, scale=sigmay, size=n)
    mask = (x + beta * y) > alpha
    return x, y, mask


def reweight(mux, sigmax, muy):
    # Calculate w_approx. Better to store this when you calculate it the first time
    original_weight = 0.5 * erfc((alpha - mux - epsilon)/(s2 * sigmax))

    # Using Monte Carlo integration of ten thousand points, use 100k to mimic paper if you like waiting
    x, y, mask = get_data(mux, sigmax, muy, n=10000)
    new_weight = mask.sum() * 1.0 / mask.size

    # Calculate $\mathcal{L}_{i2}$
    diff = np.log(original_weight) - np.log(new_weight)    
    return num_obs * diff


def run_realisation(i):
    print("Running realisation %d" % i)
    np.random.seed(i)

    # Get our data
    x, y, mask = get_data(mux, sigmax, muy)
    x = x[mask][:num_obs]
    y = y[mask][:num_obs]

    # Setup emcee and sample both posteriors
    p0 = [[np.random.normal(mux, 10), np.random.normal(sigmax, 2), np.random.normal(muy, 2)] for j in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_no_correction, args=[x, y])
    sampler.run_mcmc(p0, 2000)
    print("Finishing 1st sampling for realisation %d" % i)

    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob_approx_correction, args=[x, y])
    sampler2.run_mcmc(p0, 2000)
    print("Finishing 2nd sampling for realisation %d" % i)

    # Discard burn in and redshape our chains
    chain1 = sampler.chain[:, 100:, :].reshape((-1, ndim))
    chain2 = sampler2.chain[:, 100:, :].reshape((-1, ndim))

    # Get the weights for each sample in our approximately corrected chain
    weights = np.array([reweight(*row) for row in chain2])
    weights -= weights.max()  # As we are in log space, renormalise and convert back to real space
    weights = np.exp(weights)
    print("Finishing reweighting for realisation %d" % i)
    return chain1, chain2, weights


# Launch our jobs and collate the results
res = Parallel(n_jobs=num_cores, backend="threading")(delayed(run_realisation)(i) for i in range(num_realisations))
all_samples = np.vstack([r[0] for r in res])
all_samples_corrected = np.vstack([r[1] for r in res])
weights = np.array([r[2] for r in res]).flatten()

print("Generating plot for %d realisations" % num_realisations)
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$", r"$\mu_y$"], name="Biased")
c.add_chain(all_samples_corrected, name="Approximate")
c.add_chain(all_samples_corrected, weights=weights, name="Corrected")
c.configure(flip=False, sigmas=[0, 1, 2], colors=["#D32F2F", "#4CAF50", "#222222"],
            linestyles=[":", "--", "-"], shade_alpha=0.2, shade=True, diagonal_tick_labels=False)
c.plotter.plot(filename="../paper/fig_2_real.pdf", figsize="column", truth=[mux, sigmax], extents=[[90, 105], [6, 15]], parameters=2)
c.plotter.plot(filename="../plots/fig_2_real.png", figsize="column", truth=[mux, sigmax], extents=[[90, 105], [6, 15]], parameters=2)
