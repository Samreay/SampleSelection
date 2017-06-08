import numpy as np
import emcee
from scipy.stats import norm
from chainconsumer import ChainConsumer
from joblib import Parallel, delayed


mu, sigma, num_obs = 100, 10, 100  # Our model parameters
ndim, nwalkers = 2, 10  # emcee Parameters
num_realisations, num_cores = 8, 4  # Number of realisations to generate and number of cpu cores to use


def lnprob(theta, data):
    mu, sigma = theta
    if sigma < 0:
        return -np.inf
    return norm.logpdf(data, mu, sigma).sum()


def run_realisation(i):
    print("Running realisation %d" % i)
    np.random.seed(i)
    x = np.random.normal(loc=mu, scale=sigma, size=num_obs)  # Our observations

    # Get initial values and sample the posteriors
    p0 = [[np.random.normal(mu, 10), np.random.normal(sigma, 2)] for j in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[x])
    sampler.run_mcmc(p0, 2000)
    return sampler.chain[:, 100:, :].reshape((-1, ndim))  # Remove burnin, reshape and return

# Launch our jobs and collate the results
res = Parallel(n_jobs=num_cores, backend="threading")(delayed(run_realisation)(i) for i in range(num_realisations))
all_samples = np.vstack(res)

print("Generating plot")
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$"])
c.configure(flip=False, sigmas=[0, 1, 2], summary=False)
c.plotter.plot(filename="perfect.pdf", figsize="column", truth=[mu, sigma])
