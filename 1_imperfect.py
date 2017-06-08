import numpy as np
import emcee
from chainconsumer import ChainConsumer
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.special import erfc

mu, sigma, alpha, num_obs = 100, 10, 85, 100   # Our model parameters
ndim, nwalkers = 2, 10   # emcee Parameters
num_realisations, num_cores = 8, 4  # Number of realisations to generate and number of cpu cores to use


def lnprob_no_correction(theta, data):
    mu, sigma = theta
    if sigma < 0:
        return -np.inf
    return norm.logpdf(data, mu, sigma).sum()


def lnprob_corrected(theta, data):
    mu, sigma = theta
    if sigma < 0:
        return -np.inf
    if mu < alpha:
        return -np.inf
    return np.sum(norm.logpdf(data, mu, sigma) - np.log(0.5 * erfc((alpha - mu)/(np.sqrt(2) * sigma))))


def run_realisation(i):
    print("Running realisation %d" % i)
    np.random.seed(i)
    x = np.random.normal(loc=mu, scale=sigma, size=num_obs * 2)  # Oversample because we have to cut some out
    x = x[x > alpha][:num_obs]  # Cut out values less than our threshold and get the right number of obs

    # Get initial values and sample both posteriors
    p0 = [[np.random.normal(mu, 10), np.random.normal(sigma, 2)] for j in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_no_correction, args=[x])
    sampler.run_mcmc(p0, 2000)
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob_corrected, args=[x])
    sampler2.run_mcmc(p0, 2000)

    # Remove burnin, flatten, and return both sets of samples
    return sampler.chain[:, 100:, :].reshape((-1, ndim)), sampler2.chain[:, 100:, :].reshape((-1, ndim))

# Launch our jobs and collate the results
res = Parallel(n_jobs=num_cores, backend="threading")(delayed(run_realisation)(i) for i in range(num_realisations))
all_samples = np.vstack([r[0] for r in res])
all_sampels_corrected = np.vstack([r[1] for r in res])

print("Generating plot for %d realisations" % num_realisations)
c = ChainConsumer()
c.add_chain(all_samples, parameters=[r"$\mu$", r"$\sigma$"], name="Biased")
c.add_chain(all_sampels_corrected, name="Corrected")
c.configure(flip=False, sigmas=[0, 1, 2], colors=["#D32F2F", "#4CAF50"], linestyles=["-", "--"], shade_alpha=0.2)
c.plot(filename="img_1_imperfect.pdf", figsize="column", truth=[mu, sigma], extents=[[95, 105], [7, 14]])
c.plot(filename="img_1_imperfect.png", figsize="column", truth=[mu, sigma], extents=[[95, 105], [7, 14]])
